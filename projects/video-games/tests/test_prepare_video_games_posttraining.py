"""CPU-only fixtures for the epoch-2900 post-training data adapter."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_video_games_posttraining', ROOT / 'scripts/prepare_video_games_posttraining.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class PrepareVideoGamesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'input'
        self.source.mkdir()
        self.output = self.base / 'output'
        self.sids = {
            '0': ['<a_1>', '<b_2>', '<c_3>', '<d_0>'],
            '1': ['<a_1>', '<b_2>', '<c_3>', '<d_1>'],
            '2': ['<a_4>', '<b_5>', '<c_6>', '<d_0>'],
        }
        self.items = {'0': {'title': 'Pad "A"'}, '1': {'title': 'Mouse\twith\nlines'}, '2': {'title': "Game's edition"}}
        self.write_catalog()
        self.write_inter('train', ['7\t0\t1', '7\t0 1\t2', '8\t2\t0'])
        self.write_inter('valid', ['7\t0 1 2\t0'])
        self.write_inter('test', ['7\t0 1 2 0\t1'])

    def write_catalog(self):
        path = self.source / 'Video_Games.index.json'
        path.write_text(json.dumps(self.sids), encoding='utf-8')
        (self.source / 'Video_Games.item.json').write_text(json.dumps(self.items), encoding='utf-8')
        self.expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    def write_inter(self, split, rows, header=None):
        text = (header or '\t'.join(adapter.INTER_HEADER)) + '\n' + '\n'.join(rows) + '\n'
        (self.source / f'Video_Games.{split}.inter').write_text(text, encoding='utf-8')

    def prepare(self, check_only=False):
        return adapter.prepare(self.source, self.output, self.expected_hash, check_only)

    def test_preserves_rows_splits_four_tokens_and_list_literals(self):
        manifest = self.prepare()
        self.assertEqual(manifest['rows'], {'train': 3, 'valid': 1, 'test': 1})
        self.assertEqual(manifest['selected_sid_epoch'], 2900)
        with (self.output / 'train.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['user_id'] for row in rows], ['A7', 'A7', 'A8'])
        self.assertEqual(ast.literal_eval(rows[1]['history_item_id']), [0, 1])
        self.assertEqual(ast.literal_eval(rows[1]['history_item_title']), [self.items['0']['title'], self.items['1']['title']])
        self.assertEqual(ast.literal_eval(rows[1]['history_item_sid']), [''.join(self.sids['0']), ''.join(self.sids['1'])])
        self.assertEqual(rows[0]['item_sid'], ''.join(self.sids['1']))
        self.assertTrue(rows[0]['item_sid'].endswith('<d_1>'))
        info_rows = [line.split('\t') for line in (self.output / 'info.txt').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(info_rows), 3)
        self.assertTrue(all(len(row) == 3 for row in info_rows))
        self.assertEqual(info_rows[1][1], 'Mouse with lines')
        for name, record in manifest['outputs'].items():
            self.assertEqual(adapter.sha256(self.output / name), record['sha256'])

    def test_check_only_writes_nothing(self):
        manifest = self.prepare(check_only=True)
        self.assertTrue(manifest['check_only'])
        self.assertEqual(manifest['rows']['train'], 3)
        self.assertFalse(self.output.exists())

    def test_default_rejects_unapproved_sid_before_writing(self):
        with self.assertRaisesRegex(ValueError, 'SID hash mismatch'):
            adapter.prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_complete_sid_is_rejected(self):
        self.sids['1'] = self.sids['0'][:]
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, 'must be unique'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_token_or_catalog_id_is_rejected(self):
        self.sids['1'][0] = '<a_256>'
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.prepare()
        self.sids['1'][0] = '<a_1>'
        self.sids['3'] = self.sids.pop('2')
        self.items['3'] = self.items.pop('2')
        self.write_catalog()
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            self.prepare()

    def test_missing_target_or_history_mapping_rejected(self):
        self.write_inter('train', ['7\t0 1\t99'])
        with self.assertRaisesRegex(ValueError, 'target item ID is not in'):
            self.prepare()
        self.write_inter('train', ['7\t99\t0'])
        with self.assertRaisesRegex(ValueError, 'unmapped'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_history_bounds_and_bad_rows_rejected(self):
        for row in ['7\t\t0', '7\t' + ' '.join(['0'] * 21) + '\t1', '7\t0', '7\t0\t1\textra', '7\t0 x\t1']:
            with self.subTest(row=row):
                self.write_inter('train', [row])
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_all_splits_checked_before_output_is_touched(self):
        self.output.mkdir()
        existing = self.output / 'train.csv'
        existing.write_text('previous valid data', encoding='utf-8')
        self.write_inter('test', ['7\t99\t0'])
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(existing.read_text(encoding='utf-8'), 'previous valid data')
        self.assertFalse((self.output / 'valid.csv').exists())

    def test_header_and_missing_split_rejected(self):
        self.write_inter('valid', ['7\t0\t1'], header='user\thistory\ttarget')
        with self.assertRaisesRegex(ValueError, 'header'):
            self.prepare()
        self.write_inter('valid', ['7\t0\t1'])
        (self.source / 'Video_Games.test.inter').unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_csv_outputs_are_deterministic(self):
        first = self.prepare()['outputs']
        second = self.prepare()['outputs']
        self.assertEqual(first, second)
        self.assertFalse(list(self.base.glob('.video-games-sid2900-*')))


if __name__ == '__main__':
    unittest.main()
