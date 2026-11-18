"""Dependency-free checks for the real RL entry functions using CPU-only stubs.

No framework import, accelerator allocation, model download, or training occurs.
"""
import ast
import contextlib
import io
import math
from pathlib import Path
import posixpath
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _extract(tree, name, namespace):
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, '<actual-function:' + name + '>', 'exec'), namespace)
    return namespace[name]


class _Cuda:
    def __init__(self, events, available=True):
        self.events = events
        self.available = available

    def is_available(self):
        return self.available

    def device_count(self):
        return 4

    def set_device(self, rank):
        self.events.append(('set_device', rank))


class _Samples:
    def __init__(self):
        self.rows = [{'prompt': 'prompt'}]
        self.prompt2history = {'prompt': 'history'}
        self.history2target = {'history': 'sid'}

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


class _Concat:
    def __init__(self, groups):
        self.rows = [item for group in groups for item in group]

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


class _HFData:
    @classmethod
    def from_dict(cls, value):
        return cls()

    def shuffle(self, seed):
        return self

    def select(self, indices):
        return self

    def __len__(self):
        return 3


class _Fixture:
    """Run the actual train() body with every external operation replaced."""
    def __init__(self, rl_tree, validate, rank=0):
        self.events = events = []
        self.captured = captured = {}
        self.rank = rank
        fake_os = types.SimpleNamespace(
            environ={'LOCAL_RANK': str(rank), 'WORLD_SIZE': '4'}, path=posixpath
        )

        def tensor(value, device=None):
            events.append(('reward_tensor', device))
            return object()

        def load(path, map_location=None):
            events.append(('reward_load', path, map_location))
            return {}

        class Tokenizer:
            def save_pretrained(self, path):
                events.append(('tokenizer_save', path))

        class Tokenizers:
            @classmethod
            def from_pretrained(cls, path):
                events.append(('tokenizer_load', path))
                return Tokenizer()

        class Trainer:
            def __init__(self, **kwargs):
                captured['trainer'] = kwargs
                self.accelerator = types.SimpleNamespace(
                    wait_for_everyone=lambda: events.append(('barrier',))
                )

            def train(self):
                events.append(('stub_train',))

            def save_model(self, path):
                events.append(('trainer_save', path))

            def is_world_process_zero(self):
                return rank == 0

        class CF:
            def __init__(self, *args):
                events.append(('cf_init', args[-1]))

            def to(self, device):
                events.append(('cf_to', device))
                return self

            def load_state_dict(self, state):
                events.append(('cf_state',))

            def eval(self):
                return self

        def config(**kwargs):
            captured['config'] = kwargs
            return types.SimpleNamespace(**kwargs)

        def samples(*args, **kwargs):
            events.append(('dataset_category', kwargs.get('category')))
            return _Samples()

        def fake_open(path, mode='r'):
            return io.BytesIO(b'fake') if 'b' in mode else io.StringIO('sid\tItem title\t0\n')

        namespace = dict(
            os=fake_os,
            torch=types.SimpleNamespace(
                cuda=_Cuda(events),
                device=lambda kind, index=None: kind if index is None else f'{kind}:{index}',
                as_tensor=tensor, load=load,
            ),
            npu_support=types.SimpleNamespace(IS_NPU=False),
            validate_generation_batch_sizes=validate,
            set_seed=lambda seed: events.append(('seed', seed)),
            open=fake_open, Dataset=_HFData, SidDataset=samples,
            RLTitle2SidDataset=samples, RLSeqTitle2SidDataset=samples,
            ConcatDataset=_Concat, AutoTokenizer=Tokenizers, SASRec=CF,
            pickle=types.SimpleNamespace(load=lambda stream: [[1.]]), math=math,
            GRPOConfig=config, ReReTrainer=Trainer,
        )
        # There is deliberately no AutoModelForCausalLM stub: an extra model load fails.
        _extract(rl_tree, '_local_training_device', namespace)
        self.train = _extract(rl_tree, 'train', namespace)

    def run(self, **overrides):
        kwargs = dict(
            model_path='fake-sft-checkpoint', category='Video_Games',
            info_file='fake-info', train_file='fake-train', eval_file='fake-eval',
            output_dir='fake-output', train_batch_size=16, eval_batch_size=16,
            num_generations=16, gradient_accumulation_steps=16, beam_search=True,
            reward_type='ranking', ada_path='fake-embeddings', cf_path='fake-cf',
        )
        kwargs.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            self.train(**kwargs)
        return self


class TestRLDistributed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rl_tree = ast.parse((ROOT / 'rl.py').read_text(encoding='utf-8'))
        cls.trainer_tree = ast.parse((ROOT / 'minionerec_trainer.py').read_text(encoding='utf-8'))
        compile(cls.rl_tree, 'rl.py', 'exec')
        compile(cls.trainer_tree, 'minionerec_trainer.py', 'exec')
        cls.validate = staticmethod(_extract(cls.trainer_tree, 'validate_generation_batch_sizes', {}))

    def test_valid_complete_rollout_groups(self):
        cases = [(16, 16, 16, 4, True), (32, 16, 16, 4, True),
                 (64, 128, 16, 8, True), (4, 4, 16, 4, False)]
        for train, evaluation, generations, world, beam in cases:
            with self.subTest(case=(train, evaluation, generations, world, beam)):
                self.validate(train, evaluation, generations, world, beam_search=beam)
        self.validate(16, 0, 16, 4, beam_search=True, eval_enabled=False)

    def test_invalid_or_split_rank_local_groups(self):
        cases = [(4, 16, 16, 4, True), (16, 4, 16, 4, True),
                 (3, 16, 16, 4, False), (16, 16, 1, 4, True),
                 (0, 16, 16, 4, True), (16, 16, 16, 0, True)]
        for train, evaluation, generations, world, beam in cases:
            with self.subTest(case=(train, evaluation, generations, world, beam)):
                with self.assertRaises(ValueError):
                    self.validate(train, evaluation, generations, world, beam_search=beam)

    def test_invalid_batch_fails_before_device_or_data_loading(self):
        fixture = _Fixture(self.rl_tree, self.validate, rank=2)
        with self.assertRaisesRegex(ValueError, 'per-device train batch'):
            fixture.run(train_batch_size=4)
        self.assertEqual(fixture.events, [])
        with self.assertRaisesRegex(ValueError, 'gradient_accumulation_steps'):
            fixture.run(gradient_accumulation_steps=0)
        self.assertEqual(fixture.events, [])

    def test_bf16_policy_and_checkpointing_without_extra_llm(self):
        fixture = _Fixture(self.rl_tree, self.validate).run(gradient_checkpointing=True)
        config = fixture.captured['config']
        self.assertEqual(config['model_init_kwargs'],
                         {'torch_dtype': 'bfloat16', 'attn_implementation': 'sdpa'})
        self.assertFalse(config['use_vllm'])
        self.assertTrue(config['gradient_checkpointing'])
        self.assertEqual(config['gradient_checkpointing_kwargs'], {'use_reentrant': False})
        self.assertEqual(fixture.captured['trainer']['model'], 'fake-sft-checkpoint')
        self.assertTrue(all(event[1] == 'video games' for event in fixture.events
                            if event[0] == 'dataset_category'))
        categories = [event for event in fixture.events if event[0] == 'dataset_category']
        self.assertEqual(len(categories), 4)

    def test_non_main_rank_binding_and_safe_save(self):
        for rank in (0, 2):
            with self.subTest(rank=rank):
                fixture = _Fixture(self.rl_tree, self.validate, rank).run(attn_implementation='auto')
                events = fixture.events
                self.assertLess(events.index(('set_device', rank)),
                                events.index(('tokenizer_load', 'fake-sft-checkpoint')))
                self.assertNotIn('attn_implementation', fixture.captured['config']['model_init_kwargs'])
                self.assertFalse(fixture.captured['config']['gradient_checkpointing'])
                self.assertEqual([event for event in events if event[0] == 'trainer_save'],
                                 [('trainer_save', 'fake-output'),
                                  ('trainer_save', 'fake-output/final_checkpoint')])
                self.assertEqual(sum(event[0] == 'tokenizer_save' for event in events), int(rank == 0))
                self.assertEqual(sum(event[0] == 'barrier' for event in events), 2)

    def test_optional_rewards_use_local_rank_device(self):
        semantic = _Fixture(self.rl_tree, self.validate, rank=2).run(reward_type='semantic')
        self.assertIn(('reward_tensor', 'cuda:2'), semantic.events)
        sasrec = _Fixture(self.rl_tree, self.validate, rank=2).run(reward_type='sasrec')
        self.assertIn(('reward_load', 'fake-cf', 'cuda:2'), sasrec.events)
        self.assertIn(('cf_to', 'cuda:2'), sasrec.events)

    def test_device_fallbacks_and_invalid_local_rank(self):
        for available, rank, is_npu, expected in (
                (False, 0, False, 'cpu'), (True, 3, True, 'npu:3')):
            namespace = dict(
                os=types.SimpleNamespace(environ={'LOCAL_RANK': str(rank)}),
                torch=types.SimpleNamespace(cuda=_Cuda([], available),
                    device=lambda kind, index=None: kind if index is None else f'{kind}:{index}'),
                npu_support=types.SimpleNamespace(IS_NPU=is_npu),
            )
            with self.subTest(available=available, rank=rank, is_npu=is_npu):
                self.assertEqual(_extract(self.rl_tree, '_local_training_device', namespace)(), expected)
        namespace['os'].environ['LOCAL_RANK'] = '4'
        with self.assertRaisesRegex(ValueError, 'LOCAL_RANK'):
            _extract(self.rl_tree, '_local_training_device', namespace)()

    def test_trainer_also_calls_shared_rollout_guard(self):
        trainer_class = next(node for node in self.trainer_tree.body
                             if isinstance(node, ast.ClassDef) and node.name == 'ReReTrainer')
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == 'validate_generation_batch_sizes'
                            for node in ast.walk(trainer_class)))


if __name__ == '__main__':
    unittest.main()
