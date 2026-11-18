"""Data-contract checks for fixed Qwen features; no Torch or training required."""

import csv
import errno
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("prepare_qwen_data", Path(__file__).resolve().parents[1] / "prepare_qwen_data.py")
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


def save_npy(path, rows, float_bytes=4):
    header = repr({"descr": "<f" + str(float_bytes), "fortran_order": False, "shape": (len(rows), len(rows[0]))}).encode("ascii")
    header += b" " * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b"\n"
    values = [value for row in rows for value in row]
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<" + ("f" if float_bytes == 4 else "e") * len(values), *values))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "prepared"
        self.source.mkdir()
        self.dataset = "Tiny"
        self.embedding = self.source / "Tiny.emb-qwen-td.npy"
        save_npy(self.embedding, [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [0.0, 2.0, 0.0], [3.0, 2.0, 1.0]])
        self.write_json("Tiny.emb-qwen-td.npy.ids.json", ["2", "0", "3", "1"])
        self.write_json("Tiny.item.json", {str(i): {"title": "Item " + str(i)} for i in range(4)})
        self.write_json("Tiny.index.json", {str(i): [f"<a_{i}>", "<b_0>", "<c_0>"] for i in range(4)})
        (self.source / "Tiny.item2id").write_text("".join(f"ASIN{i}\t{i}\n" for i in range(4)), encoding="utf-8")
        self.write_split("train", [("u0", "0 2", "0"), ("u0", "0 2", "0")])
        self.write_split("valid", [("u1", "0", "1")])
        self.write_split("test", [("u2", "1", "3")])
        self.update_generation_manifest()

    def write_json(self, name, value):
        prepare_data.write_json(self.source / name, value)

    def write_split(self, split, rows):
        with (self.source / f"Tiny.{split}.inter").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(prepare_data.FIELDS)
            writer.writerows(rows)

    def update_generation_manifest(self, dtype="float32"):
        self.write_json("Tiny.emb-qwen-td.npy.manifest.json", {"shape": [4, 3], "dtype": dtype,
                        "embedding_sha256": prepare_data.sha256_file(self.embedding),
                        "item_metadata_sha256": prepare_data.sha256_file(self.source / "Tiny.item.json")})

    def run_prepare(self, **kwargs):
        return prepare_data.prepare(self.source, self.output, dataset=self.dataset, embedding_dim=3, **kwargs)

    def test_preserves_original_zero_ids_explicit_row_order_splits_and_duplicate_purchase(self):
        manifest = self.run_prepare()
        catalog = prepare_data.load_json(self.output / "catalog.json")
        self.assertEqual([row["item_id"] for row in catalog], ["2", "0", "3", "1"])
        self.assertEqual(catalog[1], {"embedding_row": 1, "item_id": "0", "asin": "ASIN0",
                         "sid_tokens": ["<a_0>", "<b_0>", "<c_0>"], "sid": "<a_0><b_0><c_0>"})
        self.assertEqual(prepare_data.load_json(self.output / "train_item_ids.json"), ["2", "0"])
        self.assertEqual((self.output / "Tiny.item").read_text(), "item_id:token\n2\n0\n3\n1\n")
        self.assertTrue((self.output / self.embedding.name).is_symlink())
        self.assertTrue((self.output / (self.embedding.name + ".ids.json")).is_symlink())
        self.assertEqual(manifest["splits"]["train"]["within_split_duplicate_rows_preserved"], 1)
        self.assertEqual(manifest["splits"]["train"]["target_already_in_history_rows"], 2)
        self.assertFalse(manifest["training_started"])
        for split in prepare_data.SPLITS:
            self.assertEqual((self.output / f"Tiny.{split}.inter").read_bytes(), (self.source / f"Tiny.{split}.inter").read_bytes())

    def test_float16_inference_missing_provenance_and_existing_sid_collisions(self):
        save_npy(self.embedding, [[1.0, 0.0, 0.5]] * 4, float_bytes=2)
        (self.source / "Tiny.emb-qwen-td.npy.ids.json").unlink()
        (self.source / "Tiny.emb-qwen-td.npy.manifest.json").unlink()
        (self.source / "Tiny.item2id").unlink()
        sids = prepare_data.load_json(self.source / "Tiny.index.json")
        sids["1"] = sids["0"]
        self.write_json("Tiny.index.json", sids)
        manifest = self.run_prepare()
        self.assertEqual(manifest["embedding"]["dtype"], "float16")
        self.assertFalse(manifest["embedding"]["generation_manifest_available"])
        self.assertFalse(manifest["embedding"]["model_identity_verified"])
        self.assertFalse(manifest["embedding"]["row_order_provenance_verified"])
        self.assertEqual(prepare_data.load_json(self.output / "Tiny.emb-qwen-td.npy.ids.json"), ["0", "1", "2", "3"])
        self.assertEqual(manifest["sid_collisions"]["group_count"], 1)
        self.assertEqual(manifest["sid_collisions"]["affected_items"], 2)
        self.assertIsNone(prepare_data.load_json(self.output / "catalog.json")[0]["asin"])

    def test_unsupported_symlinks_fall_back_to_identical_copies(self):
        with mock.patch.object(Path, "symlink_to", side_effect=OSError(errno.EOPNOTSUPP, "Operation not supported")):
            manifest = self.run_prepare()
        self.assertEqual(manifest["embedding"]["storage_mode"], "copy")
        self.assertEqual(manifest["embedding"]["row_ids_storage_mode"], "copy")
        self.assertFalse((self.output / self.embedding.name).is_symlink())
        self.assertEqual((self.output / self.embedding.name).read_bytes(), self.embedding.read_bytes())
        self.assertEqual((self.output / (self.embedding.name + ".ids.json")).read_bytes(),
                         (self.source / (self.embedding.name + ".ids.json")).read_bytes())

    def test_duplicate_row_ids_are_rejected(self):
        self.write_json("Tiny.emb-qwen-td.npy.ids.json", [0, 0, 2, 3])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_unknown_interaction_item_is_rejected_without_publishing_output(self):
        self.write_split("test", [("u2", "1 99", "3")])
        with self.assertRaisesRegex(ValueError, "unknown item"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_cross_split_duplicate_samples_are_rejected(self):
        self.write_split("valid", [("u0", "0 2", "0")])
        with self.assertRaisesRegex(ValueError, "different splits"):
            self.run_prepare()

    def test_nonfinite_and_zero_vectors_are_rejected(self):
        for invalid, message in ((float("nan"), "non-finite"), (float("inf"), "non-finite"), (0.0, "zero vectors")):
            with self.subTest(value=invalid):
                save_npy(self.embedding, [[invalid] * 3] * 4)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_prepare()

    def test_generation_hash_mismatch_is_rejected(self):
        save_npy(self.embedding, [[2.0, 2.0, 3.0]] * 4)
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.run_prepare()

    def test_sid_coverage_mismatch_is_rejected(self):
        self.write_json("Tiny.index.json", {"0": ["<a_0>"]})
        with self.assertRaisesRegex(ValueError, "cover the entire"):
            self.run_prepare()

    def test_shape_payload_mismatch_is_rejected(self):
        self.embedding.write_bytes(self.embedding.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "payload size"):
            self.run_prepare()

    def test_history_truncation_and_training_visible_ids(self):
        self.write_split("train", [("u0", "3 1 0 2", "0")])
        manifest = self.run_prepare(max_history_len=2)
        self.assertEqual((self.output / "Tiny.train.inter").read_text().splitlines()[1], "u0\t0 2\t0")
        self.assertEqual(manifest["train_item_count"], 4)


if __name__ == "__main__":
    unittest.main()
