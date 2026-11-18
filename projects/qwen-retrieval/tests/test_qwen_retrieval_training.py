"""CPU contract coverage; the separate smoke adds real four-GPU ZeRO-2 checks."""
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    import transformers
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Pinned Torch/Transformers runtime required")
class RetrievalTrainingTests(unittest.TestCase):
    def test_model_collator_and_official_trainer_contract(self):
        from qwen_retrieval.smoke import cpu_checks
        with tempfile.TemporaryDirectory(prefix="qwen-retrieval-cpu-") as directory:
            report = cpu_checks(Path(directory))
        self.assertTrue(all(report.values()))


if __name__ == "__main__":
    unittest.main()
