"""Regressions for invalid recommendation counts in the actual calc.gao function."""

import ast
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_gao():
    # The metric calculation does not need the CLI's optional fire/pandas imports.
    tree = ast.parse((ROOT / "calc.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "gao")
    scope = {"np": np, "math": math, "tqdm": lambda iterable: iterable}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(ROOT / "calc.py"), "exec"), scope)
    return scope["gao"]


class CalcMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.target = "<a_1><b_2><c_3>"
        self.catalog = self.folder / "items.txt"
        self.catalog.write_text(f"{self.target}\tTitle\t1\n", encoding="utf-8")
        self.gao = load_gao()

    def result_file(self, name, predictions):
        result = self.folder / name
        result.write_text(json.dumps([{"output": self.target, "predict": predictions}]),
                          encoding="utf-8")
        return str(result)

    def evaluate(self, paths):
        output = io.StringIO()
        with redirect_stdout(output):
            self.gao(paths, str(self.catalog))
        return output.getvalue().splitlines()

    def test_counts_invalid_predictions_after_first_target_hit(self):
        result = self.result_file("matched.json", [self.target, "<invalid>", self.target])
        lines = self.evaluate(result)
        self.assertEqual(lines[-1], "1")
        # Repeated target hits must retain the first rank for both ranking metrics.
        for prefix in ("HR", "NDCG:"):
            line = next(line for line in lines if line.startswith(prefix))
            values = np.fromstring(line.split("[", 1)[1].rstrip("]"), sep=" ")
            np.testing.assert_allclose(values, [1.0, 1.0])

    def test_resets_invalid_count_for_each_result_file(self):
        first = self.result_file("first.json", ["<invalid-1>", "<invalid-2>", self.target])
        second = self.result_file("second.json", [self.target, self.target, "<invalid-3>"])
        lines = self.evaluate([first, second])
        # The legacy standalone integer immediately follows each HR line.
        counts = [int(lines[index + 1]) for index, line in enumerate(lines)
                  if line.startswith("HR")]
        self.assertEqual(counts, [2, 1])
        self.assertEqual(lines[-1], "1")
        # Ranking metric accumulators must also start fresh for the second file.
        hr_lines = [line for line in lines if line.startswith("HR")]
        second_hr = np.fromstring(hr_lines[1].split("[", 1)[1].rstrip("]"), sep=" ")
        np.testing.assert_allclose(second_hr, [1.0, 1.0])
        ndcg_lines = [line for line in lines if line.startswith("NDCG:")]
        second_ndcg = np.fromstring(ndcg_lines[1].split("[", 1)[1].rstrip("]"), sep=" ")
        np.testing.assert_allclose(second_ndcg, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
