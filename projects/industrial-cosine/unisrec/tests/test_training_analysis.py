"""Analytic baseline/paired checks; no training or GPU."""

from pathlib import Path
import sys

import pytest
np = pytest.importorskip('numpy')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_training_run import baseline_queries, clustered_bootstrap, normalized_rows, score_metrics
from qwen_vector_metrics import evaluate_vectors


def test_batch_score_metrics_match_reference_with_ties():
    candidates = normalized_rows([[1., 0.], [0., 1.], [0., 1.], [-1., 0.]])
    queries = normalized_rows([[1., 0.], [0., 1.]])
    targets = np.array([1, 2])
    actual = score_metrics(queries@candidates.T, targets)
    expected = evaluate_vectors(queries, candidates, targets)['per_row']
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12)


def test_fully_tied_baseline_is_not_perfect():
    scores = np.ones((2, 20))
    result = score_metrics(scores, np.array([0, 5]))
    np.testing.assert_allclose(result['pairwise_accuracy'], .5)
    np.testing.assert_allclose(result['tie_expected_recall@10'], .5)
    np.testing.assert_allclose(result['all_catalog_vector_nll'], np.log(20))
    np.testing.assert_allclose(result['margin_vs_hardest_negative'], 0)


def test_mean_and_last_queries_use_source_history():
    candidates = normalized_rows([[1., 0.], [0., 1.], [-1., 0.]])
    records = [{'history':np.array([0, 1])}, {'history':np.array([1])}]
    mean = baseline_queries(records, candidates, 'raw_qwen_history_mean')
    np.testing.assert_allclose(mean, [[2**-.5, 2**-.5], [0, 1]])
    np.testing.assert_allclose(baseline_queries(records, candidates, 'raw_qwen_last_item'), [[0, 1], [0, 1]])


def test_clustered_bootstrap_preserves_row_weighted_estimand_and_pairing():
    delta = np.array([[1., -2.], [1., -2.], [5., -10.]])
    result = clustered_bootstrap(delta, np.array(['u1', 'u1', 'u2']), replicates=1000)
    np.testing.assert_allclose(result['estimate'], [7/3, -14/3])
    assert result['clusters'] == 2 and result['rows'] == 3
    assert result['ci_low'][0] == 1 and result['ci_high'][0] == 5
    assert result['ci_low'][1] == -10 and result['ci_high'][1] == -2
    constant = clustered_bootstrap(np.ones((3, 2)), np.array(['u1', 'u1', 'u2']), replicates=100)
    np.testing.assert_allclose(constant['ci_low'], [1, 1])
    np.testing.assert_allclose(constant['ci_high'], [1, 1])
