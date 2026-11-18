"""Hand-computed vector-quality checks; no learned model or training."""

import math
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_vector_metrics import evaluate_vectors


def test_correct_and_reversed_queries_have_expected_direction():
    candidates = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    good = evaluate_vectors([[1., 0.]], candidates, [0], temperature=.5, topk=(1, 2))
    bad = evaluate_vectors([[-1., 0.]], candidates, [0], temperature=.5, topk=(1, 2))
    row = good["per_row"]
    assert row["positive_cosine"][0] == 1
    assert row["negative_mean_cosine"][0] == -.5
    assert row["hardest_negative_cosine"][0] == 0
    assert row["margin_vs_mean_negative"][0] == 1.5
    assert row["margin_vs_hardest_negative"][0] == 1
    assert row["alignment_squared_l2"][0] == 0
    assert row["pairwise_accuracy"][0] == 1
    assert row["optimistic_rank"][0] == row["pessimistic_rank"][0] == 1
    assert row["tie_expected_mrr"][0] == 1
    assert good["summary"]["vector_nll"] == pytest.approx(math.log(math.exp(2) + 1 + math.exp(-2)) - 2)
    assert bad["summary"]["vector_nll"] > good["summary"]["vector_nll"]
    assert bad["per_row"]["pairwise_accuracy"][0] == 0
    assert bad["per_row"]["alignment_squared_l2"][0] == 4


def test_collapsed_vectors_do_not_score_as_accurate():
    vectors = np.tile([1., 0.], (4, 1))
    report = evaluate_vectors(vectors[:2], vectors, [0, 3], topk=(1, 2, 4))
    rows = report["per_row"]
    assert rows["positive_cosine"].tolist() == [1, 1]
    np.testing.assert_allclose(rows["pairwise_accuracy"], .5)
    np.testing.assert_allclose(rows["all_catalog_vector_nll"], math.log(4))
    np.testing.assert_allclose(rows["positive_softmax_probability"], .25)
    np.testing.assert_allclose(rows["margin_vs_hardest_negative"], 0)
    np.testing.assert_allclose(rows["tie_expected_recall@1"], .25)
    np.testing.assert_allclose(rows["tie_expected_mrr"], (1 + .5 + 1/3 + .25) / 4)
    np.testing.assert_allclose(rows["tie_expected_ndcg@2"], (1 + 1/math.log2(3)) / 4)
    np.testing.assert_allclose(rows["pessimistic_recall@1"], 0)
    assert report["summary"]["nll_gain_over_uniform"] == pytest.approx(0)
    assert report["geometry"]["candidates"]["collapsed"]
    assert report["geometry"]["candidates"]["covariance_effective_rank"] == 0


def test_partial_tie_is_handled_without_arbitrary_catalog_order():
    candidates = np.array([[1., 0.], [0., 1.], [0., 1.], [-1., 0.]])
    result = evaluate_vectors([[1., 0.]], candidates, [1], topk=(1, 2, 3))
    rows = result["per_row"]
    assert rows["optimistic_rank"][0] == 2
    assert rows["pessimistic_rank"][0] == 3
    assert rows["average_rank"][0] == 2.5
    assert rows["pairwise_accuracy"][0] == .5
    assert rows["tie_expected_recall@1"][0] == 0
    assert rows["tie_expected_recall@2"][0] == .5
    assert rows["tie_expected_recall@3"][0] == 1
    assert rows["tie_expected_mrr"][0] == pytest.approx((.5 + 1/3) / 2)


def test_warm_and_cold_groups_and_empty_group_are_explicit():
    candidates = np.eye(3)
    result = evaluate_vectors(candidates[:2], candidates, [0, 1], train_seen_mask=[True, False, True])
    assert result["summary"]["groups"]["train_seen"]["count"] == 1
    assert result["summary"]["groups"]["train_unseen"]["count"] == 1
    assert result["summary"]["groups"]["train_seen"]["positive_cosine_mean"] == 1
    empty = evaluate_vectors(candidates[:1], candidates, [0], train_seen_mask=[True, True, True])
    cold = empty["summary"]["groups"]["train_unseen"]
    assert cold["count"] == 0 and cold["vector_nll"] is None
    assert cold["nll_gain_over_uniform"] is None
    assert cold["quantiles"]["positive_cosine"]["p50"] is None


def test_geometry_matches_off_diagonal_identity_and_effective_rank():
    candidates = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    result = evaluate_vectors(candidates, candidates, np.arange(4))
    geometry = result["geometry"]["candidates"]
    assert geometry["coordinate_variance_mean"] == pytest.approx(.5)
    assert geometry["mean_off_diagonal_cosine"] == pytest.approx(-1/3)
    assert geometry["covariance_effective_rank"] == pytest.approx(2)
    assert geometry["collapsed"] is False


@pytest.mark.parametrize("queries,candidates,targets,kwargs,match", [
    ([[0., 0.]], np.eye(2), [0], {}, "normalized"),
    ([[float("nan"), 0.]], np.eye(2), [0], {}, "NaN"),
    ([[2., 0.]], np.eye(2), [0], {}, "normalized"),
    ([[1., 0.]], [[1., 0.], [0., 0.]], [0], {}, "exclude PAD"),
    ([[1., 0.]], np.eye(3), [0], {}, "dimensions"),
    ([[1., 0.]], np.eye(2), [-1], {}, "outside"),
    ([[1., 0.]], np.eye(2), [2], {}, "outside"),
    ([[1., 0.]], np.eye(2), [0.0], {}, "integer"),
    ([[1., 0.]], np.eye(2), [True], {}, "integer"),
    ([[1., 0.]], np.eye(2), [0], {"temperature": 0}, "temperature"),
    ([[1., 0.]], np.eye(2), [0], {"temperature": float("nan")}, "temperature"),
    ([[1., 0.]], np.eye(2), [0], {"tie_tolerance": -1}, "tie_tolerance"),
    ([[1., 0.]], np.eye(2), [0], {"train_seen_mask": [1, 0]}, "boolean"),
    ([[1., 0.]], np.eye(2), [0], {"topk": [0]}, "topk"),
    ([[1., 0.]], [[1., 0.]], [0], {}, "at least two"),
])
def test_invalid_vectors_and_metadata_are_rejected(queries, candidates, targets, kwargs, match):
    with pytest.raises(ValueError, match=match):
        evaluate_vectors(queries, candidates, targets, **kwargs)


def test_nll_is_stable_at_low_temperature_and_inputs_are_unchanged():
    queries = np.array([[1., 0.]])
    candidates = np.array([[1., 0.], [-1., 0.]])
    before = candidates.copy()
    result = evaluate_vectors(queries, candidates, [1], temperature=.0001)
    assert np.isfinite(result["summary"]["vector_nll"])
    assert result["summary"]["vector_nll"] == pytest.approx(20000)
    np.testing.assert_array_equal(candidates, before)
