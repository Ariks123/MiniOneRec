"""Task-relative quality of normalized sequence and item vectors.

These functions do not define an absolute correct coordinate system. They test
whether the observed next-item vector is well matched by a history query in
their shared, learned space. Other catalog items are comparison candidates,
not independently verified negative preferences. No model or optimizer is used.
"""

from __future__ import annotations

import math


def _normalized_matrix(values, name):
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty [rows, dimensions] matrix")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric vectors")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0) or not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"{name} must be nonzero and L2-normalized; exclude PAD")
    return array


def _geometry(values):
    import numpy as np

    rows, dimensions = values.shape
    centered = values - values.mean(axis=0, keepdims=True)
    coordinate_variances = np.mean(centered * centered, axis=0)
    variance_mean = float(coordinate_variances.mean())
    # Sum identity avoids allocating a rows x rows similarity matrix.
    if rows > 1:
        mean_vector_sum = values.sum(axis=0)
        off_diagonal_cosine = float(
            (mean_vector_sum @ mean_vector_sum - np.square(values).sum())
            / (rows * (rows - 1))
        )
    else:
        off_diagonal_cosine = None
    covariance = centered.T @ centered / max(1, rows - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total_variance = float(eigenvalues.sum())
    if total_variance <= 1e-12:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0] / total_variance
        probabilities = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "count": rows,
        "dimension": dimensions,
        "coordinate_variance_mean": variance_mean,
        "coordinate_variance_min": float(coordinate_variances.min()),
        "coordinate_variance_max": float(coordinate_variances.max()),
        "mean_off_diagonal_cosine": off_diagonal_cosine,
        "covariance_effective_rank": effective_rank,
        "collapsed": variance_mean < 1e-8,
        "collapse_variance_threshold": 1e-8,
        "collapse_interpretation": "diagnostic only; repeated histories can legitimately share queries",
    }


def _summarize(per_row, mask, candidate_count):
    import numpy as np

    count = int(np.count_nonzero(mask))
    summary = {"count": count, "candidate_count": candidate_count}
    for name, values in per_row.items():
        summary[name + "_mean"] = float(np.mean(values[mask])) if count else None
    summary["vector_nll"] = summary["all_catalog_vector_nll_mean"]
    summary["uniform_catalog_nll"] = math.log(candidate_count)
    summary["nll_gain_over_uniform"] = (
        summary["uniform_catalog_nll"] - summary["vector_nll"] if count else None
    )
    summary["quantiles"] = {}
    for name in ("positive_cosine", "margin_vs_mean_negative", "margin_vs_hardest_negative"):
        summary["quantiles"][name] = {
            label: float(np.quantile(per_row[name][mask], quantile)) if count else None
            for label, quantile in (("p05", .05), ("p25", .25), ("p50", .50), ("p75", .75), ("p95", .95))
        }
    return summary


def evaluate_vectors(queries, candidates, target_indices, *, temperature=0.07,
                     train_seen_mask=None, topk=(10, 20, 50), tie_tolerance=1e-6):
    """Compare normalized history queries against the complete item catalog.

    ``candidates`` has no PAD row. ``target_indices`` are zero-based candidate
    row indices, so target 0 is valid. ``train_seen_mask`` is a boolean mask over
    catalog rows; a cold target is absent from both training history and labels.

    Returns ``summary`` (JSON-compatible scalars), ``per_row`` (metric name to
    NumPy array), and ``geometry`` (JSON-compatible diagnostics). Rankings group
    scores within ``tie_tolerance`` of the target. Tie-expected MRR, Recall and
    NDCG average over uniform orderings of that target's tied group, preventing
    identical embeddings from receiving artificial perfect ranking scores.
    Raw cosine scores and the NLL are never rounded by the tie tolerance.
    """
    import numpy as np

    queries = _normalized_matrix(queries, "queries")
    candidates = _normalized_matrix(candidates, "candidates")
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("query and candidate dimensions must match")
    item_count = candidates.shape[0]
    if item_count < 2:
        raise ValueError("at least two non-PAD catalog candidates are required")
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    if isinstance(tie_tolerance, bool) or not math.isfinite(float(tie_tolerance)) or float(tie_tolerance) < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    temperature, tie_tolerance = float(temperature), float(tie_tolerance)
    targets = np.asarray(target_indices)
    if targets.shape != (len(queries),) or not np.issubdtype(targets.dtype, np.integer) or targets.dtype == np.bool_:
        raise ValueError("target_indices must contain one integer candidate row per query")
    if np.any(targets < 0) or np.any(targets >= item_count):
        raise ValueError("target_indices are outside the non-PAD candidate matrix")
    targets = targets.astype(np.int64, copy=False)
    cutoffs = []
    for cutoff in topk:
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, np.integer)) or cutoff <= 0:
            raise ValueError("topk values must be positive integers")
        if int(cutoff) not in cutoffs:
            cutoffs.append(int(cutoff))
    seen = None
    if train_seen_mask is not None:
        seen = np.asarray(train_seen_mask)
        if seen.shape != (item_count,) or seen.dtype != np.bool_:
            raise ValueError("train_seen_mask must be a boolean mask over all candidate rows")

    scores = queries @ candidates.T
    rows = np.arange(len(queries))
    positive = scores[rows, targets].copy()
    negative_mean = (scores.sum(axis=1) - positive) / (item_count - 1)
    # Stable full-catalog CE with the same fixed temperature used at training.
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = scores / temperature
    if not np.isfinite(scaled).all():
        raise ValueError("temperature is too small for finite vector logits")
    maxima = scaled.max(axis=1)
    log_denominator = maxima + np.log(np.exp(scaled - maxima[:, None]).sum(axis=1))
    vector_nll = log_denominator - positive / temperature
    differences = scores - positive[:, None]
    above = np.count_nonzero(differences > tie_tolerance, axis=1)
    tied = np.count_nonzero(np.abs(differences) <= tie_tolerance, axis=1)
    below = item_count - above - tied
    pairwise = (below + 0.5 * (tied - 1)) / (item_count - 1)
    optimistic = above + 1
    pessimistic = above + tied
    scores[rows, targets] = -np.inf
    hardest_negative = scores.max(axis=1)
    per_row = {
        "positive_cosine": positive,
        "negative_mean_cosine": negative_mean,
        "hardest_negative_cosine": hardest_negative,
        "margin_vs_mean_negative": positive - negative_mean,
        "margin_vs_hardest_negative": positive - hardest_negative,
        "pairwise_accuracy": pairwise,
        "all_catalog_vector_nll": vector_nll,
        "positive_softmax_probability": np.exp(-vector_nll),
        "alignment_squared_l2": np.maximum(0.0, 2.0 - 2.0 * positive),
        "optimistic_rank": optimistic,
        "pessimistic_rank": pessimistic,
        "average_rank": (optimistic + pessimistic) / 2.0,
    }
    harmonic_prefix = np.concatenate(([0.0], np.cumsum(1.0 / np.arange(1, item_count + 1))))
    discounts = 1.0 / np.log2(np.arange(1, item_count + 1) + 1)
    discount_prefix = np.concatenate(([0.0], np.cumsum(discounts)))
    per_row["tie_expected_mrr"] = (harmonic_prefix[pessimistic] - harmonic_prefix[above]) / tied
    for cutoff in cutoffs:
        eligible_ties = np.clip(cutoff - above, 0, tied)
        per_row[f"tie_expected_recall@{cutoff}"] = eligible_ties / tied
        last_rank = np.maximum(above, np.minimum(pessimistic, cutoff))
        per_row[f"tie_expected_ndcg@{cutoff}"] = (discount_prefix[last_rank] - discount_prefix[above]) / tied
        per_row[f"pessimistic_recall@{cutoff}"] = (pessimistic <= cutoff).astype(float)

    summary = _summarize(per_row, np.ones(len(queries), dtype=bool), item_count)
    if seen is not None:
        warm_rows = seen[targets]
        summary["groups"] = {
            "train_seen": _summarize(per_row, warm_rows, item_count),
            "train_unseen": _summarize(per_row, ~warm_rows, item_count),
        }
    summary["tie_policy"] = {
        "tolerance": tie_tolerance,
        "rank": "optimistic/pessimistic bounds; average_rank is their midpoint",
        "tie_expected_metrics": "uniform ordering of the group tied with the observed target",
        "pairwise": "1 if target wins, 0.5 if tied, 0 otherwise",
    }
    return {"summary": summary, "per_row": per_row,
            "geometry": {"queries": _geometry(queries), "candidates": _geometry(candidates)}}
