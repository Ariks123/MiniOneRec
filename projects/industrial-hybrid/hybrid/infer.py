"""Serve one Industrial hybrid recommendation request using the RL beta=0.001 checkpoint."""
import argparse
from contextlib import redirect_stdout
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

from hybrid.experiment import DRAFTERS, Data, Engine, draft_size
from hybrid.sweep import Catalog


def history_ids(text):
    try:
        history = [int(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("History must contain comma-separated numeric catalog item IDs") from exc
    if not (1 <= len(history) <= 10) or any(i < 0 for i in history):
        raise argparse.ArgumentTypeError("Supply one to ten nonnegative item IDs, oldest to most recent")
    return history


def finite_number(value, name):
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def config_for(args):
    if not 1 <= args.k <= 50:
        raise ValueError("K must be between 1 and 50")
    if args.winner or args.selection is not None:
        overrides = [key for key in ("drafter", "alpha", "rho", "prefix_beams", "threshold", "max_rounds")
                     if getattr(args, key) is not None]
        if overrides:
            names = ", ".join("--" + x.replace("_", "-") for x in overrides)
            raise ValueError(f"Frozen winner mode reads the complete validation configuration. Omit {names}")
        path = args.selection or args.run_dir/"report"/"selection.json"
        selected = json.loads(path.read_text())
        ident = selected["winners"].get(str(args.k))
        if ident is None:
            raise ValueError(f"No frozen validation winner for K={args.k} in {path}")
        matches = [c for c in selected["locked"] if c["id"] == ident]
        if len(matches) != 1 or matches[0]["k"] != args.k:
            raise ValueError(f"Inconsistent frozen configuration for {ident}")
        cfg = dict(matches[0])
        cfg.setdefault("rho", 2)
        cfg.setdefault("max_rounds", 3)
        finite_number(cfg["alpha"], "alpha")
        ratio = Fraction(str(cfg["alpha"]))
    else:
        alpha = args.alpha if args.alpha is not None else ("0" if args.drafter == "beam" else "0.4")
        finite_number(alpha, "alpha")
        ratio = Fraction(str(alpha))
        drafter = args.drafter or "ann"
        quota = int(ratio * args.k)
        threshold = None if args.threshold in (None, "all") else finite_number(args.threshold, "threshold")
        cfg = dict(k=args.k, drafter=drafter if quota else "beam", alpha=float(ratio), quota=quota,
                   prefix_beams=5 if args.prefix_beams is None else args.prefix_beams,
                   threshold=threshold, rho=2 if args.rho is None else args.rho,
                   max_rounds=3 if args.max_rounds is None else args.max_rounds)
        if Fraction(str(cfg["alpha"])) != ratio:
            cfg["alpha_decimal"] = str(alpha)
    quota = cfg["quota"]
    if not 0 <= ratio <= 1:
        raise ValueError("alpha must be between zero and one")
    if quota != int(ratio * args.k):
        raise ValueError("Retrieval quota must equal floor(K * alpha)")
    finite_number(cfg["rho"], "rho")
    draft_size(args.k, cfg["rho"])
    if cfg["max_rounds"] not in (1, 2, 3):
        raise ValueError("max-rounds must be 1, 2, or 3 for the three-token Industrial SIDs")
    if cfg["prefix_beams"] < (1 if quota or args.prefix_beams is not None else 0):
        raise ValueError("prefix-beams must be positive")
    if quota and cfg["drafter"] not in DRAFTERS:
        raise ValueError("Select ann, unisrec, or qwen when the retrieval quota is positive")
    if cfg["threshold"] is not None:
        finite_number(cfg["threshold"], "threshold")
    return cfg


def recommend(args, cfg):
    import torch
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.rqvae:
        from hybrid.rqvae_assets import RQVAEData

        if cfg["drafter"] not in ("beam", "unisrec"):
            raise ValueError("RQ-VAE inference allows only UniSRec and pure beam")
        data = RQVAEData(args.assets)
    else:
        data = Data(args.assets)
    row_id = args.test_row
    if row_id is not None:
        test = data.read("test")
        if not 0 <= row_id < len(test):
            raise ValueError(f"Test row must be in 0..{len(test)-1}")
        history = test[row_id]["history"]
    else:
        history = args.history
    unknown = sorted(set(history) - data.sids.keys())
    if unknown:
        raise ValueError(f"Unknown catalog item IDs: {unknown}")
    names = (cfg["drafter"],) if cfg["quota"] else ()
    engine = Engine(data, str(device), names=names)
    example = dict(history=history, history_sids=[data.sids[i] for i in history])
    ranking, = engine.online([example], cfg, Catalog.from_dict(data.record))
    retained = sum(item.source == "retrieval" for item in ranking)
    return dict(
        row_id=row_id, history=history, config={**cfg, "rho":float(cfg["rho"]),
                                               "draft_limit":draft_size(cfg["k"], cfg["rho"])},
        alpha_eff=retained/cfg["k"], retrieval_count=retained,
        beam_count=len(ranking)-retained, returned=len(ranking),
        recommendations=[dict(rank=rank, item_id=item.item_id, sid=data.sids[item.item_id],
                              source=item.source, score=item.score, round=item.round or None,
                              cold=bool(data.record["cold"][item.item_id]))
                         for rank,item in enumerate(ranking, start=1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--history", type=history_ids, help="One to ten comma-separated catalog IDs, oldest first")
    inputs.add_argument("--test-row", type=int, help="Zero-based test split row")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", help="Retrieval fraction, default 0.4 (zero with --drafter beam)")
    ap.add_argument("--rho", help="Draft size multiplier, default 2")
    ap.add_argument("--prefix-beams", type=int, help="Width of the legal prefix beam, default 5")
    ap.add_argument("--threshold", help="Finite mean log-probability threshold or 'all' (default)")
    ap.add_argument("--max-rounds", type=int, help="One to three draft rounds (default 3)")
    ap.add_argument("--drafter", choices=(*DRAFTERS, "beam"), help="Default ann")
    ap.add_argument("--winner", action="store_true", help="Use the frozen validation winner for K")
    ap.add_argument("--selection", type=Path, help="Read the frozen winner for K from this selection.json")
    ap.add_argument("--run-dir", type=Path, default=Path("/run/hybrid"),
                    help="Frozen winners are read from RUN_DIR/report/selection.json")
    ap.add_argument("--assets", type=Path, default=Path("/assets"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rqvae", action="store_true", help="Use sealed RQ-VAE and retrained UniSRec assets")
    args = ap.parse_args()
    try:
        cfg = config_for(args)
        # Keep stdout machine-readable even if a model loader prints diagnostics.
        with redirect_stdout(sys.stderr):
            result = recommend(args, cfg)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
