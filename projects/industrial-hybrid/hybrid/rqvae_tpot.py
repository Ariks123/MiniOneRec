"""Measure internal beam50 time per autoregressive SID token with the original fixed sample."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
from statistics import mean

import numpy as np
import torch

from hybrid.experiment import Data, Engine
from hybrid.manifest import EXPECTED_SID, EXPECTED_WEIGHTS
from hybrid.sweep import sha256, write_json


class BeamTokenTiming:
    """Mark already-selected SID tokens just before the next model forward."""

    def __init__(self, model, device, width):
        self.model = model
        self.device = device
        self.width = width
        self.start, self.first, self.third = (
            torch.cuda.Event(enable_timing=True) for _ in range(3)
        )
        self.lengths = None

    def __enter__(self):
        self.handle = self.model.register_forward_pre_hook(self.record, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.handle.remove()

    def record(self, module, args, kwargs):
        if self.lengths is None:
            return
        tokens = kwargs["input_ids"]
        if tokens.ndim != 2 or tokens.shape[0] != self.width:
            raise ValueError("Expected one request expanded to all 50 beams at each model forward")
        self.lengths.append(tokens.shape[1])
        # Transformers updates the selected running beam and reorders cache
        # before the next forward. These events precede computation of c2 and
        # newline, so the second-to-fourth forward interval spans c2 and c3.
        turn = len(self.lengths)
        if turn == 2:
            self.first.record(torch.cuda.current_stream(self.device))
        elif turn == 4:
            self.third.record(torch.cuda.current_stream(self.device))

    def run(self, prediction):
        torch.cuda.synchronize(self.device)
        self.lengths = []
        self.start.record(torch.cuda.current_stream(self.device))
        try:
            output = prediction()
            observed = self.lengths
        finally:
            self.lengths = None
        torch.cuda.synchronize(self.device)
        if len(observed) < 4 or observed[0] <= 1 or observed[1:4] != [1, 1, 1]:
            raise ValueError(f"Beam forward boundaries are not cached single-token decisions: {observed}")
        first_to_third = self.first.elapsed_time(self.third)
        ttft = self.start.elapsed_time(self.first)
        if not all(math.isfinite(x) and x > 0 for x in (first_to_third, ttft)):
            raise ValueError("Invalid CUDA event times for SID beam decoding")
        return output, {
            "first_to_third_ms": first_to_third,
            "tpot_ms": first_to_third / 2,
            "logical_ttft_ms": ttft,
            "model_forward_calls": len(observed),
        }


def check_protocol(protocol, total):
    ids = protocol["sample_row_ids"]
    expected = random.Random(42).sample(range(total), 256)
    if (protocol["seed"], protocol["repeats"], protocol["warmup_batches_per_shape"],
            protocol["latency_batch_size"], ids) != (42, 3, 3, 1, expected):
        raise ValueError("TPOT must use the original 256 histories, three warmups, and three batch=1 repeats")
    return ids


def distribution(values):
    p50, p95, p99 = np.percentile(values, (50, 95, 99))
    return dict(mean=mean(values), p50=float(p50), p95=float(p95), p99=float(p99))


def b1_amortization(benchmark):
    """E2E wall time per delivered SID atom, including the request timing loop."""
    items = benchmark["recommendations_batch1"]
    samples, k = benchmark["latency_samples"], benchmark["k"]
    if not 0 < items <= samples * k:
        raise ValueError("Impossible number of returned items in the single-request benchmark")
    wall_ms = sum(benchmark["latency_repeat_seconds"]) * 1000
    result = {
        "wall_e2e_ms": wall_ms, "delivered_sid_tokens": 3 * items,
        "mean_ms_per_delivered_sid_token": wall_ms / (3 * items),
    }
    if items == samples * k:
        for p in ("p50", "p95", "p99"):
            result[f"{p}_ms_per_delivered_sid_token"] = benchmark[f"latency_{p}_ms"] / (3 * k)
    return result


@torch.inference_mode()
def measure(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Internal SID TPOT requires CUDA timing events")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    sid_sha = sha256(args.assets / "sid.json")
    model_sha = sha256(args.assets / "rl/model.safetensors")
    if args.rqvae:
        from hybrid.rqvae_assets import MANIFEST, RQVAEData

        if protocol["asset_manifest_sha256"] != sha256(args.assets / MANIFEST):
            raise ValueError("RQ-VAE timing and benchmark use different prepared assets")
        data = RQVAEData(args.assets)
    else:
        if (sid_sha, model_sha) != (EXPECTED_SID, EXPECTED_WEIGHTS):
            raise ValueError("The original TPOT comparison requires the frozen RQ-KMeans and RL assets")
        data = Data(args.assets)
    test = data.read("test")
    ids = check_protocol(protocol, len(test))
    device_name = torch.cuda.get_device_name(device)
    if str(torch.__version__) != protocol["framework"] or device_name != protocol["device"]:
        raise ValueError("TPOT and the benchmark must run on the same GPU model and PyTorch version")
    transformers = version("transformers")
    if transformers != "4.57.1":
        raise ValueError("Beam token timing is audited for Transformers 4.57.1")
    engine = Engine(data, str(device), names=())
    verifier = engine.v
    if verifier.full_width != 50:
        raise ValueError("The original pure beam always uses width 50")

    def predict(row_id):
        return verifier.beam(verifier.prompts([test[row_id]["history_sids"]]))[0]

    parity_ids = ids[::32]
    expected = {row_id: predict(row_id) for row_id in parity_ids}
    inspected = 0
    samples = []
    with BeamTokenTiming(verifier.model, device, verifier.full_width) as marker:
        for row_id in ids[:protocol["warmup_batches_per_shape"]]:
            marker.run(lambda: predict(row_id))
        for repeat in range(protocol["repeats"]):
            for row_id in ids:
                predictions, metrics = marker.run(lambda: predict(row_id))
                if row_id in expected:
                    if predictions != expected[row_id]:
                        raise ValueError(f"Instrumented beam ordering changed for history {row_id}")
                    inspected += 1
                samples.append({"repeat": repeat + 1, "row_id": row_id, **metrics})
    count = len(ids) * protocol["repeats"]
    if len(samples) != count or inspected != len(parity_ids) * protocol["repeats"]:
        raise ValueError("Incomplete beam TPOT or rank-order parity sample")
    result = dict(
        schema=1, status="passed", variant="rqvae" if args.rqvae else "rqkmeans",
        batch_size=1, full_beam_width=verifier.full_width, output_ks=[1, 5, 10, 20, 50],
        sid_tokens_per_item=3, sample_row_ids=ids, repeats=protocol["repeats"],
        benchmark_protocol_sha256=sha256(args.protocol), sid_sha256=sid_sha, model_sha256=model_sha,
        device=device_name, torch=str(torch.__version__), transformers=transformers,
        definition=("(CUDA marker before newline computation, after the third SID selection, minus "
                    "marker before second SID computation, after first SID selection) / 2. "
                    "Newline and EOS model forwards are excluded. The online response is still returned as a complete ranking."),
        logical_ttft_definition="Request start before prompt preparation to the internal first SID selection boundary",
        rank_parity={"sample_row_ids": parity_ids, "tested": inspected, "exact_full_beam_order_matches": inspected},
        summary={"samples": count, **{field: distribution([row[field] for row in samples])
                                     for field in ("tpot_ms", "logical_ttft_ms", "first_to_third_ms")}},
        raw=samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "summary": result["summary"], "rank_parity": result["rank_parity"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("assets", "protocol", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--rqvae", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
