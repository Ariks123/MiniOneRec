"""Process-local, source-pinned correction for DeepSpeed 0.18.0 accumulation.

The original ZeRO-2 epilogue checks the boundary-only ``averaged_gradients``
buffer to decide whether previous microbatch gradients exist. Before a boundary
that buffer is absent, so it overwrites ``all_grad_tensors`` on every microbatch.
The correction checks the actual accumulation buffer instead. Installed package
files and all other DeepSpeed/GRPO behavior remain unchanged.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap


SUPPORTED_VERSION = "0.18.0"
ORIGINAL_SOURCE_SHA256 = "8aab6ad622662a54879f409636e8fef248b6ab42d51cf69ab2131f5caa99d542"
ORIGINAL_CONDITION = "if not i in self.averaged_gradients or self.averaged_gradients[i] is None:"
CORRECTED_CONDITION = "if i not in self.all_grad_tensors or self.all_grad_tensors[i] is None:"
METHOD_NAME = "independent_gradient_partition_epilogue"
MARKER = "_industrial_zero2_accumulation_compatibility"


def _apply_to_optimizer_class(optimizer_class, version, source_getter=inspect.getsource):
    if version != SUPPORTED_VERSION:
        raise RuntimeError(f"ZeRO-2 accumulation compatibility supports only DeepSpeed {SUPPORTED_VERSION}; got {version}")
    previous = optimizer_class.__dict__.get(MARKER)
    if previous is not None:
        if getattr(optimizer_class, METHOD_NAME) is not previous["method"]:
            raise RuntimeError("Previously applied ZeRO-2 accumulation compatibility was subsequently changed")
        return {**previous["metadata"], "already_applied": True}
    original = getattr(optimizer_class, METHOD_NAME)
    source = source_getter(original)
    original_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if original_hash != ORIGINAL_SOURCE_SHA256:
        raise RuntimeError("Unsupported DeepSpeed ZeRO-2 epilogue source SHA256: " + original_hash)
    if source.count(ORIGINAL_CONDITION) != 1:
        raise RuntimeError("Expected exactly one verified ZeRO-2 accumulation condition")
    patched_source = source.replace(ORIGINAL_CONDITION, CORRECTED_CONDITION, 1)
    namespace = dict(original.__globals__)
    exec(compile(textwrap.dedent(patched_source), original.__code__.co_filename, "exec"), namespace)
    patched = namespace[METHOD_NAME]
    patched.__module__, patched.__qualname__ = original.__module__, original.__qualname__
    metadata = {
        "applied": True, "version": version, "already_applied": False,
        "original_source_sha256": original_hash,
        "patched_source_sha256": hashlib.sha256(patched_source.encode("utf-8")).hexdigest(),
        "method": optimizer_class.__module__ + "." + optimizer_class.__qualname__ + "." + METHOD_NAME,
        "scope": "current_rl_process_only", "installed_package_modified": False,
        "reason": "Retain prior microbatch gradients in all_grad_tensors before the accumulation boundary",
    }
    setattr(optimizer_class, METHOD_NAME, patched)
    setattr(optimizer_class, MARKER, {"method": patched, "metadata": metadata})
    return dict(metadata)


def apply_zero2_accumulation_fix():
    """Fail closed on unknown runtimes; apply once in this RL process only."""
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    return _apply_to_optimizer_class(DeepSpeedZeroOptimizer, deepspeed.__version__)
