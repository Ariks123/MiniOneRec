# Industrial 0.5B: selected SFT → GRPO

The RL policy and initial reference start from the linear-three-epoch SFT run's
`epochs3/sft/final_checkpoint`, which contains checkpoint 5187 (epoch 2.8594).
The source model is read-only; RL starts with a new optimizer and scheduler.

```bash
source /mnt/ossdata/ziweizhao/env.sh
cd /mnt/ossdata/ziweizhao/code/amazon2018industrialandscientific-cosine
python docker/industrial_rl_experiment.py start
python docker/industrial_rl_experiment.py status
```

The detached supervisor continues after the terminal closes. The default run is
`weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1/rl`
under the workspace. Reports are in
`output/reports/industrial_05b_rl_from_linear_ep3_retry1/`. Existing run directories are
never overwritten; a new attempt requires a separate run/report/container name.

## Fixed training recipe

- Four H20 GPUs, BF16, DeepSpeed ZeRO-2, SDPA and gradient checkpointing.
- MiniOneRec's group-normalized policy-gradient/KL objective with deterministic
  constrained beam16 rollouts. Exact-hit and position-discount ranking rewards
  retain the original formulas; no ground-truth injection, alternative loss,
  semantic reward, cold-item weighting or vLLM is enabled.
- At most two epochs, learning rate `1e-5`, cosine decay, 3% warmup,
  `paged_adamw_32bit`, gradient clipping 0.3, seed 42.
- Per-device completion batch 16, accumulation 16, four devices: 1,024
  completions or 64 independent prompt groups per optimizer update.
- KL beta 0.001; reference synchronization every 512 updates with policy mixing
  coefficient 0.6. These parameters are explicitly recorded rather than inherited
  implicitly from library defaults.
- 52,775 prompt rows: 36,259 SID histories, 3,646 titles, 2,870 descriptions and
  10,000 title histories. The title-history sample uses seed 42 and its sample
  identities are preserved in the prepared task manifest.

Each reward uses the current row's `target_sid`. Duplicate prompts or histories
with different legitimate targets remain distinct through `sample_id`; the
trainer checks complete groups on each rank. Validation labels are not merged
into a global training reward dictionary.

## Validation, selection and reporting

Before updates, the supervisor's RL entry computes an independent full-validation
SFT baseline without invoking DeepSpeed's inference initialization path.
Approximately every quarter epoch, and at the final live training step, it
computes full-catalog beam50 ranking over all 4,532 validation rows using the
original evaluation prompt. This loop does not compute or label a GRPO objective
as a teacher-forced validation loss.

Strict validation NDCG@10 improvement selects the best checkpoint. Three
consecutive non-improvements trigger early stopping; ties retain the earlier
checkpoint. At most two training checkpoints remain. The final exported policy
reloads the selected checkpoint, and records the selected weight identity when
its validation is repeated after reload.

Only after training, all 4,533 test rows are evaluated with the same catalog,
prompt and beam50 protocol used for SFT. HR/NDCG at 1/3/5/10/20/50 and warm/cold
strata are recomputed from predictions. Reports compare SFT and RL; recommendation
uses full-validation NDCG@10 only, with ties favoring SFT. Tests never select a
checkpoint.

The training report includes rewards, KL, zero-reward/hit/active-advantage group
rates, learning rate and pre-clipping global gradient norm. Per-task reward
telemetry averages task-conditional microbatch means, rather than claiming an
exact epoch-wide sample-weighted mean. Validation reports include cold-row hit
counts, SID collisions and complete prediction coverage.

Before launch, the supervisor verifies the selected SFT weights, data/model/source
hashes, immutable runtime image, idle GPUs and at least 50 GiB of local storage.
It publishes logs and reports about every 60 seconds. Failed runs retain their
container, logs and available checkpoints; they are not silently resumed with a
new reference model. Completion requires training, evaluation and comparison
validation plus both PDF reports to pass.

## Runtime accumulation correction

The first RL attempt was stopped after 114 updates and is excluded from result
comparison. A two-microbatch diagnostic showed that the installed DeepSpeed
0.18.0 ZeRO-2 implementation discarded preceding gradients when its final
microbatch had zero gradient. Four-card tests also confirmed the defect and the
minimal correction, including sums, cancellation and gradient clipping.

The retry applies a process-local compatibility correction after verifying the
exact DeepSpeed version and original method source hash. It tests the persistent
gradient buffer when deciding whether to accumulate, and does not modify the
installed package or historical SFT files. The run configuration records both
source hashes and the correction reason. Unsupported source implementations are
rejected instead of patched speculatively.

The existing SFT export remains the fixed starting-point baseline. Its historical
training used the same DeepSpeed version and may also have been affected; these
results do not isolate the contribution of the runtime correction from RL itself.
No historical SFT experiment is silently retrained or relabeled.

## Checks

```bash
python -m unittest discover -s tests -q
# In the pinned image, on four H20 GPUs, with a fresh output directory:
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  docker/smoke_industrial_rl.py --output-dir /path/to/fresh/synthetic-smoke
# Also run the sparse-microbatch accumulation regression in that environment:
python docker/smoke_industrial_rl_accumulation.py --output /path/to/regression-result.json
```

The distributed smoke uses synthetic weights/data, real sixteen-candidate GRPO
updates and real full-catalog beam50 validation. Only its selection score is
scripted, to verify early stopping and best-model reload deterministically.
It also checks reference synchronization, row-local labels for duplicate
prompts, state/RNG restoration, checkpoint rotation and export reload.
