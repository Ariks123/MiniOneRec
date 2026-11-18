"""Warmup, stable, cosine-decay learning rates for SFT.

The schedule is defined in optimizer updates (after gradient accumulation), not
micro-batches or epochs. Its horizon stays fixed at the planned training maximum
when early stopping finishes training sooner. Torch is imported only by the
scheduler factory, so phase resolution and resume checks need only the stdlib.
"""
from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral, Real


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _ratio(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real ratio')
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must be between zero and one')
    return float(value)


def resolve_wsd_schedule(num_training_steps, *, warmup_ratio=0.03,
                         stable_ratio=0.07, min_lr_ratio=0.1):
    """Return a reproducible, JSON-serializable optimizer-step schedule.

    Warmup and stable steps are floored separately. All remaining updates belong
    to cosine decay, avoiding an unassigned update caused by independent ratio
    rounding. Small training horizons may resolve a requested phase to zero
    updates; the resolved counts are recorded explicitly.
    """
    total = _integer(num_training_steps, 'num_training_steps', 1)
    warmup = _ratio(warmup_ratio, 'warmup_ratio')
    stable = _ratio(stable_ratio, 'stable_ratio')
    minimum = _ratio(min_lr_ratio, 'min_lr_ratio')
    if warmup + stable >= 1:
        raise ValueError('warmup_ratio + stable_ratio must be less than one to leave cosine-decay updates')
    # Hyperparameter decimals such as .58 must not accidentally floor 100*.58
    # to 57 because their binary-float product is 57.99999999999999.
    warmup_steps = math.floor(total * Fraction(str(warmup)))
    stable_steps = math.floor(total * Fraction(str(stable)))
    decay_steps = total - warmup_steps - stable_steps
    if decay_steps < 1:
        raise ValueError('The resolved schedule must include at least one cosine-decay update')
    return {
        'type': 'warmup_stable_cosine',
        'unit': 'optimizer_updates',
        'num_training_steps': total,
        'warmup_ratio': warmup,
        'stable_ratio': stable,
        'decay_ratio': 1.0 - warmup - stable,
        'min_lr_ratio': minimum,
        'warmup_steps': warmup_steps,
        'stable_steps': stable_steps,
        'decay_steps': decay_steps,
        'warmup_end_step': warmup_steps,
        'decay_start_step': warmup_steps + stable_steps,
        'decay_end_step': total,
        'rounding': 'floor warmup and stable; assign the remainder to decay',
        'early_stopping_policy': 'Keep the planned maximum-step horizon; do not rescale phases after early stopping.',
        'step_semantics': 'LambdaLR step 0 is initialization; factor at num_training_steps is min_lr_ratio.',
    }


def validate_wsd_resume(previous, resolved):
    """Reject a changed horizon or phase after a checkpoint has been saved."""
    if not isinstance(previous, dict) or previous != resolved:
        raise ValueError('WSD resume schedule differs from its checkpoint: horizon, phases, minimum and metadata must match')


def _validate_resolved(schedule):
    if not isinstance(schedule, dict):
        raise ValueError('WSD schedule must be a resolved schedule dictionary')
    try:
        expected = resolve_wsd_schedule(schedule['num_training_steps'],
                                        warmup_ratio=schedule['warmup_ratio'],
                                        stable_ratio=schedule['stable_ratio'],
                                        min_lr_ratio=schedule['min_lr_ratio'])
    except KeyError as error:
        raise ValueError('Incomplete resolved WSD schedule') from error
    if schedule != expected:
        raise ValueError('Resolved WSD phase metadata is inconsistent')


def wsd_lr_factor(current_step, schedule):
    """Return the multiplier of the peak learning rate at an optimizer step."""
    step = _integer(current_step, 'current_step', 0)
    _validate_resolved(schedule)
    warmup_steps = schedule['warmup_steps']
    if step < warmup_steps:
        return step / warmup_steps
    if step <= schedule['decay_start_step']:
        return 1.0
    if step >= schedule['num_training_steps']:
        return schedule['min_lr_ratio']
    progress = (step - schedule['decay_start_step']) / schedule['decay_steps']
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule['min_lr_ratio'] + (1.0 - schedule['min_lr_ratio']) * cosine


def build_wsd_scheduler(optimizer, num_training_steps, *, warmup_ratio=0.03,
                        stable_ratio=0.07, min_lr_ratio=0.1, last_epoch=-1,
                        expected_schedule=None):
    """Construct a Torch LambdaLR with phase metadata and safe state loading.

    A Trainer subclass can return this from create_scheduler and serialize
    ``scheduler.wsd_schedule`` into run_config.json. Trainer checkpoint scheduler
    state also includes that metadata; load_state_dict rejects phase changes
    before restoring the learning-rate state. Existing ordinary-cosine/linear
    checkpoints cannot be silently resumed under this new schedule.
    """
    resolved = resolve_wsd_schedule(num_training_steps, warmup_ratio=warmup_ratio,
                                    stable_ratio=stable_ratio, min_lr_ratio=min_lr_ratio)
    _integer(last_epoch, 'last_epoch', -1)
    if expected_schedule is not None:
        validate_wsd_resume(expected_schedule, resolved)
    import torch

    class WarmupStableCosineLR(torch.optim.lr_scheduler.LambdaLR):
        def __init__(self):
            self.wsd_schedule = dict(resolved)
            super().__init__(optimizer, lr_lambda=lambda step: wsd_lr_factor(step, resolved),
                             last_epoch=last_epoch)

        def load_state_dict(self, state_dict):
            validate_wsd_resume(state_dict.get('wsd_schedule'), resolved)
            return super().load_state_dict(state_dict)

    return WarmupStableCosineLR()

