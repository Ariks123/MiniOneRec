"""Pure epoch-based learning-rate policy; this module never creates an optimizer."""

import math


def epoch_learning_rate(base_lr, epoch_index, total_epochs, schedule=None,
                        warmup_epochs=0, min_learning_rate=0.0):
    """Return the LR for a zero-based epoch, including both cosine endpoints.

    Five warmup epochs use 1/5, 2/5, ..., 5/5 of the peak LR. Subsequent
    epochs decay from the peak to ``min_learning_rate`` in the final epoch.
    ``None`` or ``constant`` preserves the original single-GPU policy.
    """
    base_lr = float(base_lr)
    min_learning_rate = float(min_learning_rate)
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not isinstance(epoch_index, int) or isinstance(epoch_index, bool) or epoch_index < 0:
        raise ValueError('epoch_index must be a nonnegative integer')
    if not isinstance(total_epochs, int) or total_epochs <= 0:
        raise ValueError('epochs must be a positive integer')
    if schedule is None or schedule == 'constant':
        return base_lr
    if schedule != 'warmup_cosine':
        raise ValueError('lr_schedule must be constant or warmup_cosine')
    if not isinstance(warmup_epochs, int) or not 0 <= warmup_epochs < total_epochs:
        raise ValueError('warmup_epochs must be an integer in [0, epochs)')
    if not math.isfinite(min_learning_rate) or not 0 <= min_learning_rate <= base_lr:
        raise ValueError('min_learning_rate must lie in [0, learning_rate]')
    if epoch_index < warmup_epochs:
        return base_lr * (epoch_index + 1) / warmup_epochs
    decay_intervals = total_epochs - warmup_epochs - 1
    if decay_intervals <= 0:
        return base_lr
    progress = min(1.0, (epoch_index - warmup_epochs) / decay_intervals)
    return min_learning_rate + 0.5 * (base_lr - min_learning_rate) * (1.0 + math.cos(math.pi * progress))
