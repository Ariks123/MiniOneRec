"""Exact WSD phase, rounding and checkpoint-identity checks without Torch."""
import copy
import importlib.util
import json
import math
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.sft_wsd_schedule import (build_wsd_scheduler, resolve_wsd_schedule,
                                      validate_wsd_resume, wsd_lr_factor)


class WSDScheduleTests(unittest.TestCase):
    def test_three_phase_values_boundaries_and_floor(self):
        schedule = resolve_wsd_schedule(100, warmup_ratio=.1, stable_ratio=.3, min_lr_ratio=.1)
        self.assertEqual((schedule['warmup_steps'], schedule['stable_steps'], schedule['decay_steps']), (10, 30, 60))
        self.assertEqual(wsd_lr_factor(0, schedule), 0)
        self.assertEqual(wsd_lr_factor(5, schedule), .5)
        self.assertEqual(wsd_lr_factor(10, schedule), 1)
        self.assertEqual(wsd_lr_factor(25, schedule), 1)
        self.assertEqual(wsd_lr_factor(40, schedule), 1)
        self.assertAlmostEqual(wsd_lr_factor(70, schedule), .55)
        self.assertEqual(wsd_lr_factor(100, schedule), .1)
        self.assertEqual(wsd_lr_factor(101, schedule), .1)

    def test_default_has_a_hold_before_cosine(self):
        schedule = resolve_wsd_schedule(1000)
        self.assertEqual((schedule['warmup_steps'], schedule['stable_steps'], schedule['decay_steps']), (30, 70, 900))
        self.assertEqual(wsd_lr_factor(50, schedule), 1)
        self.assertEqual(wsd_lr_factor(100, schedule), 1)
        self.assertLess(wsd_lr_factor(101, schedule), 1)

    def test_phase_monotonicity_and_total_horizon(self):
        schedule = resolve_wsd_schedule(207, warmup_ratio=.13, stable_ratio=.29)
        values = [wsd_lr_factor(step, schedule) for step in range(208)]
        warm, decay = schedule['warmup_steps'], schedule['decay_start_step']
        self.assertTrue(all(a <= b for a, b in zip(values[:warm], values[1:warm+1])))
        self.assertTrue(all(value == 1 for value in values[warm:decay+1]))
        self.assertTrue(all(a >= b for a, b in zip(values[decay:], values[decay+1:])))
        self.assertEqual(schedule['warmup_steps'] + schedule['stable_steps'] + schedule['decay_steps'], 207)
        self.assertEqual(values[-1], .1)

    def test_fractional_steps_leave_rounding_remainder_in_decay(self):
        schedule = resolve_wsd_schedule(17, warmup_ratio=.11, stable_ratio=.33)
        self.assertEqual((schedule['warmup_steps'], schedule['stable_steps'], schedule['decay_steps']), (1, 5, 11))
        self.assertEqual(schedule['decay_start_step'], 6)
        self.assertEqual(schedule['decay_end_step'], 17)
        self.assertEqual(resolve_wsd_schedule(100, warmup_ratio=.58, stable_ratio=.1)['warmup_steps'], 58)

    def test_small_horizons_and_zero_phases_are_well_defined(self):
        schedule = resolve_wsd_schedule(1)
        self.assertEqual((schedule['warmup_steps'], schedule['stable_steps'], schedule['decay_steps']), (0, 0, 1))
        self.assertEqual(wsd_lr_factor(0, schedule), 1)
        self.assertEqual(wsd_lr_factor(1, schedule), .1)
        schedule = resolve_wsd_schedule(8, warmup_ratio=0, stable_ratio=.5, min_lr_ratio=0)
        self.assertEqual([wsd_lr_factor(n, schedule) for n in range(5)], [1] * 5)
        self.assertEqual(wsd_lr_factor(8, schedule), 0)

    def test_invalid_inputs_are_rejected(self):
        for total in [0, -1, True, 1.5, float('nan')]:
            with self.subTest(total=total), self.assertRaises(ValueError):
                resolve_wsd_schedule(total)
        for kwargs in [{'warmup_ratio': -1}, {'stable_ratio': 1.1}, {'min_lr_ratio': -1},
                       {'warmup_ratio': float('nan')}, {'stable_ratio': float('inf')},
                       {'warmup_ratio': True}, {'warmup_ratio': .5, 'stable_ratio': .5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_wsd_schedule(100, **kwargs)
        for step in [-1, True, 2.1, float('inf')]:
            with self.subTest(step=step), self.assertRaises(ValueError):
                wsd_lr_factor(step, resolve_wsd_schedule(10))

    def test_resume_json_roundtrip_is_identical_but_changes_are_rejected(self):
        original = resolve_wsd_schedule(1000)
        validate_wsd_resume(json.loads(json.dumps(original)), original)
        for changed in [resolve_wsd_schedule(1001), resolve_wsd_schedule(1000, stable_ratio=.08),
                        resolve_wsd_schedule(1000, min_lr_ratio=.2), None]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'resume schedule differs'):
                validate_wsd_resume(changed, original)
        corrupt = dict(original, decay_steps=1)
        with self.assertRaisesRegex(ValueError, 'metadata is inconsistent'):
            wsd_lr_factor(1, corrupt)

    def test_factory_is_lazy_and_saved_state_binds_the_schedule(self):
        class FakeLambdaLR:
            def __init__(self, optimizer, lr_lambda, last_epoch):
                self.optimizer, self.lr_lambda, self.last_epoch = optimizer, lr_lambda, last_epoch

            def state_dict(self):
                return {'wsd_schedule': copy.deepcopy(self.wsd_schedule), 'last_epoch': self.last_epoch}

            def load_state_dict(self, value):
                self.last_epoch = value['last_epoch']

        fake = SimpleNamespace(optim=SimpleNamespace(lr_scheduler=SimpleNamespace(LambdaLR=FakeLambdaLR)))
        with mock.patch.dict(sys.modules, {'torch': fake}):
            scheduler = build_wsd_scheduler('optimizer', 1000)
            self.assertEqual(scheduler.wsd_schedule, resolve_wsd_schedule(1000))
            self.assertEqual(scheduler.lr_lambda(1000), .1)
            state = scheduler.state_dict()
            state['last_epoch'] = 75
            scheduler.load_state_dict(state)
            self.assertEqual(scheduler.last_epoch, 75)
            state['wsd_schedule'] = resolve_wsd_schedule(999)
            with self.assertRaisesRegex(ValueError, 'resume schedule differs'):
                scheduler.load_state_dict(state)
            self.assertEqual(scheduler.last_epoch, 75)
            with self.assertRaisesRegex(ValueError, 'resume schedule differs'):
                build_wsd_scheduler('optimizer', 1000, expected_schedule=resolve_wsd_schedule(2000))

    @unittest.skipIf(importlib.util.find_spec('torch') is None, 'Torch is available in the GPU runtime image only')
    def test_real_torch_optimizer_lr_and_checkpoint_resume(self):
        import torch
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=.03)
        scheduler = build_wsd_scheduler(optimizer, 10, warmup_ratio=.2, stable_ratio=.2)
        values = [optimizer.param_groups[0]['lr']]
        for step in range(1, 11):
            optimizer.step()
            scheduler.step()
            values.append(optimizer.param_groups[0]['lr'])
            if step == 5:
                saved_optimizer, saved_scheduler = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
        expected = [.03 * wsd_lr_factor(step, scheduler.wsd_schedule) for step in range(11)]
        self.assertEqual(values, expected)
        restored_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.03)
        restored = build_wsd_scheduler(restored_optimizer, 10, warmup_ratio=.2, stable_ratio=.2)
        restored_optimizer.load_state_dict(saved_optimizer)
        restored.load_state_dict(saved_scheduler)
        for step in range(6, 11):
            restored_optimizer.step()
            restored.step()
            self.assertEqual(restored_optimizer.param_groups[0]['lr'], expected[step])
        incompatible = build_wsd_scheduler(restored_optimizer, 11, warmup_ratio=.2, stable_ratio=.2)
        with self.assertRaisesRegex(ValueError, 'resume schedule differs'):
            incompatible.load_state_dict(saved_scheduler)


if __name__ == '__main__':
    unittest.main()
