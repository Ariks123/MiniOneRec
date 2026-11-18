"""Exercise the source-pinned compatibility hook without loading DeepSpeed."""
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import deepspeed_zero2_compat as compat


FIXTURE = '''    def independent_gradient_partition_epilogue(self):
        i = 0
        if not i in self.averaged_gradients or self.averaged_gradients[i] is None:
            self.all_grad_tensors[i] = self.next_gradient
        else:
            self.all_grad_tensors[i] += self.next_gradient
'''


def optimizer_class():
    class Optimizer:
        def independent_gradient_partition_epilogue(self):
            raise AssertionError("Unpatched fixture method")
    return Optimizer


class Zero2CompatibilityTests(unittest.TestCase):
    def apply(self, cls, source=FIXTURE, version="0.18.0"):
        with patch.object(compat, "ORIGINAL_SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest()):
            return compat._apply_to_optimizer_class(cls, version, lambda method: source)

    def test_accumulation_preserves_earlier_nonboundary_gradient(self):
        cls = optimizer_class()
        metadata = self.apply(cls)
        instance = cls()
        instance.averaged_gradients, instance.all_grad_tensors = {}, {}
        instance.next_gradient = 2
        instance.independent_gradient_partition_epilogue()
        instance.next_gradient = 0
        instance.independent_gradient_partition_epilogue()
        self.assertEqual(instance.all_grad_tensors[0], 2)
        instance.next_gradient = 4
        instance.independent_gradient_partition_epilogue()
        self.assertEqual(instance.all_grad_tensors[0], 6)
        instance.next_gradient = -6
        instance.independent_gradient_partition_epilogue()
        self.assertEqual(instance.all_grad_tensors[0], 0)
        self.assertTrue(metadata["applied"])
        self.assertFalse(metadata["installed_package_modified"])
        self.assertEqual(metadata["original_source_sha256"], hashlib.sha256(FIXTURE.encode()).hexdigest())
        self.assertEqual(metadata["patched_source_sha256"], hashlib.sha256(
            FIXTURE.replace(compat.ORIGINAL_CONDITION, compat.CORRECTED_CONDITION).encode()).hexdigest())

    def test_repeated_application_is_idempotent(self):
        cls = optimizer_class()
        first = self.apply(cls)
        method = cls.independent_gradient_partition_epilogue
        second = self.apply(cls)
        self.assertIs(cls.independent_gradient_partition_epilogue, method)
        self.assertFalse(first["already_applied"])
        self.assertTrue(second["already_applied"])
        first["reason"] = "caller mutation"
        self.assertNotEqual(self.apply(cls)["reason"], "caller mutation")

    def test_unknown_version_or_source_refuses_to_change_class(self):
        cls = optimizer_class()
        original = cls.independent_gradient_partition_epilogue
        with self.assertRaisesRegex(RuntimeError, "supports only"):
            self.apply(cls, version="0.18.1")
        with self.assertRaisesRegex(RuntimeError, "source SHA256"):
            compat._apply_to_optimizer_class(cls, "0.18.0", lambda method: FIXTURE + "# changed\n")
        self.assertIs(cls.independent_gradient_partition_epilogue, original)
        self.assertNotIn(compat.MARKER, cls.__dict__)

    def test_missing_or_multiple_conditions_refuse_to_patch(self):
        for source in [FIXTURE.replace(compat.ORIGINAL_CONDITION, "if True:"), FIXTURE + FIXTURE]:
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "exactly one"):
                self.apply(optimizer_class(), source=source)

    def test_later_method_replacement_is_detected(self):
        cls = optimizer_class()
        self.apply(cls)
        cls.independent_gradient_partition_epilogue = lambda self: None
        with self.assertRaisesRegex(RuntimeError, "subsequently changed"):
            self.apply(cls)


if __name__ == "__main__":
    unittest.main()
