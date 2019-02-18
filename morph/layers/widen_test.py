import unittest

from .widen import widen, nn
from .._error import ValidationError


class TestWiden_Functional(unittest.TestCase):

    DUD_LINEAR = nn.Linear(1, 1)

    def test_widen_width_factor_too_small_should_fail(self):
        with self.assertRaises(ValidationError):
            widen(self.DUD_LINEAR, 0.8)

    def test_widen_width_factor_identity_should_fail(self):
        with self.assertRaises(ValidationError):
            widen(self.DUD_LINEAR, 1.0)

    def test_widen_width_factor_increases_layer_generously(self):
        pass


if __name__ == "__main__":
    unittest.main()