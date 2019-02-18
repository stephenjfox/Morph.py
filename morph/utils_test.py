import unittest

from .utils import round

class TestGlobalUtilities(unittest.TestCase):

    def test_round_down(self):
        test = 1.2
        expected = 1
        self.assertEqual(expected, round(test), '1.2 should round DOWN, to 1')

    def test_round_up(self):
        test = 1.7
        expected = 2
        self.assertEqual(expected, round(test), '1.7 should round UP, to 2')


if __name__ == "__main__":
    unittest.main()