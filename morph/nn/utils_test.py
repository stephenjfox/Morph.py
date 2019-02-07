import torch.nn as nn
import unittest

from utils import new_input_layer, new_output_layer


class TestLayer(unittest.TestCase):
    def test_new_input_layer_only_touches_output_Linear(self):
        test = new_input_layer(nn.Linear(5, 4), 'Linear', 7)
        expected = nn.Linear(5, 7)

        self.assertEqual(expected.out_features, test.out_features,
                         "Changing the output dimension should be successful")

        self.assertEqual(expected.in_features, test.in_features,
                         "The input dimensions should be unchanged")

    def test_new_output_layer_only_touches_output_Linear(self):
        test = new_output_layer(nn.Linear(5, 4), 'Linear', 7)
        expected = nn.Linear(7, 4)

        self.assertEqual(expected.in_features, test.in_features,
                         "Changing the input dimension should be successful")

        self.assertEqual(expected.out_features, test.out_features,
                         "The output dimensions should be unchanged")

    def test_new_output_layer_only_changes_input_Conv2d(self):
        test = new_output_layer(
            nn.Conv2d(3, 12, kernel_size=3, stride=1), 'Conv2d', 6)
        expected = nn.Conv2d(6, 12, 3, 1)

        self.assertEqual(expected.in_channels, test.in_channels,
                         "The input dimension should be the same")

        self.assertEqual(expected.out_channels, test.out_channels,
                         "The output dimension should be the same")

        self.assertEqual(expected.padding, test.padding,
                         "The padding aspect shoud be the same")

    def test_new_input_layer_only_changes_output_Conv2d(self):
        test = new_input_layer(
            nn.Conv2d(3, 12, kernel_size=3, stride=1), 'Conv2d', 16)
        expected = nn.Conv2d(3, 16, 3, 1)

        self.assertEqual(expected.in_channels, test.in_channels,
                         "The input dimension should be the same")

        self.assertEqual(expected.out_channels, test.out_channels,
                         "The output dimension should be the same")

        self.assertEqual(expected.padding, test.padding,
                         "The padding aspect shoud be the same")


if __name__ == "__main__":
    unittest.main()