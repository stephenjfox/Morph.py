import torch.nn as nn
import unittest

from utils import new_input_layer, new_output_layer

class TestLayer(unittest.TestCase):
    
    def test_new_input_layer_only_touches_output(self):
        test = new_input_layer(nn.Linear(5, 4), 'Linear', 7)
        expected = nn.Linear(5, 7)
        
        self.assertEqual(expected.out_features, test.out_features,
            "Changing the output dimension should be successful")
        
        self.assertEqual(expected.in_features, test.in_features,
            "The input dimensions should be unchanged")
    
    def test_new_output_layer_only_touches_output(self):
        test = new_output_layer(nn.Linear(5, 4), 'Linear', 7)
        expected = nn.Linear(7, 4)
        
        self.assertEqual(expected.in_features, test.in_features,
            "Changing the input dimension should be successful")
        
        self.assertEqual(expected.out_features, test.out_features,
            "The output dimensions should be unchanged")

if __name__ == "__main__":
    unittest.main()