from unittest import main as test_main, TestCase, skip

from .sparse import sparsify, torch

class TestSparseFunctions(TestCase):
    
    @skip("Skipping value-wise comparison until better solution than iterating all tensor values")
    def test_sparsify_selected_indices_should_have_sub_threshold_values(self):
        test_threshold = 0.1
        test_tensor = torch.randn(3, 2)
        expected = torch.where(test_tensor > test_threshold, test_tensor, torch.zeros(3, 2))
        self.assertEqual(expected, sparsify(test_tensor, test_threshold))


if __name__ == "__main__":
    test_main()