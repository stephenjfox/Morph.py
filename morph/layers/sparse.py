import torch
import torch.nn as nn

_ZERO_TENSOR = torch.tensor(0.)

def sparsify(tensor: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Sparsify a `tensor` with `0.0` for all values that do not meet the `threshold`.
    Defaults to rejecting negative numbers.
    Note: does not convert `tensor` to an equivalent `torch.spares.*Tensor` as that API is
    currently experimental.

    Args:
        tensor: 
        threshold: the minimum required activation, else 0.
    
    Returns:
        `torch.Tensor` of the same dimensions as `tensor`
    """
    return tensor.where(tensor > threshold, _ZERO_TENSOR)

def percent_waste(layer: nn.Module) -> float:
    """Computes the number of sparse neurons in a weight matrix,
    given the supplied layer.

    Return: percentage as a float. Multiply with the `n` of an `m` x `n`
      weight matrix/tensor to determine how many neurons can be spared
    """
    w = layer.weight
    non_sparse_w = torch.nonzero(sparsify(w))
    non_zero_count = non_sparse_w.numel() // len(non_sparse_w[0])

    percent_size = non_zero_count / w.numel()

    return percent_size