import torch

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
