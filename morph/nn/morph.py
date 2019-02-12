import torch.nn as nn

def once(net: nn.Module, experimental_support=False) -> nn.Module:
    """Runs an experimental implementation of the MorphNet algorithm on `net`
      producing a new network:
      1. Shrink the layers o

    Returns: either `net` if `experimental_support == False` or a MorphNet of
        the supplied `net`.
    """
    # TODO: run the algorithm
    return net