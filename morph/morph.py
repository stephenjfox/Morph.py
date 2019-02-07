import torch.nn as nn
from .nn.morph_net import MorphNet

def morph(net: nn.Module, experimental_support=False) -> nn.Module:
    """Experimental morphing algorithm, use with caution.

    Returns: either `net` if `experimental_support == False` or a MorphNet of
        the supplied `net`.
    """
    return MorphNet(net) if experimental_support else net