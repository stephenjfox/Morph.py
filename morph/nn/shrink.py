from morph.layers.sparse import percent_waste
from morph.utils import check, round
from .resizing import Resizing
from .utils import in_dim, out_dim
from .widen import resize_layers

import torch.nn as nn


def prune(net: nn.Module) -> nn.Module:
    return resize_layers(net, width_factor=0.7)