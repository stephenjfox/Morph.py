from morph.layers.sparse import percent_waste
from morph.utils import check, round
from .resizing import Resizing
from .utils import in_dim, out_dim
from .widen import resize_layers
from .utils import group_layers_by_algo
from ._types import type_name

from typing import List

import torch.nn as nn


def uniform_prune(net: nn.Module) -> nn.Module:
    """Shrink the network down 70%. Input and output dimensions are not altered"""
    return resize_layers(net, width_factor=0.7)


def shrink_layer(layer: nn.Module) -> Resizing:
    pass


def fit_layer_sizes(layer_sizes: List[Resizing]) -> List[Resizing]:
    pass

def transform(original_layer: nn.Module, new_shape: Resizing) -> nn.Module:
    pass

def shrink_prune_fit(net: nn.Module) -> nn.Module:
    first, middle_layers, last = group_layers_by_algo(net)
    shrunk = {
        "first": shrink_layer(first),
        "middle": [shrink_layer(m) for m in middle_layers],
        "last": shrink_layer(last)
    }

    # FIXME: why doesn't the linter like `fitted_layers_in_order`
    fitted_layers_in_order = fit_layer_sizes([shrunk["first"], *shrunk["middle"], shrunk["last"]])

    # iteration very similar to `resize_layers` but matches Resizing
    # with the corresponding layer
    new_first, new_middle_layers, new_last = group_layers_by_algo(fitted_layers_in_order)

    new_net = nn.Module()

    new_net.add_module(type_name(first), transform(first, new_first))

    for old, new in zip(middle_layers, new_middle_layers):
        new_net.add_module(type_name(old), transform(old, new))
        pass  # append to new_net with the Resizing's properties

    new_net.add_module(type_name(last), transform(last, new_last))

    return new_net
