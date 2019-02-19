from morph.layers.sparse import percent_waste
from morph.utils import check, round
from .utils import in_dim, out_dim
from .widen import resize_layers
from .utils import group_layers_by_algo
from ._types import type_name

from typing import List

import torch.nn as nn


class Shrinkage:

    def __init__(self, initial_parameters: int, waste_percentage: float):
        self.initial_parameters = initial_parameters
        self.waste_percentage = waste_percentage

    @property
    def reduced_parameters(self):
        percent_keep = (1. - self.waste_percentage)
        unrounded_params_to_keep = percent_keep * self.initial_parameters
        # round digital up to the nearest integer
        return round(unrounded_params_to_keep)


def uniform_prune(net: nn.Module) -> nn.Module:
    """Shrink the network down 70%. Input and output dimensions are not altered"""
    return resize_layers(net, width_factor=0.7)


#################### the algorithm to end all algorithms ####################


def shrink_layer(layer: nn.Module) -> Shrinkage:
    waste = percent_waste(layer)
    parameter_count = layer.weight.numel()  # the count is already tracked for us
    return Shrinkage(parameter_count, waste)


def fit_layer_sizes(layer_sizes: List[Shrinkage]) -> List[Shrinkage]:

    pass


def transform(original_layer: nn.Module, new_shape: Shrinkage) -> nn.Module:
    pass


def shrink_prune_fit(net: nn.Module) -> nn.Module:
    first, middle_layers, last = group_layers_by_algo(net)
    shrunk = {
        "first": shrink_layer(first),
        "middle": [shrink_layer(m) for m in middle_layers],
        "last": shrink_layer(last)
    }

    # FIXME: why doesn't the linter like `fitted_layers`
    fitted_layers = fit_layer_sizes([shrunk["first"], *shrunk["middle"], shrunk["last"]])

    # iteration very similar to `resize_layers` but matches Shrinkage with the corresponding layer
    new_first, new_middle_layers, new_last = group_layers_by_algo(fitted_layers)

    new_net = nn.Module()

    new_net.add_module(type_name(first), transform(first, new_first))

    for old, new in zip(middle_layers, new_middle_layers):
        new_net.add_module(type_name(old), transform(old, new))
        pass  # append to new_net with the Shrinkage's properties

    new_net.add_module(type_name(last), transform(last, new_last))

    return new_net
