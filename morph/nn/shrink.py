from morph.layers.sparse import percent_waste
from morph.utils import check, round
from .resizing import Resizing
from .utils import in_dim, out_dim, group_layers_by_algo
from .widen import resize_layers
from ._types import type_name

from typing import List

import torch.nn as nn


class Shrinkage:
    """
    An intermediary for the "Shrink" step of the three step Morphing algorithm.
    Rather than have all of the state be free in the small scope of a mega-function,
    these abstractions ease the way of implementing the shrinking and prune of the
    network.
    * Given that we have access to the total count of nodes, and how wasteful a layer was
      we can deduce any necessary changes once given a new input dimension
    * We expect input dimensions to change to accomodate the trimmed down earlier layers,
      but we want an expansion further along to allow the opening of bottlenecks in the architecture
    """

    def __init__(self, input_dimension: int, initial_parameters: int,
                 waste_percentage: float):
        self.input_dimension = input_dimension  # TODO: is this relevant in any non-Linear case?
        self.initial_parameters = initial_parameters
        self.waste_percentage = waste_percentage
        self.reduced_parameters = Shrinkage.reduce_parameters(initial_parameters,
                                                              waste_percentage)

    @staticmethod
    def reduce_parameters(initial_parameters: int, waste: float) -> int:
        """Calculates the new, smaller, number of paratemers that this instance encapsulates"""
        percent_keep = (1. - waste)
        unrounded_params_to_keep = percent_keep * initial_parameters
        # round digital up to the nearest integer
        return round(unrounded_params_to_keep)


def shrink_to_resize(shrinkage: Shrinkage, new_input_dimension: int) -> Resizing:
    """Given the `new_input_dimension`, calculate a reshaping/resizing for the parameters
    of the supplied `shrinkage`.
    We round up the new output dimension, generously allowing for opening bottlenecks.
    Iteratively, any waste introduced is pruned hereafter. (Needs proof/unit test)
    """
    new_output_dimension = round(shrinkage.reduced_parameters / new_input_dimension)
    return Resizing(new_input_dimension, new_output_dimension)


#################### prove of a good implementation ####################


def uniform_prune(net: nn.Module) -> nn.Module:
    """Shrink the network down 70%. Input and output dimensions are not altered"""
    return resize_layers(net, width_factor=0.7)


#################### the algorithm to end all algorithms ####################


def shrink_layer(layer: nn.Module) -> Shrinkage:
    waste = percent_waste(layer)
    parameter_count = layer.weight.numel()  # the count is already tracked for us
    return Shrinkage(in_dim(layer), parameter_count, waste)


def fit_layer_sizes(layer_sizes: List[Shrinkage]) -> List[Resizing]:
    # TODO: where's the invocation site for shrink_to_resize
    pass


def transform(original_layer: nn.Module, new_shape: Resizing) -> nn.Module:
    # TODO: this might just be utils.redo_layer, without the primitive obsession
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
