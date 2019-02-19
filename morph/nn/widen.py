import torch.nn as nn

import logging

from morph.nn.utils import group_layers_by_algo, make_children_list, out_dim, redo_layer
from morph.utils import round
from morph.nn._types import type_name, type_supported


def widen(net: nn.Module, width_factor: float = 1.4) -> nn.Module:
    return resize_layers(net, width_factor)


def resize_layers(net: nn.Module, width_factor: float = 1.4) -> nn.Module:
    """Perform a uniform layer widening, which increases the output dimension for
    fully-connected layers and the number of filters for convolutional layers.
    """

    old_layers = make_children_list(net.named_children())
    (first_name, first_layer), middle, last = group_layers_by_algo(old_layers)

    first_layer_output_size = out_dim(first_layer)  # count of the last layer's out features

    new_out_next_in = round(first_layer_output_size * width_factor)

    # NOTE: is there a better way to do this part? Maybe nn.Sequential?
    network = nn.Module()  # new network

    network.add_module(first_name, redo_layer(first_layer, new_out=new_out_next_in))

    for name, child in middle:
        if type_supported(type_name(child)):

            new_out = round(out_dim(child) * width_factor)

            new_layer = redo_layer(child, new_in=new_out_next_in, new_out=new_out)
            new_out_next_in = new_out
            network.add_module(name, new_layer)
        elif type_is_nested(child):
            raise NotImplementedError(
                'Currently do not support for nested structures (i.e. ResidualBlock, nn.Sequntial)')
        else:
            logging.warning(f"Encountered a non-resizable layer: {type(child)}")
            network.add_module(name, child)

    last_name, last_layer = last
    network.add_module(last_name, redo_layer(last_layer, new_in=new_out_next_in))

    return network

def type_is_nested(layer: nn.Module) -> bool:
    """Returns true is the `layer` has children"""
    return bool(make_children_list(layer))