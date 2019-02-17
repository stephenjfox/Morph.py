import torch.nn as nn

import logging
# TODO: nope. This is really long
from morph.nn.utils import group_layers_by_algo, make_children_list, new_input_layer, new_output_layer, out_dim, redo_layer, type_name, type_supported
from morph._utils import round


def resize_layers(net: nn.Module, width_factor: float = 1.4) -> nn.Module:

    old_layers = make_children_list(net.named_children())
    (first_name, first_layer), middle, last = group_layers_by_algo(old_layers)

    first_layer_output_size = first_layer.out_channels  # count of the last layer's out features

    new_out_next_in = int(first_layer_output_size * width_factor)

    # NOTE: is there a better way to do this part? Maybe nn.Sequential?
    network = nn.Module()  # new network

    network.add_module(
        first_name,
        new_input_layer(first_layer, type_name(first_layer), out_dim=new_out_next_in))

    # TODO: format and utilize the functions in utils for making layers
    for name, child in middle:
        if type_supported(type_name(child)):

            new_out = round(out_dim(child) * width_factor)

            new_layer = redo_layer(child, new_in=new_out_next_in, new_out=new_out)
            new_out_next_in = new_out
            network.add_module(name, new_layer)
        else:
            logging.warning(f"Got an incompatible layer: {type(child)}")
            network.add_module(f'{name}_INCOMPATIBLE', child)

    last_name, last_layer = last
    network.add_module(
        last_name,
        new_output_layer(last_layer, type_name(last_layer), in_dim=new_out_next_in))

    return network