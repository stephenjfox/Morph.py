import torch.nn as nn

# TODO: nope. This is really long
from morph.nn.utils import group_layers_by_algo, layer_is_conv2d, make_children_list, new_input_layer, new_output_layer, redo_layer, type_name, type_supported


# TODO: refactor out width_factor
def resize_layers(net: nn.Module):

    old_layers = make_children_list(net.named_children())
    (first_name, first_layer), middle, last = group_layers_by_algo(old_layers)

    first_layer_output_size = first_layer.out_channels  # count of the last layer's out features

    new_out_next_in = int(first_layer_output_size * 1.4)

    # NOTE: is there a better way to do this part? Maybe nn.Sequential?
    network = nn.Module()  # new network

    network.add_module(
        first_name,
        new_input_layer(first_layer, type_name(first_layer), out_dim=new_out_next_in))

    # TODO: format and utilize the functions in utils for making layers
    for name, child in middle:
        # otherwise, we want to
        _t = type_name(child)
        if type_supported(_t):

            new_out = 0
            # TODO: look up performance on type name access. Maybe this could just be layer_is_conv2d(child)
            if layer_is_conv2d(_t):
                new_out = int(child.out_channels * 1.4)
            else:  # type_name == 'Linear'
                new_out = int(child.out_features * 1.4)

            new_layer = redo_layer(child, new_in=new_out_next_in, new_out=new_out)
            new_out_next_in = new_out
            network.add_module(name, new_layer)

    last_name, last_layer = last
    network.add_module(
        last_name,
        new_output_layer(last_layer, type_name(last_layer), in_dim=new_out_next_in))

    return network