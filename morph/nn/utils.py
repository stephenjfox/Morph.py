import torch.nn as nn

from morph.nn._types import type_name, type_supported
from morph._utils import check

from typing import List, Tuple, TypeVar

ML = TypeVar('MODULES', List[nn.Module])
# Type constrained to be the results of nn.Module.children() or ...named_children()
CL = TypeVar('MODULE_CHILDREN_LIST', ML, List[Tuple[str, nn.Module]])


def group_layers_by_algo(children_list: CL) -> ML:
    """Group the layers into how they will be acted upon by my implementation of the algorithm:
    1. First child in the list (the "input" layer)
    2. Slice of all the children, those that are not first nor last
    3. Last child in the list (the "output" layer)
    """

    list_len = len(children_list)

    # validate input in case I slip up
    check(list_len > 1, 'Your children_list must be more than a singleton')

    if list_len <= 2:
        return children_list  # interface?

    first = children_list[0]
    middle = children_list[1:-1]
    last = children_list[-1]

    return first, middle, last


def layer_has_bias(layer: nn.Module) -> bool:
    return not layer.bias is None


def make_children_list(children_or_named_children):
    """Receives `nn.Module.children()` or `nn.Module.named_children()`.
    Returns: that generator collected as a list
    """
    return [c for c in children_or_named_children]


#################### LAYER INSPECTION ####################


def in_dim(layer: nn.Module) -> int:
    check(type_supported(layer))

    if layer_is_linear(layer):
        return layer.in_features
    elif layer_is_conv2d(layer):
        return layer.in_channels
    else:
        raise RuntimeError('Inspecting on unsupported layer')


def out_dim(layer: nn.Module) -> int:
    check(type_supported(layer))

    if layer_is_linear(layer):
        return layer.out_features
    elif layer_is_conv2d(layer):
        return layer.out_channels
    else:
        raise RuntimeError('Inspecting on unsupported layer')


#################### NEW LAYERS ####################


def new_layer(base_layer: nn.Module, type_name: str, in_dim: int,
              out_dim: int) -> nn.Module:

    has_bias = layer_has_bias(base_layer)

    if layer_is_linear(type_name):
        return nn.Linear(in_dim, out_dim, bias=has_bias)

    if layer_is_conv2d(type_name):
        return nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            bias=has_bias)

    raise ValueError('User got around type check ;)')


def new_input_layer(base_layer: nn.Module, type_name: str, out_dim: int) -> nn.Module:
    has_bias = layer_has_bias(base_layer)

    if layer_is_linear(type_name):
        return nn.Linear(base_layer.in_features, out_features=out_dim, bias=has_bias)

    if layer_is_conv2d(type_name):
        return nn.Conv2d(
            base_layer.in_channels,
            out_channels=out_dim,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            bias=has_bias)


def new_output_layer(base_layer: nn.Module, type_name: str, in_dim: int) -> nn.Module:
    has_bias = layer_has_bias(base_layer)

    if layer_is_linear(type_name):
        return nn.Linear(in_dim, base_layer.out_features, bias=has_bias)

    if layer_is_conv2d(type_name):
        return nn.Conv2d(
            in_dim,
            base_layer.out_channels,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            bias=has_bias)


def redo_layer(layer: nn.Module, new_in=None, new_out=None) -> nn.Module:
    if new_in is None and new_out is None:
        return layer

    _type = type_name(layer)
    if not type_supported(_type):
        raise ValueError('Unsupported layer type:', _type)

    received_new_input = new_in is not None
    received_new_output = new_out is not None

    if received_new_input and received_new_output:
        return new_layer(layer, _type, new_in, new_out)

    if received_new_input:
        # we need a new input dim, but retain the same output dim
        return new_output_layer(layer, _type, new_in)

    if received_new_output:
        # we need a new output dim, but retain the same input dim
        return new_input_layer(layer, _type, new_out)


#################### TYPE HELPERS ####################


def layer_is_conv2d(name: str):
    return name == 'Conv2d'


def layer_is_linear(name: str):
    return name == 'Linear'
