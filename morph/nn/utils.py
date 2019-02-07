import torch.nn as nn

def layer_has_bias(layer: nn.Module) -> bool:
    return not layer.bias is None

def make_children_list(children_or_named_children):
    """Receives `nn.Module.children()` or `nn.Module.named_children()`.
    Returns: that generator collected as a list
    """
    return [c for c in children_or_named_children]

def new_layer(base_layer: nn.Module, type_name: str, in_dim: int, out_dim: int) -> nn.Module:

    has_bias = layer_has_bias(base_layer)

    if layer_is_linear(type_name):
        return nn.Linear(in_dim, out_dim, bias=has_bias)
    
    if layer_is_conv2d(type_name):
        return nn.Conv2d(in_dim, out_dim, kernel_size=base_layer.kernel_size,
            stride=base_layer.stride, bias=has_bias)

    raise ValueError('User got around type check ;)')

def new_input_layer(base_layer: nn.Module, type_name: str, out_dim: int) -> nn.Module:
    has_bias = layer_has_bias(base_layer)
    
    if layer_is_linear(type_name):
        return nn.Linear(base_layer.in_features, out_features=out_dim, bias=has_bias)
        
    if layer_is_conv2d(type_name):
        return nn.Conv2d(base_layer.in_channels, out_channels=out_dim,
            kernel_size=base_layer.kernel_size, stride=base_layer.stride, bias=has_bias)

def new_output_layer(base_layer: nn.Module, type_name: str, in_dim: int) -> nn.Module:
    has_bias = layer_has_bias(base_layer)
    
    if layer_is_linear(type_name):
        return nn.Linear(in_dim, base_layer.out_features, bias=has_bias)
        
    if layer_is_conv2d(type_name):
        return nn.Conv2d(in_dim, base_layer.out_channels,
            kernel_size=base_layer.kernel_size, stride=base_layer.stride, bias=has_bias)


def redo_layer(layer: nn.Module, new_in=None, new_out=None) -> nn.Module:
    if new_in is None and new_out is None:
        return layer
    
    _type = type_name(layer)
    if not type_supported(_type):
        raise ValueError('Unsupported layer type:', _type)
    
    if new_in is not None and new_out is not None:
        return new_layer(layer, _type, new_in, new_out)
    
    if new_in is not None:
        # we need a new input dim, but retain the same output dim
        return new_output_layer(layer, _type, new_in)
    
    if new_out is not None:
        # we need a new output dim, but retain the same input dim
        return new_input_layer(layer, _type, new_out)

#################### TYPE HELPERS ####################

def layer_is_conv2d(name: str):
    return name == 'Conv2d'

def layer_is_linear(name: str):
    return name == 'Linear'

def type_name(o):
    '''Returns the simplified type name of the given object.
    Eases type checking, rather than any(isinstance(some_obj, _type) for _type in [my, types, to, check])
    '''
    return type(o).__name__

def type_supported(type_name: str) -> bool:
    return type_name in ['Conv2d', 'Linear']