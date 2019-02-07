import logging

import torch
import torch.nn as nn

from ..nn.utils import layer_has_bias

# NOTE: should factor be {smaller, default at all}?
# TODO: Research - is there a better type for layer than nn.Module?
def widen(layer: nn.Module, factor=1.4, in_place=False) -> nn.Module:
    """
    Args:
        layer: a torch.nn.Module that is a particular layer. I'm __sure__ the abstraction I'm about
          to articulate does __not__ handle the prospect of 
        factor: a float of some reasonable size. Given that exploding the width of your layers may be
          a bad idea, we encourage values in [1.0, 2.0].
          * 1.0 for the sake of proving that we didn't break anything
          * 2.0 for rapid up-sizing and wild experiments
          NOTE: All multiplications are rounded to nice integers, then to the next even number
            i.e. 1 * 1.3 = 1.3; round(1.3) == 1; 4 * 1.4 = 5.6; round(5.6) = 6

    Returns:
        A new layer of the base type (e.g. nn.Linear) or `None` if in_place=True
    """
    if factor < 1.0:
        raise ValueError('Cannot shrink with the widen() function')
    if factor == 1.0:
        raise ValueError("You shouldn't waste compute time if you're not changing anything")
    # we know that layer.weight.size()[0] is the __output__ dimension in the linear case
    output_dim = 0
    if isinstance(layer, nn.Linear):
        output_dim = layer.weight.size()[0] # FIXME: switch to layer.out_features?
        input_dim = layer.weight.size()[1] # FIXME: switch to layer.in_features?
    # TODO: other classes, for robustness?
    # TODO: Use dictionary look-ups instead, because they're faster?
    else:
        raise ValueError('unsupported layer type:', type(layer))
    
    logging.debug(f"current dimensions: {(output_dim, input_dim)}")
    
    new_size = round(factor * output_dim + .5) # round up, not down, if we can
    
    # We're increasing layer width from output_dim to new_size, so let's save that for later
    size_diff = new_size - output_dim
    # make the difference even, because it's easier to surrond the old weights
    if size_diff % 2 == 1: size_diff += 1
    
    with torch.no_grad():        
        
        expanded_weights = _expand_bias_or_weight(layer.weight, size_diff)
        expanded_bias = _expand_bias_or_weight(layer.bias, size_diff)

        # parameterize weights and bias
        p_weights, p_bias = nn.Parameter(expanded_weights), nn.Parameter(expanded_bias)

        # TODO: cleanup duplication? Missing properties that will effect usability?
        if in_place:
            layer.out_features = new_size
            layer.weight = p_weights
            layer.bias = p_bias
            logging.warning('Using experimental "in-place" version. May have unexpected affects on activation.')
            return layer
        else:
            print(f"New shape = {expanded_weights.shape}")
            l = nn.Linear(*expanded_weights.shape[::-1], bias=utils.layer_has_bias(layer))
            l.weight = p_weights
            l.bias = p_bias
            return l

def _expand_bias_or_weight(t: nn.Parameter, increase: int) -> torch.Tensor:
    """Returns a tensor of shape `t`, with padding values drawn from a Guassian distribution
    with \mu = `torch.mean(layer.weight, dim=0)` and \sigma = `torch.std(layer.weight, dim=0)`.
    Effectively does a column-wise normal distribution.
    
    Args:
        t: `weight` or `bias` attribute of some torch.nn.<Layer> (e.g. `nn.Linear`)
        increase: the amount by which to increase the parameter `t`.
            For example: if `t` has size (5, 2) - meaning it has an output dimension of 5, 
                input dimension of 2 - and `count` is 2, the dimensions of the returned
                Tensor is (7, 2)                    

    Returns:
        the original Parameter, surrounded by `count // 2` addition neurons on either side
    """
    logging.debug(f"_expand_bias_or_weight() t = {t}")
    logging.debug(f"_expand_bias_or_weight() increase = {increase}")
    with torch.no_grad():
        # take the std and mean so we can sample from that normal distribution
        std = torch.std(t, dim=0)
        mean = torch.mean(t, dim=0)
    
        # We sandwich the existing tensor between new tensors with tensor.cat(t_0, t_1, ... t_n)
        # the data takes the following shape (all at the same dimension)
        # [samples * (increase // 2), t.data, more_samples * (increase // 2)],

        new_values = torch.normal(mean, std).unsqueeze(0)
        
        for _ in range(increase // 2 - 1):
            new_values = torch.cat((torch.normal(mean, std).unsqueeze(0), new_values))
        
        new_values = torch.cat((new_values, t.data))
        
        for _ in range(increase // 2):
            new_values = torch.cat((new_values, torch.normal(mean, std).unsqueeze(0)))
        
        return new_values
