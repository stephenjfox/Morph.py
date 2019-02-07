from ..layers.sparsify import sparsify

import torch
import torch.nn as nn



#################### HELPERS ####################

def _group_layers_by_algo(children_list):
    """Group the layers into how they will be acted upon by my implementation of the algorithm:
    1. First child in the list
    2. Slice of all the child, those that are not first nor last
    3. Last child in the list
    """
    
    list_len = len(children_list)
    
    # validate input in case I slip up
    if list_len < 1:
        raise ValueError('Invalid argument:', children_list)
    
    if list_len <= 2:
        return children_list # interface?

    
    first = children_list[0]
    middle = children_list[1:-1]
    last = children_list[-1]
    
    return first, middle, last

def _percent_waste(layer: nn.Module) -> float:
    """Computes the number of sparse neurons in a weight matrix,
    given the supplied layer.

    Return: percentage as a float. Multiply with the `n` of an `m` x `n`
      weight matrix/tensor to determine how many neurons can be spared
    """
    w = layer.weight
    non_sparse_w = torch.nonzero(sparsify(w))
    non_zero_count = non_sparse_w.numel() // len(non_sparse_w[0])

    percent_size = non_zero_count / w.numel()

    return percent_size