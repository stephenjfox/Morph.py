from morph.layers.sparse import percent_waste
from morph._utils import check, round
from morph.nn.utils import in_dim, out_dim

import torch.nn as nn


def calc_reduced_size(layer: nn.Module) -> (int, int):
    """Calculates the reduced size of the layer, post training (initial or morphed re-training)
    so the layers can be resized.
    """
    # TODO: remove this guard when properly we protect access to this function
    check(
        type(layer) == nn.Conv2d or type(layer) == nn.Linear,
        'Invalid layer type: ' + type(layer))

    percent_keep = 1 - percent_waste(layer)
    shrunk_in, shrunk_out = percent_keep * in_dim(layer), percent_keep * out_dim(layer)

    return round(shrunk_in), round(shrunk_out)
