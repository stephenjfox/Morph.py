from ._error import ValidationError
import torch.nn as nn

def check(pred: bool, message='Validation failed'):
    if not pred: raise ValidationError(message)


def round(value: float) -> int:
    """Rounds a `value` up to the next integer if possible.
    Performs differently from the standard Python `round`
    """
    return int(value + .5)


# courtesy of https://pytorch.org/tutorials/beginner/nn_tutorial.html#nn-sequential
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
