from ._error import ValidationError


def check(pred: bool, message='Validation failed'):
    if not pred: raise ValidationError(message)


def round(value: float) -> int:
    """Rounds a `value` up to the next integer if possible.
    Performs differently from the standard Python `round`
    """
    return int(value + .5)