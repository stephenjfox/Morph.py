def type_name(o):
    '''Returns the simplified type name of the given object.
    Eases type checking, rather than any(isinstance(some_obj, _type) for _type in [my, types, to, check])
    '''
    return type(o).__name__


def type_supported(type_name: str) -> bool:
    # NOTE: already considerd a constants file. I don't like that precident
    return type_name in ['Conv2d', 'Linear']
