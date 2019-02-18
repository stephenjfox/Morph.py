from collections import namedtuple

Resizing = namedtuple('Resizing', ['input_size', 'output_size'], defaults=[0, 0])
Resizing.__doc__ += ': Baseclass for a type that encapsulates a resized layer'
Resizing.input_size.__doc__ = "The layer's \"new\" input dimension size (Linear -> in_features, Conv2d -> in_channels)"
Resizing.output_size.__doc__ = "The layer's \"new\" output dimension size (Linear -> out_features, Conv2d -> out_channels)"
