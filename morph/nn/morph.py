import torch.nn as nn # TODO do we even need to import this __just__ for the type?


def once(net: nn.Module, experimental_support=False) -> nn.Module:
    """Runs an experimental implementation of the MorphNet algorithm on `net`
      producing a new network:
      1. Shrink the layers
      2. Widen the network
        a. If everything mathematically fits together nicely, try to run inference
          i. initialize those new weights with my random sampling technique
        b. If things aren't snug, apply the more robust layer fitting approach
          i. the layer widths will be what they will and that logic is handled in
            morph.nn.widen.py
      3. Present the new model in a simple dataclass
        a. takes advantage of the generated __repr__ and __eq__
        b. that class will have analysis functions (like `pd.DataFrame.summary()`)
    Returns: either `net` if `experimental_support == False` or a MorphNet of
        the supplied `net`.
    """
    # TODO: run the algorithm
    return net