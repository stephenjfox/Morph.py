import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import morph
import morph.nn as net
from morph.layers.sparse import sparsify

from morph.testing.models import EasyMnist


def random_dataset():
    return TensorDataset(torch.randn(2, 28, 28))

def main():
    my_model = EasyMnist()
    # do one pass through the algorithm
    modified = morph.once(my_model)

    print(modified) # take a peek at the new layers. You take it from here

    my_dataloader = DataLoader(random_dataset())

    # get back the class that will do work
    morphed = net.Morph(my_model, epochs=5, dataloader=my_dataloader)

    # TODO: we need your loss function, but this is currentry __unsupported__
    morphed.run_training()


if __name__ == '__main__':
    main()  # TODO: add commandline arguments?
