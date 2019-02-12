import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import morph
import morph.nn as net
from morph.layers.sparse import sparsify

from morph._models import EasyMnist

def main():
    my_model = EasyMnist()
    # do one pass through the algorithm
    modified = morph.once(my_model)

    print(modified) # proof that the thing wasn't tampered with

    my_dataloader = DataLoader(TensorDataset(torch.randn(2, 28, 28)))

    # get back the class that will do work
    morphed = net.Morph(my_model, epochs=5, dataloader=my_dataloader)
    morphed.run_training()


if __name__ == '__main__':
    main() # TODO: add commandline arguments?