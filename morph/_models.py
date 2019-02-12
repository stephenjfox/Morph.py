import torch
import torch.nn as nn
import torch.nn.functional as F

class EasyMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 1000)
        self.linear2 = nn.Linear(1000, 30)
        self.linear3 = nn.Linear(30, 10)

    def forward(self, x_batch: torch.Tensor):
        """Simple ReLU-based activations through all layers of the DNN.
        Simple and effectively deep neural network. No frills.
        """
        _input = x_batch.view(-1, 784) # shape for our linear1
        out1 = F.relu(self.linear1(x_batch))
        out2 = F.relu(self.linear2(out1))
        out3 = F.relu(self.linear3(out2))

        return out3