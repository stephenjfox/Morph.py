import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Lambda

class EasyMnist(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 1000)
        self.linear2 = nn.Linear(1000, 30)
        self.linear3 = nn.Linear(30, 10)

    def forward(self, x_batch: torch.Tensor):
        """Simple ReLU-based activations through all layers of the DNN.
        Simple and sufficiently deep neural network. No frills.
        """
        _input = x_batch.view(-1, 784)  # shape for our linear1
        out1 = F.relu(self.linear1(x_batch))
        out2 = F.relu(self.linear2(out1))
        out3 = F.relu(self.linear3(out2))

        return out3


# for comparison with the above
def EasyMnistSeq():
    return nn.Sequential(
        Lambda(lambda x: x.reshape(-1, 784)),
        nn.Linear(784, 1000),
        nn.Relu(),
        nn.Linear(1000, 300),
        nn.Relu(),
        nn.Linear(300, 10),
        nn.Relu(),
    ) 


class MnistConvNet(nn.Module):
    def __init__(self, interim_size=16):
        """
        A simple and shallow deep CNN to show that morph will shrink this architecture,
          which will inherently be wasteful on the task of classifying MNIST digits with
          accuracy above 95%.
        By default produces a 1x16 -> 16x16 -> 16x10 convnet
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, interim_size, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(interim_size, interim_size, kernel_size=3, stride=2, padding=1)        
        self.conv3 = nn.Conv2d(interim_size, 10, kernel_size=3, stride=2, padding=1)
    
    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28) # any batch_size, 1 channel, 28x28 pixels
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        
        # reshape the output to the second dimension of the pool size, and just fill the rest to whatever.
        return xb.view(-1, xb.size(1))
 