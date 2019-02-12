from morph.layers.sparse import percent_waste
import torch.nn as nn
from torch.utils.data import DataLoader

class Morph(nn.Module):
    """An encapsulation of the benefits of MorphNet, namely:
    1. automatically shrinking and widening, to produce a new architecture w.r.t. layer widths
    2. Training of the network, to match (or beat) model performance
    3. 
    """
    
    @classmethod
    def shrink_out(cls, child_layer):
        new_out = int(child_layer.out_features * percent_waste(child_layer))
        return nn.Linear(child_layer.in_features, new_out)
    
    def __init__(self, net: nn.Module, epochs: int, dataloader: DataLoader):
        super().__init__()
        self.layers = nn.ModuleList([
            Morph.shrink_out(c) for c in net.children()
        ])

    def run_training(self):
        """Performs the managed training for this instance"""
        pass