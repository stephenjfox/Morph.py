import torch.nn as nn

class MorphNet(nn.Module):
    
    @classmethod
    def shrink_out(cls, child_layer):
        new_out = int(child_layer.out_features * _percent_waste(child_layer))
        return nn.Linear(child_layer.in_features, new_out)
    
    def __init__(self, net: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([
            MorphNet.shrink_out(c) for c in net.children()
        ])
            