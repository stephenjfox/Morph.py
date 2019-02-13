import torch.nn as nn

from morph.nn.utils import make_children_list, group_layers_by_algo

# TODO: refactor out width_factor
def new_resize_layers(net: nn.Module):
    
    old_layers = make_children_list(net.named_children())
    (first_name, first_layer), middle, last = group_layers_by_algo(old_layers)
    
    last_out = first_layer.out_channels # count of the last layer's out features
    
    new_out_next_in = int(last_out * 1.4)
    
    # NOTE: is there a better way to do this part?
    network = nn.Module() # new network
    
    network.add_module(first_name, nn.Conv2d(
        first_layer.in_channels, new_out_next_in, kernel_size=first_layer.kernel_size,
        stride=first_layer.stride
    ))
    
    # TODO: format and utilize the functions in utils for making layers
    for name, child in middle:        
        # otherwise, we want to 
        type_name = type(child).__name__
        if type_name in ['Conv2d', 'Linear']:

            temp = 0
            if type_name == 'Conv2d':
                temp = int(child.out_channels * 1.4)
                network.add_module(name, nn.Conv2d(
                    new_out_next_in, out_channels=temp,
                    kernel_size=child.kernel_size, stride=child.stride
                ))
            else: # type_name == 'Linear'
                temp = int(child.out_features * 1.4)
                network.add_module(name, nn.Linear(
                    in_features=new_out_next_in, out_features=temp
                ))
                
            new_out_next_in = temp
    
    last_name, last_layer = last
    network.add_module(last_name, nn.Conv2d(
        new_out_next_in, last_layer.out_channels,
        kernel_size=last_layer.kernel_size, stride=last_layer.stride
    ))
    
    return network