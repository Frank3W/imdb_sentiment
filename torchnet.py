import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

## TODO: support dtype for returned data from __getitem__- float is for network input.
class InMemTorchData(Dataset):
    """Simple pytorch dataset class for in-memory feature and label data that support __getitem__ function.
    """
    def __init__(self, data, label, transformer=None):
        self.data = np.array(data)
        self.label = np.array(label)

        if transformer is not None and not callable(transformer):
            raise ValueError('Transformer {} must be callable if not None.'.format(transformer))
        else:
            self.transformer = transformer
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        if self.transformer is not None:
            return self.transformer([self.data[i]]), self.label[i], i
        else:
            return self.data[i], np.array([self.label[i]]), i
        
        
class FullNet(torch.nn.Module):
    def __init__(self, actfunc_type, * args):
        """Define network components
        """
        
        super(FullNet, self).__init__()
        
        self.layers = []
        
        layer_size_list = list(args)
        # if last item is not 1, then 1 is appended.
        if layer_size_list[-1] != 1:
            layer_size_list.append(1)
        
        for i in range(len(layer_size_list)-1):
            self.layers.append(torch.nn.Linear(layer_size_list[i], layer_size_list[i+1]))
            
        self.actfunc_type = actfunc_type
            
        
    def forward(self, x):
        num_layers = len(self.layers)
        
        for i in range(num_layers-1):
            x = self.layers[i](x)
            if self.actfunc_type == 'relu':
                x = F.relu(x)
            else:
                raise ValueError('active function type only support relu.')
        
        x = self.layers[-1](x)
        
        return x
