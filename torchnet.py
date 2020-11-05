import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

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
        # Weights and bias of nn layers are of float32 type by default. But float numpy array is float64 by default.
        # To be consistent, float method is invoked to convert float64 to float32.
        if self.transformer is not None:
            return torch.from_numpy(self.transformer([self.data[i]])).float(), self.label[i, np.newaxis], i
        else:
            return torch.from_numpy(self.data[i]).float(), self.label[i, np.newaxis], i
        
        
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
