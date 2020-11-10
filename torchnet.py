import logging

import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

logger = logging.getLogger(__name__)


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
        
        layers = []
        
        layer_size_list = list(args)
        # if last item is not 1, then 1 is appended.
        if layer_size_list[-1] != 1:
            layer_size_list.append(1)
        
        for i in range(len(layer_size_list) - 1):
            layers.append(torch.nn.Linear(layer_size_list[i], layer_size_list[i+1]))
           
        # nn.ModuleList ensure property parameters populated for layers.
        self.layers = torch.nn.ModuleList(layers)
            
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

def train_bclassif(model, optimizer, epoch_num, fit_dataloader, val_data, val_label):
    # use GPU if exists
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    val_loss_list = []
    val_rocauc_list = []

    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epoch_num):
        logger.debug(f'epoch {epoch} starts')
        for fit_data_batch, fit_label_batch, idx_batch in fit_dataloader:
            output = model(fit_data_batch.to(device))
            loss_output = loss_fn(output, fit_label_batch.float().to(device))
            loss_output.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            val_output = model(val_data.to(device))
            val_loss = loss_fn(val_output, torch.from_numpy(val_label).reshape((-1, 1)).float().to(device))
            val_loss_list.append(val_loss)

            val_pred_np = val_output.cpu().numpy().flatten()

            val_rocauc = metrics.roc_auc_score(val_label, val_pred_np)
            val_rocauc_list.append(val_rocauc)
            logger.debug(f'roc auc on validation {val_rocauc}')
            logger.debug(f'loss on validation {val_loss}')
        
    return val_loss_list, val_rocauc_list
