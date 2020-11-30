"""
Module for helper functions.
"""

import numpy as np

def random_split(data, labels, ratio=0.7):
    # convert to numpy array if not
    data = np.array(data)
    labels = np.array(labels)
    
    data1_num = int(data.shape[0] * ratio)
    data1_idx = np.random.choice(range(data.shape[0]), size=data1_num, replace=False)
    data2_idx = [i for i in range(data.shape[0]) if i not in data1_idx]
    
    return data[data1_idx], labels[data1_idx], data[data2_idx], labels[data2_idx]
