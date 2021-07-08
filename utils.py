from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from math import floor 

def train_test_split(dataset, shuffle=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split , test_split = int(floor(dataset_size * .2 )) , int(floor(dataset_size * .3))

    test_indices , val_indices = indices[0:test_split] , indices[test_split:test_split + validation_split]
    
    train_indices = indices[test_split + validation_split]

    train_samples = SubsetRandomSampler(train_indices,)
    validation_samples = SubsetRandomSampler(val_indices)
    test_samples = SubsetRandomSampler(test_indices)

    return train_samples , test_samples , validation_samples 
