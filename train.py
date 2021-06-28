import torch 
from torch.utils.data import DataLoader
from lstm import ContextualEmotionLSTM, predict 
import torch.nn as nn
from torch import optim

def train(model, dataset, epochs, batch_size=32,):
    
    dataloader_config = { 
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 6
    }

    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dataset_loader = DataLoader(
        dataset,
        **dataloader_config
    )
    for epoch in epochs:
        for text , context in dataset_loader:
            output , (hidden_state, control_state), emo_label= model(text,context )
            loss = criterion(emo_label, context)
            
    return 0