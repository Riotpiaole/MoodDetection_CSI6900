import torch 
from torch.utils.data import DataLoader
from lstm import ContextualEmotionLSTM, predict 
import torch.nn as nn
from torch import optim
from pdb import set_trace

def train(model, dataset, epochs, batch_size=32,):
    
    dataloader_config = { 
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 6
    }

    catgorical_cross_critierion = nn.CrossEntropyLoss(
        reduction='mean', 
        weight= torch.ones(
         [8]))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dataset_loader = DataLoader(
        dataset,
        **dataloader_config
    )
    for epoch in range(epochs):
        for text , context in dataset_loader:
            output , next_word_prob , emo_label= model(text,context )
            set_trace()
            emotion_loss = catgorical_cross_critierion(emo_label, context)
            
    return 0

if __name__ == '__main__':
    model = ContextualEmotionLSTM()
    train(model,)