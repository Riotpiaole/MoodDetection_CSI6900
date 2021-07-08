from platform import processor
import torch 
from prepare_dataset import load_dataset, SequenctialTextDataSet
from lstm import ContextualEmotionLSTM
from pdb import set_trace
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import random
from torch.autograd import Variable
from constants import MAX_SEQUENCE, VOCAB_SIZE , MAXNUM_STEPS

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))



from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset_dict = load_dataset('embedding')
    
    batch_size = 5
    input_size = 128
    hidden_size = 128
    contextual_type_size = 8
    df = dataset_dict['dataset']
    loader = DataLoader(
        df, batch_size=batch_size,
        num_workers=3
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = ContextualLSTM(batch_size, 10, input_size, hidden_size, contextual_type_size).to(device)
    model = ContextualEmotionLSTM(
        num_steps=batch_size,
        num_layers=20,
        input_size=input_size,
        hidden_size=hidden_size,
        contextual_type=contextual_type_size).to(device)

    for token, next_token, context, next_context in loader:
        break
    token = token.long().to(device)
    context = context.long().to(device)
    states = model.init_state()
    lstm_output_prediciton , next_word_predicition , next_emotion_prediction, next_state = model(token, context, states)  
    catgorical_cross_critierion = nn.CrossEntropyLoss(reduction='mean') 
         
    # output, (h_final, c_final) = model.forward(process, contexts)
    # context = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # lines = []
    # line = []
    # for i in range(5):
    #     line = []
    #     for j in range(200):
    #         item = random.choice(context)
    #         line.append(item)
    #     lines.append(line)
    # inputs = Variable(torch.rand(5, 200, 10))
    # content = np.array(lines)