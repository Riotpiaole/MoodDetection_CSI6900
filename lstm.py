from torch.nn.modules import dropout
from torch.nn.modules.sparse import Embedding
from transformers import BertTokenizer
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
from torch.autograd import Variable
from torch import optim
import math
import numpy as np

from constants import MAX_SEQUENCE, VOCAB_SIZE

from pdb import set_trace

class ContextualEmotionLSTM(nn.Module):
    def __init__(self, 
        num_steps=5, 
        num_layers=20, 
        input_size=128, 
        hidden_size= 128, 
        contextual_type = 8, 
        output_size=1,
        is_training = True,
        bias=True):

        super(ContextualEmotionLSTM, self).__init__()

        self.num_steps = num_steps
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.contextual_type = contextual_type
        self.bias = bias
        self.output_size = output_size
        self.max_sequence_size = MAX_SEQUENCE
        

        self.contextual_lstm = nn.LSTM(
            input_size = self.input_size * 2,
            hidden_size = self.hidden_size ,
            num_layers=self.num_layers,
            dropout= 0.2 if is_training else 0.0
        )
    
        self.embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=input_size)
        
        self.emotion_linear = nn.Linear(
            self.hidden_size,
            contextual_type
        )

        self.next_word_linear= nn.Linear( self.hidden_size , VOCAB_SIZE)

    def init_state(self):

        if next(self.parameters()).is_cuda:
            return ( 
                torch.zeros(self.num_layers, self.max_sequence_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, self.max_sequence_size, self.hidden_size).cuda()
            )
        return ( 
            torch.zeros(self.num_layers, self.max_sequence_size, self.hidden_size),
            torch.zeros(self.num_layers, self.max_sequence_size, self.hidden_size)
        )

    def merge_text_context(self, inputs, context):
        
        embedded_tokens = self.embedding(inputs)
        embedded_context = self.embedding(context)
        concated_contextual_tokens = torch.cat([embedded_tokens, embedded_context], dim=2)
        return concated_contextual_tokens


    def forward(self, inputs, context, prev_state):
        inputs = self.merge_text_context(inputs, context)

        lstm_output, state = self.contextual_lstm(inputs, prev_state)

        emotion_prediction = self.emotion_linear(lstm_output)
        next_context_prediction = torch.sigmoid(emotion_prediction)
        
        linearized_next_word = self.next_word_linear(lstm_output)
        next_word_prediction = torch.sigmoid(linearized_next_word) 
        
        return lstm_output, next_word_prediction , next_context_prediction, state



def predict(model, text, next_words=1):
    words = text.split(' ')
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    state_h, state_c = model.init_state(len(words))
    x = tokenizer.encode_plus(
             text, add_special_tokens=True,
                max_length = MAX_SEQUENCE,           # Pad & truncate all sentences.
                truncation=True,
                padding='max_length',
                return_tensors='pt')
    outputs = []
    for i in range(0, next_words):
        
        y_pred, next_word = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        outputs.append(tokenizer.decode([word_index]))

    return words