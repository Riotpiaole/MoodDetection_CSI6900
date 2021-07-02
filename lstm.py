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

# 先定义Contextual LSTM的一个cell
class ContextualLSTMCell(nn.Module):
    """basic Contextual LSTM Cell"""

    def __init__(self, input_size, hidden_size, contextual_type, bias=True, embedding_dim=100, vocab_size= 100000 ):
        super(ContextualLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.contextual_type = contextual_type
        self.bias = bias

        # input gate parameter
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.w_ci = Parameter(Tensor(hidden_size, hidden_size))
        self.w_bi = Parameter(Tensor(hidden_size, contextual_type))
        self.bias_i = Parameter(Tensor(hidden_size, 1))

        # forget gate parameter
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.w_cf = Parameter(Tensor(hidden_size, hidden_size))
        self.w_bf = Parameter(Tensor(hidden_size, contextual_type))
        self.bias_f = Parameter(Tensor(hidden_size, 1))

        # cell memory parameter
        self.w_ic = Parameter(Tensor(hidden_size, input_size))
        self.w_hc = Parameter(Tensor(hidden_size, hidden_size))
        self.w_bc = Parameter(Tensor(hidden_size, contextual_type))
        self.bias_c = Parameter(Tensor(hidden_size, 1))

        # output gate parameter
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.w_co = Parameter(Tensor(hidden_size, hidden_size))
        self.w_bo = Parameter(Tensor(hidden_size, contextual_type))
        self.bias_o = Parameter(Tensor(hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            init.uniform_(parameter, -stdv, stdv)


    def forward(self, x, h, c, topic):
        """
        :param x: 当前时刻的输入
        :param h: 上一时刻的隐状态
        :param c: 上一时刻的记忆单元
        :param b: 当前时刻输入的上下文
        :return:
        """
        # input gate
        ci = torch.sigmoid(self.w_ii @ x + self.w_hi @ h + self.w_ci @ c + self.w_bi @ topic + self.bias_i)

        # forget gate
        cf = torch.sigmoid(self.w_if @ x + self.w_hf @ h + self.w_cf @ c + self.w_bf @ topic + self.bias_f)

        # cell memory
        cc = cf * c + ci * torch.tanh(self.w_ic @ x + self.w_hc @ h + self.w_bc @ topic + self.bias_c)

        # output gate
        co = torch.sigmoid(self.w_io @ x - self.w_ho @ h + self.w_co @ c + self.w_bo @ topic + self.bias_o)

        # hidden state
        ch = co * torch.tanh(cc)

        return ch, cc

    
    def init_state(self, batch_size, hidden_size):
        if self.w_ii.is_cuda:
            h_init = Variable(torch.rand(batch_size, hidden_size).t()).cuda()
            c_init = Variable(torch.rand(batch_size, hidden_size).t()).cuda()
        else:
            h_init = Variable(torch.rand(batch_size, hidden_size).t())
            c_init = Variable(torch.rand(batch_size, hidden_size).t())
        return h_init, c_init


# 定义完整的Contextual LSTM模型
class ContextualLSTM(nn.Module):
    """Contextual LSTM model"""

    def __init__(self, 
        num_steps, 
        num_layers, 
        input_size, 
        hidden_size, 
        contextual_type, 
        bias=True):

        super(ContextualLSTM, self).__init__()

        # 序列长度
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.contextual_type = contextual_type
        self.bias = bias


        self._all_layers = []
        for k in range(self.num_layers):
            layer_name = 'cell{}'.format(k)
            cell = ContextualLSTMCell(self.input_size, self.hidden_size, self.contextual_type, self.bias)
            setattr(self, layer_name, cell)
            self._all_layers.append(cell)

    def forward(self, inputs, contexts):
        internal_state = []
        outputs = []
        for step in range(self.num_steps):
            
            x_step = inputs[step].t()  
            context_step = contexts[step].t()  
            
            for layer in range(self.num_layers):

                layer_name = 'cell{}'.format(layer)

                if step == 0:
                    batch_size = inputs[step].size()[0]

                    h, c = getattr(self, layer_name).init_state(batch_size=batch_size, hidden_size=self.hidden_size)
                    internal_state.append((h, c))

                (h, c) = internal_state[layer]
                x_step, c_new = getattr(self, layer_name)(x_step, h, c, context_step)
                internal_state[layer] = (x_step, c_new)

            outputs.append(x_step)

        return outputs, (x_step, c_new)

class ContextualEmotionLSTM(nn.Module):
    def __init__(self, 
        num_steps=5, 
        num_layers=20, 
        input_size=8, 
        hidden_size= 8, 
        contextual_type = 8, 
        output_size=1,
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
        
        self.contextual_lstm = ContextualLSTM( 
            self.num_steps,
            self.num_layers,
            self.input_size,
            self.hidden_size,
            self.contextual_type, 
            self.bias)

        self.emotion_linear = nn.Linear(
            self.max_sequence_size,
            contextual_type
        )

        self.next_word_linear= nn.Linear( MAX_SEQUENCE , VOCAB_SIZE)
        self.embedding = nn.Embedding(VOCAB_SIZE + 1, input_size)

    def forward(self, inputs, contexts):
        embedded_tokens = self.embedding(inputs)
        output, (h_final, _) = self.contextual_lstm(embedded_tokens, contexts)
        emotion_prediction = self.emotion_linear(h_final[-1])
        result = torch.sigmoid(emotion_prediction)
        
        linearized_next_word = self.next_word_linear(h_final)
        next_word = torch.sigmoid(linearized_next_word) 
        return output , next_word , result



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