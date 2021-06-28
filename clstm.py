import torch.nn as nn
from functools import partial

class WordPredictionModel( nn.Module):
    def __init__(self, is_training, config):
        super().__init__(self)
        self.config = config 
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.num_step = config.num_steps

        self._all_layers = []
        for k in range(self.config.num_layers):
            layer_name = 'cell{}'.format(k)
            setattr(self, layer_name, cell)
            self._all_layers.append( 
                nn.LSTMCell(
                    self.config.input_size,
                    dropout=self.config.keep_prob
                )
            )

    def forward(self, input)