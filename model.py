import torch
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss
from pytorch_pretrained_bert.modeling import( 
    BertForPreTraining, 
    BertPreTrainedModel, 
    BertModel, 
    BertConfig, 
    BertForMaskedLM, 
    BertForSequenceClassification,
)

class MobileBertSimilarityComparison(Module):
    def __init__(self):
        return True
    