import torch
import torch.nn as nn
from torch.nn.modules import dropout
from transformers import  MobileBertForMaskedLM
from pdb import set_trace
import pickle


class ContextualBert(nn.Module):
    def __init__(self):
        super().__init__()
        D_in, H, D_out = 30522, 50, 8
        self.bert_maskedLM_word_predictor = MobileBertForMaskedLM.from_pretrained(
            'google/mobilebert-uncased',
            return_dict=True,
            output_hidden_states =False,
            output_attentions= False,)
            
        self.linear_in = nn.Linear(D_in, H)
        self.relu = nn.ReLU(inplace =False)
        self.drop_out = nn.Dropout(0.2)
        self.output = nn.Linear(H, D_out)
        
        self.classifier_loss = torch.nn.BCEWithLogitsLoss()

    def conditionizing_freeze_bert(self, cond=False):
        for param in self.bert_maskedLM_word_predictor.parameters():
            param.requires_grad = cond


    def forward(self, input_ids, attention_mask , masked_label=None):
        res = self.bert_maskedLM_word_predictor(
            input_ids=input_ids,
            labels=masked_label,
            attention_mask=attention_mask)
        try:
            loss , logits = res['loss'] , res['logits']
        except KeyError:
            loss , logits = -1, res['logits']
        last_hidden_state_cls = logits[: , 0 , : ]
        
        outs= self.linear_in(last_hidden_state_cls)
        relued_outs = self.relu(outs)
        dropped_outs = self.drop_out(relued_outs)

        clsifer_loss = self.output(dropped_outs)
        clsifer_loss = torch.exp(clsifer_loss)
        return loss, clsifer_loss , logits



if __name__ == "__main__":
    from dataset.constants import LMBERT_PKL
    from dataset.ConditionalSequenceDataSet import ConditionalSequenceDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim

    with open(LMBERT_PKL, 'rb') as handler:
        dicts = pickle.load(handler)
    train_data, test_data , val_data = dicts['train_dataset'], dicts['test_dataset'] , dicts['val_dataset']
    dataset = ConditionalSequenceDataset(*train_data)
    loader = DataLoader(dataset, 32, False)
    model = ContextualBert()
    
    bert_optimizer = optim.Adam(model.parameters(), lr=1e-5)

    binary_cross_entropy = torch.nn.BCEWithLogitsLoss()

    for i, (masked_lm_text, attention_mask, masked_label , emo_token ) in enumerate(loader):
        
        emo_token = emo_token.float()

        model.conditionizing_freeze_bert(True)
        loss, clsifer_loss = model(masked_lm_text, attention_mask,  masked_label)
        bert_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        model.conditionizing_freeze_bert(False)
        cross_nn_loss = binary_cross_entropy(clsifer_loss, emo_token)
        cross_nn_loss.backward()

        bert_optimizer.step()