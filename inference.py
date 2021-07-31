import torch 
from contextual_bert import ContextualBert 
from transformers import MobileBertForMaskedLM, BertTokenizer
from dataset.constants import LMBERT_PKL , get_model_file_path
import pickle
import string
from pdb import set_trace

def decode_multi_index(tokenizer, pred_logits, attention_masks, top_clean):
    res = []
    for attention_mask in attention_masks:
        res.append(
            decode(tokenizer, pred_logits , attention_mask, top_clean))
    return res

def decode(tokenizer, pred_logits, attention_mask, top_clean=5):
    pred_idx = pred_logits[:, attention_mask, :][0].topk(10).indices.tolist()
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for word in pred_idx:
        token = tokenizer.decode([word])
        if '##' not in token or token not in ignore_tokens:
            tokens.append(token)
        elif '##' in token:
            res = word.replace('##', "" )
            tokens[-1] += res
    
    return tokens[:top_clean]

def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('[MASK]', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    
    tokenized_text = tokenizer.tokenize(text_sentence)
    if '[MASK]' not in tokenized_text:
        text_sentence +=  " [MASK]"

    encoded_res= tokenizer.encode_plus( text_sentence, add_special_tokens=add_special_tokens)
    input_ids, attention_mask = torch.tensor(encoded_res['input_ids']) ,encoded_res['attention_mask']  

    if torch.where(input_ids == tokenizer.mask_token_id)[0].sum() > 0:
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[0]
    else: 
        mask_idx = torch.LongTensor()

    return input_ids.unsqueeze(0), mask_idx , torch.tensor(attention_mask).unsqueeze(0)

if __name__ == "__main__":
    from transformers import BertTokenizer
    with open(LMBERT_PKL, 'rb') as handler:
        dicts = pickle.load(handler)
    
    train_data, test_data , val_data = dicts['train_dataset'], dicts['test_dataset'] , dicts['val_dataset']
    tokenizer = BertTokenizer.from_pretrained('google/mobilebert-uncased')
    text , emos = "The kitchen stinks", 0
    experiment_text = "I am [MASK]"
    class_text , emos = "All right .", 4
    
    from dataset.constants import MASK

    input_ids, masked_id ,  attention_mask = encode(tokenizer,  experiment_text )

    # class_based_input_id , masked_id, classed_masked_id = encode(tokenizer, class_text)

    contextbert = ContextualBert()
    loaded_state_dict = torch.load(get_model_file_path(0, "masked_lm",))
    contextbert.load_state_dict(loaded_state_dict)
    masked_bert = MobileBertForMaskedLM.from_pretrained(
            'google/mobilebert-uncased',
            return_dict=True,
            output_hidden_states =False,
            output_attentions= False,)
    
    with torch.no_grad():
        masked_out = masked_bert( input_ids=input_ids,attention_mask=torch.tensor(attention_mask).unsqueeze(0))
        masked_out = masked_out['logits']
        res = decode_multi_index(tokenizer, masked_out, masked_id , 5)
        
        bert_loss, bert_classes , bert_logits= contextbert(input_ids, attention_mask )
        res = decode_multi_index(tokenizer, bert_logits, masked_id , 5)