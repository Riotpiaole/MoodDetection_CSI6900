try:

    from constants import (
        ROOT_DIR , 
        DATASET_DIR , 
        STOKEN , 
        MAXNUM_STEPS , 
        MAX_SEQUENCE, 
        BERT_PKL,
        EMBEDDING_PKL,
        BERT_PKL,
        LMBERT_PKL
    )
    from constants import ( SEP, MASK, TMP_PLACEHOLDER , PAD, LABEL_MASK)
except ModuleNotFoundError:
    from dataset.constants import (
        ROOT_DIR , 
        DATASET_DIR , 
        STOKEN , 
        MAXNUM_STEPS , 
        MAX_SEQUENCE, 
        BERT_PKL,
        EMBEDDING_PKL,
        BERT_PKL,
        LMBERT_PKL
    )
    from dataset.constants import ( SEP, MASK, TMP_PLACEHOLDER , PAD, LABEL_MASK)


from os.path import exists

# bert special token constants
from torch import tensor
import sys
import os
from transformers import BertTokenizer
from tqdm import tqdm
import torch
import pickle


def daily_dialogue_emotion_textization(index: int, size: int ):
    mapper = [ 'emotion', 'anger',  'disgust', 'fear', 'happiness', 'sadness',  'surprise']
    return [mapper[index] for i in range(size)]

def encode_dict(tokenizer, line):
    return tokenizer.encode_plus(
        line, add_special_tokens=True,
        max_length = MAX_SEQUENCE,           # Pad & truncate all sentences.
        truncation=True,
        padding='max_length',
        return_tensors='pt')


def sentence_perplexity_estimation(input_id,masked_token = MASK):
    mask_length = torch.nonzero(input_id)[-1].item()
    
    if mask_length <= 2:
        repeat_input = input_id.repeat(1, 1)
    else:
        repeat_input = input_id.repeat(mask_length - 2, 1)


    tril_input = torch.tril(repeat_input, diagonal=1)
    tril_input[repeat_input == SEP] = SEP
    
    tril_input[repeat_input == PAD] = TMP_PLACEHOLDER
    tril_input[tril_input == PAD] = masked_token
    tril_input[tril_input == TMP_PLACEHOLDER] = PAD

    label = tensor(repeat_input).clone().detach().\
        masked_fill(tril_input != MASK, LABEL_MASK)
    origin_input_label = tensor([ LABEL_MASK for _ in range(len(label[0]))]).unsqueeze(0)
    label = torch.cat(( label, origin_input_label))
    tril_input = torch.cat( ( tril_input, input_id.unsqueeze(0)))
    return tril_input , label

def load_daily_dialogue_dataset(mode= 'train', types= 'embedding'):
    print(f"preparing {types} {mode} datasets")
    assert mode == 'train' or mode == 'test' or mode == 'validation'
    dpath = DATASET_DIR
    dial_f = os.path.join(dpath, '{}/dialogues_{}.txt'.format(mode, mode))
    emo_f = os.path.join(dpath, '{}/dialogues_emotion_{}.txt'.format(mode, mode))

    tokenizer = BertTokenizer.from_pretrained('google/mobilebert-uncased')

    input_ids , masks, emos = [] , [] , [] 

    # fix size for num of steps
    num_steps = MAXNUM_STEPS
    vocab_size = tokenizer.vocab_size
    with open(emo_f, 'r') as emo_f:
        with open(dial_f, 'r') as text_f:
            text_lines = text_f.readlines()
            emo_lines = emo_f.readlines()
            for  text ,  emo in tqdm(zip(text_lines, emo_lines), total=len(text_lines)):
                stop_word_lines = text.strip().split(STOKEN) 
                emo_tag =  [ int(d) for d in emo.strip().split(' ')]
                text_tokens , masks_tokens, emo_tokens = [] , [], []
                if not stop_word_lines[-1].isalnum():  stop_word_lines = stop_word_lines[:-1]
                if emo_tag[-1] == '': emo_tag = emo_tag[:-1]
                assert len(stop_word_lines) == len(emo_tag) , "length is not match"

                for dialogue , emotion_label in zip(stop_word_lines, emo_tag):
                    if types == 'bert':
                        encoded_ids= encode_dict(tokenizer, dialogue)
                        text_tokens.append( encoded_ids['input_ids'][0].tolist()) 
                        masks_tokens.append( encoded_ids['attention_mask'][0].tolist())
                        emo_tokens.append(one_hot_parse(emotion_label, one_hot=True))
                    
                    elif types == 'embedding':
                        encoded_ids = encode_dict(tokenizer, dialogue)
                        filtered_token_id = list(filter(lambda x: x != 0, encoded_ids['input_ids'][0].tolist()))
                        text_tokens += filtered_token_id
                    elif types == 'lm_bert':
                        encoded_ids= encode_dict(tokenizer, dialogue)
                        input_id = encoded_ids['input_ids'][0]
                        attention_mask = encoded_ids['attention_mask'][0]

                        lm_input_id , lm_mask_label = sentence_perplexity_estimation(input_id)
                        text_tokens += lm_input_id.tolist()
                        attention_mask = attention_mask.repeat(len(lm_mask_label), 1)
                        
                        zipped_list = zip(lm_mask_label.tolist(), attention_mask.tolist())
                        masks_tokens += list(zipped_list)

                        labels = one_hot_parse(emotion_label, 
                                          one_hot=True, 
                                          max_tokenize_size= len(lm_input_id))
                        emo_tokens+= labels
                        
                input_ids += text_tokens
                masks += masks_tokens
                emos += emo_tokens

    return vocab_size, input_ids,  masks, emos

def one_hot_parse( index , num_label=8, max_tokenize_size = MAX_SEQUENCE, one_hot=False):
    if one_hot:
        encoded_res = [ 0 for _ in range(num_label)]
        encoded_res[index] = 1 
        encoded_final_res = [ encoded_res.copy()  for _ in range(max_tokenize_size) ]
        return encoded_final_res
    return [ index for _ in range(max_tokenize_size)]

def switch_pkl_name(types='embedding'):
    if types == 'embedding':
        return EMBEDDING_PKL
    elif types == 'bert':
        return BERT_PKL
    elif types == 'lm_bert':
        return LMBERT_PKL
    raise ValueError("Invalid types for pkl")

def prepare_dataset(types='embedding'):
    assert types in [ 'bert', 'embedding', 'lm_bert' ,None ]
    
    dataset_pt_loc = switch_pkl_name(types)
    vocab_size,  train_data, train_attention_mask, train_emo_data = load_daily_dialogue_dataset( 'train', types)
    vocab_size,  test_data,   test_attention_mask, test_emo_data  = load_daily_dialogue_dataset('test', types)
    vocab_size,  val_data,  val_attention_mask, val_emo_data  = load_daily_dialogue_dataset('validation', types)
    
    masks = train_attention_mask + test_attention_mask + val_attention_mask

    if types == 'embedding':
        train_dataset = [train_data, train_emo_data, MAX_SEQUENCE]
        test_dataset = [test_data, test_emo_data, MAX_SEQUENCE]
        val_dataset = [val_data, val_emo_data, MAX_SEQUENCE]
    else:
        if (len(masks) != 0):
            train_dataset = [train_data, train_attention_mask, train_emo_data]
            test_dataset = [test_data, test_attention_mask, test_emo_data]
            val_dataset = [val_data, val_attention_mask, val_emo_data]
        else:
            train_dataset = [train_data, train_emo_data]
            test_dataset = [test_data, test_emo_data]
            val_dataset = [val_data, val_emo_data]
        
    with open(dataset_pt_loc, 'wb') as handle:
        pickle.dump({ 
            'train_dataset':train_dataset,
            'test_dataset':test_dataset,
            'val_dataset':val_dataset,
            'vocab': vocab_size
        }, handle)
        print("Saved the dataset")
    return train_dataset , test_dataset , val_dataset

def load_dataset(types='bert'):
    
    if exists(f"{ROOT_DIR}/EMNLP_dataset/{types}.pt"):
        with open(f"{ROOT_DIR}/EMNLP_dataset/{types}.pt", 'rb') as handle:
            return pickle.load(handle)
    return prepare_dataset(types)



        
if __name__ == "__main__":
    args = sys.argv
    
    prepare_dataset( 'lm_bert' )
    # prepare_dataset( 'embedding' )
    