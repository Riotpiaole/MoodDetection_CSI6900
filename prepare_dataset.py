from os.path import exists
from pdb import set_trace
import pickle
from constants import (ROOT_DIR , DATASET_DIR , STOKEN , MAXNUM_STEPS , MAX_SEQUENCE)
import sys
import os
import pandas as pd 
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from itertools import chain


class SequenctialTextDataSet(torch.utils.data.Dataset):
    def __init__(self, entire_tokens, labels, sequence_length):
        self.sequence_length = sequence_length
        self.words = torch.Tensor(entire_tokens)
        self.emotion_labels = torch.Tensor(labels)
        
    def __len__(self):
        return len(self.words) - self.sequence_length
    
    def __getitem__(self, index):
        return (
            self.words[index: index + self.sequence_length],
            self.words[index + 1: index + self.sequence_length + 1],
            
            self.emotion_labels[index: index + self.sequence_length],
            self.emotion_labels[index + 1: index + self.sequence_length + 1]
        )

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

def load_daily_dialogue_dataset(mode= 'train', types= 'embedding'):
    print(f"preparing {types} {mode} datasets")
    assert mode == 'train' or mode == 'test' or mode == 'validation'
    dpath = DATASET_DIR
    dial_f = os.path.join(dpath, '{}/dialogues_{}.txt'.format(mode, mode))
    emo_f = os.path.join(dpath, '{}/dialogues_emotion_{}.txt'.format(mode, mode))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
                num_steps = max(num_steps, len(emo_tag))

                for dialogue , emotion_label in zip(stop_word_lines, emo_tag):
                    if types == 'bert':
                        encoded_ids= encode_dict(tokenizer, dialogue)
                        text_tokens.append( encoded_ids['input_ids'][0].tolist()) 
                        masks_tokens.append( encoded_ids['attention_mask'][0].tolist())
                    elif types == 'embedding':
                        encoded_ids = encode_dict(tokenizer, dialogue)
                        filtered_token_id = list(filter(lambda x: x != 0, encoded_ids['input_ids'][0].tolist()))
                        text_tokens += filtered_token_id
                        
                        one_hot_emo_token = one_hot_parse(emotion_label, max_tokenize_size=len(filtered_token_id))
                        emo_tokens +=  one_hot_emo_token


                        
                    
                input_ids += text_tokens
                masks += masks_tokens
                emos += emo_tokens

    return vocab_size, num_steps, input_ids,  masks, emos

def one_hot_parse( index , num_label=8, max_tokenize_size = MAX_SEQUENCE, one_hot=False):
    if one_hot:
        encoded_res = [ 0 for i in range(num_label)]
        encoded_res[index] = 1 
        encoded_final_res = [ encoded_res.copy()  for i in range(max_tokenize_size) ]
        return encoded_final_res
    return [ index for i in range(max_tokenize_size)]

def prepare_dataset(types='embedding'):
    assert types in [ 'bert', 'embedding' ,None ]
    vocab_size, num_steps_train, train_data, train_attention_mask, train_emo_data = load_daily_dialogue_dataset( 'train', types)
    vocab_size, num_steps_test, test_data,   test_attention_mask, test_emo_data  = load_daily_dialogue_dataset('test', types)
    vocab_size, num_steps_val, val_data,  val_attention_mask, val_emo_data  = load_daily_dialogue_dataset('validation', types)
    
    num_steps = max( max(num_steps_train, num_steps_test), num_steps_val)
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
        
    with open(f"./EMNLP_dataset/{types}.pt", 'wb') as handle:
        pickle.dump({ 
            'num_steps': num_steps,  
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
    if len(args) == 1:
        types = 'embedding'
    else:
        types = sys.argv[-1] if len(sys.argv) > 1 else 'embedding'
    
    print(f"working directory {ROOT_DIR}")
    if not exists(f"{ROOT_DIR}/EMNLP_dataset/{types}.pt"):
        prepare_dataset( types)
    else:
        print("Dataset is preprocessed")
        dataset = load_dataset(types)
        print(dataset)
    