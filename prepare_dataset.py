from os.path import exists
import pickle
from constants import ROOT_DIR , DATASET_DIR , STOKEN

import os
import pandas as pd 

from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from tqdm import tqdm
import torch

DATASET_FOLDER = " a"

def encode_dict(tokenizer, line):
    return tokenizer.encode_plus(
        line, add_special_tokens=True,
        max_length = 256,           # Pad & truncate all sentences.
        truncation=True,
        padding='max_length',
        return_tensors='pt')

def load_daily_dialogue_dataset(mode= 'train', types= 'bert'):
    print(f"preparing {mode} datasets")
    assert mode == 'train' or mode == 'test' or mode == 'validation'
    dpath = DATASET_DIR
    dial_f = os.path.join(dpath, '{}/dialogues_{}.txt'.format(mode, mode))
    emo_f = os.path.join(dpath, '{}/dialogues_emotion_{}.txt'.format(mode, mode))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []
    masks = []
    emos =[]
    
    with open(emo_f, 'r') as emo_f:
        with open(dial_f, 'r') as text_f:
            text_lines = text_f.readlines()
            emo_lines = emo_f.readlines()
            for  text ,  emo in tqdm(zip(text_lines, emo_lines)):
                stop_word_lines = text.strip().split(STOKEN) 
                if not stop_word_lines[-1].isalnum():  stop_word_lines = stop_word_lines[:-1]
                emo_tag =  [ int(d) for d in emo.strip().split(' ')]
                if emo_tag[-1] == '': emo_tag = emo_tag[:-1]
                assert len(stop_word_lines) == len(emo_tag) 
                
                for dialogue , emotion_label in zip(stop_word_lines, emo_tag):
                    if types == 'bert':
                        encoded_ids= encode_dict(tokenizer, dialogue)
                        input_ids.append(encoded_ids['input_ids']) 
                        masks.append(encoded_ids['attention_mask'])
                # else:
                #     turns['inpupt_ids'].append([t.strip().split(' ') for t in l.split(STOKEN)])
                #     if turns[-1] == ['']:

                    emos.append(torch.tensor([emotion_label]))
    return input_ids, masks, emos

def prepare_dataset(types='bert'):
    assert types in [ 'bert', None ]
    train_data, train_attention_mask, train_emo_data = load_daily_dialogue_dataset( 'train', types)
    test_data,  test_attention_mask, test_emo_data  = load_daily_dialogue_dataset('test', types)
    val_data, val_attention_mask, val_emo_data  = load_daily_dialogue_dataset('validation', types)
    
    data = train_data + test_data + val_data
    masks = train_attention_mask + test_attention_mask + val_attention_mask
    labels = train_emo_data + test_emo_data + val_emo_data

    data = torch.cat(data, dim=0)
    masks = torch.cat(masks, dim=0)
    labels = torch.cat(labels, dim=0)
    dataset = TensorDataset(data, masks, labels)
    with open(f"./dataset.pt", 'wb') as handle:
        pickle.dump(dataset, handle)
        print("Saved the dataset")
    return dataset

def load_dataset(types='bert'):
    
    if exists(f"{ROOT_DIR}/EMNLP_dataset/dataset.pt"):
        with open(f"{ROOT_DIR}/EMNLP_dataset/dataset.pt", 'rb') as handle:
            return pickle.load(handle)
    return prepare_dataset(types)

if __name__ == "__main__":

    print(f"working directory {ROOT_DIR}")
    if not exists(f"{ROOT_DIR}/EMNLP_dataset/dataset.pt"):
        prepare_dataset()
    else:
        print("Dataset is preprocessed")
        dataset = load_dataset()
        print(dataset)
        exit(0)
    