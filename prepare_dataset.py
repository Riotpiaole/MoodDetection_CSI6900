from os.path import exists
from constants import ROOT_DIR , DATASET_DIR , STOKEN

import os
import pandas as pd 

DATASET_FOLDER = " a"

def load_daily_dialogue_dataset(mode= 'train'):
    print("preparing datasets")
    assert mode == 'train' or mode == 'test' or mode == 'validation'
    dpath = DATASET_DIR
    dial_f = os.path.join(dpath, '{}/dialogues_{}.txt'.format(mode, mode))
    emo_f = os.path.join(dpath, '{}/dialogues_emotion_{}.txt'.format(mode, mode))
    dlg_data, act_data, emo_data = [], [], []
    with open(dial_f, 'r') as f:
        lines = f.readlines()
        for l in lines:
            turns = [t.strip().split(' ') for t in l.split(STOKEN)]
            if turns[-1] == ['']:
                turns = turns[:-1]
            dlg_data.append(turns)

    with open(emo_f, 'r') as f:
        lines = f.readlines()
        for l in lines:
            emos = [int(d) for d in l.strip().split(' ')]
            emo_data.append(emos)
    return dlg_data, emo_data

def prepare_dataset():
    train_data, train_act_data, train_emo_data = load_daily_dialogue_dataset( 'train')
    test_data, test_act_data, test_emo_data  = load_daily_dialogue_dataset('test')
    val_data, val_act_data, val_emo_data  = load_daily_dialogue_dataset('validation')
    data = train_data + test_data + val_data
    vocab = ['_UNK_'] + sorted(set(w for d in data for s in d for w in s))
    
    w2i = {w:i for i, w in enumerate(vocab)}
    i2w = {i:w for w, i in enumerate(w2i)}

if __name__ == "__main__":
    print(f"working directory {ROOT_DIR}")
    if not exists("./dataset.pt"):
        train_data, train_act_data, train_emo_data = load_daily_dialogue_dataset( 'train')
        test_data, test_act_data, test_emo_data  = load_daily_dialogue_dataset('test')
        val_data, val_act_data, val_emo_data  = load_daily_dialogue_dataset('validation')
        data = train_data + test_data + val_data

    else:
        print("Dataset is preprocessed")
        exit(0)
    