import os 
from os.path import abspath , basename , exists
if basename(os.getcwd()) == 'dataset': 
    ROOT_DIR = abspath("..")
else:
    ROOT_DIR = abspath('.')


DATASET_DIR = f"{ROOT_DIR}/EMNLP_dataset"
EMBEDDING_PKL=f"{DATASET_DIR}/embedding.pt"
LMBERT_PKL=f"{DATASET_DIR}/lm_bert.pt"
BERT_PKL=f"{DATASET_DIR}/bert.pt"
MODEL_STATE_DICT_FOLDER=f"{ROOT_DIR}/outs/"
STOKEN = "__eou__"
MAXNUM_STEPS = 36
VOCAB_SIZE = 30522
REGEX_PUNCUATION = r'\s+|[,;.-]\s*'
MAX_SEQUENCE = 36 + 2
DEFAULT_EPOCHS=20
CLS=101
SEP=102
MASK=103
TMP_PLACEHOLDER=-1000
PAD=0
LABEL_MASK=-100

def get_model_file_path(epoch, name):
    if not exists(f"{MODEL_STATE_DICT_FOLDER}/{name}/"):
        os.mkdir(f"{MODEL_STATE_DICT_FOLDER}/{name}/")
    return f"{MODEL_STATE_DICT_FOLDER}/{name}/{epoch}_model.pt" 

def get_meta_file_path(name):
    if not exists(f"{MODEL_STATE_DICT_FOLDER}/{name}/"):
        os.mkdir(f"{MODEL_STATE_DICT_FOLDER}/{name}/")
    return f"{MODEL_STATE_DICT_FOLDER}/{name}/meta.pkl"


def get_loss_record_meta_file_path(name, tag):
    if not exists(f"{MODEL_STATE_DICT_FOLDER}/{name}/"):
        os.mkdir(f"{MODEL_STATE_DICT_FOLDER}/{name}/")
    return f"{MODEL_STATE_DICT_FOLDER}/{name}/{tag}_loss.csv"
    