from pdb import set_trace
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LsiModel, LdaMulticore, LdaModel, HdpModel
from constants import DATASET_DIR , STOKEN
from gensim.test.utils import datapath
import os

def preclean_daily_dialogue(text, emo):
    stop_word_lines = text.strip().split(STOKEN) 
    emo_tag =  [ int(d) for d in emo.strip().split(' ')]
    stop_word_lines = list(map(lambda x: x.strip(), stop_word_lines))
    if not stop_word_lines[-1].isalnum():  stop_word_lines = stop_word_lines[:-1]
    if emo_tag[-1] == '': emo_tag = emo_tag[:-1]
    return stop_word_lines , emo_tag

def daily_dialogue_generator(mode):
    assert mode == 'train' or mode == 'test' or mode == 'validation'
    dpath = DATASET_DIR
    dial_f = os.path.join(dpath, '{}/dialogues_{}.txt'.format(mode, mode))
    emo_f = os.path.join(dpath, '{}/dialogues_emotion_{}.txt'.format(mode, mode))

    text_lines , emo_lines = None , None
    with open(emo_f, 'r') as emo_f:
        with open(dial_f, 'r') as text_f:
            text_lines = text_f.readlines()
            emo_lines = emo_f.readlines()
    for text , emo in zip(text_lines, emo_lines):
        splitted_text , splitted_emos = preclean_daily_dialogue(text , emo)
        yield  splitted_text , splitted_emos


def topic_analysis(corpus, dictionary, models_path, technique):
    generator = daily_dialogue_generator('train')
    dailogues , emos = [] , []
    for i ,(dailogue , emotion_label) in enumerate(generator):
        dailogues.append(dailogue)
        emos.append(emotion_label)
    Dicts = Dictionary(dailogues)
    corpus = [ Dicts.doc2bow(text) for text in dailogues]
    set_trace()

if __name__ == "__main__":
    topic_analysis(1,2,3,4)
    # dictionary = Dictionary()