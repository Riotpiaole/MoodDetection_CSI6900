from os.path import dirname ,abspath
ROOT_DIR = abspath(".")
DATASET_DIR = f"{ROOT_DIR}/EMNLP_dataset"
STOKEN = "__eou__"
MAXNUM_STEPS = 1
VOCAB_SIZE = 30522
REGEX_PUNCUATION = r'\s+|[,;.-]\s*'
MAX_SEQUENCE = 36
DEFAULT_EPOCHS=20