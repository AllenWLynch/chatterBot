


import sentencepiece as spm
from data_utils import TOKENS
import os

DATA_DIR = './data/text_corpus.csv'

param_str = '--input={} --vocab_size=8000 --character_coverage=1.0 --model_prefix=sentpiece_model --user_defined_symbols={}'\
    .format(','.join([os.path.join(DATA_DIR, filename) for filename in os.listdir(DATA_DIR)]), ','.join([token.strip() for token in TOKENS]))

spm.SentencePieceTrainer.Train(param_str)