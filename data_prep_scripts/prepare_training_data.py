
#%%

from config import DATA_DIR
import numpy as np
import json
import preprocess_text
import os
import random
import re


#%%
class MultiFileLineGenerator():

    def __init__(self, dirname, normalization_fn = lambda x : x):
        self.dirname= dirname
        self.normalization_fn = normalization_fn

    def __iter__(self):
        for filename in os.listdir(self.dirname):
            print('Reading: {}'.format(filename))
            for line in open(os.path.join(self.dirname, filename), 'r', encoding='utf8'):
                yield self.normalization_fn(line)

#print(transform(sample1))
# %%

#Encoding hyperparameters
MAX_SEQ_LEN = 128
LOAD_DIR = os.path.join(DATA_DIR, 'mined_conversations')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_samples.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_samples.csv')

TEST_FILE_SAMPLE_RATE = 0.003
SEED = 1234

random.seed(SEED)

max_response_len = 0
samples_generated = 0

for conversation in MultiFileLineGenerator(LOAD_DIR):

    try:
        data = transform(conversation)

        samples_generated += 1

        save_filename = TEST_FILE if random.random() < TEST_FILE_SAMPLE_RATE else TRAIN_FILE

        with open(save_filename, 'a+') as f:
            f.write(save_str + '\n')

        print('\rSamples generated: {}'.format(str(samples_generated)), end = '')

        if samples_generated > 10:
            assert(False)

    except BadDataException:
        pass

print('')
print('Done!')