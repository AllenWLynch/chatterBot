
#%%

import sentencepiece as spm
import numpy as np
import json
import data_utils
import importlib
import os
import random
import re

#%%
data_utils = importlib.reload(data_utils)

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


#%%
sp = spm.SentencePieceProcessor()
sp.Load("./model_components/spm/sentpiece_model.model")

SPANISH_WORDS = ['el','de','del','que','porque','tu','la','pero','mi','en']
SPANISH_REs = re.compile(r'\b({})\b'.format('|'.join(SPANISH_WORDS)), re.IGNORECASE)

#%%
class BadDataException(Exception):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def encode_as_digits(text):
    return [1] + sp.EncodeAsIds(text) + [2]

#sample2 = '''{"sender":["378148","XboxSupport","378148","XboxSupport","378148","XboxSupport","378148","XboxSupport"],"tweets":["Keep getting this error for some games while transferring to a new hard drive. Anybody know why? @378149 @115787 @XboxSupport https://t.co/9Jjh2TtIcM","@378148 Hi there! What happens if you select the resume all option? Do they start and then stop right away?  ^BL","@XboxSupport Yes indeed. It's only like 10 games out of 200 so far.","@378148 Thank you for letting us know that. To be sure do you have enough room on the new hard drive for the content?  ^BL","@XboxSupport Yes, plenty","@378148 If you cancel those other games and just try one at a time does it still happen?  ^BL","@XboxSupport Not sure, tried 2 at a time, will try 1. Transferring some others, will be a while.","@378148 Not a problem, please let us know if that helps with this.  ^BL"],"response":"@XboxSupport No luck, stops even as a solo download","author_id":"378148","id":13821204758528}'''
#sample1 = '''{"sender":["393346","SpotifyCares","393346","SpotifyCares","393346","SpotifyCares","393346"],"tweets":["Dear @SpotifyCares - my desktop application is constantly offline despite strong internet connection. How do I connect? @117168","@393346 Hey there! Can you let us know your device's make/model, operating system, and Spotify version? We'll see what we can suggest /GT","@SpotifyCares Intel Xeon CPU, Windows 7 Enterprise, Spotify 1.0.66.478.g1296534d (not sure where I check version)","@393346 Thanks. Does logging out, restarting your computer, and logging back in help? Keep us posted /GT","@SpotifyCares Nope - nothing helps. I reinstalled and it worked briefly, but no longer.","@393346 Got it. Are you currently connected to a home or work/public WiFi network? Keep us in the loop /JI","@SpotifyCares Work, non-public hard line @SpotifyCares Happy to do this faster off Twitter too - knittelf(at)https://t.co/h2mmwrcQpE"],"response":"@393346 Hey! For your security, we'd recommend deleting your previous post. Can we have you DM us your username? We'll continue talking there /JI https://t.co/ldFdZRiNAt","author_id":"SpotifyCares","id":13821204758530}'''

def binarize_author_id(id_):
    return int(id_ > 0)

def transform(conversation_obj):

    data = json.loads(conversation_obj)
    
    data['encoded_tweet'] = [encode_as_digits(data_utils.apply_filters(tweet)) for tweet in data['tweets']]
    data['encoded_response'] = encode_as_digits(data_utils.apply_filters(data['response']))

    if len(data['response'].split(' ')) < 3:
        raise BadDataException()

    data['sender_ids'] = [
        sender_id 
        for sender, tweet in zip(data['sender'],data['encoded_tweet']) 
        for sender_id in [binarize_author_id(data_utils.encode_authors(sender))] * len(tweet) 
    ]

    data['author_id'] = [binarize_author_id(data_utils.encode_authors(data['author_id']))]*len(data['encoded_response'])

    data['encoded_tweet'] = [_id for tweet in data['encoded_tweet'] for _id in tweet]

    
    is_english = not re.search(SPANISH_REs, x['response']) is None
    if not is_english:
        

    return data

#print(transform(sample1))
# %%

#Encoding hyperparameters
MAX_SEQ_LEN = 128
LOAD_DIR = './data/mined_conversations'
TRAIN_FILE = 'data/train_samples.json'
TEST_FILE = 'data/test_samples.json'

TEST_FILE_SAMPLE_RATE = 0.003
SEED = 1234

random.seed(SEED)

max_response_len = 0
samples_generated = 0

for conversation in MultiFileLineGenerator(LOAD_DIR):

    try:
        data = transform(conversation)

        max_response_len = max(max_response_len, len(data['encoded_response']))

        data['encoded_response'] = data['encoded_response'][0 : MAX_SEQ_LEN]
        data['author_id'] = data['author_id'][0 : MAX_SEQ_LEN]

        data['encoded_tweet'] = data['encoded_tweet'][-MAX_SEQ_LEN : ]
        data['sender_ids'] = data['sender_ids'][-MAX_SEQ_LEN : ]

        save_str = json.dumps(data)

        samples_generated += 1

        save_filename = TEST_FILE if random.random() < TEST_FILE_SAMPLE_RATE else TRAIN_FILE

        with open(save_filename, 'a+') as f:
            f.write(save_str + '\n')

        print('\rSamples generated: {}'.format(str(samples_generated)), end = '')

        #if samples_generated > 10:
        #    assert(False)

    except BadDataException:
        pass

print('')
print('Done!')