
#%%
import bpemb
import re
import os
from config import DATA_DIR
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

encoder = bpemb.BPEmb(lang = 'en', dim = 300, vs = 10000)

#%%
class BadDataException(Exception):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

with open(os.path.join(DATA_DIR, 'bot_names.csv'), 'r') as f:
    bot_names = [line.strip() for line in f.readlines()]

bot_names_str = '|'.join(bot_names)

ALLOWED_CHARS = re.compile(r'[^A-Za-z0-9/_$@ .!?\'\":\-#*\(\)<>\n&]', re.UNICODE)

URL_RE = re.compile(r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))', re.UNICODE)
URL_TOKEN = '<url>'

USERNAME_RE = re.compile(r'(@[0-9]{3,})', re.UNICODE)
USER_TOKEN = '<usr>'

CHATBOT_RE = re.compile(r"(@(" + bot_names_str + r"))", re.UNICODE)
CHATBOT_TOKEN = '<bot>'

PHONE_NUMBER_RE = re.compile(r"([+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]{7,15})", re.UNICODE)
PHONE_TOKEN = '<phone> '

SIGNATURE_RE = re.compile(r"([-/\^] *[A-Z]{1,2}[\w]*)$")
SIGNATURE_TOKEN = '<sig>'

FILTERS = (URL_RE, PHONE_NUMBER_RE, USERNAME_RE, CHATBOT_RE, SIGNATURE_RE, ALLOWED_CHARS)
SUBSTITUTE_WITH = (URL_TOKEN, PHONE_TOKEN, USER_TOKEN, CHATBOT_TOKEN, SIGNATURE_TOKEN, "")

SPANISH_WORDS = ['el','de','del','que','porque','tu','la','pero','mi','en']
SPANISH_REs = re.compile(r'\b({})\b'.format('|'.join(SPANISH_WORDS)), re.IGNORECASE)

SPLIT_RE = re.compile(r'(<\w+>)', re.UNICODE)

with open(os.path.join(DATA_DIR, 'vocab.txt'), 'r', encoding = 'utf-8') as f:
    vocab = [line.strip() for line in f.readlines()]

VOCAB_DICT = {word : i for i, word in enumerate(vocab + list(SUBSTITUTE_WITH))}

with open(os.path.join(DATA_DIR, 'bot_names.csv'), 'r') as f:
    AUTHORS = [line.strip() for line in f.readlines()]

AUTHORS = { author : i+1 for (i, author) in enumerate(AUTHORS)}

def encode_authors(author):
    return AUTHORS.get(author, int(0))

def get_vocab_id(subword):
    try:
        return VOCAB_DICT[subword]
    except KeyError:
        return 0

def is_english(text, threshold = 0.4):
    if not text:
        raise BadDataException()
    return str(sum([(char.isalpha() and ord(char) <= 127) for char in text])/len(text) > threshold)

def apply_filters(text):
    for _re, substitute in zip(FILTERS, SUBSTITUTE_WITH):
        text = _re.sub(substitute, text)
    return text.strip()

test_str = "Hi @442762 Thank you for responding to us. We've forwarded your request to our Account Specialist Team at https://amz.com. They'll get back to you soon. ^SF"
test_str2 = "Hi Thank you for responding to us. We've forwarded your request to our Account Specialist Team"

def preprocess(text):

    if len(text) == 0 or not is_english(text) or not re.search(SPANISH_REs, text) is None:
        raise BadDataException()

    text = apply_filters(text)
    text = re.split(SPLIT_RE, text)

    text, tokens = text[::2], text[1::2]

    encoding = encoder.encode(text)

    repaired_sequence = [
        entry
        for fixed_segment in [segment + [token] for segment, token in zip(encoding, tokens + [''])]
        for entry in fixed_segment
    ]

    id_encoded = [1] + [get_vocab_id(subword) for subword in repaired_sequence] + [2]

    return id_encoded

def binarize_author_id(id_):
    binarized_id = int(id_ > 0) + 1
    return binarized_id

def transform(conversation_obj, sequence_length):

    data = json.loads(conversation_obj)
    
    #process text to get id array
    encoded_tweet = [preprocess(tweet) for tweet in data['tweets']]
    
    #replicate sender_ids and extend to match length of tweet encoding
    sender_ids = [
        sender_id
        for sender, tweet in zip(data['sender'],encoded_tweet) 
        for sender_id in [binarize_author_id(encode_authors(sender))] * len(tweet) 
    ]
    #flatten context
    encoded_tweet = [_id for tweet in encoded_tweet for _id in tweet]

    encoded_response = preprocess(data['response'])

    if len(encoded_response) > 70:
        raise BadDataException()

    #replicate author_ids
    author_ids = [binarize_author_id(encode_authors(data['author_id']))]*len(encoded_response)

    sequences = [np.array(a) for a in (encoded_tweet, sender_ids, encoded_response, author_ids)]

    sequences = pad_sequences(sequences, sequence_length)

    sequences = sequences.reshape(-1)

    return ','.join([str(num) for num in list(sequences)])
# %%
