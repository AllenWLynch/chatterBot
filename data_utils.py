
#%%
import re
import pandas as pd
import findspark
findspark.init()

import pyspark
from pyspark.sql import functions
from pyspark.sql import types

#%%
#regexes for filtering tweets:
with open('./data/bot_names.csv', 'r') as f:
    bot_names = [line.strip() for line in f.readlines()]

bot_names_str = '|'.join(bot_names)
#%%
ALLOWED_CHARS = re.compile(r'[^A-Za-z0-9/_$@ .!?\'\":\-#*\(\)<>\n&]', re.UNICODE)

URL_RE = re.compile(r'(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))', re.UNICODE)
URL_TOKEN = '<url>'

USERNAME_RE = re.compile(r'(@[0-9]{3,})', re.UNICODE)
USER_TOKEN = '<usr>'

CHATBOT_RE = re.compile(r"(@(" + bot_names_str + r"))", re.UNICODE)
CHATBOT_TOKEN = '<bot>'

PHONE_NUMBER_RE = re.compile(r"([+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]{7,15})", re.UNICODE)
PHONE_TOKEN = '<phone> '

SIGNATURE_RE = re.compile(r"([-/\^]* *[A-Z]{1,2}[\w]*)$")
SIGNATURE_TOKEN = '<sig>'

#%%
FILTERS = (URL_RE, PHONE_NUMBER_RE, USERNAME_RE, CHATBOT_RE, SIGNATURE_RE, ALLOWED_CHARS)
SUBSTITUTE_WITH = (URL_TOKEN, PHONE_TOKEN, USER_TOKEN, CHATBOT_TOKEN, SIGNATURE_TOKEN, "")
TOKENS = (URL_TOKEN, CHATBOT_TOKEN, USER_TOKEN, PHONE_TOKEN, SIGNATURE_TOKEN)

def apply_filters(text):
    for _re, substitute in zip(FILTERS, SUBSTITUTE_WITH):
        text = _re.sub(substitute, text)
    return text.strip()

FILTER_UDF = functions.udf(apply_filters, types.StringType())
#%%

# %%

def is_english(text, threshold = 0.4):
    if not text:
        return 'False'
    return str(sum([(char.isalpha() and ord(char) <= 127) for char in text])/len(text) > threshold)

ENGLISH_UDF = functions.udf(is_english, types.StringType())


#%%
with open('./data/bot_names.csv', 'r') as f:
    AUTHORS = [line.strip() for line in f.readlines()]

AUTHORS = { author : i+1 for (i, author) in enumerate(AUTHORS)}

def encode_authors(author):
    return AUTHORS.get(author, int(0))

AUTHORS_UDF = functions.udf(encode_authors, types.IntegerType())

#%%

if __name__ == "__main__":
    sample_tweet = "@VirginAmerica have & you're 11 Guys ever Seen the information at https://www.regular-expressions.info/alternation it's hella cool! Get me back at @1115679"

    output_tweet = sample_tweet
    for _re, substitute in zip(FILTERS, SUBSTITUTE_WITH):
        output_tweet = _re.sub(substitute, output_tweet)

    print(output_tweet)

    tests = pd.read_csv('./data/sample.csv')

    filtered = tests['text'].apply(apply_filters)
    filtered.values

