
#%%
import os
import gensim
import sentencepiece as spm
#%%
import numpy as np

#%%
class MultiFileGenerator():

    def __init__(self, dirname, normalization_fn = lambda x : x):
        self.dirname= dirname
        self.normalization_fn = normalization_fn

    def __iter__(self):
        for filename in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, filename), 'r'):
                yield self.normalization_fn(line)

#%%

sp = spm.SentencePieceProcessor()
sp.Load("./spm/sentpiece_model.model")

#%%

def encode_as_digits(text):
    return ['1'] + [str(num_encoding) for num_encoding in sp.EncodeAsIds(text)] + ['2']

sentences = MultiFileGenerator('./data/text_corpus.csv', encode_as_digits)

#%%
model = gensim.models.Word2Vec(sentences = sentences, size = 512, min_count = 1, workers = 12)

#%%
#save the output
model.save('./w2v_model')

#%%
converted_embeddings = np.random.randn(8000, 512)

for word in model.wv.vocab:
    converted_embeddings[int(word)] = model.wv[word]

#%%
np.save('./w2v_embeddings', converted_embeddings)

# %%
