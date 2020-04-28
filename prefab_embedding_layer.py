
#%%
import numpy as np
import bpemb
import preprocess_text
import tensorflow as tf
#%%

encoder = bpemb.BPEmb(lang = 'en', dim = 300, vs = 10000)
#%%
total_embeddings = len(encoder.emb.vocab) + len(preprocess_text.FILTERS) + 1

# %%
embedding_weights = np.zeros((total_embeddings, 300))

embedder = tf.keras.layers.Embedding(total_embeddings - len(encoder.emb.vocab), 300)

embedder.build((1,10))

other_weights = embedder.embeddings.numpy()
# %%

embedding_weights[-7:] = other_weights

# %%
for i, word in enumerate(encoder.emb.vocab.keys()):
    embedding_weights[i] = encoder.emb.wv[word]


# %%
embedder = tf.keras.layers.Embedding(total_embeddings, 300)
embedder.build((1,3))

# %%
embedder.set_weights([embedding_weights])

# %%
output = embedder(np.array([[0,1,2]])).numpy()

# %%
np.sum(np.squeeze(output) - embedding_weights[:3])

# %%
np.save('embedding_layer_weights.npy', embedding_weights)

# %%
