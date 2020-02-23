
#%%
import tensorflow as tf
import numpy as np

#%%
import model
import pipelines
import importlib
import sentencepiece

model = importlib.reload(model)
pipelines = importlib.reload(pipelines)
#%%
subword_processor = sentencepiece.SentencePieceProcessor()
subword_processor.Load("./model_components/spm/sentpiece_model.model")

#%%
architecture_specs =  dict(
    num_subwords = 8000,
    num_speakers = 2,
    d_model = 64,
    num_shared_layers = 1,
    num_decoder_layers = 1,
    num_encoder_layers = 1,
    )
transformer_specs = dict(
        dff = 256,
    )

chatbot = model.ChatBotModel(subword_processor, architecture_specs, transformer_specs)

#%%
#Pipeline hyperparams
SEQ_LEN = 128
BATCH_SIZE = 32

train_pipeline = pipelines.chatbot_training_stream('./data/train_samples', SEQ_LEN, BATCH_SIZE)
test = pipelines.chatbot_training_stream('./data/test_samples', SEQ_LEN, BATCH_SIZE)

inference_pipeline = pipelines.example_stream('./data/test_samples', CONTEXT_LENGTH)

#%%

chatbot.add_train_metric(tf.keras.metrics.SparseCategoricalCrossentropy(name = '[Train] Sparse Categorical Crossentropy', from_logits=True))

chatbot.add_test_metric(tf.keras.metrics.SparseCategoricalCrossentropy(name = '[Test] Sparse Categorical Crossentropy', from_logits=True))

example = next(iter(train_pipeline))

#l = chatbot.train_step(example[0], example[1])
chatbot.model(example[0])

# %%
sample = next(iter(inference_pipeline))

chatbot.respond(*sample[:3], 128)

# %%
