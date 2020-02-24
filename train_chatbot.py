
#%%
import tensorflow as tf
import numpy as np


#%%
import model
import pipelines
import importlib
import sentencepiece
import chatbot_estimator
model = importlib.reload(model)
pipelines = importlib.reload(pipelines)
chatbot_estimator = importlib.reload(chatbot_estimator)

#%%
#Pipeline hyperparams
SEQ_LEN = 128
BATCH_SIZE = 32

train_pipeline = pipelines.chatbot_training_stream('./data/train_samples', SEQ_LEN, BATCH_SIZE)
test = pipelines.chatbot_training_stream('./data/test_samples', SEQ_LEN, BATCH_SIZE)

inference_pipeline = pipelines.example_stream('./data/test_samples', SEQ_LEN)

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
#chatbot = model.ChatBotModel(subword_processor, architecture_specs, transformer_specs)
chatbot_model = model.Transformer(**architecture_specs, transformer_layer_kwargs = transformer_specs)

#%%

chatbot = chatbot_estimator.ChatBotTrainer(subword_processor, chatbot_model, model.TransformerOptimizer(64), load_from_checkpoint = True)

chatbot.add_train_metric(tf.keras.metrics.SparseCategoricalCrossentropy(name = '[Train] Sparse Categorical Crossentropy', from_logits=True))
chatbot.add_test_metric(tf.keras.metrics.SparseCategoricalCrossentropy(name = '[Test] Sparse Categorical Crossentropy', from_logits=True))

# %%

chatbot.fit(train_pipeline, train_pipeline, inference_pipeline, epochs = 1, steps_per_epoch = 10, evaluation_steps = 10)

# %%
