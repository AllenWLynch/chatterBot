#%%
import tensorflow as tf
import numpy as np
from model import SelectiveChatbotModel, TransformerOptimizer
from selective_chatbot_estimator import ChatBotTrainer
from pipeline import get_datafeed
from config import DATA_DIR
import os
#%%
tf.keras.backend.clear_session()

trans_kwargs = dict(
    num_subwords=10007,
    num_speakers=3,
    d_model = 512,
    embedding_dim = 300,
    num_encoder_layers=3,
    num_highway_layers=1,
)

trans_layer_kwargs = dict(
    conv_width = 5,
    dff = 1048,
    heads = 8,
    max_relative_distance = 24,
)

#small shape for testing
'''trans_kwargs = dict(
    num_subwords=200,
    num_speakers=3,
    d_model = 64,
    embedding_dim = 48,
    num_encoder_layers=1,
    num_highway_layers=1,
)
trans_layer_kwargs = dict(
    conv_width = 5,
    dff = 128,
    heads = 4,
    max_relative_distance = 4,
)'''

chatbot_model = SelectiveChatbotModel(**trans_kwargs, transformer_layer_kwargs = trans_layer_kwargs)

BATCH_SIZE = 32
train_pipeline = get_datafeed(os.path.join(DATA_DIR, 'train_samples.csv'), batch_size= BATCH_SIZE)
test_pipeline = get_datafeed(os.path.join(DATA_DIR, 'test_samples.csv'), batch_size= BATCH_SIZE)

example = next(iter(train_pipeline))

chatbot_model(example)

pre_trained_embeddings = np.load('./embedding_layer_weights.npy')

chatbot_model.representation_model.embedding_layer.set_weights([pre_trained_embeddings])

chatbot_model.summary()

chatbot = ChatBotTrainer(
        chatbot_model, 
        TransformerOptimizer(0.0001, warmup_steps = 10000, step_reduction = 1),
        load_from_checkpoint=False)

chatbot.fit(train_pipeline, test_pipeline, 0.5,
        epochs = 100, steps_per_epoch = 5000, evaluation_steps = 30, checkpoint_every = 10,
        logfreq = 50)

