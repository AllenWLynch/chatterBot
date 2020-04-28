#%%
import tensorflow as tf
import numpy as np
from model import SelectiveChatbotModel, TransformerOptimizer
from selective_chatbot_estimator import ChatBotTrainer
from pipeline import get_datafeed
from config import DATA_DIR
import os


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

chatbot = ChatBotTrainer(
        chatbot_model, 
        TransformerOptimizer(0.001, warmup_steps = 10000, step_reduction = 10))

train_pipeline = get_datafeed(os.path.join(DATA_DIR, 'train_samples.csv'))
test_pipeline = get_datafeed(os.path.join(DATA_DIR, 'test_samples.csv'))

example = next(iter(train_pipeline))

chatbot.train_step(example, 0.5)

chatbot.model.summary()

pre_trained_embeddings = np.load('./embedding_layer_weights.npy')

chatbot.model.representation_model.embedding_layer.set_weights([pre_trained_embeddings])

chatbot.fit(train_pipeline, test_pipeline,
        epochs = 2, steps_per_epoch = 20, evaluation_steps = 20, checkpoint_every = 1,
        logfreq = 1)

