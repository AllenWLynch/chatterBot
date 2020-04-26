#%%
import tensorflow as tf
import numpy as np
from model import SelectiveChatbotModel, TransformerOptimizer
from selective_chatbot_estimator import ChatBotTrainer
import pipelines
import sentencepiece
#%%
trans_kwargs = dict(
    num_subwords=200,
    num_speakers=2,
    d_model = 64,
    num_encoder_layers=1,
    num_highway_layers=1,
)

chatbot_model = SelectiveChatbotModel(**trans_kwargs)

m, t, d = 3, 10, 64

X = (
    np.random.randint(1, 199, size = (1, t)),
    np.random.randint(1,2, size= (1, t)),
    np.random.randint(1, 199, size = (m, t)),
    np.random.randint(1,2, size= (m, t)),
)

output = chatbot_model(X)

print("Output shape: ", output.shape)

train_pipeline = pipelines.chatbot_datafeed('./data/train_samples', t, 70, m)
test_pipeline = pipelines.chatbot_datafeed('./data/test_samples', t, 70, m)

#%%
subword_processor = sentencepiece.SentencePieceProcessor()
subword_processor.Load("./model_components/spm/sentpiece_model.model")

#%%
chatbot = ChatBotTrainer(
        subword_processor, 
        chatbot_model, 
        TransformerOptimizer(0.001, warmup_steps = 10000, step_reduction = 10))

#%%

loss, _ = chatbot.train_step(X, 0.3)

print(loss)


# %%
