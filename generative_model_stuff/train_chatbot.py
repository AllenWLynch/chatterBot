
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
if __name__ == "__main__":
    SEQ_LEN = 128
    RESPONSE_MAXLEN = 70
    BATCH_SIZE = 48

    train_pipeline = pipelines.chatbot_datafeed('./data/train_samples', SEQ_LEN, RESPONSE_MAXLEN, BATCH_SIZE)
    test_pipeline = pipelines.chatbot_datafeed('./data/test_samples', SEQ_LEN, RESPONSE_MAXLEN, BATCH_SIZE)

    inference_pipeline = pipelines.example_stream('./data/test_samples', SEQ_LEN)

    #%%
    subword_processor = sentencepiece.SentencePieceProcessor()
    subword_processor.Load("./model_components/spm/sentpiece_model.model")

    #small specs for testing
    '''transformer_specs = dict(
            dff = 128,
            heads = 4,
            attn_dropout = 0.1,
            fcnn_dropout = 0.1,
            conv_width = 3,
            max_relative_distance = 8,
        )
    architecture_specs =  dict(
        num_subwords = 8000,
        num_speakers = 2,
        d_model = 64,
        num_decoder_layers = 2,
        num_encoder_layers = 2,
        num_highway_layers = 2,
        highway_dropout = 0.1,
        embedding_dropout = 0.1,
        )'''
    #architecture specifications
    transformer_specs = dict(
            dff = 2048,
            heads = 8,
            attn_dropout = 0.1,
            fcnn_dropout = 0.1,
            conv_width = 7,
            max_relative_distance = 24,
        )
    architecture_specs =  dict(
        num_subwords = 8000,
        num_speakers = 2,
        d_model = 512,
        num_decoder_layers = 7,
        num_encoder_layers = 4,
        num_highway_layers = 2,
        highway_dropout = 0.1,
        embedding_dropout = 0.1,
        )
    tf.keras.backend.clear_session()

    chatbot_model = model.Transformer(**architecture_specs, transformer_layer_kwargs = transformer_specs)
#%%
    test_data_sample = next(iter(test_pipeline))

    chatbot_model(test_data_sample[0])

    word2v_embeddings = np.load('./model_components/w2v_embeddings.npy')

    chatbot_model.embedding_layer.set_weights([word2v_embeddings])

    chatbot_model.summary()

    chatbot = chatbot_estimator.ChatBotTrainer(
        subword_processor, 
        chatbot_model, 
        model.TransformerOptimizer(0.001, warmup_steps = 0, initial_step = 1e6, step_reduction = 10), 
        load_from_checkpoint = True)

    chatbot.add_train_metric(tf.keras.metrics.SparseCategoricalCrossentropy(name = '[Train] Sparse Categorical Crossentropy', from_logits=True))
    chatbot.add_test_metric(tf.keras.metrics.SparseCategoricalCrossentropy(name = '[Test] Sparse Categorical Crossentropy', from_logits=True))

    # %%
    try:
        chatbot.fit(train_pipeline, train_pipeline, inference_pipeline, epochs = 10000, steps_per_epoch = 5000, debugging = False, checkpoint_every=5)
    except KeyboardInterrupt as err:
        print('\nCaught execption >:(\nSaving model...')
        chatbot.prompt_for_save()
        

# %%
