


#%%

import tensorflow as tf
import json
import numpy as np
import os
import re

#%%
SPANISH_WORDS = ['el','de','del','que','porque','tu','la','pero','mi','en']
SPANISH_REs = re.compile(r'\b({})\b'.format('|'.join(SPANISH_WORDS)), re.IGNORECASE)

def convert_from_json(seq_len, response_len_threshold, json_str):

    x = json.loads(json_str.numpy())

    sequences = [
        np.array(x) for x in
        (x['encoded_tweet'], x['sender_ids'], x['encoded_response'][:-1], x['author_id'][:-1], x['encoded_response'][1:])
    ]
    sequences[1] += 1
    sequences[3] += 1

    response_len = sequences[2].shape[-1]

    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, seq_len)

    is_english = re.search(SPANISH_REs, x['response']) is None
    
    return response_len < response_len_threshold and is_english, sequences[0], sequences[1], sequences[2], sequences[3], sequences[4]

def process_json(seq_len, response_len_threshold, textline):

    return tf.py_function(convert_from_json, [seq_len, response_len_threshold, textline], (tf.bool, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))

def group_xy(len_requirement, context, sender, response_input, author, response_target):

    return (context, sender, response_input, author), response_target
 

def chatbot_datafeed(datadir, seq_len, response_len_threshold, batch_size, shuffle_size = 400):

    filenames = [os.path.join(datadir, filename) for filename in os.listdir(datadir)]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.interleave(lambda filename : 
        tf.data.TextLineDataset(filename)\
            .map(lambda x : process_json(seq_len, response_len_threshold, x), num_parallel_calls = tf.data.experimental.AUTOTUNE),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.filter(lambda *x : x[0])

    dataset = dataset.map(lambda *x : group_xy(*x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.repeat().shuffle(shuffle_size)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

test = chatbot_datafeed('./data/train_samples', 128, 70, 32)

#%%

def read_example(context_len, json_str):

    x = json.loads(json_str.numpy())

    sequences = [
        np.array(x) for x in
        (x['encoded_tweet'], x['sender_ids'])
    ]
    sequences[1] += 1

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, context_len)

    context_str = '\n'.join(['<{}>: {}'.format(sender, tweet) for sender, tweet in zip(x['sender'], x['tweets'])]) + '\nResponse: '

    is_english = re.search(SPANISH_REs, context_str) is None

    return is_english, padded_sequences[0], padded_sequences[1], [x['author_id'][0]], context_str

def process_example(context_len, textline):

    return tf.py_function(read_example, [context_len, textline], (tf.bool, tf.int32, tf.int32, tf.int32, tf.string))

def inference_group_xy(spanish_filter, context, sender, author_id, context_str):

    return (context, sender, author_id), context_str

def example_stream(test_dir, context_len):

    filenames = [os.path.join(test_dir, filename) for filename in os.listdir(test_dir)]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.interleave(lambda filename : 
        tf.data.TextLineDataset(filename).map(lambda x : process_example(context_len, x))
    )

    dataset = dataset.filter(lambda *x : x[0])

    dataset = dataset.map(lambda *x : inference_group_xy(*x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.repeat().shuffle(50)

    dataset = dataset.batch(1)

    return dataset

test2 = example_stream('./data/test_samples',128)
# %%
