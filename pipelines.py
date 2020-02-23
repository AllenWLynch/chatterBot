


#%%

import tensorflow as tf
import json
import numpy as np
import os

#%%
def convert_from_json(seq_len, json_str):

    x = json.loads(json_str.numpy())

    sequences = [
        np.array(x) for x in
        (x['encoded_tweet'], x['sender_ids'], x['encoded_response'][:-1], x['author_id'][:-1], x['encoded_response'][1:])
    ]
    sequences[1] += 1
    sequences[3] += 1

    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, seq_len)
    
    return sequences[:]

def process_json(context_len, response_len, textline):

    return tf.py_function(convert_from_json, [context_len, response_len, textline], (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))

def group_xy(context, sender, response_input, author, response_target):

    return (context, sender, response_input, author), response_target


def chatbot_training_stream(datadir, context_len, response_len, batch_size, shuffle_size = 400):

    filenames = [os.path.join(datadir, filename) for filename in os.listdir(datadir)]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.interleave(lambda filename : 
        tf.data.TextLineDataset(filename).map(lambda x : process_json(context_len, response_len, x), num_parallel_calls = tf.data.experimental.AUTOTUNE),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.map(lambda *x : group_xy(*x), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.repeat().shuffle(shuffle_size)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

#%%

def read_example(context_len, json_str):

    x = json.loads(json_str.numpy())

    sequences = [
        np.array(x) for x in
        (x['encoded_tweet'], x['sender_ids'])
    ]
    sequences[1] += 1

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, context_len)

    return padded_sequences[0], padded_sequences[1], [x['author_id'][0]], '\n'.join(x['tweets'])

def process_example(context_len, textline):

    return tf.py_function(read_example, [context_len, textline], (tf.int32, tf.int32, tf.int32, tf.string))

def example_stream(test_dir, context_len):

    filenames = [os.path.join(test_dir, filename) for filename in os.listdir(test_dir)]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.interleave(lambda filename : 
        tf.data.TextLineDataset(filename).map(lambda x : process_example(context_len, x))
    )

    dataset = dataset.repeat().shuffle(50)

    dataset = dataset.batch(1)

    return dataset
    




# %%
