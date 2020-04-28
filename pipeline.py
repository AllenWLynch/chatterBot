
import tensorflow as tf
from config import DATA_DIR
import os

def line_2_tensor(str_line):

    indices = tf.strings.to_number(tf.strings.split(str_line, ','), tf.int32)

    indices = tf.reshape(indices, (4, 128))

    split_indices = tf.split(indices, 4, axis = 0)

    return tf.squeeze(split_indices[0]), tf.squeeze(split_indices[1]), tf.squeeze(split_indices[2]), tf.squeeze(split_indices[3])


def get_datafeed(filename, shuffle_size = 500, batch_size = 32):

    dataset = tf.data.TextLineDataset(filename)\
        .map(line_2_tensor, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    dataset = dataset.repeat().shuffle(shuffle_size)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
