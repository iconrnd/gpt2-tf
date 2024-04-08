import tensorflow as tf
from config.config import *


def to_dataset(sequence, length, shuffle=False, seed=model_config.seed, batch_size=global_config.batch_per_replica, buffer=global_config.buffer_size):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length+1))
    if shuffle:
        ds = ds.shuffle(buffer_size=global_config.buffer_size, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


def get_datasets(batch_size):
    shakespear_url = "https://homl.info/shakespeare"
    filepath = tf.keras.utils.get_file('shakespear.txt', shakespear_url)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        shakespear_txt = f.read()

    text_vec_layer = tf.keras.layers.TextVectorization(split='character',
                                                    standardize='lower')
    text_vec_layer.adapt([shakespear_txt])                                                    
    encoded = text_vec_layer([shakespear_txt])[0]
    encoded -= 2

    # n_tokens = text_vec_layer.vocabulary_size() - 2
    # n_tokens == 32

    ds = tf.data.Dataset.from_tensor_slices(encoded)
    train_set = to_dataset(encoded[:1_060_000], batch_size=batch_size, length=model_config.block_size, shuffle=True, seed=1337)
    valid_set = to_dataset(encoded[1_060_000:], batch_size=batch_size, length=model_config.block_size)

    return train_set, valid_set
