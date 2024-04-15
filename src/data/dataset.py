import tensorflow as tf
import tiktoken
from config.config import global_config


def to_dataset(sequence, length, seed=1337, shuffle=False, batch_size=global_config.batch_per_replica, buffer=global_config.buffer_size):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length+1))
    if shuffle:
        ds = ds.shuffle(buffer_size=global_config.buffer_size, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


def get_datasets(batch_size, model_config):
    shakespear_url = "https://homl.info/shakespeare"
    filepath = tf.keras.utils.get_file('shakespear.txt', shakespear_url)

    with open(filepath, 'r', encoding='utf-8') as f:
        shakespear_txt = f.read()

    text_vec_layer = tf.keras.layers.TextVectorization(split='character',
                                                       standardize='lower')
    text_vec_layer.adapt([shakespear_txt])
    encoded = text_vec_layer([shakespear_txt])[0]
    encoded_len = len(encoded)

    n_tokens = text_vec_layer.vocabulary_size() - 2

    dataset = to_dataset(encoded, batch_size=batch_size, seed=model_config.seed, length=model_config.block_size, shuffle=True)

    dataset = dataset.shuffle(buffer_size=4096)

    train_set = dataset.take(int(0.9 * encoded_len / (128 * 64)))
    valid_set = dataset.skip(int(0.9 * encoded_len / (128 * 64)))

    return train_set, valid_set, n_tokens


def get_datasets_tiktok(batch_size, model_config):
    shakespear_url = "https://homl.info/shakespeare"
    filepath = tf.keras.utils.get_file('shakespear.txt', shakespear_url)

    with open(filepath, 'r', encoding='utf-8') as f:
        shakespear_txt = f.read()

    encoder = tiktoken.get_encoding("gpt2")

    encoded = encoder.encode_ordinary(shakespear_txt)

    encoded_len = len(encoded)

    dataset = to_dataset(encoded, batch_size=batch_size, seed=model_config.seed, length=model_config.block_size, shuffle=True)

    train_set = dataset.take(int(0.9 * encoded_len / (model_config.block_size * batch_size)))
    valid_set = dataset.skip(int(0.9 * encoded_len / (model_config.block_size * batch_size)))

    return train_set, valid_set, model_config.vocab_size

