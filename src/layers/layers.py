import tensorflow as tf
import numpy as np


class MyLayerNorm(tf.keras.layers.Layer):
    def __init__(self, bias=True, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.bias = bias

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',
                                      shape=input_shape[-1:],  # [-1:] gives last elem but keeps dims
                                      initializer=tf.keras.initializers.Ones(),
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=input_shape[-1:],  # [-1:] gives last elem but keeps dims
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True) if self.bias else None

        super(MyLayerNorm, self).build(input_shape)

    @tf.function(jit_compile=True)
    def call(self, x):
        # Can also use tf.nn.moments(inputs, axes=-1, keepdims=True),
        # but then additionally one needs to take the sqrt to get \sigma
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)

        return self.weight * (x - mean) / (std + self.eps) + self.bias


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must divide number of heads"
        # key, query, value computed at once and splitted later
        self.initializer_proj = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02 / tf.math.sqrt(2. * config.n_layer), seed=None)
        self.c_attn = tf.keras.layers.Dense(3 * config.n_embd,
                                            activation=None,
                                            use_bias=config.bias)
        # output projection
        self.c_proj = tf.keras.layers.Dense(config.n_embd,
                                            activation=None,
                                            kernel_initializer=self.initializer_proj,
                                            use_bias=config.bias)
        self.dropout = config.dropout
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout)
        self.resid_dropout = tf.keras.layers.Dropout(self.dropout)

        self.mask = tf.experimental.numpy.tril(
            tf.ones([config.block_size, config.block_size]))[tf.newaxis, tf.newaxis, :, :]

    @tf.function(jit_compile=True)
    def forward(self, x):

        B, T, C = x.size()  # batch, sequence and channel, which is the embedding dim

        q, k, v = self.c_attn(x).split(self.n_embd, axis=2)

        k = tf.transpose(tf.reshape(k, [B, T, self.n_head, C // self.n_head]),
                         perm=[0, 2, 1, 3])
        q = tf.transpose(tf.reshape(q, [B, T, self.n_head, C // self.n_head]),
                         perm=[0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [B, T, self.n_head, C // self.n_head]),
                         perm=[0, 2, 1, 3])

        att = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * (1.0 / tf.math.sqrt(k.shape[-1]))
        mask = tf.experimental.numpy.tril(tf.ones([T, T]))[tf.newaxis, tf.newaxis, :, :]
        att = tf.where(mask != 0, att, tf.constant(-np.inf))
        att = tf.nn.softmax(att, axis=3)
        att = self.attn_dropout(att)
        y = att @ v
        y = tf.reshape(tf.transpose(y, perm=[0, 2, 1, 3]), [B, T, C])

        return self.resid_dropout(self.c_proj(y))


class MLP(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.initializer_proj = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02 / tf.math.sqrt(2. * config.n_layer), seed=None)
        # Streching and shrinking in channel/embedding dimension,
        # like for large resnets
        self.c_fc = tf.keras.layers.Dense(4 * config.n_embd, activation=None, use_bias=config.bias)
        self.c_proj = tf.keras.layers.Dense(config.n_embd, activation=None, kernel_initializer=self.initializer_proj, use_bias=config.bias)
        self.gelu = tf.keras.activations.gelu
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    @tf.function(jit_compile=True)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = MyLayerNorm(bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = MyLayerNorm(bias=config.bias)
        self.mlp = MLP(config)

    @tf.function(jit_compile=True)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))
