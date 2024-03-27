import tensorflow as tf
import tensorflow_probability as tfp
from layers.layers import *



class GPT(tf.keras.models.Model):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.initializer_dense = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=self.config.seed)
        self.initializer_embed = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=self.config.seed)
        self.initializer_bias = tf.keras.initializers.Zeros()
        
        self.wte = tf.keras.layers.Embedding(self.config.vocab_size, 
                                             self.config.n_embd, 
                                             embeddings_initializer=self.initializer_embed, 
                                             name='wte')
        
        self.wpe = tf.keras.layers.Embedding(self.config.block_size, 
                                             self.config.n_embd, 
                                             embeddings_initializer=self.initializer_embed, 
                                             name='wpe')
        
        self.drop = tf.keras.layers.Dropout(self.config.dropout, name='drop')
        
        self.h = [Block(self.config) for _ in range(self.config.n_layer)]
        
        self.ln_f = MyLayerNorm(bias=self.config.bias, name='ln_f')
        
        # Crucial for mixed precision: final model output should be of dtype='float32'
        self.lm_head = tf.keras.layers.Dense(self.config.vocab_size, use_bias=self.config.bias, dtype='float32')

    def build(self, input_shape):
        self.wte.build(input_shape=[self.config.vocab_size])
        self.lm_head.build(input_shape=[self.config.n_embd])
        self.wte.trainable_weights[0].assign(tf.transpose(self.lm_head.trainable_weights[0]))
    
    @tf.function(jit_compile=True)
    def call(self, idx):
        b, t = idx.shape
        #assert t <= self.config.block_size, f'sequence too long for the defined context of {self.config.block_size}'
        pos = tf.range(0, t, dtype=tf.int64)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def crop_block_size(self, block_size):
        assert block_size < self.config.block_size
        self.config.block_size = block_size
        self.wpe.weights[0] = tf.Variable(self.wpe.weights[0][:block_size], trainable=True)
        for block in self.h:
            #if hasattr(block.attn, 'bias'):
            if len(block.attn.weights) == 2:
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
                  
    def get_num_params(self, non_embedding=True):
        n_params = self.count_params()
        if non_embedding:
            n_params -= self.wpe.count_params()
        return n_params
        
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        L, H, Q, T = self.config.n_layer, self.config.n_head, self.config.n_embd / self.config.n_head, self.config.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 # A100 at bfloat16
        mfu = flops_achieved / flops_promised
        return mfu
        
    @tf.function(jit_compile=True)
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for i in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond, training=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = tf.math.top_k(logits, min(top_k, logits.shape[-1]))
                # Returned top_k values are sorted, so [-1] is the smallest of top_k
                # and we cut all below that value
                logits[logits < v[:, -1]] = -float('Inf')
            probs = tf.keras.activations.softmax(logits, axis=-1)
            idx_dist = tfp.distributions.Multinomial(total_count=1, probs=probs)
            idx_next = idx_dist.sample(1)
            idx_next = tf.reshape(tf.math.argmax(idx_next, axis=-1), shape=(-1, 1))
            idx = tf.concat([idx, idx_next], axis=1)
        return idx