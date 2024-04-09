import tensorflow as tf
from tensorflow.experimental import numpy as tnp


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 learning_rate: float = 6e-4,
                 warmup_iters: int = 10,
                 min_lr: float = 6e-5,
                 lr_decay_iters: int = 100):

        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.min_lr = min_lr
        self.lr_decay_iters = lr_decay_iters

    def warmup(self, step):
        def res():
            return self.learning_rate * float(step) / self.warmup_iters
        return res

    def late(self):
        return self.min_lr

    def middle(self, step):
        def res():
            decay_ratio = (float(step) - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
            # assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + tf.math.cos(tnp.pi * decay_ratio))
            return self.min_lr + coeff * (self.learning_rate - self.min_lr)
        return res

    @tf.function(jit_compile=True)
    def __call__(self, step):
        lr = tf.case([(tf.less(step, self.warmup_iters), self.warmup(step)),
                     (tf.greater(step, self.lr_decay_iters), self.late)],
                     default=self.middle(step), exclusive=True)

        return lr


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=512):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length+1))
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


class Context():
    def __init__(self,
                 optimizer=None,
                 loss_object=None,
                 train_accuracy=None,
                 val_accuracy=None,
                 val_loss=None,
                 callbacks=None,
                 strategy=None,
                 manager=None,
                 ckpt=None,
                 best_val_loss=None,
                 step=None,
                 train_set_dist=None,
                 valid_set_dist=None):
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.train_accuracy = train_accuracy
        self.val_accuracy = val_accuracy
        self.val_loss = val_loss
        self.callbacks = callbacks
        self.strategy = strategy
        self.manager = manager
        self.ckpt = ckpt
        self.best_val_loss = best_val_loss
        self.step = step
        self.train_set_dist = train_set_dist
        self.valid_set_dist = valid_set_dist
