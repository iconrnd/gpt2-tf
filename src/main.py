import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from training.training import *

model, ctx = get_model_and_ctx()

training(model, ctx)




if __name__ == '__main__':
    run()