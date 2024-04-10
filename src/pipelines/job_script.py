import tensorflow as tf
from config.config import model_config
from training.training import get_model_and_ctx, training
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

model, ctx = get_model_and_ctx(model_config, restore=False)

training(model, ctx)