import sys
import tensorflow as tf
from config.config import model_config
import argparse
from training.training import get_model_and_ctx, training
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')


def main():
    parser = argparse.ArgumentParser(description='nanoGPT2 model training')
    parser.add_argument('--restore', default=False, type=bool, help='Resume training?')
    parser.add_argument('--n_layer', default=model_config.n_layer, type=int, help='Number of attention layers')
    parser.add_argument('--n_head', default=model_config.n_head, type=int, help='Number of attention heads per layer')
    parser.add_argument('--n_embd', default=model_config.n_embd, type=int, help='Embedding dimension')
    parser.add_argument('--block_size', default=model_config.block_size, type=int, help='Input context length')
    parser.add_argument('--bias', default=model_config.bias, type=bool, help='Use bias?')
    parser.add_argument('--seed', default=model_config.seed, type=int, help='Seed')

    args = parser.parse_args()

    model_config.n_layer = args.n_layer
    model_config.n_head = args.n_head
    model_config.n_embd = args.n_embd
    model_config.block_size = args.block_size
    model_config.bias = args.bias
    model_config.seed = args.seed

    model, ctx = get_model_and_ctx(model_config, args.restore)

    training(model, ctx)


if __name__ == '__main__':
    sys.exit(main())
    print('\nDone\n')
