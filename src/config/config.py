import sys
import pathlib
from ruamel.yaml import YAML


class GlobalConfig():
    def __init__(self,
                batch_per_replica: int = 512,
                best_val_loss: float = 1e9,
                weight_decay: float = 1e-1,
                warmup_iters: int = 50,
                learning_rate: float = 6e-4,
                max_iters: int = 250,
                lr_decay_iters: int = 250,
                min_lr: float = 6e-5,
                eval_iters: int = 20,
                grad_clip: float = 1.0,
                eval_only: bool = False,
                eval_interval: int = 1,
                always_save_checkpoint: bool = True,
                restore: bool = False,
                checkpoint_directory: str = '',
                log_dir: str = '',
                buffer_size: int = 100_000):
        self.batch_per_replica = batch_per_replica
        self.best_val_loss = best_val_loss
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.eval_iters = eval_iters
        self.grad_clip = grad_clip
        self.eval_only = eval_only
        self.eval_interval = eval_interval
        self.always_save_checkpoint = always_save_checkpoint
        self.restore = restore
        self.checkpoint_directory = checkpoint_directory
        self.log_dir = log_dir
        self.buffer_size = buffer_size


class GPTConfig():
    def __init__(self,
                 block_size: int = 8,
                 vocab_size: int = 39,
                 n_layer: int = 2,
                 n_head: int = 2,
                 n_embd: int = 10,
                 dropout: float = 0.0,
                 bias: bool = False,
                 seed: int = 1337):

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.seed = seed


yaml = YAML(typ='safe', pure=True)

# configs = yaml.load( ( pathlib.Path('./config/config.yaml')) .read_text() )
configs = yaml.load((pathlib.Path(sys.argv[0]).parent / 'config/config.yaml').read_text())

global_config = GlobalConfig(**configs['globals'])
model_config = GPTConfig(**configs['gpt'])
