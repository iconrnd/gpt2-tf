# GPT2 in TensorFlow

NanoGPT (GPT2) Transformer by Andrei Carpathy total conversion to TensorFlow with some additions.

Based on the repo in PyTorch:
https://github.com/karpathy/nanoGPT

### Some features:
*  Distributed custom training loop
*  "Pipeline" for Vertex AI, job runner script
*  Mixed precision training
*  JIT compiled tf.functions()
*  Callbacks with custom training loop
*  Custom LR Scheduler
*  TikToken and TextVectorization encoders
*  Code in src divided into separate modules as a standalone commandline app
*  Custom events writer for TensorBoard
*  Checkpointing
*  Tiny Shakespeare dataset for fast experiments
*  Kept weird variables naming by Andrei to easily match with his PyTorch code :)
*  flake8 compliant :) (except line lengths)
*  block_size has the most signifianct impact on accuracy
