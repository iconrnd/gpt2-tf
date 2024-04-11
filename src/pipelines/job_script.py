import os

# This can be put to requirements.txt
os.system('pip install ruamel.yaml')
os.system('pip install tensorflow==2.15.1')
os.system('pip install tensorflow_probability==0.22.0')
os.system('pip install tiktoken==0.6.0')

os.system('git clone https://github.com/iconrnd/gpt2-tf.git')

os.system('cd ./gpt2-tf/src/; python main.py')
