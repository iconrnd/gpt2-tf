import os
from pathlib import  Path
import numpy as np
import json
import tensorflow as tf
from google.cloud import aiplatform
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='./credentials/tf101.json'

project_id = 'just-aloe-414315'
bucket_name = 'tf101_bucket'
location = 'us-central1'

storage_client = storage.Client(project=project_id)

# Bucket creation, done only once
# bucket = storage_client.create_bucket(bucket_name, location=location)

bucket = storage_client.get_bucket(bucket_name)

aiplatform.init(project=project_id, location=location)

server_image = "gcr.io/cloud-aiplatform/prediction/tf2-gpu.2-8:latest"

custom_training_job = aiplatform.CustomTrainingJob(
    display_name="gpt2_training_job",
    script_path="job_script.py",
    # Training image containter - for GPU training
    # container_uri='gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest',
    # No quota for GPU, changing to CPU
    container_uri='gcr.io/cloud-aiplatform/training/tf-cpu.2-4:latest',
    # Inference image container
    model_serving_container_image_uri=server_image,
    # Library dependences example
    requirements=["gcsfs==2022.3.0"],
    # Training script storage and model saving bucket
    staging_bucket=f'gs://{bucket_name}/staging'
)

job_run = custom_training_job.run(
    machine_type="n1-standard-4",
    replica_count=4,
    # No GPU quota available, changing to CPU
    #accelerator_type="NVIDIA_TESLA_K80",
    #accelerator_count=0,
)
# Remote TB access with the run id (accessible after authoriation)
# tensorboard --logdir gs://projects/just-aloe-414315/logs/ml.googleapis.com%2F5106815277148930048