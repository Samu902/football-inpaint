# general
import os
import shutil

# conversion
from my_util import base64_to_zip_buffer, team_name_to_code
import zipfile

# other
from huggingface_hub import snapshot_download
import subprocess

# celery
from celery import Celery

celery = Celery(
    'pipeline',
    broker='redis://127.0.0.1:6379/0',
    backend='redis://127.0.0.1:6379/0'
)
#celery -A pipeline.celery worker --loglevel=INFO

def init_enviroment():
    print('Initializing pipeline environment...')

    # setup env var
    print('working directory: ' + os.getcwd())

    # ignore proxies
    os.environ['HTTP_PROXY'] = ''
    os.environ['http_proxy'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['https_proxy'] = ''

    # download sdxl model if not already present
    os.makedirs('./models/sdxl_inpainting_model', exist_ok=True)
    print('Loading inpainting model...')
    if len(os.listdir('./models/sdxl_inpainting_model')) == 0:
        snapshot_download("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", local_dir='./models/sdxl_inpainting_model')

    print('Cleaning data directory...')
    shutil.rmtree('./data', ignore_errors=True)
    os.makedirs('./data/training_images', exist_ok=True)

@celery.task
def start_new_task(zip_base64: str, team_name: str, steps: int):
    print('Starting pipeline task...')

    # prepare environment
    init_enviroment()

    # convert and unzip file to training images folder
    zip_buffer = base64_to_zip_buffer(zip_base64)
    with zipfile.ZipFile(zip_buffer, 'r') as zip_file: 
        zip_file.extractall('./data/training_images')

    # team name translation
    team_code = team_name_to_code(team_name)

    # training parameters
    model_path = "./models/sdxl_inpainting_model"
    instance_prompt = f"ftbllplyr {team_code}"
    instance_data_dir = "./data/training_images"
    output_dir = os.getcwd()
    train_steps = steps

    # setup training command
    command = [
        "accelerate", "launch", "./train-lora-sdxl-inpaint/diffusers/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint_lora_sdxl.py",
        "--pretrained_model_name_or_path", model_path,
        "--instance_prompt", instance_prompt,
        "--instance_data_dir", instance_data_dir,
        "--output_dir", output_dir,
        "--train_text_encoder",
        "--mixed_precision", "fp16",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", f"{train_steps}",
        "--seed", "42"
    ]

    # launch training process
    try:
        print(f"Starting training with prompt '{instance_prompt}', training images from '{instance_data_dir}' and outputting to '{output_dir}'...")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as exc:
        raise Exception(exc.output)

    # not returning anything, since the lora file was already saved to disk
    print(f"Training finished, .safetensors file available in {output_dir}.")