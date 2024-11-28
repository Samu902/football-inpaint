# stage 0
import os
import shutil
import gdown
from my_util import base64_to_PIL, PIL_to_base64

# stage 1
from ultralytics import YOLO
from PIL import Image

# stage 2a
from ultralytics import SAM
import numpy as np
import copy

# stage 2b
from sports.common.team import TeamClassifier

# stage 3
from diffusers import AutoPipelineForInpainting
import torch
import cv2 as cv

# stage 4
import gc

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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # make huggingface cache dir point to custom location (to keep everything inside the project)
    # used by sxdl and siglip model
    os.makedirs('./models/huggingface_cache/hub', exist_ok=True)
    os.environ['HF_HOME'] = os.getcwd() + '/models/huggingface_cache'
    os.environ['HF_HUB_CACHE'] = os.environ['HF_HOME'] + '/hub'

    # ignore proxies
    os.environ['HTTP_PROXY'] = ''
    os.environ['http_proxy'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['https_proxy'] = ''

    # download roboflow model and loras from google drive if not present
    if not os.path.isfile('./models/roboflow_model/best.pt'):
        gdown.download(id='103DgLujAKKLlfETz-rgDO0-ibvQh7evQ', output='./roboflow_model/best.pt')
    if not os.path.isdir('./models/sdxl_lora_weights'):
        gdown.download_folder(id='1VvzOiPwhkv7fuK7P3IktuEzuXdnKg3Le', output='./models/sdxl_lora_weights')

    # initialize models not to waste time and memory every time
    ROBOFLOW_DETECTION_MODEL = YOLO("models/roboflow_model/best.pt")            # pretrained Roboflow YOLO model (training_model_1.ipynb)
    SAM2_SEGMENT_MODEL = SAM("models/sam2_model/sam2_t.pt")                     # SAM2 tiny model (good quality and speed)
    TEAM_CLASSIFIER_MODEL = TeamClassifier(device=DEVICE)                       # Roboflow all-in-one Team Classifier model
    SDXL_INPAINTING_PIPELINE = AutoPipelineForInpainting.from_pretrained(       # SDXL inpainting model
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(DEVICE)
    # load team lora weights on sdxl model
    SDXL_INPAINTING_PIPELINE.load_lora_weights(f"./models/sdxl_lora_weights/sqjvnts", weight_name="pytorch_lora_weights.safetensors", adapter_name="sqjvnts")
    SDXL_INPAINTING_PIPELINE.load_lora_weights(f"./models/sdxl_lora_weights/sqfrntn", weight_name="pytorch_lora_weights.safetensors", adapter_name="sqfrntn")
    SDXL_INPAINTING_PIPELINE.load_lora_weights(f"./models/sdxl_lora_weights/sqntrxx", weight_name="pytorch_lora_weights.safetensors", adapter_name="sqntrxx")
    SDXL_INPAINTING_PIPELINE.load_lora_weights(f"./models/sdxl_lora_weights/sqmlnxx", weight_name="pytorch_lora_weights.safetensors", adapter_name="sqmlnxx")
    SDXL_INPAINTING_PIPELINE.load_lora_weights(f"./models/sdxl_lora_weights/sqnplxx", weight_name="pytorch_lora_weights.safetensors", adapter_name="sqnplxx")
    SDXL_INPAINTING_PIPELINE.load_lora_weights(f"./models/sdxl_lora_weights/sqrmxxx", weight_name="pytorch_lora_weights.safetensors", adapter_name="sqrmxxx")

    print('Cleaning data directory...')

    shutil.rmtree('./data', ignore_errors=True)

    os.makedirs('./data/stage_0', exist_ok=True)
    os.makedirs('./data/stage_1', exist_ok=True)
    os.makedirs('./data/stage_1/players', exist_ok=True)
    os.makedirs('./data/stage_1/players_no_pad', exist_ok=True)
    os.makedirs('./data/stage_2a', exist_ok=True)
    os.makedirs('./data/stage_2a/masks', exist_ok=True)
    os.makedirs('./data/stage_2a/cropped_masks', exist_ok=True)
    os.makedirs('./data/stage_2b', exist_ok=True)
    os.makedirs('./data/stage_2b/team_0', exist_ok=True)
    os.makedirs('./data/stage_2b/team_1', exist_ok=True)
    os.makedirs('./data/stage_3', exist_ok=True)
    os.makedirs('./data/stage_4', exist_ok=True)

    return (DEVICE, ROBOFLOW_DETECTION_MODEL, SAM2_SEGMENT_MODEL, TEAM_CLASSIFIER_MODEL, SDXL_INPAINTING_PIPELINE)

@celery.task
def start_new_task(image_base64: str, team1: str, team2: str):
    print('Starting pipeline task...')

    ## STAGE 0
    # preparazione dell'ambiente prima dell'esecuzione dei modelli della pipeline

    # init_environment
    DEVICE, ROBOFLOW_DETECTION_MODEL, SAM2_SEGMENT_MODEL, TEAM_CLASSIFIER_MODEL, SDXL_INPAINTING_PIPELINE = init_enviroment()

    # translate team codes
    team_dict = { 'Juventus': 'sqjvnts', 'Fiorentina': 'sqfrntn', 'Inter': 'sqntrxx', 'Milan': 'sqmlnxx', 'Napoli': 'sqnplxx', 'Roma': 'sqrmxxx' }
    team1_code = team_dict[team1]
    team2_code = team_dict[team2]

    # save input image
    input_image = base64_to_PIL(image_base64).convert('RGB')
    input_image.save(fp=f"data/stage_0/input_image.jpg")

    # ----------------------------

    ## STAGE 1
    # questo modello trova i giocatori nell'immagine di partenza, crea le bounding box e ritaglia le immagini dei singoli giocatori

    # Run batched inference on input image
    result_stage_1 = ROBOFLOW_DETECTION_MODEL(input_image)[0]

    # Process result and save
    boxes = result_stage_1.boxes.xyxy                   # bounding boxes (tensor)
    classes = result_stage_1.boxes.cls.tolist()         # class of each bounding box
    result_stage_1.save(filename=f"data/stage_1/result.jpg")

    # Crop players from images (filter out referees, goalkeepers and balls) with and without padding
    pad = 50
    i = 0
    for b in boxes:
        if classes[i] == 2:       # only players
            b = b.tolist()
            im = input_image.crop((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))
            im.save(f"data/stage_1/players/player_{i}.jpg")
            im_no_pad = input_image.crop(b)
            im_no_pad.save(f"data/stage_1/players_no_pad/player_{i}.jpg")
        i += 1

    # ----------------------------

    ## STAGE 2a
    # questo modello, guidato dalle bounding box fornite dal modello precedente, crea le maschere dei giocatori in due versioni: immagine integrale e ritagliata a bounding box

    # Run inference on input_image: segment inside bounding boxes
    i = 0
    result_stage_2a = SAM2_SEGMENT_MODEL(input_image, bboxes=copy.deepcopy(boxes))[0]     # deep copy to avoid array "contamination"

    # Process result
    i = 0
    for m in result_stage_2a.masks:
        if classes[i] == 2:       # only players
            mfile = Image.fromarray((m.data.cpu().numpy().squeeze() * 255).astype(np.uint8)).convert('L')   # create mask image
            mfile.save(fp=f"data/stage_2a/masks/mask_{i}.jpg")                                               # save to disk
            b = boxes[i].tolist()
            cmfile = mfile.crop((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))                           # create cropped mask image                                                                          # display to screen
            cmfile.save(fp=f"data/stage_2a/cropped_masks/cropped_mask_{i}.jpg")                              # save to disk
        i += 1
    result_stage_2a.save(filename=f"data/stage_2a/result.jpg")

    # ----------------------------

    ## STAGE 2b
    # questo modello prima impara autonomamente le differenze tra le due squadre e poi divide i giocatori nei due gruppi

    # load player crops
    i = 0
    train_players_crops = []
    predict_player_crops = []
    for b in boxes:
        if not os.path.isfile(f"data/stage_1/players_no_pad/player_{i}.jpg"):      # only players (other classes' images haven't been processed, so skip them)
            i += 1
            continue
        im = Image.open(f"data/stage_1/players_no_pad/player_{i}.jpg")
        train_players_crops.append(np.asarray(im))
        #train_players_crops.append(np.asarray(im.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)))
        #train_players_crops.append(np.asarray(im.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)))
        #train_players_crops.append(np.asarray(im.transpose(method=Image.Transpose.ROTATE_90)))
        #train_players_crops.append(np.asarray(im.transpose(method=Image.Transpose.ROTATE_180)))
        #train_players_crops.append(np.asarray(im.transpose(method=Image.Transpose.ROTATE_270)))
        predict_player_crops.append(np.asarray(im))
        i += 1
    #train_players_crops = train_players_crops * 4

    # fit TeamClassifier
    TEAM_CLASSIFIER_MODEL.fit(train_players_crops)

    # predict teams
    team_ids = list(TEAM_CLASSIFIER_MODEL.predict(predict_player_crops))
    print('team_ids: ' + str(team_ids))

    # fill list with holes (goalkeepers, referees, ball) to pass correct indices to next steps
    i = 0
    for b in boxes:
        if not os.path.isfile(f"data/stage_1/players_no_pad/player_{i}.jpg"):
            team_ids.insert(i, None)
        i += 1
    print('team_ids (filled): \n' + '\n'.join([f'subject {i}\'s team: {t}' for (i, t) in enumerate(team_ids)]))

    # save player crops to corresponding team folder to help debugging
    i = 0
    for b in boxes:
        if not os.path.isfile(f"data/stage_1/players_no_pad/player_{i}.jpg"):      # only players (other classes' images haven't been processed, so skip them)
            i += 1
            continue
        im = Image.open(f"data/stage_1/players_no_pad/player_{i}.jpg")
        im.save(fp=f"data/stage_2b/team_{team_ids[i]}/player_{i}.jpg")
        i += 1

    # DEBUG
    return PIL_to_base64(input_image)

    # ----------------------------

    ## STAGE 3
    # questo modello, a partire dalle immagini ritagliate dei giocatori e alle rispettive maschere, ridisegna i giocatori (inpainting) usando le squadre fornite dall'utente
    # da repo nikgli, sdxl-inpainting-lora.py: "this script is for doing inpainting on the sd/sdxl inpainting model with lora weights"
    
    # inner function to dilate a mask image
    def dilated_mask(mask, dilate_iterations = 6):
        return (
            Image.fromarray(
                cv.cvtColor(
                    np.array(
                        cv.dilate(
                            cv.cvtColor(
                                np.array(
                                    mask
                                ),
                                cv.COLOR_RGB2BGR
                            ),
                            np.ones((5,5), np.uint8),
                            iterations=dilate_iterations
                        )
                    ),
                    cv.COLOR_BGR2RGB
                )
            )
        )
    
    # Run inference (inpainting)

    generator = torch.Generator(device=DEVICE).manual_seed(9)
    output_inpaint_size = (1024, 1024)

    i = 0
    for player_box in boxes:
        if not os.path.isfile(f"data/stage_1/players/player_{i}.jpg"):      # only players (other classes' images haven't been processed, so skip them)
            i += 1
            continue
        
        init_image = Image.open(f"data/stage_1/players/player_{i}.jpg").convert('RGB').resize((1024, 1024))     # load and resize player image
        #init_image = Image.new("RGB", (1024, 1024), (15, 191, 88))     # se la qualità fa schifo, provare questa come input
        mask_image = dilated_mask(Image.open(f"data/stage_2a/cropped_masks/cropped_mask_{i}.jpg").convert('L').resize(output_inpaint_size), 6) # load, dilate and resize mask
        #mask_image = Image.new("L", (1024, 1024), 255)

        team_code = team1_code if team_ids[i] == 0 else team2_code

        # set prompt and adapter based on team classification
        prompt = f"ftbllplyr {team_code}"
        negative_prompt = "orange pink red blue yellow brown green grey purple"
        #negative_prompt = "weird shape, unrealistic"
        #prompt = "soccer player, realistic, vertical striped jersey, vertical white stripes, vertical black stripes"
        #negative_prompt = "blurred, blur, unrealistic, distorted, unnatural pose, cartoon style, vector style, child, orange, red, blue, yellow, brown, green, grey, purple"
        SDXL_INPAINTING_PIPELINE.set_adapters(team_code)
        
        image = SDXL_INPAINTING_PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            guidance_scale=8.0,
            num_inference_steps=30,
            strength=0.99,
            generator=generator,
            width=output_inpaint_size[0],
            height=output_inpaint_size[1]
        ).images[0]                                             # inference (inpainting)
        image.save(f"data/stage_3/result_{i}.jpg")              # save to disk
        i += 1
        ### SHORTCUT DEBUG
        if(i >= 1):
            break
        ###

    # libera RAM e VRAM
    del DEVICE, ROBOFLOW_DETECTION_MODEL, SAM2_SEGMENT_MODEL, TEAM_CLASSIFIER_MODEL, SDXL_INPAINTING_PIPELINE
    torch.cuda.empty_cache()
    gc.collect()

    # ----------------------------

    ## STAGE 4
    # ricomposizione dell'immagine originale con i giocatori ridisegnati
    
    # Apri l'immagine di sfondo
    general_image = Image.open('data/stage_0/input_image.jpg')
    i = 0
    for player_box in boxes:
        # round box coordinates and convert to list
        player_box = [round(b) for b in player_box.tolist()]     
        # Apri l'immagine da incollare e la maschera per ritagliarla
        player_image = Image.open(f"data/stage_3/result_{i}.jpg").resize((player_box[2] - player_box[0] + pad * 2, player_box[3] - player_box[1] + pad * 2))
        mask_image = dilated_mask(Image.open(f"data/stage_2a/cropped_masks/cropped_mask_{i}.jpg").resize((player_box[2] - player_box[0] + pad * 2, player_box[3] - player_box[1] + pad * 2)), 2).convert('L')
        # Incolla l'immagine
        general_image.paste(player_image, (player_box[0] - pad, player_box[1] - pad), mask_image)
        i += 1
        ### SHORTCUT DEBUG
        if(i >= 2):
            break
        ###
    # Salva l'immagine risultante
    general_image.save(f"data/stage_4/result.jpg")

    return PIL_to_base64(general_image)
