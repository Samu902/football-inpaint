# stage 0
import os
import shutil
import gdown

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

#ready = False

def init(): # andrà prob spostato in dockerfile
    #global ready

    clean_data_dir()

    gdown.download(id='103DgLujAKKLlfETz-rgDO0-ibvQh7evQ', output='./roboflow-model/best.pt')
    gdown.download_folder(id='1VvzOiPwhkv7fuK7P3IktuEzuXdnKg3Le', output='./sdxl_lora_weights')

    #ready = True

def clean_data_dir():
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

def start(input_image: Image.Image, team1: str, team2: str):
    
    ## STAGE 0
    # preparazione dell'ambiente prima dell'esecuzione dei modelli della pipeline

    # clean data directory
    clean_data_dir()

    # translate team codes
    team_dict = { 'Juventus': 'sqjvnts', 'Fiorentina': 'sqfrntn', 'Inter': 'sqntrxx', 'Milan': 'sqmlnxx', 'Napoli': 'sqnplxx', 'Roma': 'sqrmxxx' }
    team1_code = team_dict[team1]
    team2_code = team_dict[team2]

    # setup env var
    print('working directory: ' + os.getcwd())
    my_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # save input image
    input_image = input_image.convert('RGB')
    input_image.save(fp=f"data/stage_0/input_image.jpg")

    # ----------------------------

    ## STAGE 1
    # questo modello trova i giocatori nell'immagine di partenza, crea le bounding box e ritaglia le immagini dei singoli giocatori

    # Load pretrained Roboflow YOLO model (training_model_1.ipynb)
    model = YOLO("roboflow-model/best.pt")

    # Run batched inference on input image
    result_stage_1 = model(input_image)[0]

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

    # Load SAM2 tiny model (good quality and speed)
    model = SAM("sam2_t.pt")

    # Run inference on input_image: segment inside bounding boxes
    i = 0
    result_stage_2a = model(input_image, bboxes=copy.deepcopy(boxes))[0]     # deep copy to avoid array "contamination"

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

    # ----------------------------

    ## STAGE 2b
    # questo modello prima impara autonomamente le differenze tra le due squadre e poi divide i giocatori nei due gruppi

    # load player crops
    i = 0
    players_crops = []
    for b in boxes:
        if not os.path.isfile(f"data/stage_1/players_no_pad/player_{i}.jpg"):      # only players (other classes' images haven't been processed, so skip them)
            i += 1
            continue
        im = Image.open(f"data/stage_1/players_no_pad/player_{i}.jpg")
        players_crops.append(np.asarray(im))
        i += 1

    # fit TeamClassifier
    team_classifier = TeamClassifier(device=my_device)
    team_classifier.fit(players_crops)

    # predict teams
    team_ids = list(team_classifier.predict(players_crops))
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
    return input_image

    # ----------------------------

    ## STAGE 3
    # questo modello, a partire dalle immagini ritagliate dei giocatori e alle rispettive maschere, ridisegna i giocatori (inpainting) usando le squadre fornite dall'utente

    # da repo nikgli sdxl-inpainting-lora.py: "this script is for doing inpainting on the sd/sdxl inpainting model with lora weights"
    # caricamento della pipeline e dei due LORA per le squadre
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        #torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.load_lora_weights(f"./sdxl_lora_weights/{team1_code}", weight_name="pytorch_lora_weights.safetensors", adapter_name="sq1")
    pipe.load_lora_weights(f"./sdxl_lora_weights/{team2_code}", weight_name="pytorch_lora_weights.safetensors", adapter_name="sq2")
    #https://huggingface.co/docs/diffusers/using-diffusers/loading

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

    prompt = f"ftbllplyr {team1_code}"
    negative_prompt = "orange pink red blue yellow brown green grey purple"
    #negative_prompt = "weird shape, unrealistic"
    #prompt = "soccer player, realistic, vertical striped jersey, vertical white stripes, vertical black stripes"
    #negative_prompt = "blurred, blur, unrealistic, distorted, unnatural pose, cartoon style, vector style, child, orange, red, blue, yellow, brown, green, grey, purple"
    generator = torch.Generator(device=my_device).manual_seed(9)
    output_inpaint_size = (1024, 1024)

    i = 0
    for player_box in boxes:
        if not os.path.isfile(f"data/stage_1/players/player_{i}.jpg"):      # only players (other classes' images haven't been processed, so skip them)
            i += 1
            continue
        init_image = Image.open(f"data/stage_1/players/player_{i}.jpg").convert('RGB').resize((1024, 1024))     # load and resize player image
        #init_image = Image.new("RGB", (1024, 1024), (15, 191, 88))
        mask_image = dilated_mask(Image.open(f"data/stage_2a/cropped_masks/cropped_mask_{i}.jpg").convert('L').resize(output_inpaint_size), 6) # load, dilate and resize mask
        #mask_image = Image.new("L", (1024, 1024), 255)
        pipe.set_adapters("sq1")
        image = pipe(
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

    # ----------------------------

    ## STAGE 4
    # ricomposizione dell'immagine originale con i giocatori ridisegnati
    
    # Apri l'immagine di sfondo
    general_image = Image.open('data/stage_0/input_image.jpg')
    i = 0
    for player_box in boxes:
        player_box = list(map(lambda b: round(b), player_box.tolist()))     # round box coordinates and convert to list
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

    return general_image

#init()
