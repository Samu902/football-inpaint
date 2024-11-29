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
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

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

def init_environment(team_1: str, team_2: str):
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
    SDXL_INPAINTING_PIPELINE.load_lora_weights(f"./models/sdxl_lora_weights/{team_1}", weight_name="pytorch_lora_weights.safetensors", adapter_name=f"{team_1}")
    SDXL_INPAINTING_PIPELINE.load_lora_weights(f"./models/sdxl_lora_weights/{team_2}", weight_name="pytorch_lora_weights.safetensors", adapter_name=f"{team_2}")

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

    # translate team codes and colors
    team_dict = { 'Juventus': 'sqjvnts', 'Fiorentina': 'sqfrntn', 'Inter': 'sqntrxx', 'Milan': 'sqmlnxx', 'Napoli': 'sqnplxx', 'Roma': 'sqrmxxx' }
    team1_code = team_dict[team1]
    team2_code = team_dict[team2]

    total_colors = ['white', 'black', 'orange', 'pink', 'red', 'blue', 'lightblue', 'yellow', 'brown', 'green', 'grey', 'purple']
    team_colors_dict = {
        'sqjvnts': ['white', 'black'],
        'sqfrntn': ['purple'],
        'sqntrxx': ['blue', 'black'],
        'sqmlnxx': ['red', 'black'],
        'sqnplxx': ['lightblue'],
        'sqrmxxx': ['red', 'orange']
    }

    # init_environment
    DEVICE, ROBOFLOW_DETECTION_MODEL, SAM2_SEGMENT_MODEL, TEAM_CLASSIFIER_MODEL, SDXL_INPAINTING_PIPELINE = init_environment(team1_code, team2_code)

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

    # ----------------------------

    ## STAGE 3
    # questo modello, a partire dalle immagini ritagliate dei giocatori e alle rispettive maschere, ridisegna i giocatori (inpainting) usando le squadre fornite dall'utente
    # da repo nikgli, sdxl-inpainting-lora.py: "this script is for doing inpainting on the sd/sdxl inpainting model with lora weights"

    # Load a pre-trained ResNet model
    resnet = models.resnet50(pretrained=True)
    resnet.eval()  # Set the model to evaluation mode

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension
        with torch.no_grad():
            features = resnet(img_tensor)
        return features.squeeze().numpy()

    def find_most_similar(input_image_path, image_set_paths):
        input_features = extract_features(input_image_path)
        similarities = []
        for img_path in image_set_paths:
            try:
                features = extract_features(img_path)
                similarity = cosine_similarity([input_features], [features])[0][0]
                similarities.append((img_path, similarity))
            except Exception as e:  # Handle potential errors like invalid images
                print(f"Error processing image {img_path}: {e}")
                similarities.append((img_path, -1)) # Assign -1 for errors so they are ranked last

        similarities.sort(key=lambda x: x[1], reverse=True) # Sort in descending order
        return similarities[0][0]

    # find most similar hi-res mask and base starting from player cropped mask
    hi_res_masks_dir = './hi_res/mask'
    hi_res_masks = [os.path.join(hi_res_masks_dir, m) for m in os.listdir(hi_res_masks_dir) if os.path.isfile(os.path.join(hi_res_masks_dir, m))]
    mask_to_hi_res_dict = {}
    for in_cmask in os.listdir('./data/stage_2a/cropped_masks'):
        hi_res_mask_path = find_most_similar(os.path.join('./data/stage_2a/cropped_masks', in_cmask), hi_res_masks)
        hi_res_base_path = hi_res_mask_path.replace('/mask/', '/base/')
        mask_to_hi_res_dict[os.path.join('./data/stage_2a/cropped_masks', in_cmask)] = {
            "mask": hi_res_mask_path,
            "base": hi_res_base_path,
        }

    # ----------------------------

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

        team_code = team1_code if team_ids[i] == 0 else team2_code
        team1_weight = 1.0 if team_ids[i] == 0 else 0.0
        team2_weight = 1.0 if team_ids[i] == 1 else 0.0

        team_colors_str = ' '.join([c for c in total_colors if c in team_colors_dict[team_code]])
        not_team_colors_str = ' '.join([c for c in total_colors if c not in team_colors_dict[team_code]])

        # set prompt and adapter based on team classification
        prompt = f"ftbllplyr {team_code} {team_colors_str}"
        negative_prompt = f"{not_team_colors_str}"
        SDXL_INPAINTING_PIPELINE.set_adapters([team1_code, team2_code], adapter_weights=[team1_weight, team2_weight])

        init_image = Image.open(mask_to_hi_res_dict[f"./data/stage_2a/cropped_masks/cropped_mask_{i}.jpg"]["base"]).convert('RGB').resize(output_inpaint_size)                  # load and resize player image
        mask_image = dilated_mask(Image.open(mask_to_hi_res_dict[f"./data/stage_2a/cropped_masks/cropped_mask_{i}.jpg"]["mask"]).convert('L').resize(output_inpaint_size), 6)   # load, dilate and resize mask

        print(f'player {i}: prompt={prompt}, negative_prompt={negative_prompt}, team1_weight={team1_weight}, team2_weight={team2_weight}')

        image = SDXL_INPAINTING_PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            guidance_scale=10.0,   # 8.0 oppure 5.0
            num_inference_steps=30,
            strength=0.99,
            generator=generator,
            width=output_inpaint_size[0],
            height=output_inpaint_size[1]
        ).images[0]                                             # inference (inpainting)
        image.save(f"data/stage_3/result_{i}.jpg")              # save to disk
        i += 1

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
        player_image = Image.open(f"data/stage_3/result_{i}.jpg").convert('RGB').resize((player_box[2] - player_box[0] + pad, player_box[3] - player_box[1] + pad))
        mask_image = dilated_mask(Image.open(f"./data/stage_2a/cropped_masks/cropped_mask_{i}.jpg").resize(player_image.size), 0).convert('L')
        # Incolla l'immagine
        general_image.paste(player_image, (player_box[0] - round(pad / 2), player_box[1] - round(pad / 2)), mask_image)
        i += 1
    # Salva l'immagine risultante
    general_image.save(f"data/stage_4/result.jpg")

    return PIL_to_base64(general_image)
