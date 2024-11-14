import base64
from io import BytesIO
from PIL import Image
import traceback

def PIL_to_base64(img: Image.Image):
    img_bytes = BytesIO()
    img = img.convert('RGB')
    img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    b64_str = base64.b64encode(img_bytes).decode('utf-8')
    return b64_str

def base64_to_PIL(b64_str: str):
    img_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_bytes))

def log_exc_to_file():
    with open('app.log', 'w+') as f:
        traceback.print_exc(file=f)