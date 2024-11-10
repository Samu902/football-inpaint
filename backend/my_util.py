import base64
from io import BytesIO
from PIL import Image

def PIL_to_base64(img: Image.Image):
    img = img.convert('RGB')
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()               # im_bytes: image in binary format.
    return base64.b64encode(im_bytes)

def base64_to_PIL(b64: bytes):
    im_bytes = base64.b64decode(b64)         # im_bytes is a binary image
    im_file = BytesIO(im_bytes)                 # convert image to file-like object
    return Image.open(im_file)                  # img is now PIL Image object
