import base64
from io import BytesIO
import traceback

def zip_bytes_to_base64(zip_bytes: bytes):
    b64_str = base64.b64encode(zip_bytes).decode('utf-8')
    return b64_str

def base64_to_zip_buffer(b64_str: str):
    byte_data = base64.b64decode(b64_str)
    zip_buffer = BytesIO(byte_data)
    zip_buffer.seek(0)
    return zip_buffer

def log_exc_to_file():
    with open('app.log', 'w+') as f:
        traceback.print_exc(file=f)

def team_name_to_code(name: str):
    name = name.replace(' ', '')
    name = name.lower()
    name = ''.join([c for c in name if c not in 'aeiou'])
    name = name.ljust(5, 'x')
    name = name[:5]
    return f'sq{name}'