from PIL import Image
import numpy as np
from io import BytesIO

def convert_jpeg_to_bmp(jpeg_path, bmp_path):
    with Image.open(jpeg_path) as img:
        img = img.convert("RGB")
        img.save(bmp_path, "BMP")

def extract_bmp_header_and_data(file_path):
    with open(file_path, 'rb') as f:
        bmp = f.read()
    header = bmp[:54]
    pixel_data = np.frombuffer(bmp[54:], dtype=np.uint8)
    return header, pixel_data

def wave_buffer_to_high_quality_jpeg(filtered_buffer, header, output_path_jpeg, orig_length):
    import soundfile as sf
    from utils.audio_io import ulaw_encode
    data, sr = sf.read(filtered_buffer, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    data = np.clip(data, -1.0, 1.0)
    ulaw = ulaw_encode(data)
    pixel_data = ((ulaw + 1.0) / 2.0 * 255).astype(np.uint8)
    if len(pixel_data) < orig_length:
        pixel_data = np.pad(pixel_data, (0, orig_length - len(pixel_data)), mode='constant')
    elif len(pixel_data) > orig_length:
        pixel_data = pixel_data[:orig_length]
    bmp_bytes = header + pixel_data.tobytes()
    with BytesIO(bmp_bytes) as bmp_stream:
        img = Image.open(bmp_stream).convert("RGB")
        img.save(output_path_jpeg, "JPEG", quality=100, subsampling=0, optimize=True, progressive=True)