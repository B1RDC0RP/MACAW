import numpy as np
from numba import njit
from io import BytesIO
import wave

@njit
def ulaw_encode(signal, μ=255.0):
    return np.sign(signal) * np.log1p(μ * np.abs(signal)) / np.log1p(μ)

def image_bytes_to_ulaw_wave_buffer(pixel_data, sample_rate=22050):
    normalized = (pixel_data.astype(np.float32) - 128.0) / 128.0
    ulaw = ulaw_encode(normalized)
    ulaw_uint8 = ((ulaw + 1.0) / 2.0 * 255).astype(np.uint8)
    wave_buffer = BytesIO()
    with wave.open(wave_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sample_rate)
        wf.writeframes(ulaw_uint8.tobytes())
    wave_buffer.seek(0)
    return wave_buffer