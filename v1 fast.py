import numpy as np
import wave
import scipy.signal
import soundfile as sf
from io import BytesIO
from numba import njit
from PIL import Image
import os
from datetime import datetime

# ====================== Select Effect ======================
# Choose: "highpass", "reverb", "distortion", or "echo"
SELECTED_EFFECT = "echo"

# ====================== Base directories for portability ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER_ROOT = os.path.join(BASE_DIR, "output")

# ====================== μ-law Encoding ======================

@njit
def ulaw_encode(signal, μ=255.0):
    return np.sign(signal) * np.log1p(μ * np.abs(signal)) / np.log1p(μ)

@njit
def ulaw_decode(ulaw_encoded, μ=255.0):
    return np.sign(ulaw_encoded) * (1.0 / μ) * ((1.0 + μ) ** np.abs(ulaw_encoded) - 1.0)

# ====================== JPEG to BMP ======================

def convert_jpeg_to_bmp(jpeg_path, bmp_path):
    with Image.open(jpeg_path) as img:
        img.save(bmp_path, "BMP")

# ====================== Extract BMP Header and Data ======================

def extract_bmp_header_and_data(file_path):
    with open(file_path, 'rb') as f:
        bmp = f.read()
    return bmp[:54], np.frombuffer(bmp[54:], dtype=np.uint8)

# ====================== μ-law Waveform ======================

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

# ====================== Effects ======================

def apply_highpass_filter_to_buffer(wave_buffer, cutoff_freq=1200.0, sample_rate=44100):
    data, sr = sf.read(wave_buffer)
    sos = scipy.signal.butter(4, cutoff_freq, 'hp', fs=sample_rate, output='sos')
    filtered = scipy.signal.sosfilt(sos, data)
    return save_to_buffer(filtered, sample_rate)

def apply_reverb_to_buffer(wave_buffer, decay=0.5, sample_rate=44100, num_reflections=50):
    data, sr = sf.read(wave_buffer)
    
    ir_length = int(sample_rate * decay)
    ir = np.zeros(ir_length)
    reflection_positions = np.linspace(0, ir_length - 1, num_reflections, dtype=int)
    
    for i, pos in enumerate(reflection_positions):
        ir[pos] = np.exp(-3 * (i / num_reflections))
    
    ir /= np.max(np.abs(ir))
    convolved = scipy.signal.fftconvolve(data, ir, mode='full')[:len(data)]
    convolved = convolved / np.max(np.abs(convolved)) * 0.9
    
    return save_to_buffer(convolved, sample_rate)

def apply_distortion_to_buffer(wave_buffer, gain=2.0, sample_rate=44100):
    data, sr = sf.read(wave_buffer)
    distorted = np.tanh(data * gain)
    return save_to_buffer(distorted, sample_rate)

def apply_echo_to_buffer(wave_buffer, delay_seconds=0.01, decay=1.5, repeats=10, sample_rate=44100):
    data, sr = sf.read(wave_buffer)
    delay_samples = int(delay_seconds * sample_rate)
    
    output_length = len(data) + delay_samples * repeats
    output = np.zeros(output_length)
    output[:len(data)] = data
    
    for i in range(1, repeats + 1):
        start = delay_samples * i
        end = start + len(data)
        output[start:end] += data * (decay ** i)
    
    output = output / np.max(np.abs(output)) * 0.9
    return save_to_buffer(output, sample_rate)

def save_to_buffer(audio_array, sample_rate):
    buffer = BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer

# ====================== Reconstruct BMP ======================

def wave_buffer_to_high_quality_jpeg(filtered_buffer, header, output_path_jpeg):
    data, sr = sf.read(filtered_buffer)

    if data.dtype != np.uint8:
        data = np.clip(data, -1.0, 1.0)
        ulaw = ulaw_encode(data)
        pixel_data = ((ulaw + 1.0) / 2.0 * 255).astype(np.uint8)
    else:
        pixel_data = data

    bmp_bytes = header + pixel_data.tobytes()

    with BytesIO(bmp_bytes) as bmp_stream:
        img = Image.open(bmp_stream).convert("RGB")
        img.save(output_path_jpeg, "JPEG", quality=100, subsampling=0, optimize=True, progressive=True)

# ====================== Pipeline ======================

def run_databending_pipeline_multiple_filters(input_path, effect_values, output_root):
    if input_path.lower().endswith(".jpg") or input_path.lower().endswith(".jpeg"):
        bmp_path = input_path.replace(".jpg", ".bmp").replace(".jpeg", ".bmp")
        print(f"Converting {input_path} to BMP...")
        convert_jpeg_to_bmp(input_path, bmp_path)
    else:
        bmp_path = input_path

    print(f"Extracting BMP header and pixel data from {bmp_path}...")
    header, pixel_data = extract_bmp_header_and_data(bmp_path)

    print("Converting pixel data to μ-law waveform in memory...")
    wave_buffer = image_bytes_to_ulaw_wave_buffer(pixel_data)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(output_root, timestamp)
    os.makedirs(output_folder, exist_ok=True)

    for i, param in enumerate(effect_values, start=1):
        print(f"\n=== Pass {i}: Applying {SELECTED_EFFECT} @ {param} ===")
        buffer_copy = BytesIO(wave_buffer.getvalue())

        if SELECTED_EFFECT == "highpass":
            processed = apply_highpass_filter_to_buffer(buffer_copy, cutoff_freq=param)
        elif SELECTED_EFFECT == "reverb":
            processed = apply_reverb_to_buffer(buffer_copy, decay=param)
        elif SELECTED_EFFECT == "distortion":
            processed = apply_distortion_to_buffer(buffer_copy, gain=param)
        elif SELECTED_EFFECT == "echo":
            processed = apply_echo_to_buffer(buffer_copy, delay_seconds=param / 1000.0)
        else:
            raise ValueError(f"Unsupported effect: {SELECTED_EFFECT}")

        output_filename_jpeg = os.path.join(output_folder, f"{SELECTED_EFFECT}_{param}.jpg")
        wave_buffer_to_high_quality_jpeg(processed, header, output_filename_jpeg)
        print(f"Saved JPEG: {output_filename_jpeg}")

# ====================== Example ======================

if __name__ == "__main__":
    if SELECTED_EFFECT == "highpass":
        effect_values = [0.5, 1, 3, 5, 7, 10, 20, 50, 100, 1000, 10000]
    elif SELECTED_EFFECT == "reverb":
        effect_values = [0.00025, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01]
    elif SELECTED_EFFECT == "distortion":
        effect_values = [0.5, 1, 2, 3, 5]
    elif SELECTED_EFFECT == "echo":
        effect_values = [0.5, 1, 2, 3, 4, 4.5, 5, 6, 7, 8, 9, 10, 20, 100, 250, 500, 5000]  # in ms
    else:
        raise ValueError("Set SELECTED_EFFECT to a valid value.")

    input_file = os.path.join(INPUT_FOLDER, "input85.jpg")
    run_databending_pipeline_multiple_filters(input_file, effect_values, OUTPUT_FOLDER_ROOT)



