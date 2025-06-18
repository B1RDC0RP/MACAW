import os
import argparse
from effects.highpass import apply_highpass_filter_to_buffer
# from effects.reverb import apply_reverb_to_buffer
# from effects.distortion import apply_distortion_to_buffer
# from effects.echo import apply_echo_to_buffer
from utils.image_io import convert_jpeg_to_bmp, extract_bmp_header_and_data, wave_buffer_to_high_quality_jpeg
from utils.audio_io import image_bytes_to_ulaw_wave_buffer

from datetime import datetime

def run_databending_pipeline_multiple_filters(input_path, effect, effect_values, sample_rate=22050):
    if input_path.lower().endswith((".jpg", ".jpeg")):
        bmp_path = input_path.rsplit(".", 1)[0] + ".bmp"
        print(f"Converting {input_path} to BMP...")
        convert_jpeg_to_bmp(input_path, bmp_path)
    else:
        bmp_path = input_path

    print(f"Extracting BMP header and pixel data from {bmp_path}...")
    header, pixel_data = extract_bmp_header_and_data(bmp_path)
    print("Converting pixel data to μ-law waveform in memory...")
    wave_buffer = image_bytes_to_ulaw_wave_buffer(pixel_data, sample_rate=sample_rate)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join("output", timestamp)
    os.makedirs(output_folder, exist_ok=True)
    for i, param in enumerate(effect_values, start=1):
        print(f"\n=== Pass {i}: Applying {effect} @ {param} ===")
        buffer_copy = wave_buffer.__class__(wave_buffer.getvalue())
        if effect == "highpass":
            processed = apply_highpass_filter_to_buffer(buffer_copy, cutoff_freq=param, sample_rate=sample_rate)
        # elif effect == "reverb":
        #     processed = apply_reverb_to_buffer(buffer_copy, decay=param, sample_rate=sample_rate)
        # elif effect == "distortion":
        #     processed = apply_distortion_to_buffer(buffer_copy, gain=param, sample_rate=sample_rate)
        # elif effect == "echo":
        #     processed = apply_echo_to_buffer(buffer_copy, delay_seconds=param / 1000.0, sample_rate=sample_rate)
        else:
            raise ValueError(f"Unsupported effect: {effect}")
        output_filename_jpeg = os.path.join(output_folder, f"{effect}_{param}.jpg")
        wave_buffer_to_high_quality_jpeg(processed, header, output_filename_jpeg, len(pixel_data))
        print(f"Saved JPEG: {output_filename_jpeg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Databending Glitcher: Apply audio effects to image bytes")
    parser.add_argument("input", help="Input image (JPEG or BMP)")
    parser.add_argument("--effect", choices=["highpass"], default="highpass")
    parser.add_argument("--params", nargs="+", type=float, help="Effect parameters list")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    args = parser.parse_args()
    if not args.params:
        if args.effect == "highpass":
            effect_values = [500, 1000, 5000, 10000]
        # elif args.effect == "reverb":
        #     effect_values = [0.001, 0.003, 0.005, 0.01]
        # elif args.effect == "distortion":
        #     effect_values = [1, 2, 3, 5]
        # elif args.effect == "echo":
        #     effect_values = [1, 3, 5, 10, 50, 100, 500, 1000]
    else:
        effect_values = args.params
    run_databending_pipeline_multiple_filters(args.input, args.effect, effect_values, args.sample_rate)