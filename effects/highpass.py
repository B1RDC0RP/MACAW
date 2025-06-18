import scipy.signal
import soundfile as sf
from io import BytesIO

def apply_highpass_filter_to_buffer(wave_buffer, cutoff_freq=1200.0, sample_rate=22050):
    data, sr = sf.read(wave_buffer, dtype='float32')
    sos = scipy.signal.butter(4, cutoff_freq, 'hp', fs=sample_rate, output='sos')
    filtered = scipy.signal.sosfilt(sos, data)
    buffer = BytesIO()
    sf.write(buffer, filtered, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer