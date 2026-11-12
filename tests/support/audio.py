"""Audio de test généré en mémoire : signaux simples et fichiers encodés (soundfile ou PyAV).

Aucun fichier binaire n'est versionné pour ces tests : chaque format est encodé à la demande, avec le
FFmpeg embarqué dans PyAV, comme le produirait un navigateur.
"""

import io

import av
import numpy as np
import soundfile as sf

# (conteneur PyAV, codec) des formats compressés produits par les navigateurs ou les dictaphones.
COMPRESSED_FORMATS = {
    "webm_opus": ("webm", "libopus"),  # MediaRecorder de Chrome et Firefox
    "ogg_opus": ("ogg", "libopus"),  # MediaRecorder de Firefox
    "mp4_aac": ("mp4", "aac"),  # MediaRecorder de Safari, mémos vocaux
    "mp3": ("mp3", "libmp3lame"),
}


def tone(frequency: float = 440.0, seconds: float = 1.0, sample_rate: int = 16_000, amplitude: float = 0.3) -> np.ndarray:
    """Sinusoïde float32 : (n,)."""
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def as_frames(samples: np.ndarray) -> np.ndarray:
    """(n,) ou (n, canaux) -> (n, canaux)."""
    return samples[:, None] if samples.ndim == 1 else samples


def soundfile_bytes(samples: np.ndarray, sample_rate: int, format: str = "WAV", subtype: str | None = None) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, as_frames(samples), sample_rate, format=format, subtype=subtype)
    return buffer.getvalue()


def pyav_bytes(samples: np.ndarray, sample_rate: int, container: str, codec: str) -> bytes:
    """Encode (n,) ou (n, canaux) avec FFmpeg, par trames de 20 ms."""
    planar = np.ascontiguousarray(as_frames(samples).T, dtype=np.float32)  # (canaux, n)
    layout = "mono" if planar.shape[0] == 1 else "stereo"
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container) as output:
        stream = output.add_stream(codec, rate=sample_rate, layout=layout)
        step = sample_rate // 50
        for start in range(0, planar.shape[1], step):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(planar[:, start : start + step]), format="fltp", layout=layout
            )
            frame.sample_rate = sample_rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return buffer.getvalue()


def encoded(kind: str, samples: np.ndarray, sample_rate: int) -> bytes:
    """Fichier d'un format compressé de COMPRESSED_FORMATS."""
    container, codec = COMPRESSED_FORMATS[kind]
    return pyav_bytes(samples, sample_rate, container, codec)


def dominant_frequency(signal: np.ndarray, sample_rate: int) -> float:
    """Fréquence du pic du spectre d'amplitude."""
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    return float(np.fft.rfftfreq(signal.size, 1 / sample_rate)[spectrum.argmax()])
