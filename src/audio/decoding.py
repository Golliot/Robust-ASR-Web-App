"""Décodage d'un fichier audio quelconque en échantillons float32, à sa fréquence d'origine.

Deux décodeurs, essayés dans cet ordre :
    soundfile   WAV, FLAC, OGG (Vorbis, Opus), MP3… C'est le lecteur de `load_audio` dans le notebook :
                les échantillons sont donc identiques à ceux que voyait le pipeline de recherche.
    PyAV        tout ce que décode FFmpeg, embarqué dans la roue : WebM/Opus (Chrome, Firefox),
                MP4/AAC (Safari), etc. Le navigateur ne produit pas forcément du WAV.

Aucune conversion ici : le passage en mono et à 16 kHz se fait dans src/audio/preprocessing.py.

Le décodage s'arrête dès que la durée dépasse la limite, avec une marge : un fichier qui annonce ou
contient des heures d'audio n'est jamais décodé entièrement. La vérification exacte de la durée se
fait au prétraitement, sur les échantillons réellement décodés.
"""

import io
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf

from src.audio.errors import AudioError, AudioTooLongError, EmptyAudioError, InvalidAudioError

# Les métadonnées d'un conteneur arrondissent parfois la durée : la limite de décodage a donc une marge.
DURATION_MARGIN_S = 1.0

UNREADABLE = "Fichier audio illisible : format non pris en charge ou fichier corrompu."

# Noms des démultiplexeurs FFmpeg qui couvrent plusieurs formats.
_FORMAT_LABELS = {
    "matroska,webm": "WebM/Matroska",
    "mov,mp4,m4a,3gp,3g2,mj2": "MP4/M4A",
}


@dataclass(frozen=True)
class DecodedAudio:
    """Audio décodé, avant toute conversion."""

    samples: np.ndarray  # (trames, canaux), float32
    sample_rate: int
    format: str  # conteneur : « WAV », « FLAC », « WebM/Matroska »…
    codec: str  # « PCM_16 », « opus », « aac »…
    decoder: str  # « soundfile » ou « pyav »

    @property
    def n_frames(self) -> int:
        return self.samples.shape[0]

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sample_rate

    @property
    def description(self) -> str:
        """Format lisible, par exemple « WebM/Matroska (opus) »."""
        return f"{self.format} ({self.codec})" if self.codec else self.format


def decode_audio(source: bytes | str | Path, *, max_duration_s: float | None = None) -> DecodedAudio:
    """Décode un fichier (chemin) ou son contenu (octets) en float32, à sa fréquence d'origine.

    Lève EmptyAudioError, InvalidAudioError ou AudioTooLongError, avec un message pour l'utilisateur.
    """
    if isinstance(source, (str, Path)):
        if Path(source).stat().st_size == 0:
            raise EmptyAudioError("Le fichier est vide.")
    elif not source:
        raise EmptyAudioError("Le fichier est vide.")

    limit_s = None if max_duration_s is None else max_duration_s + DURATION_MARGIN_S
    decoded = _decode_with_soundfile(source, limit_s, max_duration_s)
    if decoded is None:
        decoded = _decode_with_pyav(source, limit_s, max_duration_s)
    if decoded.n_frames == 0:
        raise EmptyAudioError("Le fichier ne contient aucun échantillon audio.")
    return decoded


def _open(source: bytes | str | Path):
    """Un nouvel objet à lire à chaque tentative : un décodeur qui échoue a pu avancer dans le flux."""
    return io.BytesIO(source) if isinstance(source, bytes) else str(source)


def _decode_with_soundfile(source, limit_s: float | None, max_duration_s: float | None) -> DecodedAudio | None:
    """Décode avec libsndfile, comme `load_audio`. Renvoie None si le format ne lui est pas connu."""
    try:
        with sf.SoundFile(_open(source)) as file:
            if file.samplerate <= 0 or file.channels <= 0:
                return None
            if limit_s is not None and file.frames > limit_s * file.samplerate:
                raise AudioTooLongError(file.frames / file.samplerate, max_duration_s)
            samples = file.read(dtype="float32", always_2d=True)
            return DecodedAudio(samples, file.samplerate, file.format, file.subtype, "soundfile")
    except (sf.SoundFileError, RuntimeError):
        # Format inconnu de libsndfile, ou données illisibles : PyAV tente sa chance.
        return None


def _declared_duration(container, stream) -> float | None:
    """Durée annoncée par le conteneur, si elle existe (absente des WebM de MediaRecorder)."""
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def _decode_with_pyav(source, limit_s: float | None, max_duration_s: float | None) -> DecodedAudio:
    """Décode avec FFmpeg (PyAV) la première piste audio, en float32 planaire."""
    try:
        with av.open(_open(source), mode="r") as container:
            if not container.streams.audio:
                raise InvalidAudioError("Le fichier ne contient aucune piste audio.")
            stream = container.streams.audio[0]
            codec = stream.codec_context
            rate = codec.sample_rate
            if not rate or codec.layout.nb_channels <= 0:
                raise InvalidAudioError(UNREADABLE)

            declared = _declared_duration(container, stream)
            if limit_s is not None and declared is not None and declared > limit_s:
                raise AudioTooLongError(declared, max_duration_s)

            # Seul le format des échantillons change (float32 planaire) : ni la fréquence ni les canaux.
            resampler = av.AudioResampler(format="fltp", layout=codec.layout, rate=rate)
            max_frames = None if limit_s is None else int(limit_s * rate)
            chunks, n_frames = [], 0
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunk = converted.to_ndarray()
                    chunks.append(chunk)
                    n_frames += chunk.shape[1]
                if max_frames is not None and n_frames > max_frames:
                    raise AudioTooLongError(n_frames / rate, max_duration_s, at_least=True)
            chunks += [converted.to_ndarray() for converted in resampler.resample(None)]
            container_name, codec_name = container.format.name, codec.name
    except AudioError:
        raise
    except (av.FFmpegError, ValueError, TypeError, IndexError, OSError) as error:
        raise InvalidAudioError(UNREADABLE) from error

    if not chunks:
        return DecodedAudio(np.zeros((0, 1), dtype=np.float32), rate, _label(container_name), codec_name, "pyav")
    samples = np.ascontiguousarray(np.concatenate(chunks, axis=1).T, dtype=np.float32)
    return DecodedAudio(samples, rate, _label(container_name), codec_name, "pyav")


def _label(container_name: str) -> str:
    return _FORMAT_LABELS.get(container_name, container_name.upper())
