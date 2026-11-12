"""Moteur de transcription : une version de modèle chargée une fois, utilisable sans FastAPI.

    from src.asr.engine import ASREngine

    engine = ASREngine.load("models/v1")            # GPU si utilisable, sinon CPU
    print(engine.transcribe_file("voix.webm").text)

Le chargement lit le checkpoint une seule fois, sur CPU, construit le modèle, puis le place sur le
périphérique retenu par src/runtime/device_manager.py et le valide par une transcription de chauffe.
Une transcription reproduit `transcribe` du notebook (cellule 31) : forme d'onde 16 kHz mono, beam
search joint CTC / attention, décodage des caractères, correction optionnelle.

Une seule inférence à la fois : un verrou protège le modèle, même si plusieurs threads l'appellent.
Le décodage et la conversion de l'audio, sans état partagé, se font hors du verrou.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.asr.decoding import ctc_greedy, joint_beam_search
from src.asr.model import RobustASR, autocast
from src.asr.model_store import ModelSpec, ModelConfigError, load_weights, read_spec
from src.asr.tokenizer import CharTokenizer
from src.audio.decoding import DecodedAudio, decode_audio
from src.audio.errors import InvalidAudioError
from src.audio.preprocessing import prepare_waveform
from src.runtime.device_manager import DeviceInfo, place_model, select_device

logger = logging.getLogger(__name__)

WARMUP_SECONDS = 1.0


@dataclass(frozen=True)
class Transcription:
    """Résultat d'une transcription. Les durées sont en secondes."""

    text: str
    audio_duration: float
    inference_time: float  # modèle et décodage seuls
    preprocessing_time: float  # décodage du fichier et conversion ; 0 pour un signal déjà prêt
    device: str  # « cuda » ou « cpu »
    model_version: str
    sample_rate: int
    input_format: str | None = None  # format du fichier reçu, par exemple « WebM/Matroska (opus) »

    @property
    def processing_time(self) -> float:
        return self.preprocessing_time + self.inference_time


class ASREngine:
    """Un modèle chargé sur un périphérique. À construire avec ASREngine.load()."""

    def __init__(
        self,
        spec: ModelSpec,
        model: RobustASR,
        device_info: DeviceInfo,
        *,
        max_duration_s: float | None = None,
        corrector=None,
        metadata: dict | None = None,
    ) -> None:
        self.spec = spec
        self.model = model
        self.device_info = device_info
        self.tokenizer = model.tok
        self.corrector = corrector
        self.metadata = metadata or {}
        self.min_duration_s = spec.limits.min_duration_s
        self.max_duration_s = spec.limits.max_duration_s
        if max_duration_s is not None:
            # Un réglage de l'application ne peut que réduire la limite fixée pour le modèle.
            if not self.min_duration_s < max_duration_s:
                raise ValueError(f"durée maximale {max_duration_s} s inférieure au minimum du modèle")
            self.max_duration_s = min(self.max_duration_s, max_duration_s)
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        model_dir: Path | str,
        device: str = "auto",
        *,
        max_duration_s: float | None = None,
        num_threads: int | None = None,
        verify_checksums: bool = True,
    ) -> "ASREngine":
        """Charge une version de modèle (dossier models/<version>) sur le périphérique demandé."""
        started = time.perf_counter()
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        # Les formes changent à chaque requête : benchmark rechercherait l'algorithme à chaque fois.
        torch.backends.cudnn.benchmark = False

        spec = read_spec(model_dir)
        weights = load_weights(spec, verify_checksums=verify_checksums)
        model = RobustASR(spec.model, CharTokenizer())
        model.load_state_dict(weights.state_dict, strict=True)
        model.eval()
        corrector = _load_corrector(spec)

        engine = cls(spec, model, select_device(device), max_duration_s=max_duration_s, corrector=corrector,
                     metadata=weights.metadata)  # fmt: skip
        engine.device_info = place_model(model, engine.device_info, engine._warmup)
        logger.info(
            "Modèle %s chargé sur %s (%s) en %.1f s",
            spec.version,
            engine.device_info.type,
            engine.device_info.name,
            time.perf_counter() - started,
        )
        return engine

    @property
    def device(self) -> torch.device:
        return self.device_info.torch_device

    @property
    def sample_rate(self) -> int:
        return self.spec.model.sample_rate

    @property
    def busy(self) -> bool:
        """Vrai pendant une inférence."""
        return self._lock.locked()

    def transcribe(self, waveform, sample_rate: int | None = None) -> Transcription:
        """Transcrit un signal : 1-D (mono) ou 2-D (trames, canaux), numpy ou torch, à sample_rate Hz.

        Par défaut, le signal est supposé au format du modèle (16 kHz). Il est converti en mono et
        rééchantillonné si besoin, comme un fichier.
        """
        samples = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2 or not np.issubdtype(samples.dtype, np.floating):
            raise InvalidAudioError("Le signal doit être un tableau de flottants, de forme (n,) ou (n, canaux).")
        audio = DecodedAudio(samples.astype(np.float32, copy=False), sample_rate or self.sample_rate, "PCM", "", "")
        return self._transcribe_decoded(audio, input_format=None, preprocessing_started=None)

    def transcribe_bytes(self, data: bytes) -> Transcription:
        """Transcrit le contenu d'un fichier audio (WAV, FLAC, OGG, MP3, WebM, MP4…)."""
        started = time.perf_counter()
        audio = decode_audio(data, max_duration_s=self.max_duration_s)
        return self._transcribe_decoded(audio, input_format=audio.description, preprocessing_started=started)

    def transcribe_file(self, path: Path | str) -> Transcription:
        """Transcrit un fichier audio."""
        started = time.perf_counter()
        audio = decode_audio(Path(path), max_duration_s=self.max_duration_s)
        return self._transcribe_decoded(audio, input_format=audio.description, preprocessing_started=started)

    def _transcribe_decoded(self, audio: DecodedAudio, *, input_format, preprocessing_started) -> Transcription:
        prepare_started = time.perf_counter()
        waveform = prepare_waveform(
            audio,
            min_duration_s=self.min_duration_s,
            max_duration_s=self.max_duration_s,
            target_sample_rate=self.sample_rate,
        )
        preprocessing_time = time.perf_counter() - (preprocessing_started or prepare_started)

        with self._lock:
            started = time.perf_counter()
            ids = self._decode(self.model, waveform.to(self.device)[None])
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            text = self.tokenizer.decode(ids)
            if self.corrector is not None:
                text = self.corrector(text)
            inference_time = time.perf_counter() - started

        return Transcription(
            text=text,
            audio_duration=waveform.size(0) / self.sample_rate,
            inference_time=inference_time,
            preprocessing_time=preprocessing_time,
            device=self.device.type,
            model_version=self.spec.version,
            sample_rate=self.sample_rate,
            input_format=input_format,
        )

    @torch.no_grad()
    def _decode(self, model: RobustASR, wav: torch.Tensor) -> list[int]:
        """Tokens de la meilleure hypothèse pour un énoncé (1, n), selon la méthode de config.yaml."""
        wav_lens = torch.tensor([wav.size(1)], device=wav.device)
        decoding = self.spec.decoding
        if decoding.method == "joint_beam_search":
            return joint_beam_search(model, wav, wav_lens, beam_size=decoding.beam_size, ctc_weight=decoding.ctc_weight)[0]
        feats, feat_lens = model.features(wav, wav_lens)
        with autocast(wav.device.type):
            memory, memory_lens = model.encode(feats, feat_lens)
            ctc_logits = model.ctc_head(memory)
        return ctc_greedy(ctc_logits.float().log_softmax(-1), memory_lens, model.tok.blank)[0]

    def _warmup(self, model: RobustASR, device: torch.device) -> None:
        """Transcription d'une seconde de silence : vérifie que le périphérique exécute tout le modèle."""
        wav = torch.zeros(1, int(WARMUP_SECONDS * self.sample_rate), device=device)
        self._decode(model, wav)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def describe(self) -> dict:
        """Version chargée, limites effectives et périphérique, pour l'API et les logs."""
        summary = self.spec.summary()
        summary["limits"] = {"min_duration_s": self.min_duration_s, "max_duration_s": self.max_duration_s}
        summary["checkpoint"] = {key: value for key, value in self.metadata.items()}
        return summary


def _load_corrector(spec: ModelSpec):
    """Correcteur du notebook, s'il est activé dans config.yaml. Il exige rapidfuzz."""
    if not spec.decoding.use_corrector:
        return None
    try:
        from src.asr.corrector import WordCorrector
    except ImportError as error:
        raise ModelConfigError("decoding.use_corrector exige le paquet rapidfuzz") from error
    lines = spec.corrector_transcripts_path.read_text(encoding="utf-8").splitlines()
    return WordCorrector(line for line in lines if line.strip())

