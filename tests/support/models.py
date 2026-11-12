"""Mini-modèles RobustASR aléatoires, écrits comme une vraie version de models/.

Même architecture et même format de fichiers que le modèle réel, mais de quelques milliers de
paramètres : les tests du chargement et du moteur tournent en une fraction de seconde, sans le
checkpoint de 102 Mo.
"""

from pathlib import Path

import torch
import yaml

from src.asr.config import ModelConfig
from src.asr.model import RobustASR
from src.asr.model_store import sha256_file
from src.asr.tokenizer import CharTokenizer

TINY_MODEL = {
    **ModelConfig().to_dict(),
    "n_mels": 16,
    "d_model": 16,
    "n_heads": 2,
    "d_ffn": 32,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "conv_channels": 4,
}


def tiny_state_dict(seed: int = 0, config: dict | None = None) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = RobustASR(ModelConfig.from_dict(config or TINY_MODEL), CharTokenizer())
    n_mels = model.cfg.n_mels
    model.logmel.mean.copy_(torch.linspace(3.0, 5.0, n_mels))
    model.logmel.std.copy_(torch.linspace(3.5, 5.0, n_mels))
    return model.state_dict()


def tiny_config(version: str = "v1", **overrides) -> dict:
    """Contenu de config.yaml pour un mini-modèle ; overrides remplace des sections entières."""
    config = {
        "format": 1,
        "version": version,
        "description": "mini-modèle de test",
        "language": "en",
        "files": {"checkpoint": "best.pt", "cmvn": "cmvn.pt"},
        "sha256": {},
        "model": dict(TINY_MODEL),
        "tokenizer": {"type": "char", "chars": CharTokenizer().chars},
        "decoding": {"method": "joint_beam_search", "beam_size": 2, "ctc_weight": 0.3, "use_corrector": False},
        "limits": {"min_duration_s": 0.1, "max_duration_s": 30.0},
        "provenance": {"epoch": 0},
    }
    config.update(overrides)
    return config


def write_model_dir(
    root: Path,
    version: str = "v1",
    *,
    seed: int = 0,
    checksums: bool = True,
    checkpoint: object | None = None,
    cmvn: object | None = None,
    **overrides,
) -> Path:
    """Écrit root/<version>/ : best.pt, cmvn.pt et config.yaml (avec empreintes si checksums)."""
    directory = root / version
    directory.mkdir(parents=True)
    state_dict = tiny_state_dict(seed)
    torch.save({"model": state_dict, "epoch": 3} if checkpoint is None else checkpoint, directory / "best.pt")
    default_cmvn = {"mean": state_dict["logmel.mean"], "std": state_dict["logmel.std"]}
    torch.save(default_cmvn if cmvn is None else cmvn, directory / "cmvn.pt")
    config = tiny_config(version, **overrides)
    if checksums:
        config["sha256"] = {"checkpoint": sha256_file(directory / "best.pt"), "cmvn": sha256_file(directory / "cmvn.pt")}
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return directory
