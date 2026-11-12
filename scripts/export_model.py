"""Exporte un checkpoint de recherche en version de modèle pour l'application : models/<version>/.

Écrit dans models/<version>/ :
    best.pt       {"model": state_dict} et la provenance, sans l'état de l'optimiseur ni l'historique
                  (environ 102 Mo au lieu de 306 Mo). Le notebook peut le relire tel quel.
    cmvn.pt       copie à l'octet près des statistiques CMVN.
    config.yaml   hyperparamètres (valeurs du Config du notebook), tokenizer, décodage, limites,
                  provenance et empreintes SHA-256 des deux fichiers.

Les fichiers source ne sont que lus. Avant d'écrire, le script vérifie que la CMVN est identique à
celle du checkpoint, et que les hyperparamètres reconstruisent exactement le modèle sauvegardé
(chargement strict, banc de filtres Mel recalculé à l'identique). Après l'écriture, il recharge la
version exportée comme le fera l'application et compare chaque tenseur à la source. Une version
existante n'est jamais écrasée.

Exemple :
    python scripts/export_model.py \\
        --checkpoint "../Robust ASR/checkpoints/best.pt" \\
        --cmvn "../Robust ASR/data/manifests/cmvn.pt" \\
        --version v1
"""

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.config import ModelConfig  # noqa: E402
from src.asr.model import RobustASR  # noqa: E402
from src.asr.model_store import CONFIG_NAME, FORMAT_VERSION, load_weights, read_spec, sha256_file  # noqa: E402
from src.asr.tokenizer import CharTokenizer  # noqa: E402

logger = logging.getLogger("export_model")

CONFIG_HEADER = """\
# Version de modèle de l'application Robust-ASR, écrite par scripts/export_model.py.
#
# model      : hyperparamètres du Config du notebook (mêmes noms). Ils ne figurent pas dans le
#              checkpoint : le chargement strict des poids vérifie qu'ils lui correspondent.
# decoding   : beam search joint CTC / attention du notebook (beam 10, poids CTC 0.3), ou
#              ctc_greedy, plus rapide sur CPU.
# limits     : durées acceptées. Au-delà de 30 s, la transcription se dégrade fortement (le modèle
#              n'a pas vu d'énoncé plus long à l'entraînement) ; au-delà d'environ 240 s, elle échoue.
# sha256     : empreintes vérifiées à chaque chargement.
"""


class ExportError(Exception):
    """L'export est impossible : le message explique pourquoi."""


def read_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ExportError(f"{path} ne contient pas de clé « model » : est-ce un checkpoint du notebook ?")
    provenance = {
        key: checkpoint[key] for key in ("epoch", "best_wer", "val_set", "val_set_sha256") if key in checkpoint
    }
    return checkpoint["model"], provenance


def check_source(state_dict: dict[str, torch.Tensor], cmvn: dict, config: ModelConfig) -> None:
    """Refuse une CMVN étrangère ou des hyperparamètres qui ne reconstruisent pas ce modèle."""
    if not isinstance(cmvn, dict) or not {"mean", "std"} <= set(cmvn):
        raise ExportError("le fichier CMVN doit contenir les tenseurs mean et std")
    for key in ("mean", "std"):
        if not torch.equal(cmvn[key], state_dict[f"logmel.{key}"]):
            raise ExportError(f"la CMVN ({key}) diffère de celle du checkpoint : statistiques d'un autre modèle ?")

    model = RobustASR(config, CharTokenizer())
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ExportError(f"les hyperparamètres ne correspondent pas au checkpoint : {error}") from error
    # Le banc de filtres et la fenêtre dépendent de sample_rate, n_fft, win_length et n_mels.
    for name in ("logmel.melspec.spectrogram.window", "logmel.melspec.mel_scale.fb"):
        if not torch.equal(expected[name], state_dict[name]):
            raise ExportError(f"{name} du checkpoint ne correspond pas à la configuration des features")


def build_config(version: str, description: str, config: ModelConfig, provenance: dict) -> dict:
    return {
        "format": FORMAT_VERSION,
        "version": version,
        "description": description,
        "language": "en",
        "files": {"checkpoint": "best.pt", "cmvn": "cmvn.pt"},
        "sha256": {},
        "model": config.to_dict(),
        "tokenizer": {"type": "char", "chars": CharTokenizer().chars},
        "decoding": {
            "method": "joint_beam_search",
            "beam_size": config.beam_size,
            "ctc_weight": config.ctc_weight,
            "use_corrector": False,
        },
        "limits": {"min_duration_s": 0.1, "max_duration_s": 30.0},
        "provenance": provenance,
    }


def write_config(path: Path, config: dict) -> None:
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
    path.write_text(CONFIG_HEADER + "\n" + text, encoding="utf-8")


def verify_export(directory: Path, state_dict: dict[str, torch.Tensor]) -> None:
    """Recharge la version comme l'application et compare chaque tenseur à la source."""
    spec = read_spec(directory)
    weights = load_weights(spec)
    if weights.state_dict.keys() != state_dict.keys():
        raise ExportError("les tenseurs exportés ne correspondent pas à ceux du checkpoint")
    for name, tensor in state_dict.items():
        exported = weights.state_dict[name]
        if exported.dtype != tensor.dtype or not torch.equal(exported, tensor):
            raise ExportError(f"{name} diffère après l'export")
    model = RobustASR(spec.model, CharTokenizer())
    model.load_state_dict(weights.state_dict, strict=True)


def export(checkpoint: Path, cmvn_path: Path, version: str, models_dir: Path, description: str) -> Path:
    target = models_dir / version
    if target.exists():
        raise ExportError(f"{target} existe déjà : une version publiée ne s'écrase pas, choisissez un autre nom")
    for path in (checkpoint, cmvn_path):
        if not path.is_file():
            raise ExportError(f"{path} introuvable")

    logger.info("Lecture de %s", checkpoint)
    state_dict, provenance = read_checkpoint(checkpoint)
    cmvn = torch.load(cmvn_path, map_location="cpu", weights_only=True)
    config = ModelConfig()
    check_source(state_dict, cmvn, config)

    provenance = {
        "source_checkpoint": checkpoint.name,
        "source_sha256": sha256_file(checkpoint),
        "source_cmvn_sha256": sha256_file(cmvn_path),
        **provenance,
        "exported_on": dt.date.today().isoformat(),
    }

    # Écriture dans un dossier temporaire voisin, renommé à la fin : jamais de version à moitié écrite.
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_dir, prefix=f".{version}-") as temporary:
        staging = Path(temporary) / version
        staging.mkdir()
        torch.save({"model": state_dict, "format": FORMAT_VERSION, **provenance}, staging / "best.pt")
        shutil.copyfile(cmvn_path, staging / "cmvn.pt")
        content = build_config(version, description, config, provenance)
        content["sha256"] = {"checkpoint": sha256_file(staging / "best.pt"), "cmvn": sha256_file(staging / "cmvn.pt")}
        write_config(staging / CONFIG_NAME, content)
        verify_export(staging, state_dict)
        staging.rename(target)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint du notebook (best.pt)")
    parser.add_argument("--cmvn", type=Path, required=True, help="statistiques CMVN (data/manifests/cmvn.pt)")
    parser.add_argument("--version", required=True, help="nom de la version, par exemple v1")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models", help="dossier des versions")
    parser.add_argument(
        "--description",
        default="Transformer hybride CTC / attention avec QK-norm, LibriSpeech 960 h, 60 époques",
        help="description affichée par l'application",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        target = export(args.checkpoint, args.cmvn, args.version, args.models_dir, args.description)
    except ExportError as error:
        logger.error("Export impossible : %s", error)
        return 1
    spec = read_spec(target)
    size_mb = (target / "best.pt").stat().st_size / 1e6
    logger.info("Version %s écrite dans %s (best.pt : %.0f Mo), vérifiée tenseur par tenseur.", spec.version, target, size_mb)
    logger.info("Provenance : %s", json.dumps(spec.provenance, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
