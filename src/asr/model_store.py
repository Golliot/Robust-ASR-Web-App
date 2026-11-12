"""Versions de modèle : lecture de models/<version>/config.yaml et chargement vérifié des poids.

Une version est un dossier :

    models/v1/
        config.yaml   hyperparamètres, tokenizer, décodage, limites, provenance, empreintes SHA-256
        best.pt       {"model": state_dict, …} : export allégé, ou checkpoint d'entraînement complet
        cmvn.pt       {"mean", "std"} : doit être identique à la CMVN contenue dans le checkpoint

Le checkpoint fait foi : sa CMVN est celle qui a servi à l'entraînement. Un cmvn.pt différent révèle
un mélange de fichiers, et le chargement est refusé. Les hyperparamètres ne figurent pas dans le
checkpoint : config.yaml les fournit tous, sans valeur implicite, et le chargement strict des poids
vérifie qu'ils correspondent au modèle sauvegardé.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from src.asr.config import ModelConfig
from src.asr.tokenizer import CharTokenizer

CONFIG_NAME = "config.yaml"
FORMAT_VERSION = 1
DECODING_METHODS = ("joint_beam_search", "ctc_greedy")
TOP_LEVEL_KEYS = {"format", "version", "description", "language", "files", "sha256", "model", "tokenizer",
                  "decoding", "limits", "provenance"}  # fmt: skip
REQUIRED_KEYS = ("format", "version", "files", "model", "tokenizer", "decoding", "limits")
# Positions de l'encodage positionnel du notebook (PositionalEncoding, max_length=6000).
POSITIONAL_MAX_LENGTH = 6000


class ModelConfigError(ValueError):
    """config.yaml absent, invalide ou incohérent."""


class ModelIntegrityError(RuntimeError):
    """Fichier de modèle manquant, empreinte différente ou CMVN incohérente."""


@dataclass(frozen=True)
class DecodingConfig:
    method: str = "joint_beam_search"
    beam_size: int = 10
    ctc_weight: float = 0.3
    use_corrector: bool = False


@dataclass(frozen=True)
class Limits:
    min_duration_s: float = 0.1
    max_duration_s: float = 30.0


@dataclass(frozen=True)
class ModelSpec:
    """Une version de modèle décrite par son config.yaml. Aucun tenseur n'est chargé ici."""

    version: str
    directory: Path
    description: str
    language: str
    model: ModelConfig
    chars: str
    decoding: DecodingConfig
    limits: Limits
    files: dict[str, str]
    sha256: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / self.files["checkpoint"]

    @property
    def cmvn_path(self) -> Path:
        return self.directory / self.files["cmvn"]

    @property
    def corrector_transcripts_path(self) -> Path | None:
        name = self.files.get("corrector_transcripts")
        return None if name is None else self.directory / name

    def missing_files(self) -> list[str]:
        paths = [self.checkpoint_path, self.cmvn_path]
        if self.decoding.use_corrector and self.corrector_transcripts_path is not None:
            paths.append(self.corrector_transcripts_path)
        return [path.name for path in paths if not path.is_file()]

    def summary(self) -> dict[str, Any]:
        """Description publique de la version, sans chemin absolu."""
        return {
            "version": self.version,
            "description": self.description,
            "language": self.language,
            "decoding": vars(self.decoding).copy(),
            "limits": vars(self.limits).copy(),
            "sample_rate": self.model.sample_rate,
            "provenance": dict(self.provenance),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def positional_limit_s(config: ModelConfig) -> float:
    """Durée au-delà de laquelle l'encodage positionnel du modèle déborde (environ 240 s).

    Le beam search ajoute jusqu'à n_frames + 2 positions au décodeur : la limite porte sur n_frames + 2.
    """
    max_encoder_frames = POSITIONAL_MAX_LENGTH - 2
    # Le frontend divise le nombre de trames de features par 4 ; il y en a n_samples // hop + 1.
    max_feature_frames = 4 * max_encoder_frames - 4
    return (max_feature_frames - 1) * config.hop_length / config.sample_rate


def _section(raw: Mapping, name: str) -> Mapping:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ModelConfigError(f"section {name} absente ou invalide")
    return value


def _number(section: Mapping, key: str, kind: type, where: str):
    value = section.get(key)
    valid = isinstance(value, kind) and not isinstance(value, bool) if kind is int else (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if not valid:
        raise ModelConfigError(f"{where}.{key} doit être un nombre{' entier' if kind is int else ''} (reçu : {value!r})")
    return kind(value)


def parse_spec(raw: Any, directory: Path) -> ModelSpec:
    """Valide le contenu d'un config.yaml et le traduit en ModelSpec."""
    if not isinstance(raw, Mapping):
        raise ModelConfigError("le fichier doit contenir un dictionnaire YAML")
    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if unknown or missing:
        problems = ([f"sections absentes : {', '.join(missing)}"] if missing else []) + (
            [f"sections inconnues : {', '.join(unknown)}"] if unknown else []
        )
        raise ModelConfigError(" ; ".join(problems))
    if raw["format"] != FORMAT_VERSION:
        raise ModelConfigError(f"format {raw['format']!r} non pris en charge (attendu : {FORMAT_VERSION})")

    version = raw["version"]
    if not isinstance(version, str) or version != directory.name:
        raise ModelConfigError(f"version {version!r} différente du nom du dossier {directory.name!r}")

    files = _section(raw, "files")
    for key in ("checkpoint", "cmvn"):
        if not isinstance(files.get(key), str) or not files[key]:
            raise ModelConfigError(f"files.{key} doit nommer un fichier du dossier")
    for key, name in files.items():
        if key not in {"checkpoint", "cmvn", "corrector_transcripts"}:
            raise ModelConfigError(f"files.{key} inconnu")
        if name is not None and (not isinstance(name, str) or Path(name).name != name):
            raise ModelConfigError(f"files.{key} doit être un nom de fichier sans chemin (reçu : {name!r})")

    try:
        model = ModelConfig.from_dict(_section(raw, "model"))
    except ValueError as error:
        raise ModelConfigError(str(error)) from error

    chars = _section(raw, "tokenizer").get("chars")
    if chars != CharTokenizer().chars:
        raise ModelConfigError("tokenizer.chars différent du tokenizer caractère du notebook : non pris en charge")

    raw_decoding = _section(raw, "decoding")
    decoding = DecodingConfig(
        method=raw_decoding.get("method"),
        beam_size=_number(raw_decoding, "beam_size", int, "decoding"),
        ctc_weight=_number(raw_decoding, "ctc_weight", float, "decoding"),
        use_corrector=raw_decoding.get("use_corrector", False),
    )
    if decoding.method not in DECODING_METHODS:
        raise ModelConfigError(f"decoding.method doit valoir {' ou '.join(DECODING_METHODS)}")
    if decoding.beam_size < 1:
        raise ModelConfigError("decoding.beam_size doit être au moins 1")
    if decoding.method == "joint_beam_search" and not 0 < decoding.ctc_weight <= 1:
        # Sans terme CTC, le beam search favorise les hypothèses tronquées (section 9 du notebook).
        raise ModelConfigError("decoding.ctc_weight doit être dans ]0, 1] pour le beam search joint")
    if not isinstance(decoding.use_corrector, bool):
        raise ModelConfigError("decoding.use_corrector doit être un booléen")
    if decoding.use_corrector and not files.get("corrector_transcripts"):
        raise ModelConfigError("decoding.use_corrector exige files.corrector_transcripts")

    raw_limits = _section(raw, "limits")
    limits = Limits(
        min_duration_s=_number(raw_limits, "min_duration_s", float, "limits"),
        max_duration_s=_number(raw_limits, "max_duration_s", float, "limits"),
    )
    # Le spectrogramme centré exige plus de n_fft / 2 échantillons (padding « reflect »).
    if limits.min_duration_s * model.sample_rate <= model.n_fft // 2:
        raise ModelConfigError(f"limits.min_duration_s doit dépasser {model.n_fft // 2} échantillons")
    if not limits.min_duration_s < limits.max_duration_s < positional_limit_s(model):
        raise ModelConfigError(
            f"limits.max_duration_s doit être entre min_duration_s et {positional_limit_s(model):.0f} s "
            "(limite de l'encodage positionnel)"
        )

    sha256 = raw.get("sha256") or {}
    if not isinstance(sha256, Mapping) or not all(isinstance(value, str) for value in sha256.values()):
        raise ModelConfigError("sha256 doit associer un nom de fichier (checkpoint, cmvn) à une empreinte")
    provenance = raw.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise ModelConfigError("provenance doit être un dictionnaire")

    return ModelSpec(
        version=version,
        directory=directory,
        description=str(raw.get("description", "")),
        language=str(raw.get("language", "")),
        model=model,
        chars=chars,
        decoding=decoding,
        limits=limits,
        files={key: value for key, value in files.items() if value is not None},
        sha256=dict(sha256),
        provenance=dict(provenance),
    )


def read_spec(directory: Path | str) -> ModelSpec:
    """Lit et valide models/<version>/config.yaml."""
    directory = Path(directory)
    path = directory / CONFIG_NAME
    if not path.is_file():
        raise ModelConfigError(f"{CONFIG_NAME} introuvable dans {directory}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ModelConfigError(f"{CONFIG_NAME} n'est pas un YAML valide : {error}") from error
    try:
        return parse_spec(raw, directory)
    except ModelConfigError as error:
        raise ModelConfigError(f"{path} : {error}") from error


@dataclass(frozen=True)
class VersionEntry:
    """Une version trouvée dans models/ : sa description, ou l'erreur qui l'empêche d'être chargée."""

    version: str
    spec: ModelSpec | None
    error: str | None


def list_versions(models_dir: Path | str) -> list[VersionEntry]:
    """Toutes les versions de models/, triées par nom, y compris celles qui sont invalides."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    entries = []
    for directory in sorted(path for path in models_dir.iterdir() if (path / CONFIG_NAME).is_file()):
        try:
            spec = read_spec(directory)
        except ModelConfigError as error:
            entries.append(VersionEntry(directory.name, None, str(error).removeprefix(f"{directory / CONFIG_NAME} : ")))
            continue
        missing = spec.missing_files()
        entries.append(VersionEntry(spec.version, spec, f"fichiers absents : {', '.join(missing)}" if missing else None))
    return entries


@dataclass(frozen=True)
class LoadedWeights:
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def load_weights(spec: ModelSpec, *, verify_checksums: bool = True) -> LoadedWeights:
    """Lit le checkpoint et la CMVN une seule fois, sur CPU, et vérifie leur cohérence."""
    missing = spec.missing_files()
    if missing:
        raise ModelIntegrityError(f"fichiers absents de {spec.directory} : {', '.join(missing)}")

    if verify_checksums:
        for key in ("checkpoint", "cmvn"):
            expected = spec.sha256.get(key)
            if expected is not None and sha256_file(spec.directory / spec.files[key]) != expected:
                raise ModelIntegrityError(
                    f"{spec.files[key]} ne correspond pas à l'empreinte de {CONFIG_NAME} : fichier modifié ou remplacé"
                )

    # weights_only : seuls des tenseurs et des types simples sont acceptés, jamais du code arbitraire.
    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("model"), Mapping):
        state_dict = dict(checkpoint["model"])
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model", "optimizer", "scheduler", "history"} and isinstance(value, (str, int, float))
        }
    elif isinstance(checkpoint, Mapping) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state_dict, metadata = dict(checkpoint), {}
    else:
        raise ModelIntegrityError(f"{spec.files['checkpoint']} ne contient pas de state_dict reconnaissable")

    cmvn = torch.load(spec.cmvn_path, map_location="cpu", weights_only=True)
    if not isinstance(cmvn, Mapping) or not {"mean", "std"} <= set(cmvn):
        raise ModelIntegrityError(f"{spec.files['cmvn']} doit contenir les tenseurs mean et std")
    for key in ("mean", "std"):
        stored = state_dict.get(f"logmel.{key}")
        if stored is None:
            # Checkpoint sans CMVN : celle du fichier dédié est la seule disponible.
            state_dict[f"logmel.{key}"] = cmvn[key].to(torch.float32)
        elif not torch.equal(stored, cmvn[key]):
            raise ModelIntegrityError(
                f"{spec.files['cmvn']} diffère de la CMVN du checkpoint ({key}) : statistiques d'un autre modèle ?"
            )
    return LoadedWeights(state_dict, metadata)
