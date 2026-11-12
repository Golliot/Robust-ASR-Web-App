"""Jeux de validation figés : relecture, conditions et résumé de la grille.

Un jeu est généré une fois par build_val_set.py, avec sa propre configuration, puis seulement
relu : il n'y a aucun lien avec la configuration d'entraînement. Une ablation sur l'exposant,
p_rir, les probabilités ou la graine ne change donc pas les données de validation.

Un même fichier sert deux vues :
    mixed   chaque énoncé reçoit une seule condition, selon les proportions d'entraînement,
            RIR comprise. Le WER global de cette vue sélectionne le modèle.
    grid    chaque énoncé passe dans toutes les conditions, sans RIR sauf `rir_sim`. Diagnostic.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import torch

from src.augment.babble import cut, normalize_rms
from src.augment.noise_mix import BABBLE, draw_index, read_segment

CLEAN = "clean"  # grille : parole vraiment propre
NO_NOISE = "no_noise"  # vue mélangée : sans bruit, mais réverbérée pour une partie des énoncés
RIR_ONLY = "rir_sim"


@dataclass(frozen=True)
class ValCategory:
    """Catégorie de bruit figée. training_snr_db sert seulement à contrôler la grille."""

    name: str
    probability: float
    snrs_db: tuple[float, ...]
    training_snr_db: tuple[float, float]


@dataclass(frozen=True)
class ValSetSpec:
    """Configuration figée d'un jeu de validation, écrite avec lui."""

    version: str
    seed: int
    subsets: tuple[str, ...]
    p_noise: float
    p_rir: float
    duration_exponent: float
    rir_room_fraction: float
    categories: tuple[ValCategory, ...]

    def __post_init__(self) -> None:
        if not all(subset.startswith("dev-") for subset in self.subsets):
            raise ValueError(f"La validation n'utilise que les sous-ensembles dev : {self.subsets}")
        if not math.isclose(sum(category.probability for category in self.categories), 1.0):
            raise ValueError("Les probabilités des catégories doivent sommer à 1")
        for category in self.categories:
            low, high = category.training_snr_db
            if not category.snrs_db or not all(low <= snr_db <= high for snr_db in category.snrs_db):
                raise ValueError(f"SNR de validation de {category.name!r} hors de la plage d'entraînement")


@dataclass(frozen=True)
class Degradation:
    """Dégradation exacte d'un énoncé : chemins résolus et offsets déjà tirés."""

    name: str
    rir: str | None = None
    category: str | None = None
    sources: tuple[tuple[str, int], ...] = ()
    snr_db: float | None = None


def condition_name(category: str, snr_db: float) -> str:
    return f"{category}_{snr_db:g}db"


def draw_validation_noise(
    categories: Sequence[ValCategory],
    p_noise: float,
    generator: torch.Generator | None = None,
) -> tuple[ValCategory, float] | None:
    """Tire la condition d'un énoncé : None s'il reste sans bruit, sinon (catégorie, SNR)."""
    if torch.rand(1, generator=generator).item() >= p_noise:
        return None
    probabilities = torch.tensor([category.probability for category in categories], dtype=torch.float64)
    category = categories[draw_index(probabilities, generator)]
    return category, category.snrs_db[int(torch.randint(len(category.snrs_db), (1,), generator=generator))]


def render_noise(degradation: Degradation, n_samples: int) -> torch.Tensor:
    """Reconstruit le bruit figé d'une dégradation, sans aucun tirage."""
    if degradation.category != BABBLE:
        path, start = degradation.sources[0]
        return read_segment(path, start, n_samples)
    babble = torch.zeros(n_samples)
    for path, start in degradation.sources:
        audio, _ = sf.read(path, dtype="float32")
        babble += cut(normalize_rms(torch.from_numpy(audio)), start, n_samples)
    return babble


def category_wers(
    condition_wers: Mapping[str, float],
    categories: Sequence[ValCategory],
    base: str = CLEAN,
) -> dict[str, float]:
    """Réduit les WER par condition : la base, puis chaque catégorie en moyenne sur ses SNR."""
    if base not in condition_wers:
        raise ValueError(f"Condition {base!r} absente")
    wers = {base: condition_wers[base]}
    for category in categories:
        names = [condition_name(category.name, snr_db) for snr_db in category.snrs_db]
        values = [condition_wers[name] for name in names if name in condition_wers]
        if not values:
            raise ValueError(f"Aucune condition pour la catégorie {category.name!r}")
        wers[category.name] = sum(values) / len(values)
    return wers


def grid_summary_wer(
    condition_wers: Mapping[str, float],
    categories: Sequence[ValCategory],
    p_noise: float,
) -> float:
    """Résume la grille : (1 - p_noise) WER_clean + p_noise somme(p_c WER_c). Ne sélectionne rien."""
    wers = category_wers(condition_wers, categories)
    noisy = sum(category.probability * wers[category.name] for category in categories)
    return (1 - p_noise) * wers[CLEAN] + p_noise * noisy


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_from_config(config: Mapping) -> ValSetSpec:
    categories = tuple(
        ValCategory(
            name=category["name"],
            probability=category["probability"],
            snrs_db=tuple(category["snrs_db"]),
            training_snr_db=tuple(category["training_snr_db"]),
        )
        for category in config["spec"]["categories"]
    )
    return ValSetSpec(**{**config["spec"], "subsets": tuple(config["spec"]["subsets"]), "categories": categories})


@dataclass
class ValSet:
    """Un jeu figé relu depuis le disque. Ne contient aucun lien avec la configuration du train."""

    spec: ValSetSpec
    sha256: str
    rir_rooms: list[str]
    entries: dict[str, dict]
    data_dir: Path
    musan_root: Path
    grid_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.grid_names = [
            CLEAN,
            RIR_ONLY,
            *[condition_name(category.name, snr) for category in self.spec.categories for snr in category.snrs_db],
        ]

    def entry(self, utterance_id: str) -> dict:
        if utterance_id not in self.entries:
            raise KeyError(f"{utterance_id} absent de {self.spec.version}")
        return self.entries[utterance_id]

    def _rir(self, entry: dict) -> str:
        return str(self.data_dir / entry["rir"])

    def _sources(self, entry: dict, category: str) -> tuple[tuple[str, int], ...]:
        root = self.data_dir / "LibriSpeech" if category == BABBLE else self.musan_root
        return tuple((str(root / source["path"]), source["start"]) for source in entry["sources"][category])

    def mixed(self, items: Iterable[Mapping]) -> list[Degradation]:
        """Vue mélangée : une condition par énoncé, RIR comprise."""
        degradations = []
        for item in items:
            entry = self.entry(item["id"])
            rir = self._rir(entry) if entry["reverberant"] else None
            if entry["category"] is None:
                degradations.append(Degradation(NO_NOISE, rir=rir))
            else:
                degradations.append(
                    Degradation(
                        condition_name(entry["category"], entry["snr_db"]),
                        rir=rir,
                        category=entry["category"],
                        sources=self._sources(entry, entry["category"]),
                        snr_db=entry["snr_db"],
                    )
                )
        return degradations

    def grid(self, items: Iterable[Mapping], name: str) -> list[Degradation]:
        """Vue grille : la même condition pour tous, sans RIR sauf `rir_sim`."""
        if name not in self.grid_names:
            raise ValueError(f"Condition inconnue : {name!r}")
        if name == CLEAN:
            return [Degradation(CLEAN) for _ in items]
        if name == RIR_ONLY:
            return [Degradation(RIR_ONLY, rir=self._rir(self.entry(item["id"]))) for item in items]

        category, snr_db = next(
            (category.name, snr_db)
            for category in self.spec.categories
            for snr_db in category.snrs_db
            if condition_name(category.name, snr_db) == name
        )
        return [
            Degradation(name, category=category, sources=self._sources(self.entry(item["id"]), category), snr_db=snr_db)
            for item in items
        ]


def load_val_set(version: str, data_dir: Path | str, val_dir: Path | str | None = None) -> ValSet:
    """Relit un jeu figé et refuse tout changement de ce jeu ou du partage MUSAN dont il dépend."""
    data_dir = Path(data_dir)
    directory = Path(val_dir) if val_dir is not None else data_dir / "manifests" / "val"
    manifest, config_path = directory / f"{version}.jsonl", directory / f"{version}_config.json"
    if not (manifest.exists() and config_path.exists()):
        raise FileNotFoundError(f"{version} absent de {directory} : lancez `python build_val_set.py`")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    sha256 = file_sha256(manifest)
    if sha256 != config["sha256"]:
        raise ValueError(f"{manifest} a changé depuis sa génération : un jeu figé ne se modifie pas")

    musan_dir = data_dir / "manifests" / "musan"
    musan_val = musan_dir / "musan_val.jsonl"
    # Un autre partage peut placer dans le train des fichiers de bruit figés dans ce jeu.
    if not musan_val.exists() or file_sha256(musan_val) != config["musan_val_sha256"]:
        raise ValueError(
            f"{musan_val} n'est plus le partage MUSAN de {version} : ses bruits pourraient être dans le train. "
            "Restaurez ce partage, ou générez une nouvelle version du jeu."
        )

    entries = {entry["id"]: entry for entry in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())}
    musan_config = json.loads((musan_dir / "musan_config.json").read_text(encoding="utf-8"))
    return ValSet(
        spec=spec_from_config(config),
        sha256=sha256,
        rir_rooms=config["rir_rooms"],
        entries=entries,
        data_dir=data_dir,
        musan_root=Path(musan_config["musan_root"]),
    )
