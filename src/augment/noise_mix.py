"""Tirage du bruit d'entraînement en deux étages : la catégorie, puis le fichier.

Étage 1 : la catégorie suit des probabilités fixées et porte son propre SNR, car la musique
et le babble, proches de la parole, gênent plus qu'un bruit stationnaire au même SNR.

Étage 2 : le fichier est pondéré par une puissance de sa durée. L'exposant 1 donne à chaque
seconde la même chance, mais concentre les tirages sur quelques longs fichiers ; 0.5 (racine
carrée) préserve la diversité, et 0 revient au tirage uniforme par fichier.

La validation ne passe pas par ce module : elle est figée, voir src/augment/val_set.py.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

from src.augment.babble import BabbleBank

BABBLE = "babble"


@dataclass(frozen=True)
class NoiseCategory:
    """Une catégorie de bruit : `babble`, ou une catégorie des manifestes MUSAN."""

    name: str
    probability: float
    snr_db: tuple[float, float]


def draw_index(weights: torch.Tensor, generator: torch.Generator | None = None) -> int:
    return int(torch.multinomial(weights, 1, generator=generator))


def duration_weights(durations: Sequence[float], exponent: float) -> torch.Tensor:
    return torch.tensor(durations, dtype=torch.float64) ** exponent


def read_segment(path: str | Path, start: int, n_samples: int) -> torch.Tensor:
    """Lit n_samples du premier canal depuis start, en bouclant si le fichier est trop court."""
    with sf.SoundFile(path) as file:
        n_read = min(n_samples, file.frames)
        if start + n_read > file.frames:
            raise ValueError(f"{path} : segment [{start}, {start + n_read}[ hors du fichier ({file.frames} échantillons)")
        file.seek(start)
        noise = file.read(n_read, dtype="float32", always_2d=True)[:, 0].copy()
    noise = torch.from_numpy(noise)
    if n_read < n_samples:
        noise = noise.repeat(n_samples // n_read + 1)[:n_samples]
    return noise


class NoiseSampler:
    """Tire un segment de bruit et son SNR pour un énoncé d'entraînement."""

    def __init__(
        self,
        categories: Sequence[NoiseCategory],
        musan_entries: Iterable[dict],
        babble: BabbleBank,
        read_noise: Callable[..., torch.Tensor],
        duration_exponent: float = 1.0,
    ) -> None:
        if not math.isclose(sum(category.probability for category in categories), 1.0):
            raise ValueError("Les probabilités des catégories doivent sommer à 1")
        if any(category.snr_db[0] > category.snr_db[1] for category in categories):
            raise ValueError("Chaque plage de SNR doit aller du minimum au maximum")
        if duration_exponent < 0:
            raise ValueError("L'exposant de durée doit être positif ou nul")

        self.categories = list(categories)
        self.probabilities = torch.tensor([category.probability for category in categories], dtype=torch.float64)
        self.babble = babble
        self.read_noise = read_noise

        entries = list(musan_entries)
        self.files: dict[str, list[str]] = {}
        self.weights: dict[str, torch.Tensor] = {}
        self.hours: dict[str, float] = {}
        for category in self.categories:
            if category.name == BABBLE:
                continue
            selected = [entry for entry in entries if entry["category"] == category.name]
            if not selected:
                raise ValueError(f"Aucun fichier MUSAN pour la catégorie {category.name!r}")
            durations = [entry["duration"] for entry in selected]
            self.files[category.name] = [entry["path"] for entry in selected]
            self.weights[category.name] = duration_weights(durations, duration_exponent)
            self.hours[category.name] = sum(durations) / 3600

    def sample(
        self,
        n_samples: int,
        speaker: int | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Renvoie (bruit de n_samples échantillons, SNR en dB)."""
        category = self.categories[draw_index(self.probabilities, generator)]
        low, high = category.snr_db
        snr_db = low + (high - low) * torch.rand(1, generator=generator).item()

        if category.name == BABBLE:
            return self.babble.build(n_samples, speaker, generator), snr_db
        path = self.files[category.name][draw_index(self.weights[category.name], generator)]
        return self.read_noise(path, n_samples, generator), snr_db

    def summary(self) -> str:
        """Décrit chaque catégorie en une ligne."""
        lines = []
        for category in self.categories:
            if category.name == BABBLE:
                content = f"{len(self.babble.speakers)} locuteurs"
            else:
                content = f"{len(self.files[category.name])} fichiers, {self.hours[category.name]:.1f} h"
            low, high = category.snr_db
            lines.append(f"{category.name:<7} p = {category.probability:.2f}  SNR ~ U({low:g}, {high:g}) dB  {content}")
        return "\n".join(lines)
