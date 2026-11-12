"""Bruit de babble : somme d'énoncés LibriSpeech d'autres locuteurs que le locuteur cible.

Utilisé à l'entraînement par src/augment/noise_mix.py. Les tirages suivent la convention du
notebook : generator=None à l'entraînement. En validation, build_val_set.py fige les tirages
une fois pour toutes dans un fichier, relu ensuite par src/augment/val_set.py.

Exemple :
    bank = BabbleBank("train")
    noise = bank.build(n_samples=wav.size(0), target_speaker=speaker)
    wav = mix_at_snr(wav, noise, snr_db)
"""

from pathlib import Path

import soundfile as sf
import torch

MIN_TALKERS = 3
MAX_TALKERS = 7
MIN_RMS = 1e-5  # évite de diviser par zéro sur un fichier silencieux

# Le test est absent volontairement : la parole de test ne doit jamais servir de bruit.
SPLIT_SUBSETS = {
    "train": ("train-clean-100", "train-clean-360", "train-other-500"),
    "val": ("dev-clean", "dev-other"),
}


def normalize_rms(audio: torch.Tensor) -> torch.Tensor:
    """Ramène audio à un RMS de 1."""
    return audio / audio.pow(2).mean().sqrt().clamp_min(MIN_RMS)


def draw_start(length: int, n_samples: int, generator: torch.Generator | None = None) -> int:
    """Tire le début d'un recadrage, ou la phase d'une boucle si l'énoncé est plus court."""
    # Phase aléatoire : les énoncés bouclés ne commencent pas tous au même instant.
    high = length - n_samples + 1 if length >= n_samples else length
    return int(torch.randint(high, (1,), generator=generator))


def cut(audio: torch.Tensor, start: int, n_samples: int) -> torch.Tensor:
    """Découpe n_samples depuis start, en bouclant audio s'il est plus court."""
    if audio.size(0) >= n_samples:
        return audio[start : start + n_samples]
    return audio.repeat(n_samples // audio.size(0) + 2)[start : start + n_samples]


def fit_length(audio: torch.Tensor, n_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Recadre audio à n_samples, ou le boucle s'il est plus court."""
    return cut(audio, draw_start(audio.size(0), n_samples, generator), n_samples)


class BabbleBank:
    """Énoncés LibriSpeech d'un split, regroupés par locuteur."""

    def __init__(self, split: str, librispeech_root: Path | str = Path("data/LibriSpeech")) -> None:
        if split not in SPLIT_SUBSETS:
            raise ValueError(f"Split de babble invalide : {split!r} (autorisés : {', '.join(SPLIT_SUBSETS)})")

        self.split = split
        self.utterances: dict[str, list[Path]] = {}
        root = Path(librispeech_root)
        for subset in SPLIT_SUBSETS[split]:
            if not (root / subset).is_dir():
                raise FileNotFoundError(f"{root / subset} introuvable : lancez `python download_librispeech.py`")
            # Structure LibriSpeech : <sous-ensemble>/<locuteur>/<chapitre>/<énoncé>.flac
            for path in sorted((root / subset).glob("*/*/*.flac")):
                self.utterances.setdefault(path.parent.parent.name, []).append(path)

        self.speakers = sorted(self.utterances)
        # Le locuteur cible peut figurer dans la banque : il en faut un de plus que MAX_TALKERS.
        if len(self.speakers) <= MAX_TALKERS:
            raise ValueError(f"{len(self.speakers)} locuteurs dans le split {split} : il en faut plus de {MAX_TALKERS}")

    def pick_utterances(self, target_speaker: int | str, generator: torch.Generator | None = None) -> list[Path]:
        """Tire N énoncés, N dans [3, 7], de N locuteurs distincts autres que le locuteur cible."""
        # torchaudio donne le locuteur en int, les dossiers en str.
        candidates = [speaker for speaker in self.speakers if speaker != str(target_speaker)]
        n_talkers = int(torch.randint(MIN_TALKERS, MAX_TALKERS + 1, (1,), generator=generator))
        chosen = torch.randperm(len(candidates), generator=generator)[:n_talkers].tolist()

        paths = []
        for index in chosen:
            utterances = self.utterances[candidates[index]]
            paths.append(utterances[int(torch.randint(len(utterances), (1,), generator=generator))])
        return paths

    def build(
        self,
        n_samples: int,
        target_speaker: int | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Renvoie n_samples de babble. Le niveau final dépend de N : mix_at_snr le rééchelonne."""
        babble = torch.zeros(n_samples)
        for path in self.pick_utterances(target_speaker, generator):
            audio, _ = sf.read(path, dtype="float32")
            babble += fit_length(normalize_rms(torch.from_numpy(audio)), n_samples, generator)
        return babble
