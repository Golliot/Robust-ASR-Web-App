"""Hyperparamètres d'un modèle RobustASR, repris du `Config` du notebook.

Le `Config` du notebook mêle les données, l'entraînement et le modèle. `ModelConfig` n'en garde que les
champs que lisent les modules du modèle et le beam search, sous les mêmes noms et avec les mêmes
valeurs par défaut : `RobustASR(ModelConfig(), CharTokenizer())` construit exactement le modèle du
notebook. Pour une version de modèle, toutes les valeurs viennent de la section `model` de son
config.yaml, sans valeur implicite.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    # Log-Mel : 80 bandes, fenêtre de 25 ms, pas de 10 ms
    sample_rate: int = 16_000
    n_mels: int = 80
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160

    # SpecAugment : inactif en évaluation, mais lu par le constructeur du modèle
    freq_masks: int = 2
    freq_mask_width: int = 27
    time_masks: int = 2
    time_mask_width: int = 40

    # Modèle
    d_model: int = 256
    n_heads: int = 4
    d_ffn: int = 2048
    encoder_layers: int = 12
    decoder_layers: int = 6
    conv_channels: int = 32
    dropout: float = 0.1
    qk_norm: bool = True

    # Loss hybride ; ctc_weight est aussi le poids CTC par défaut du beam search
    ctc_weight: float = 0.3
    label_smoothing: float = 0.1

    # Beam search : largeur par défaut
    beam_size: int = 10

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Construit la configuration depuis un dictionnaire complet : un champ absent ou inconnu est refusé."""
        names = [field.name for field in fields(cls)]
        missing = [name for name in names if name not in values]
        unknown = sorted(set(values) - set(names))
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f"champs absents : {', '.join(missing)}")
            if unknown:
                problems.append(f"champs inconnus : {', '.join(unknown)}")
            raise ValueError("section model incomplète ou invalide (" + " ; ".join(problems) + ")")

        checked = {}
        for field in fields(cls):
            value = values[field.name]
            expected = field.type
            # bool est un sous-type d'int en Python : il faut l'exclure explicitement des entiers.
            if expected is bool:
                valid = isinstance(value, bool)
            elif expected is int:
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                value = float(value) if valid else value
            if not valid:
                raise ValueError(f"model.{field.name} doit être de type {expected.__name__} (reçu : {value!r})")
            checked[field.name] = value
        return cls(**checked)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
