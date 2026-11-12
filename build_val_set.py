"""Génère un jeu de validation figé, une fois pour toutes.

Chaque énoncé dev reçoit sa dégradation exacte, écrite dans un fichier : RIR, fichiers de bruit,
offsets, locuteurs du babble, condition et SNR. Le notebook relit ce fichier et ne tire plus rien.
Sa configuration est ici, séparée de celle de l'entraînement : une ablation ne change donc pas
les données de validation, et les runs restent comparables.

Écrit dans --out-dir :
    <version>.jsonl         une ligne par énoncé
    <version>_config.json   configuration figée, salles RIR réservées, empreintes

Une version existante n'est jamais écrasée. Pour changer quoi que ce soit, ajoutez une nouvelle
entrée à VAL_SETS, puis réévaluez dessus les checkpoints à comparer.

Les salles RIR réservées sont retirées du train par le notebook : elles aussi sont figées ici,
pour qu'une autre graine d'entraînement ne fasse pas entrer une salle de validation dans le train.

Exemple :
    python build_val_set.py
"""

import argparse
import dataclasses
import json
import logging
import random
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf
import torch

from src.augment.babble import BabbleBank, draw_start
from src.augment.noise_mix import BABBLE, draw_index, duration_weights
from src.augment.val_set import ValCategory, ValSetSpec, draw_validation_noise, file_sha256

logger = logging.getLogger("build_val_set")

VAL_SETS = {
    "val_v2": ValSetSpec(
        version="val_v2",
        seed=1234,
        subsets=("dev-clean", "dev-other"),
        p_noise=0.6,
        p_rir=0.3,
        duration_exponent=0.5,
        rir_room_fraction=0.1,
        categories=(
            ValCategory("noise", 0.4, (0, 5, 10), training_snr_db=(0.0, 15.0)),
            ValCategory("music", 0.3, (5, 10, 15), training_snr_db=(5.0, 15.0)),
            ValCategory(BABBLE, 0.3, (13, 17, 20), training_snr_db=(13.0, 20.0)),
        ),
    ),
}


class BuildError(Exception):
    """Le jeu ne peut pas être généré : le message explique pourquoi."""


def make_generator(spec: ValSetSpec, purpose: str, utterance_id: str) -> torch.Generator:
    """Graine stable, propre à l'énoncé et au tirage : un tirage ne décale jamais les autres."""
    return torch.Generator().manual_seed(zlib.crc32(f"{spec.seed}/{purpose}/{utterance_id}".encode()))


def list_utterances(librispeech: Path, subsets: Sequence[str]) -> list[dict]:
    """Liste les énoncés, triés par identifiant, en ne lisant que leurs en-têtes."""
    utterances = []
    for subset in subsets:
        paths = sorted((librispeech / subset).glob("*/*/*.flac"))
        if not paths:
            raise BuildError(f"Aucun énoncé dans {librispeech / subset} : lancez `python download_librispeech.py`")
        utterances += [{"id": path.stem, "subset": subset, "frames": sf.info(path).frames} for path in paths]
    return sorted(utterances, key=lambda utterance: utterance["id"])


def select_rir_rooms(data_dir: Path, spec: ValSetSpec) -> tuple[list[str], list[str]]:
    """Réserve des salles simulées entières. Renvoie (salles, fichiers), relatifs à data_dir."""
    files = sorted(path.relative_to(data_dir).as_posix() for path in data_dir.glob("simulated_rirs_16k/*/*/*.wav"))
    if not files:
        raise BuildError(f"Aucune RIR simulée dans {data_dir} : lancez `python download_noise.py --datasets sim_rir`")
    rooms = sorted({str(Path(file).parent.as_posix()) for file in files})
    random.Random(f"{spec.seed}/rir_rooms").shuffle(rooms)
    reserved = sorted(rooms[: round(len(rooms) * spec.rir_room_fraction)])
    return reserved, [file for file in files if str(Path(file).parent.as_posix()) in set(reserved)]


def load_musan_val(data_dir: Path) -> tuple[Path, list[dict], str]:
    """Relit le split de validation de MUSAN. Renvoie (racine, entrées, empreinte du manifeste)."""
    directory = data_dir / "manifests" / "musan"
    manifest, config = directory / "musan_val.jsonl", directory / "musan_config.json"
    if not (manifest.exists() and config.exists()):
        raise BuildError(f"Partage MUSAN absent de {directory} : lancez `python split_musan.py`")
    root = Path(json.loads(config.read_text(encoding="utf-8"))["musan_root"])
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    return root, entries, file_sha256(manifest)


def draw_musan_source(
    musan_root: Path,
    entries: list[dict],
    weights: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
) -> list[dict]:
    entry = entries[draw_index(weights, generator)]
    frames = sf.info(musan_root / entry["path"]).frames
    start = int(torch.randint(max(frames - n_samples, 0) + 1, (1,), generator=generator))
    return [{"path": entry["path"], "start": start}]


def draw_babble_sources(
    bank: BabbleBank,
    librispeech: Path,
    utterance_id: str,
    n_samples: int,
    generator: torch.Generator,
) -> list[dict]:
    # BabbleBank retire le locuteur cible, donc aussi l'énoncé cible.
    speaker = utterance_id.split("-")[0]
    return [
        {
            "path": path.relative_to(librispeech).as_posix(),
            "start": draw_start(sf.info(path).frames, n_samples, generator),
        }
        for path in bank.pick_utterances(speaker, generator)
    ]


def build_entries(spec: ValSetSpec, data_dir: Path) -> tuple[list[dict], list[str], str]:
    """Tire tout le jeu. Renvoie (entrées, salles RIR réservées, empreinte du manifeste MUSAN)."""
    librispeech = data_dir / "LibriSpeech"
    musan_root, musan_entries, musan_sha256 = load_musan_val(data_dir)
    rooms, rir_files = select_rir_rooms(data_dir, spec)
    babble = BabbleBank("val", librispeech)

    musan_banks = {}
    for category in spec.categories:
        if category.name == BABBLE:
            continue
        selected = [entry for entry in musan_entries if entry["category"] == category.name]
        if not selected:
            raise BuildError(f"Aucun fichier MUSAN de validation pour la catégorie {category.name!r}")
        musan_banks[category.name] = selected, duration_weights([e["duration"] for e in selected], spec.duration_exponent)

    entries = []
    for utterance in list_utterances(librispeech, spec.subsets):
        utterance_id, n_samples = utterance["id"], utterance["frames"]

        mix_generator = make_generator(spec, "mix", utterance_id)
        reverberant = torch.rand(1, generator=mix_generator).item() < spec.p_rir
        draw = draw_validation_noise(spec.categories, spec.p_noise, mix_generator)

        rir_generator = make_generator(spec, "rir", utterance_id)
        rir = rir_files[int(torch.randint(len(rir_files), (1,), generator=rir_generator))]

        # Une source par catégorie : un énoncé garde le même bruit à tous les SNR de la grille.
        sources = {}
        for category in spec.categories:
            generator = make_generator(spec, f"noise/{category.name}", utterance_id)
            if category.name == BABBLE:
                sources[BABBLE] = draw_babble_sources(babble, librispeech, utterance_id, n_samples, generator)
            else:
                sources[category.name] = draw_musan_source(musan_root, *musan_banks[category.name], n_samples, generator)

        entries.append(
            {
                **utterance,
                "reverberant": reverberant,
                "rir": rir,
                "category": draw[0].name if draw else None,
                "snr_db": draw[1] if draw else None,
                "sources": sources,
            }
        )
    return entries, rooms, musan_sha256


def run(version: str, data_dir: Path, out_dir: Path) -> None:
    """Génère un jeu, sans jamais écraser une version existante."""
    spec = VAL_SETS[version]
    manifest, config_path = out_dir / f"{version}.jsonl", out_dir / f"{version}_config.json"
    if manifest.exists() or config_path.exists():
        raise BuildError(f"{version} existe déjà dans {out_dir} : un jeu figé ne se régénère pas")

    entries, rooms, musan_sha256 = build_entries(spec, data_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    config = {
        "spec": dataclasses.asdict(spec),
        "sha256": file_sha256(manifest),
        "musan_val_sha256": musan_sha256,
        "rir_rooms": rooms,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    logger.info("%s : %d énoncés, %d salles RIR réservées, écrit dans %s", version, len(entries), len(rooms), out_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", choices=VAL_SETS, default="val_v2", help="jeu à générer")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="dossier des données")
    parser.add_argument("--out-dir", type=Path, help="dossier de sortie (défaut : <data-dir>/manifests/val)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args.version, args.data_dir, args.out_dir or args.data_dir / "manifests" / "val")
    except BuildError as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
