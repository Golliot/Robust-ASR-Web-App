"""Inventorie MUSAN et le partage en train/validation, par fichier et de façon reproductible.

Le partage est stratifié par (catégorie, source) : le ratio s'applique séparément dans
chaque groupe. Le sous-dossier LibriVox de speech/ est exclu par défaut, car LibriSpeech
provient de LibriVox : il pourrait partager des locuteurs et des textes avec la parole.

Écrit dans --out-dir :
    musan_train.jsonl, musan_val.jsonl   une ligne par fichier, chemins relatifs à la racine
    musan_config.json                    racine MUSAN et paramètres du partage
    musan_split_summary.md               récapitulatif par catégorie, source et split

Pour déplacer MUSAN, il suffit de modifier `musan_root` dans musan_config.json. Une racine
relative s'entend depuis la racine du projet.

Le script ne lit que les en-têtes audio et n'écrit rien dans MUSAN. Il échoue, sans rien
écrire, si un fichier n'est pas en 16 kHz mono : aucun rééchantillonnage n'est fait.

Exemples :
    python split_musan.py
    python split_musan.py --val-ratio 0.1 --seed 7
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

logger = logging.getLogger("split_musan")

CATEGORIES = ("music", "noise", "speech")
EXPECTED_SR = 16_000
EXPECTED_CHANNELS = 1


class ValidationError(Exception):
    """Une vérification a échoué : le message explique laquelle."""


@dataclass(frozen=True)
class AudioFile:
    """Métadonnées d'un fichier MUSAN, avec son chemin relatif à la racine."""

    path: str
    category: str
    source: str
    duration: float
    sr: int
    channels: int

    @property
    def group(self) -> tuple[str, str]:
        return self.category, self.source

    def manifest_entry(self) -> dict:
        return {
            "path": self.path,
            "category": self.category,
            "source": self.source,
            "duration": self.duration,
            "sr": self.sr,
        }


def inventory(root: Path) -> list[AudioFile]:
    """Liste les .wav de root, triés par chemin, en ne lisant que leurs en-têtes."""
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    if not paths:
        raise ValidationError(f"Aucun fichier .wav dans {root}")

    files, misplaced, unreadable = [], [], []
    for path in paths:
        relative = path.relative_to(root)
        # Attendu : <catégorie>/<source>/.../<fichier>.wav
        if len(relative.parts) < 3 or relative.parts[0] not in CATEGORIES:
            misplaced.append(relative.as_posix())
            continue
        try:
            info = sf.info(path)
        except sf.SoundFileError:
            unreadable.append(relative.as_posix())
            continue
        files.append(
            AudioFile(
                path=relative.as_posix(),
                category=relative.parts[0],
                source=relative.parts[1],
                duration=info.duration,
                sr=info.samplerate,
                channels=info.channels,
            )
        )

    if misplaced:
        raise ValidationError(
            f"{len(misplaced)} fichier(s) hors de <{'|'.join(CATEGORIES)}>/<source>/ :\n" + bullet_list(misplaced)
        )
    if unreadable:
        raise ValidationError(f"{len(unreadable)} fichier(s) illisibles :\n" + bullet_list(unreadable))
    return files


def is_librivox(file: AudioFile) -> bool:
    return file.category == "speech" and file.source.lower() == "librivox"


def filter_librivox(files: Iterable[AudioFile], keep: bool) -> tuple[list[AudioFile], list[AudioFile]]:
    """Sépare les fichiers en (conservés, exclus) et journalise la décision."""
    files = list(files)
    librivox = [file for file in files if is_librivox(file)]
    if keep:
        logger.warning(
            "LibriVox conservé (--keep-librivox) : %d fichiers, risque de fuite de locuteurs et de textes "
            "vers LibriSpeech",
            len(librivox),
        )
        return files, []
    if librivox:
        logger.info("LibriVox exclu : %d fichiers, %.1f h", len(librivox), total_hours(librivox))
    else:
        logger.warning("Aucun dossier LibriVox trouvé dans speech/ : rien n'a été exclu")
    return [file for file in files if not is_librivox(file)], librivox


def check_format(files: Iterable[AudioFile]) -> None:
    """Échoue en listant tout fichier qui n'est pas en 16 kHz mono."""
    wrong = [file for file in files if file.sr != EXPECTED_SR or file.channels != EXPECTED_CHANNELS]
    if wrong:
        details = [f"{file.path} : {file.sr} Hz, {file.channels} canal(aux)" for file in wrong]
        raise ValidationError(
            f"{len(wrong)} fichier(s) ne sont pas en 16 kHz mono (rien n'a été rééchantillonné) :\n"
            + bullet_list(details)
        )


def val_size(n_files: int, val_ratio: float) -> int:
    """Nombre de fichiers de validation d'un groupe, avec au moins 1 fichier de chaque côté."""
    if n_files < 2:
        return 0
    return min(max(round(n_files * val_ratio), 1), n_files - 1)


def split_files(files: Iterable[AudioFile], val_ratio: float, seed: int) -> tuple[list[AudioFile], list[AudioFile]]:
    """Partage par fichier, stratifié par (catégorie, source), indépendant de l'ordre d'entrée."""
    groups: dict[tuple[str, str], list[AudioFile]] = defaultdict(list)
    for file in files:
        groups[file.group].append(file)

    train, val = [], []
    for (category, source), members in sorted(groups.items()):
        members = sorted(members, key=lambda file: file.path)
        # Une graine par groupe : ajouter ou retirer une source ne change pas le partage des autres.
        random.Random(f"{seed}/{category}/{source}").shuffle(members)
        n_val = val_size(len(members), val_ratio)
        val += members[:n_val]
        train += members[n_val:]
        if len(members) * min(val_ratio, 1 - val_ratio) < 1:
            logger.warning(
                "Groupe trop petit pour le ratio demandé : %s/%s, %d fichier(s) (%d en validation)",
                category,
                source,
                len(members),
                n_val,
            )
    return sorted(train, key=lambda file: file.path), sorted(val, key=lambda file: file.path)


def check_disjoint(train: Iterable[AudioFile], val: Iterable[AudioFile]) -> None:
    """Échoue si un fichier figure dans les deux splits."""
    overlap = sorted({file.path for file in train} & {file.path for file in val})
    if overlap:
        raise ValidationError(f"{len(overlap)} fichier(s) dans les deux splits :\n" + bullet_list(overlap))


def check_groups(train: Sequence[AudioFile], val: Sequence[AudioFile]) -> None:
    """Échoue si un groupe d'au moins 2 fichiers manque dans un split."""
    counts = Counter(file.group for file in [*train, *val])
    train_groups = {file.group for file in train}
    val_groups = {file.group for file in val}
    missing = [
        "/".join(group)
        for group, count in sorted(counts.items())
        if count >= 2 and not (group in train_groups and group in val_groups)
    ]
    if missing:
        raise ValidationError(f"{len(missing)} groupe(s) absents d'un split :\n" + bullet_list(missing))


def write_manifest(path: Path, files: Iterable[AudioFile]) -> None:
    lines = [json.dumps(file.manifest_entry(), ensure_ascii=False) for file in files]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def summary_markdown(
    train: Sequence[AudioFile],
    val: Sequence[AudioFile],
    excluded: Sequence[AudioFile],
    val_ratio: float,
    seed: int,
) -> str:
    """Produit le récapitulatif par groupe et par split."""
    excluded_groups = sorted({file.group for file in excluded})
    excluded_text = (
        ", ".join(
            f"{category}/{source} ({len(members)} fichiers, {total_hours(members):.2f} h)"
            for category, source in excluded_groups
            for members in [[file for file in excluded if file.group == (category, source)]]
        )
        or "aucune"
    )
    lines = [
        "# Partage train/validation de MUSAN",
        "",
        f"- Graine : {seed}",
        f"- Ratio : {1 - val_ratio:g}/{val_ratio:g}, appliqué dans chaque groupe (catégorie, source)",
        f"- Sources exclues : {excluded_text}",
        "",
        "| Catégorie | Source | Fichiers train | Heures train | Fichiers val | Heures val |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in sorted({file.group for file in [*train, *val]}):
        group_train = [file for file in train if file.group == group]
        group_val = [file for file in val if file.group == group]
        lines.append(summary_row(*group, group_train, group_val))
    lines.append(summary_row("**Total**", "", train, val))
    return "\n".join(lines) + "\n"


def summary_row(category: str, source: str, train: Sequence[AudioFile], val: Sequence[AudioFile]) -> str:
    return (
        f"| {category} | {source} | {len(train)} | {total_hours(train):.2f} "
        f"| {len(val)} | {total_hours(val):.2f} |"
    )


def total_hours(files: Iterable[AudioFile]) -> float:
    return sum(file.duration for file in files) / 3600


def bullet_list(items: Iterable[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)


def run(root: Path, out_dir: Path, val_ratio: float, seed: int, keep_librivox: bool) -> None:
    """Vérifie tout, puis écrit les sorties : un échec ne laisse aucun fichier partiel."""
    if not root.is_dir():
        raise ValidationError(f"{root} introuvable : lancez `python download_noise.py --datasets musan`")
    if out_dir.resolve().is_relative_to(root.resolve()):
        raise ValidationError(f"--out-dir ({out_dir}) ne doit pas être dans MUSAN ({root})")

    files = inventory(root)
    logger.info("%d fichiers .wav, %.1f h", len(files), total_hours(files))

    kept, excluded = filter_librivox(files, keep_librivox)
    # Les fichiers exclus ne servent jamais : leur format n'a pas d'importance.
    check_format(kept)

    train, val = split_files(kept, val_ratio, seed)
    check_disjoint(train, val)
    check_groups(train, val)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir / "musan_train.jsonl", train)
    write_manifest(out_dir / "musan_val.jsonl", val)
    config = {"musan_root": str(root), "val_ratio": val_ratio, "seed": seed, "keep_librivox": keep_librivox}
    (out_dir / "musan_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    summary = summary_markdown(train, val, excluded, val_ratio, seed)
    (out_dir / "musan_split_summary.md").write_text(summary, encoding="utf-8")

    logger.info(
        "Train : %d fichiers, %.1f h. Validation : %d fichiers, %.1f h. Sorties dans %s",
        len(train),
        total_hours(train),
        len(val),
        total_hours(val),
        out_dir,
    )


def ratio(value: str) -> float:
    """Type argparse : un réel strictement entre 0 et 1."""
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("le ratio doit être strictement entre 0 et 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/musan"), help="racine de MUSAN")
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests/musan"), help="dossier de sortie")
    parser.add_argument("--val-ratio", type=ratio, default=0.2, help="part de validation dans chaque groupe")
    parser.add_argument("--seed", type=int, default=42, help="graine du partage")
    parser.add_argument(
        "--keep-librivox",
        action="store_true",
        help="conserve speech/librivox, au risque d'une fuite vers LibriSpeech",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args.root, args.out_dir, args.val_ratio, args.seed, args.keep_librivox)
    except ValidationError as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
