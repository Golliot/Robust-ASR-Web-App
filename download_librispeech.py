"""Télécharge LibriSpeech via torchaudio dans le dossier data/.

Par défaut, les 7 sous-ensembles sont récupérés (environ 60 Go d'archives) :
    train-clean-100, train-clean-360, train-other-500   -> entraînement (960 h)
    dev-clean, dev-other                                -> validation
    test-clean, test-other                              -> test

Les fichiers extraits arrivent dans data/LibriSpeech/<sous-ensemble>/. Un
sous-ensemble déjà extrait est ignoré : le script peut être relancé sans risque.
Un téléchargement interrompu reprend là où il s'est arrêté.

Exemples :
    python download_librispeech.py
    python download_librispeech.py --subsets dev-clean test-clean
"""

import argparse
import hashlib
from pathlib import Path

from torchaudio.datasets import LIBRISPEECH
from torchaudio.datasets.librispeech import _CHECKSUMS

from download_noise import download

BASE_URL = "http://www.openslr.org/resources/12"
SUBSETS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]


def fetch_archive(subset: str, root: Path) -> None:
    """Dépose l'archive vérifiée dans root : torchaudio n'a plus qu'à l'extraire."""
    url = f"{BASE_URL}/{subset}.tar.gz"
    archive = download(url, root / f"{subset}.tar.gz")
    digest = hashlib.sha256()
    with archive.open("rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    # torchaudio ne contrôle la somme que s'il télécharge lui-même, sans reprise possible.
    if digest.hexdigest() != _CHECKSUMS[url]:
        archive.unlink()
        raise RuntimeError(f"Somme SHA-256 invalide pour {archive.name} : fichier supprimé, relancez le script")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data"), help="dossier de destination")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=SUBSETS)
    parser.add_argument("--keep-archives", action="store_true", help="conserve les .tar.gz après extraction")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        print(f"[{subset}] téléchargement et extraction...", flush=True)
        if not (args.root / "LibriSpeech" / subset).is_dir():
            fetch_archive(subset, args.root)
        dataset = LIBRISPEECH(root=args.root, url=subset, download=True)
        print(f"[{subset}] {len(dataset)} énoncés", flush=True)
        if not args.keep_archives:
            (args.root / f"{subset}.tar.gz").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
