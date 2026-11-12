"""Télécharge les bruits et les réponses impulsionnelles (RIR) du notebook dans data/.

Jeu        Rôle                     Taille archive   Résultat
---------  -----------------------  ---------------  ------------------------------------------
musan      bruit train + val        11 Go            data/musan/{music,noise,speech}/
sim_rir    RIR train + val          179 Mo           data/simulated_rirs_16k/
real_rir   RIR test                 1,3 Go           data/RIRS_NOISES/real_rirs_isotropic_noises/
ace        RIR test                 417 Mo           data/ACE/
wham       bruit test (split tt)    18 Go            data/wham_noise/tt/
demand     bruit test               2,1 Go           data/DEMAND/<environnement>/ch01.wav

Seul le nécessaire est conservé : le split tt de WHAM!, le canal 1 de DEMAND, les
RIR réelles (sans les bruits) d'OpenSLR 28, et les RIR de la configuration
« Single » d'ACE. Tout est livré à 16 kHz : ACE et l'environnement SCAFE de DEMAND,
publiés seulement en 48 kHz, sont rééchantillonnés ici.

Un téléchargement interrompu reprend là où il s'est arrêté. Un jeu déjà prêt est
ignoré : le script peut être relancé sans risque.

Exemples :
    python download_noise.py
    python download_noise.py --datasets demand sim_rir
"""

import argparse
import hashlib
import http.client
import json
import shutil
import tarfile
import tempfile
import time
import urllib.error
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

OPENSLR = "https://www.openslr.org/resources"
WHAM_URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
ZENODO_DEMAND = "https://zenodo.org/api/records/1227121"
ZENODO_ACE = "https://zenodo.org/api/records/6257551"

TARGET_SR = 16_000
MAX_ATTEMPTS = 20
NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)


def download(url: str, dest: Path, md5: str | None = None) -> Path:
    """Télécharge url vers dest, avec reprise sur un fichier .part existant."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urlopen(Request(url, method="HEAD"), timeout=60) as response:
                total = int(response.headers["Content-Length"])
            done = part.stat().st_size if part.exists() else 0
            if done < total:
                request = Request(url, headers={"Range": f"bytes={done}-"})
                with urlopen(request, timeout=60) as response:
                    # 200 au lieu de 206 : le serveur ignore le Range, on repart de zéro.
                    done = done if response.status == 206 else 0
                    with (
                        part.open("ab" if done else "wb") as file,
                        tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=dest.name) as bar,
                    ):
                        while chunk := response.read(1 << 20):
                            file.write(chunk)
                            bar.update(len(chunk))
            if part.stat().st_size == total:
                break
            print(f"{dest.name} : téléchargement incomplet, nouvelle tentative", flush=True)
        except NETWORK_ERRORS as error:
            print(f"{dest.name} : {error!r} (tentative {attempt}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(min(2**attempt, 60))
    else:
        raise RuntimeError(f"Échec du téléchargement de {url} après {MAX_ATTEMPTS} tentatives")

    if md5 is not None:
        digest = hashlib.md5()
        with part.open("rb") as file:
            while chunk := file.read(1 << 24):
                digest.update(chunk)
        if digest.hexdigest() != md5:
            part.unlink()
            raise RuntimeError(f"Somme MD5 invalide pour {dest.name} : fichier supprimé, relancez le script")

    part.rename(dest)
    return dest


def zenodo_files(record_url: str) -> dict[str, dict]:
    """Renvoie {nom de fichier : métadonnées} pour un enregistrement Zenodo."""
    with urlopen(record_url, timeout=60) as response:
        return {entry["key"]: entry for entry in json.load(response)["files"]}


def download_zenodo(entry: dict, downloads: Path) -> Path:
    md5 = entry["checksum"].removeprefix("md5:")
    return download(entry["links"]["self"], downloads / entry["key"], md5=md5)


def convert_to_16k_mono(source: Path, target: Path) -> None:
    """Écrit le premier canal de source dans target, à 16 kHz."""
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    resampled = torchaudio.functional.resample(torch.from_numpy(audio[:, 0]), sample_rate, TARGET_SR)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, resampled.numpy(), TARGET_SR, subtype="FLOAT")


def prepare_musan(root: Path, downloads: Path) -> list[Path]:
    archive = download(f"{OPENSLR}/17/musan.tar.gz", downloads / "musan.tar.gz")
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return [archive]


def prepare_sim_rir(root: Path, downloads: Path) -> list[Path]:
    archive = download(f"{OPENSLR}/26/sim_rir_16k.zip", downloads / "sim_rir_16k.zip")
    with zipfile.ZipFile(archive) as archive_zip:
        archive_zip.extractall(root)
    return [archive]


def prepare_real_rir(root: Path, downloads: Path) -> list[Path]:
    archive = download(f"{OPENSLR}/28/rirs_noises.zip", downloads / "rirs_noises.zip")
    with zipfile.ZipFile(archive) as archive_zip:
        # rir_list recense les 325 RIR réelles ; le dossier contient aussi des bruits isotropes.
        listing = archive_zip.read("RIRS_NOISES/real_rirs_isotropic_noises/rir_list").decode()
        archive_zip.extractall(root, members=[line.split()[-1] for line in listing.splitlines()])
    return [archive]


def prepare_ace(root: Path, downloads: Path) -> list[Path]:
    entry = zenodo_files(ZENODO_ACE)["ACE_Corpus_RIRN_Single.tbz2"]
    archive = download_zenodo(entry, downloads)
    with tempfile.TemporaryDirectory(dir=root) as temporary, tarfile.open(archive) as tar:
        members = [member for member in tar.getmembers() if member.name.endswith("_RIR.wav")]
        tar.extractall(temporary, members=members, filter="data")
        for source in Path(temporary).rglob("*_RIR.wav"):
            convert_to_16k_mono(source, root / "ACE" / source.name)
    return [archive]


def prepare_wham(root: Path, downloads: Path) -> list[Path]:
    archive = download(WHAM_URL, downloads / "wham_noise.zip")
    with zipfile.ZipFile(archive) as archive_zip:
        members = [
            name for name in archive_zip.namelist() if name.startswith("wham_noise/tt/") and name.endswith(".wav")
        ]
        archive_zip.extractall(root, members=members)
    return [archive]


def prepare_demand(root: Path, downloads: Path) -> list[Path]:
    files = zenodo_files(ZENODO_DEMAND)
    environments = sorted(name.split("_")[0] for name in files if name.endswith("_48k.zip"))
    archives = []
    for environment in environments:
        # SCAFE n'existe pas en 16 kHz sur Zenodo.
        entry = files.get(f"{environment}_16k.zip") or files[f"{environment}_48k.zip"]
        archive = download_zenodo(entry, downloads)
        archives.append(archive)
        member = f"{environment}/ch01.wav"
        target = root / "DEMAND" / member
        with tempfile.TemporaryDirectory(dir=root) as temporary, zipfile.ZipFile(archive) as archive_zip:
            source = Path(archive_zip.extract(member, temporary))
            if sf.info(source).samplerate == TARGET_SR:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, target)
            else:
                convert_to_16k_mono(source, target)
    return archives


DATASETS: dict[str, Callable[[Path, Path], list[Path]]] = {
    "musan": prepare_musan,
    "sim_rir": prepare_sim_rir,
    "real_rir": prepare_real_rir,
    "ace": prepare_ace,
    "wham": prepare_wham,
    "demand": prepare_demand,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data"), help="dossier de destination")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--keep-archives", action="store_true", help="conserve les archives après extraction")
    args = parser.parse_args()

    downloads = args.root / "downloads"
    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        marker = args.root / f".{name}.complete"
        if marker.exists():
            print(f"[{name}] déjà prêt", flush=True)
            continue
        print(f"[{name}] préparation...", flush=True)
        archives = DATASETS[name](args.root, downloads)
        marker.touch()
        if not args.keep_archives:
            for archive in archives:
                archive.unlink()
        print(f"[{name}] terminé", flush=True)


if __name__ == "__main__":
    main()
