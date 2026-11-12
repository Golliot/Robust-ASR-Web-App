"""Transcrit des fichiers audio avec le moteur, sans FastAPI ni Docker.

Le modèle est chargé une fois, sur le GPU s'il est utilisable, sinon sur le CPU, puis chaque fichier
est transcrit : WAV, FLAC, OGG, MP3, WebM, MP4/M4A…, à n'importe quelle fréquence, mono ou stéréo.

Exemples :
    python scripts/transcribe.py enregistrement.webm
    python scripts/transcribe.py tests/fixtures/librispeech/*.flac --device cpu --json
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.engine import ASREngine  # noqa: E402
from src.audio.errors import AudioError  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="fichiers audio à transcrire")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "v1", help="dossier de la version de modèle")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="périphérique de calcul")
    parser.add_argument("--json", action="store_true", help="une ligne JSON par fichier")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    engine = ASREngine.load(args.model, device=args.device)
    info = engine.device_info
    if not args.json:
        print(f"Modèle {engine.spec.version} sur {info.type} ({info.name})", file=sys.stderr)
        if info.fallback_reason:
            print(f"CUDA écarté : {info.fallback_reason}", file=sys.stderr)

    failures = 0
    for path in args.files:
        try:
            result = engine.transcribe_file(path)
        except (AudioError, OSError) as error:
            failures += 1
            message = error.message if isinstance(error, AudioError) else str(error)
            print(json.dumps({"file": str(path), "error": message}, ensure_ascii=False) if args.json else f"{path} : {message}")
            continue
        if args.json:
            print(json.dumps({"file": str(path), **asdict(result)}, ensure_ascii=False))
        else:
            print(f"{path} ({result.audio_duration:.1f} s, {result.processing_time:.2f} s) : {result.text}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
