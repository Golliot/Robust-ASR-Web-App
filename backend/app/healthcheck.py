"""Sonde de santé de l'image Docker : code 0 si le modèle est prêt (GET /api/health renvoie 200)."""

import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    url = f"http://127.0.0.1:{os.environ.get('ASR_PORT', '8000')}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            return 0 if response.status == 200 else 1
    except (urllib.error.URLError, OSError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
