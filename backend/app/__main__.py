"""Lance le backend : python -m backend.app

Un seul processus Uvicorn, propriétaire du modèle. Les réglages viennent des variables ASR_* ;
la journalisation (JSON par défaut) remplace celle d'Uvicorn.
"""

import sys

import uvicorn

from backend.app.core.logging import configure_logging
from backend.app.core.settings import Settings, SettingsError


def main() -> int:
    try:
        settings = Settings.from_env()
    except SettingsError as error:
        print(f"Configuration invalide : {error}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level, settings.log_format)
    uvicorn.run(
        "backend.app.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,
        log_config=None,
        access_log=False,
        server_header=False,
        timeout_graceful_shutdown=5,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
