"""Journalisation structurée : une ligne JSON par événement sur la sortie standard (logs Docker).

Chaque ligne porte l'identifiant de la requête en cours, s'il y en a une. Les champs passés en
`extra` sont ajoutés tels quels. Le texte transcrit n'est jamais journalisé : seules des mesures
(durées, format, périphérique) le sont.
"""

import datetime as dt
import json
import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# color_message : doublon coloré (codes ANSI) que joint Uvicorn à certains messages.
_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "taskName", "color_message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        payload.update({key: value for key, value in vars(record).items() if key not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Format lisible pour le développement, avec l'identifiant de requête."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        request_id = request_id_var.get()
        return f"{line} [{request_id}]" if request_id else line


def configure_logging(level: str = "INFO", log_format: str = "json") -> None:
    """Remplace la configuration existante, celle d'Uvicorn comprise."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if log_format == "json" else TextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    # Les accès sont journalisés par RequestContextMiddleware, avec l'identifiant de requête.
    logging.getLogger("uvicorn.access").disabled = True
