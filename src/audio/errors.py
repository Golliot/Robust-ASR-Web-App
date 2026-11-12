"""Erreurs du traitement audio.

Chacune porte un code stable, repris tel quel par l'API, et un message en français destiné à
l'utilisateur : il ne contient ni chemin ni détail interne. Le détail technique (erreur de libsndfile
ou de FFmpeg) reste accessible par la chaîne des exceptions, pour les logs.
"""


def format_seconds(value: float, digits: int = 1) -> str:
    """Durée lisible en français : 42.0 -> « 42,0 »."""
    return f"{value:.{digits}f}".replace(".", ",")


def format_limit(value: float) -> str:
    """Limite lisible en français, sans zéros inutiles : 30.0 -> « 30 », 0.1 -> « 0,1 »."""
    return f"{value:g}".replace(".", ",")


class AudioError(ValueError):
    """Audio inutilisable. `code` identifie le cas, `message` s'affiche tel quel."""

    code = "INVALID_AUDIO"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EmptyAudioError(AudioError):
    code = "EMPTY_AUDIO"


class InvalidAudioError(AudioError):
    code = "INVALID_AUDIO"


class AudioTooShortError(AudioError):
    code = "AUDIO_TOO_SHORT"

    def __init__(self, duration_s: float, min_duration_s: float) -> None:
        super().__init__(
            f"L'audio dure {format_seconds(duration_s, 2)} s : il doit durer au moins {format_limit(min_duration_s)} s."
        )
        self.duration_s = duration_s
        self.limit_s = min_duration_s


class AudioTooLongError(AudioError):
    code = "AUDIO_TOO_LONG"

    def __init__(self, duration_s: float, max_duration_s: float, *, at_least: bool = False) -> None:
        # at_least : le décodage s'est arrêté à la limite, la durée réelle est supérieure.
        prefix = "plus de " if at_least else ""
        super().__init__(
            f"L'audio dure {prefix}{format_seconds(duration_s)} s : la limite est de {format_limit(max_duration_s)} s. "
            "Au-delà, la transcription du modèle se dégrade fortement."
        )
        self.duration_s = duration_s
        self.limit_s = max_duration_s
