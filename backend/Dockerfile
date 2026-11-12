# syntax=docker/dockerfile:1
#
# Image du backend : API FastAPI et moteur ASR. Elle ne contient ni données ni checkpoint :
# les versions de modèle sont montées dans /models (volume en lecture seule).
#
#   docker build -f backend/Dockerfile --build-arg TORCH_VARIANT=cpu   -t robust-asr-backend:cpu .
#   docker build -f backend/Dockerfile --build-arg TORCH_VARIANT=gpu -t robust-asr-backend:gpu .
#
# TORCH_VARIANT (cpu ou gpu) choisit requirements-cpu.txt (PyTorch sans CUDA) ou requirements-gpu.txt
# (PyTorch CUDA 13.0, runtime CUDA embarqué dans les roues : seul le pilote vient de l'hôte).

ARG PYTHON_IMAGE=python:3.14.10-slim-trixie

FROM ${PYTHON_IMAGE} AS dependencies
ARG TORCH_VARIANT=cpu
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
COPY backend/requirements-base.txt backend/requirements-${TORCH_VARIANT}.txt /tmp/requirements/
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r /tmp/requirements/requirements-${TORCH_VARIANT}.txt


FROM ${PYTHON_IMAGE}
ARG TORCH_VARIANT=cpu
LABEL org.opencontainers.image.title="robust-asr-backend" \
      org.opencontainers.image.description="API de transcription Robust ASR (FastAPI, PyTorch ${TORCH_VARIANT})"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    ASR_HOST=0.0.0.0 \
    ASR_PORT=8000 \
    ASR_MODELS_DIR=/models \
    ASR_TORCH_VARIANT=${TORCH_VARIANT}

RUN groupadd --system --gid 10001 asr \
 && useradd --system --uid 10001 --gid asr --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin asr

COPY --from=dependencies /opt/venv /opt/venv

WORKDIR /srv
# Seul le code d'inférence est copié : ni notebook, ni scripts de recherche, ni données.
COPY src/__init__.py src/__init__.py
COPY src/asr src/asr
COPY src/audio src/audio
COPY src/runtime src/runtime
COPY backend/__init__.py backend/__init__.py
COPY backend/app backend/app
# Bytecode compilé à la construction : le système de fichiers est en lecture seule à l'exécution.
RUN python -m compileall -q src backend

USER asr
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-m", "backend.app.healthcheck"]
CMD ["python", "-m", "backend.app"]
