#!/usr/bin/env bash
# Démarre Robust ASR avec Docker Compose : sur GPU NVIDIA s'il est utilisable depuis Docker, sinon sur CPU.
#
#   docker/start.sh            détection automatique du GPU
#   docker/start.sh --cpu      force la configuration CPU
#   docker/start.sh --gpu      exige le GPU (erreur s'il n'est pas utilisable)
#   docker/start.sh --down     arrête l'application
#   docker/start.sh --logs     affiche les journaux en continu
#
# Étapes : présence de Docker et de Compose v2, présence du modèle, NVIDIA Container Toolkit, accès
# réel au GPU depuis un conteneur, puis lancement de la configuration retenue et attente de l'état sain.
# Réglages : fichier .env à la racine (voir .env.example) ; une variable d'environnement le remplace.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Image de base du backend : déjà nécessaire à la construction, elle sert aussi au test du GPU
# (le NVIDIA Container Toolkit y monte nvidia-smi et le pilote de l'hôte).
PROBE_IMAGE="python:3.14.10-slim-trixie"
CPU_FILES=(-f docker-compose.yml)
GPU_FILES=(-f docker-compose.yml -f docker-compose.gpu.yml)

if [ -t 1 ]; then BOLD=$'\033[1m' YELLOW=$'\033[33m' RED=$'\033[31m' GREEN=$'\033[32m' RESET=$'\033[0m'
else BOLD="" YELLOW="" RED="" GREEN="" RESET=""; fi
say() { printf '%s==>%s %s\n' "$BOLD" "$RESET" "$*"; }
warn() { printf '%s[attention]%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die() { printf '%s[erreur]%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }

MODE=auto
ACTION=up
for argument in "$@"; do
  case "$argument" in
    --cpu) MODE=cpu ;;
    --gpu) MODE=gpu ;;
    --auto) MODE=auto ;;
    --down) ACTION=down ;;
    --logs) ACTION=logs ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "Option inconnue : $argument" ;;
  esac
done

# Valeur d'un réglage : variable d'environnement, sinon fichier .env, sinon valeur par défaut.
setting() {
  local key=$1 default=$2 line value
  if [ -n "${!key:-}" ]; then
    printf '%s' "${!key}"
    return
  fi
  line=$(grep -E "^${key}=" .env 2>/dev/null | tail -n 1 || true)
  value=${line#*=}
  value=${value%\"} value=${value#\"} value=${value%\'} value=${value#\'}
  printf '%s' "${value:-$default}"
}

# --- 1. Docker et Compose ---------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "Docker n'est pas installé : voir https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 ||
  die "Le démon Docker est injoignable : est-il démarré, et votre utilisateur est-il dans le groupe docker ?"
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 est requis (commande « docker compose »)."

if [ "$ACTION" = down ]; then
  say "Arrêt de Robust ASR"
  docker compose "${GPU_FILES[@]}" down --remove-orphans
  exit 0
fi
if [ "$ACTION" = logs ]; then
  exec docker compose "${CPU_FILES[@]}" logs --follow --tail=100
fi

# --- 2. Modèle --------------------------------------------------------------------------------------
MODELS_DIR=$(setting ASR_MODELS_HOST_DIR ./models)
VERSION=$(setting ASR_MODEL_VERSION v1)
if [ ! -f "$MODELS_DIR/$VERSION/config.yaml" ] || [ ! -f "$MODELS_DIR/$VERSION/best.pt" ]; then
  die "Modèle $VERSION absent de $MODELS_DIR. Exportez-le depuis le checkpoint de recherche :
    python scripts/export_model.py --checkpoint <checkpoints/best.pt> --cmvn <data/manifests/cmvn.pt> --version $VERSION"
fi

# --- 3. GPU : NVIDIA Container Toolkit, puis accès réel depuis un conteneur ------------------------
gpu_problem=""
if [ "$MODE" != cpu ]; then
  say "Recherche d'un GPU NVIDIA utilisable par Docker"
  if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' && ! command -v nvidia-ctk >/dev/null 2>&1; then
    gpu_problem="NVIDIA Container Toolkit absent : aucun runtime « nvidia » déclaré dans Docker."
  elif ! docker image inspect "$PROBE_IMAGE" >/dev/null 2>&1 && ! docker pull --quiet "$PROBE_IMAGE" >/dev/null 2>&1; then
    gpu_problem="image de test $PROBE_IMAGE indisponible (pas d'accès au registre Docker ?)."
  elif ! gpu_list=$(docker run --rm --gpus all "$PROBE_IMAGE" nvidia-smi -L 2>&1); then
    gpu_problem="GPU inaccessible depuis Docker : $(printf '%s' "$gpu_list" | tail -n 1)"
  else
    say "GPU accessible depuis Docker : $(printf '%s' "$gpu_list" | head -n 1)"
  fi
fi

if [ "$MODE" = gpu ] && [ -n "$gpu_problem" ]; then
  die "GPU exigé, mais inutilisable : $gpu_problem"
fi
if [ "$MODE" != cpu ] && [ -z "$gpu_problem" ]; then
  FILES=("${GPU_FILES[@]}")
  LABEL="GPU (CUDA)"
else
  [ "$MODE" = auto ] && warn "Configuration CPU retenue : $gpu_problem"
  FILES=("${CPU_FILES[@]}")
  LABEL="CPU"
fi

# --- 4. Port de l'interface ---------------------------------------------------------------------------
PORT=$(setting ASR_PORT 8080)
BIND=$(setting ASR_BIND_ADDRESS 127.0.0.1)
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN &&
  [ -z "$(docker compose "${CPU_FILES[@]}" ps --quiet frontend 2>/dev/null)" ]; then
  die "Le port $PORT est déjà utilisé par un autre programme : choisissez-en un autre avec ASR_PORT dans .env."
fi

# --- 5. Lancement -------------------------------------------------------------------------------------
say "Construction et démarrage en configuration $LABEL"
[ "$LABEL" = CPU ] || say "La première construction télécharge PyTorch CUDA (plusieurs gigaoctets)."
if ! docker compose "${FILES[@]}" up --detach --build --remove-orphans --wait --wait-timeout 600; then
  docker compose "${FILES[@]}" logs --tail=40 backend >&2 || true
  die "L'application n'a pas atteint un état sain : voir les journaux ci-dessus (docker/start.sh --logs)."
fi

HOST=$BIND
[ "$BIND" = 127.0.0.1 ] || [ "$BIND" = 0.0.0.0 ] && HOST=localhost
URL="http://$HOST:$PORT"
DEVICE=""
if command -v curl >/dev/null 2>&1; then
  DEVICE=$(curl --silent --max-time 5 "$URL/api/health" | sed -n 's/.*"device":"\([a-z]*\)".*/\1/p')
fi
printf '\n%sRobust ASR est prêt :%s %s\n' "$GREEN" "$RESET" "$URL"
printf '   configuration %s%s\n' "$LABEL" "${DEVICE:+, calcul sur $DEVICE}"
printf '   arrêt : docker/start.sh --down   journaux : docker/start.sh --logs\n'
