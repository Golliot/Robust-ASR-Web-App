#!/usr/bin/env bash
# Test de bout en bout de l'interface, contre l'application démarrée (docker/start.sh), dans un
# Chromium piloté par Playwright. Rien à installer sur l'hôte : tout tourne dans l'image Playwright.
#
#   tests/e2e/run.sh                      contre http://localhost:8080
#   BASE_URL=http://localhost:9090 tests/e2e/run.sh
#
# Captures d'écran : tests/e2e/screenshots/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
IMAGE=mcr.microsoft.com/playwright:v1.65.0-noble
mkdir -p screenshots
docker run --rm --network host --ipc=host \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -e npm_config_update_notifier=false \
  -e BASE_URL="${BASE_URL:-http://localhost:8080}" \
  -v "$PWD:/e2e" -v "$PWD/../fixtures:/fixtures:ro" -w /e2e \
  "$IMAGE" sh -c "npm install --no-audit --no-fund --silent && node ui_smoke.mjs"
