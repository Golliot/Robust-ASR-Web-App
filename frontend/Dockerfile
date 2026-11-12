# syntax=docker/dockerfile:1
#
# Image du frontend : l'application Vue compilée, servie par Nginx (sans privilèges), qui relaie
# /api vers le backend sur le réseau interne Docker.
#
#   docker build -t robust-asr-frontend frontend

FROM node:26.13.1-alpine3.24 AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm test && npm run build


FROM nginxinc/nginx-unprivileged:1.30.6-alpine
LABEL org.opencontainers.image.title="robust-asr-frontend" \
      org.opencontainers.image.description="Interface web Robust ASR (Vue 3, Nginx)"

ENV ASR_BACKEND_URL=http://backend:8000 \
    ASR_ALLOWED_HOSTS="localhost 127.0.0.1" \
    ASR_MAX_UPLOAD_MB=25 \
    ASR_INFERENCE_TIMEOUT_S=120

COPY --chmod=0755 nginx/limits.envsh /docker-entrypoint.d/05-robust-asr-limits.envsh
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY nginx/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
