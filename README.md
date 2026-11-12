# Robust-ASR : Transformer hybride CTC / attention sur LibriSpeech

Reconnaissance vocale robuste au bruit et à la réverbération, de la forme d'onde à la transcription.
Un Transformer encodeur-décodeur, entraîné de zéro sur les 960 h de LibriSpeech avec des dégradations
réalistes (RIR, bruit, musique, babble), puis évalué sur des conditions jamais vues à l'entraînement :
RIR réelles, bruits WHAM! et DEMAND de −5 à 20 dB.

Le dépôt réunit deux choses :

- **le projet de recherche** : tout le pipeline (données, modèle, entraînement, décodage, test) est
  dans le notebook [`robust_asr.ipynb`](robust_asr.ipynb), qui documente aussi chaque choix et les
  mesures qui l'ont motivé ;
- **une application web locale** : enregistrer ou importer un audio dans le navigateur, et le faire
  transcrire par le modèle entraîné, sur GPU NVIDIA ou sur CPU, avec Docker Compose.

```
Navigateur ──► Nginx (interface Vue, 127.0.0.1:8080) ──► FastAPI ──► moteur ASR (src/asr)
                                                         backend      décodage audio, 16 kHz mono,
                                                         isolé        modèle du notebook, beam search
```

L'architecture de l'application, ses choix et ses limites sont détaillés dans
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Sommaire

**A. Application web**

1. [Prérequis](#1-prérequis)
2. [Préparer le modèle](#2-préparer-le-modèle)
3. [Lancer l'application](#3-lancer-lapplication)
4. [GPU et CUDA](#4-gpu-et-cuda)
5. [Utilisation](#5-utilisation)
6. [API REST](#6-api-rest)
7. [Configuration](#7-configuration)
8. [Développement](#8-développement)
9. [Tests](#9-tests)
10. [Problèmes fréquents](#10-problèmes-fréquents)

**B. Projet de recherche**

11. [Points saillants](#11-points-saillants)
12. [Répartition des données](#12-répartition-des-données)
13. [Installation et utilisation du notebook](#13-installation-et-utilisation-du-notebook)
14. [Résultats](#14-résultats)
15. [Arborescence](#15-arborescence)
16. [Données et licences](#16-données-et-licences)

---

# A. Application web

## 1. Prérequis

| | Mode CPU | Mode GPU |
|---|---|---|
| Docker Engine et Compose v2 | requis (testé avec Docker 29.9, Compose v5.7) | requis |
| GPU | — | NVIDIA, architecture Turing (sm_75) ou plus récente, jusqu'à Blackwell (sm_120) |
| Pilote NVIDIA | — | ≥ 580 (CUDA 13.0 ; `nvidia-smi` doit afficher « CUDA Version: 13.0 » ou plus) |
| NVIDIA Container Toolkit | — | requis (testé avec la version 1.20.2) |
| Espace disque pour Docker | environ 2 Go | environ 10 Go, plus 10 Go de cache pendant la première construction |
| Modèle exporté | `models/v1/` (102 Mo), voir ci-dessous | idem |

Un navigateur récent (Chrome, Edge, Firefox ou Safari) suffit côté client. Le micro n'est accessible
que sur `http://localhost` ou `http://127.0.0.1`, que les navigateurs considèrent comme sûrs.

Installation du NVIDIA Container Toolkit (Ubuntu), d'après la
[documentation de NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) :

```bash
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all python:3.14.10-slim-trixie nvidia-smi -L   # doit lister le GPU
```

## 2. Préparer le modèle

Les images Docker ne contiennent ni données ni checkpoint. Le modèle vit dans `models/<version>/`,
monté en lecture seule dans le backend :

```text
models/v1/
├── best.pt       poids (export allégé du checkpoint d'entraînement : 102 Mo au lieu de 306 Mo)
├── cmvn.pt       statistiques CMVN, identiques à celles contenues dans le checkpoint
└── config.yaml   hyperparamètres du notebook, décodage, limites, provenance, empreintes SHA-256
```

`config.yaml` est versionné, mais pas les poids. On les produit une fois à partir du checkpoint de
recherche, qui n'est que lu :

```bash
python scripts/export_model.py \
    --checkpoint "../Robust ASR/checkpoints/best.pt" \
    --cmvn "../Robust ASR/data/manifests/cmvn.pt" \
    --version v1
```

Le script vérifie que la CMVN et les hyperparamètres correspondent au checkpoint, écrit la version,
puis la recharge et la compare tenseur par tenseur. Il refuse d'écraser une version existante.
Sans environnement Python (voir [Développement](#8-développement)), l'image du backend peut
l'exécuter, une fois construite :

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/repo" -v "/chemin/vers/Robust ASR:/recherche:ro" \
    -w /repo robust-asr-backend:cpu python scripts/export_model.py \
    --checkpoint /recherche/checkpoints/best.pt --cmvn /recherche/data/manifests/cmvn.pt --version v1
```

## 3. Lancer l'application

```bash
docker/start.sh            # GPU s'il est utilisable depuis Docker, sinon CPU
docker/start.sh --cpu      # force le CPU
docker/start.sh --gpu      # exige le GPU (erreur explicite sinon)
docker/start.sh --logs     # journaux en continu
docker/start.sh --down     # arrêt
```

Puis ouvrir **http://localhost:8080**.

Le script vérifie, dans l'ordre :

1. la présence de Docker, de son démon et de Compose v2 ;
2. la présence du modèle ;
3. la présence du NVIDIA Container Toolkit ;
4. l'accès réel au GPU depuis un conteneur (`nvidia-smi`).

Il lance ensuite la configuration retenue et attend que les deux conteneurs soient sains. La première
construction prend environ 2 minutes en CPU et 8 minutes en GPU, où PyTorch CUDA pèse plusieurs
gigaoctets.

Sans le script :

```bash
docker compose up -d --build                                                   # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down            # arrêt
```

Deux conteneurs tournent alors :

- **frontend** : Nginx sert l'interface et relaie `/api`. Il est publié sur `127.0.0.1:8080`
  seulement.
- **backend** : FastAPI et le modèle. Il ne publie aucun port et se trouve sur un réseau Docker
  interne, sans accès à Internet.

Les deux conteneurs tournent sans privilèges, avec un système de fichiers en lecture seule.

## 4. GPU et CUDA

- **Deux images.** L'image `robust-asr-backend:cpu` contient PyTorch sans CUDA (1,7 Go).
  L'image `robust-asr-backend:gpu` contient PyTorch 2.14.1 pour CUDA 13.0 (9,3 Go) : les roues
  embarquent le runtime CUDA et cuDNN, et seul le pilote vient de l'hôte, monté par le NVIDIA
  Container Toolkit. Les versions de PyTorch et torchaudio sont celles de l'environnement de
  recherche.
- **Double détection.** `docker/start.sh` vérifie que Docker voit le GPU. Le backend
  ([`src/runtime/device_manager.py`](src/runtime/device_manager.py)) vérifie ensuite que PyTorch
  peut réellement s'en servir :
  1. build CUDA de PyTorch ;
  2. pilote présent ;
  3. architecture du GPU couverte par le build ;
  4. calcul de test ;
  5. transcription de chauffe du modèle complet.

  En cas d'échec, le modèle déjà chargé passe sur CPU, sans relecture du checkpoint. La raison
  s'affiche dans la vue **Système**.
- **`ASR_DEVICE`** : `auto` (par défaut), `cuda` (GPU exigé : erreur explicite plutôt que repli) ou
  `cpu`.
- **Précision.** Sur GPU, le modèle tourne en bfloat16 (autocast), comme à l'entraînement ; sur CPU,
  en float32, comme dans le notebook. Les transcriptions des énoncés de test sont identiques ; les
  écarts numériques mesurés et leurs tolérances sont dans
  [`tests/regression/test_notebook_equivalence.py`](tests/regression/test_notebook_equivalence.py).
- **Vitesse** (beam search de largeur 10, mesurée sur une RTX 5060 Ti et sur 6 cœurs) :

  | Audio | GPU | CPU |
  |---|---|---|
  | 10 s | 0,5 s | 2,4 s |
  | 30 s, pire cas | 2,3 s | 20 à 30 s |

- **Pilote plus ancien.** Les roues CUDA 13.0 exigent un pilote ≥ 580. Avec un pilote plus ancien, le
  backend passe sur CPU et l'indique. On peut aussi construire l'image GPU pour une autre version
  CUDA publiée par PyTorch, en adaptant l'index et la version de `torch` dans
  [`backend/requirements-gpu.txt`](backend/requirements-gpu.txt).

## 5. Utilisation

- **Transcription.**
  - Enregistrez votre voix au micro : l'enregistrement s'arrête automatiquement avant la limite, avec
    un compte à rebours.
  - Ou importez un fichier : WAV, FLAC, OGG, MP3, WebM, M4A…, à n'importe quelle fréquence, mono ou
    stéréo.
  - Réécoutez-le, puis cliquez sur **Transcrire**.

  Le résultat affiche le texte, le temps de traitement, la durée de l'audio, le périphérique (GPU ou
  CPU), la version du modèle et le format reçu.
- **Historique.** Les transcriptions de ce navigateur (texte et mesures seulement), effaçables.
- **Système.** Le périphérique utilisé et, le cas échéant, la raison du repli sur CPU, le modèle et
  sa provenance, les limites, les formats et les versions logicielles.

À savoir :

- **Langue.** Le modèle reconnaît l'**anglais** (LibriSpeech) et écrit **en majuscules, sans
  ponctuation**.
- **Durée.** L'audio doit durer **de 0,1 à 30 s**. Au-delà de 30 s, la transcription se dégrade
  fortement, car le modèle n'a jamais vu d'énoncé plus long : 18 % de WER entre 30 et 35 s, plus de
  100 % vers 75 s. L'audio est donc refusé.
- **Confidentialité.** L'audio n'est jamais conservé : il est traité en mémoire puis oublié. Pour le
  garder, utilisez **Télécharger l'audio**, qui l'enregistre depuis le navigateur.
- **Traitement du signal.** Il se fait côté serveur : décodage (soundfile, ou FFmpeg via PyAV),
  moyenne des canaux, rééchantillonnage à 16 kHz. Aucune normalisation n'est ajoutée, comme à
  l'entraînement.

## 6. API REST

| Endpoint | Méthode | Fonction |
|---|---|---|
| `/api/transcribe` | POST | Transcription d'un fichier audio (multipart/form-data, champ `file`) |
| `/api/system` | GET | Périphérique, modèle chargé, limites, formats, versions |
| `/api/health` | GET | État du backend : 200 si le modèle est prêt, 503 pendant son chargement ou en cas d'échec |
| `/api/models` | GET | Versions présentes dans `models/` et version chargée |

```bash
curl -F "file=@enregistrement.webm" http://localhost:8080/api/transcribe
```

```json
{
  "transcription": "HE HOPED THERE WOULD BE STEW FOR DINNER",
  "processing_time": 0.47,
  "inference_time": 0.45,
  "preprocessing_time": 0.02,
  "audio_duration": 10.44,
  "device": "cuda",
  "model_version": "v1",
  "sample_rate": 16000,
  "input_format": "WebM/Matroska (opus)",
  "request_id": "3f2c9a1b7e4d4c10"
}
```

Les erreurs ont toutes la forme `{"error": {"code", "message", "request_id"}}`, sans trace Python :

| Code HTTP | Codes d'erreur |
|---|---|
| 400 | `EMPTY_FILE`, `INVALID_REQUEST`, `INVALID_HOST` |
| 403 | `FORBIDDEN_ORIGIN` |
| 409 | `ENGINE_BUSY` (une transcription à la fois) |
| 413 | `FILE_TOO_LARGE` |
| 422 | `INVALID_AUDIO`, `EMPTY_AUDIO`, `AUDIO_TOO_SHORT`, `AUDIO_TOO_LONG` |
| 503 | `MODEL_NOT_READY` |
| 504 | `INFERENCE_TIMEOUT` |

La documentation interactive est sur http://localhost:8080/api/docs. Les requêtes d'écriture venant
d'un autre site ouvert dans le navigateur sont refusées ; curl et les scripts restent acceptés.

## 7. Configuration

Copiez [`.env.example`](.env.example) en `.env` à la racine, puis modifiez-le : `docker/start.sh` et
`docker compose` le lisent.

| Variable | Défaut | Rôle |
|---|---|---|
| `ASR_BIND_ADDRESS`, `ASR_PORT` | `127.0.0.1`, `8080` | adresse de l'interface (locale seulement par défaut) |
| `ASR_ALLOWED_HOSTS` | `localhost 127.0.0.1` | noms d'hôte acceptés dans la barre d'adresse |
| `ASR_MODELS_HOST_DIR`, `ASR_MODEL_VERSION` | `./models`, `v1` | versions de modèle et version chargée |
| `ASR_DEVICE` | `auto` | `auto`, `cuda` ou `cpu` |
| `ASR_NUM_THREADS` | automatique | threads CPU de PyTorch |
| `ASR_MAX_UPLOAD_MB` | `25` | taille maximale d'un fichier (Nginx s'aligne seul) |
| `ASR_MAX_DURATION_S` | celle du modèle (30) | durée maximale, qui ne peut qu'être réduite |
| `ASR_INFERENCE_TIMEOUT_S` | `120` | délai maximal d'une transcription (Nginx s'aligne seul) |
| `ASR_LOG_LEVEL`, `ASR_ENABLE_DOCS` | `INFO`, `true` | journaux, documentation de l'API |

**Ajouter une version de modèle.** Exportez-la sous un nouveau nom (`--version v2`), puis indiquez
`ASR_MODEL_VERSION=v2`. Pour essayer le décodage CTC glouton, plus rapide sur CPU :

1. copiez `models/v1` en `models/v1-greedy` ;
2. dans son `config.yaml`, mettez `version: v1-greedy` et `decoding.method: ctc_greedy`.

## 8. Développement

Environnement Python (3.14, comme la recherche), avec [uv](https://docs.astral.sh/uv/) :

```bash
uv venv .venv --python 3.14
uv pip install -r requirements.txt -r requirements-dev.txt \
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match
```

Sans GPU, remplacez `cu130` par `cpu`.

Le moteur s'utilise sans FastAPI :

```python
from src.asr.engine import ASREngine

engine = ASREngine.load("models/v1")            # GPU si utilisable, sinon CPU
result = engine.transcribe_file("enregistrement.webm")
print(result.text, result.device, result.inference_time)
```

En ligne de commande : `python scripts/transcribe.py fichier.flac autre.webm [--device cpu] [--json]`.

Backend et frontend en local, sans Docker :

```bash
ASR_LOG_FORMAT=text .venv/bin/python -m backend.app      # API sur http://127.0.0.1:8000
cd frontend && npm ci && npm run dev                     # interface sur http://127.0.0.1:5173
```

Le serveur Vite relaie `/api` vers le port 8000. Sans Node.js sur la machine, les commandes npm
tournent dans un conteneur :

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/frontend:/app" -w /app \
    node:26.13.1-alpine3.24 npm run build
```

| Dossier | Contenu |
|---|---|
| `src/asr/` | moteur extrait du notebook : modèle, features, décodage, versions de modèle, `ASREngine` |
| `src/audio/` | décodage des fichiers reçus, conversion en 16 kHz mono float32 |
| `src/runtime/` | choix du périphérique et repli CPU |
| `backend/app/` | API FastAPI : routes, schémas, exécution des transcriptions, sécurité, journaux |
| `frontend/` | interface Vue 3 + TypeScript, configuration Nginx |

## 9. Tests

```bash
.venv/bin/python -m pytest                      # 307 tests, environ 40 s
```

| Dossier | Tests | Contenu |
|---|---|---|
| `tests/test_*.py` | 122 | recherche : `src/augment/`, `split_musan.py`, `build_val_set.py` |
| `tests/unit/` | 111 | décodage audio (formats des navigateurs générés en mémoire, fichiers corrompus, stéréo, rééchantillonnage, limites), choix du périphérique (CUDA absent, indisponible, architecture non prise en charge, erreur d'initialisation, repli), versions de modèle, moteur sur un mini-modèle |
| `tests/regression/` | 33 | non-régression avec le notebook (voir ci-dessous) |
| `tests/api/` | 41 | API : codes d'erreur, taille, sécurité, garde de concurrence, délais, requêtes successives, modèle réel |

Les tests marqués `gpu` sont ignorés sans GPU utilisable, et ceux qui exigent le modèle sans
`models/v1`.

**Non-régression.** Les tests de `tests/regression/` exécutent les cellules du notebook lui-même,
extraites par AST, sans les corpus. Ils exigent du moteur, sur les mêmes fichiers, les mêmes
résultats :

- **sur CPU, à l'identique** : prétraitement, CMVN, poids, features, log-probabilités CTC, tokens du
  beam search et texte ;
- **sur GPU** : la même transcription que le notebook sur GPU ;
- **entre CPU et GPU** : des tolérances adaptées au bfloat16.

Un autre test compare le code de `src/asr/` à celui du notebook, définition par définition : toute
divergence le fait échouer. Après un changement voulu du modèle, regénérez la référence avec
`python tests/regression/make_golden.py`.

Frontend, sans Node.js local, puis parcours complet dans un navigateur, application démarrée :

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/frontend:/app" -w /app \
    node:26.13.1-alpine3.24 npm test          # 22 tests Vitest
tests/e2e/run.sh
```

Le parcours complet (Chromium piloté par Playwright) couvre l'import d'un fichier, l'enregistrement au
micro (micro simulé par Chromium), l'historique, la vue Système, un fichier illisible et l'affichage
mobile. Les captures d'écran sont écrites dans `tests/e2e/screenshots/`.

## 10. Problèmes fréquents

| Symptôme | Cause et solution |
|---|---|
| `Modèle v1 absent de ./models` | Le modèle n'a pas été exporté : voir [Préparer le modèle](#2-préparer-le-modèle). |
| `NVIDIA Container Toolkit absent`, ou démarrage sur CPU malgré un GPU | Installez et configurez le toolkit, puis redémarrez Docker. Testez avec `docker run --rm --gpus all python:3.14.10-slim-trixie nvidia-smi -L`. |
| Vue Système : « CUDA indisponible » ou « non pris en charge » en mode GPU | Pilote antérieur à 580, ou GPU antérieur à Turing. Vérifiez la version CUDA affichée par `nvidia-smi`. Le CPU prend le relais. |
| Le micro ne démarre pas | Ouvrez l'application sur `http://localhost:8080`, et non sur une adresse IP : le navigateur exige un contexte sécurisé. Autorisez ensuite le micro pour ce site. |
| Erreur 421, ou `INVALID_HOST` | L'application est ouverte sous un autre nom d'hôte : ajoutez-le à `ASR_ALLOWED_HOSTS`. |
| `Le port 8080 est déjà utilisé` | Choisissez un autre port avec `ASR_PORT` dans `.env`. |
| `AUDIO_TOO_LONG` | Le modèle n'est fiable que jusqu'à 30 s : découpez l'enregistrement. |
| `INFERENCE_TIMEOUT` sur une machine lente | Augmentez `ASR_INFERENCE_TIMEOUT_S`, ou passez au décodage CTC glouton (voir [Configuration](#7-configuration)). |
| `no space left on device` pendant la construction | L'image GPU pèse 9,3 Go, et la première construction laisse environ 10 Go de cache. `docker builder prune` libère ce cache. |
| Transcription incohérente | Le modèle ne connaît que l'anglais lu. Il se dégrade sur un audio très faible ou saturé (aucune normalisation de niveau à l'entraînement). |

---

# B. Projet de recherche

```
Audio 16 kHz
  │                          ◄── [train] vitesse, RIR, bruit MUSAN ou babble, sur la forme d'onde
Log-Mel (80 bandes, 25 ms)
  │                          ◄── [train] SpecAugment
Frontend CNN (2 Conv2d, sous-échantillonnage ×4)
  │
Encodeur Transformer (12 couches, d=256, QK-norm)
  ├── Tête CTC ──────────────────────┐
  └── Décodeur (6 couches) ──────────┤ ──► [train] loss = 0.3 CTC + 0.7 CE
                                     ▼
                      Beam search joint CTC / attention
                                     │
                      Correction d'erreurs (optionnelle)
                                     ▼
                               Transcription
```

## 11. Points saillants

- **Aucune fuite entre train, validation et test.** Les RIR simulées sont séparées par salle, MUSAN par
  fichier (80/20, stratifié par catégorie et source), le babble de validation n'utilise que des
  énoncés `dev`, et le test n'emploie que des bruits et des salles absents de l'entraînement.
  `speech/librivox` est exclu de MUSAN, car LibriSpeech provient lui aussi de LibriVox.
- **Validation figée et versionnée.** `build_val_set.py` écrit une fois pour toutes la dégradation
  exacte de chaque énoncé (RIR, fichiers, offsets, SNR) dans `data/manifests/val/val_v2.jsonl`. Une
  ablation sur la configuration d'entraînement ne change donc pas la validation, et les runs restent
  comparables.
- **Test reproductible sans écrire d'audio.** Chaque dégradation de test est tirée par un générateur
  dont la graine dérive de l'identifiant de l'énoncé : même résultat à chaque évaluation, et même
  segment de bruit à tous les SNR.
- **Beam search joint CTC / attention vectorisé.** Le score de préfixe CTC est réécrit comme une somme
  cumulée (`torch.logcumsumexp`) : plus de boucle sur les trames, et un batch entier d'énoncés est
  décodé d'un coup.
- **QK-norm contre la divergence.** Le premier entraînement complet a divergé à l'époque 19 : les
  logits d'attention avaient atteint 45 millions. Une RMSNorm par tête sur q et k les borne, pour 1 %
  de débit en moins. Le diagnostic chiffré est dans la section 8 du notebook.
- **Garde-fous d'entraînement.** Les pas au gradient non fini sont sautés et comptés, et l'entraînement
  s'arrête de lui-même sur une série de pas sautés ou une remontée du WER de validation.

## 12. Répartition des données

| | Train | Validation | Test |
|---|---|---|---|
| **Parole** | `train-clean-100`, `train-clean-360`, `train-other-500` (960 h) | `dev-clean`, `dev-other` | `test-clean`, `test-other` |
| **RIR** | OpenSLR 26 simulées, p = 0.3 | Salles simulées réservées, figé | Réelles (OpenSLR 28, ACE), figé |
| **MUSAN** | 80 % des fichiers, `noise` et `music` | 20 % des fichiers, figé | — |
| **Babble** | 3 à 7 énoncés du train | 3 à 7 énoncés `dev`, figé | — |
| **SpecAugment** | 2 masques fréq. (F ≤ 27), 2 masques temps (T ≤ 40) | — | — |
| **WHAM!** | — | — | Split `tt`, SNR ∈ {−5, 0, 5, 10, 20} dB |
| **DEMAND** | — | — | SNR ∈ {−5, 0, 5, 10, 20} dB |

Le bruit d'entraînement s'applique avec p = 0.6 : noise (p = 0.4, SNR ~ U(0, 15) dB), music
(p = 0.3, U(5, 15) dB) ou babble (p = 0.3, U(13, 20) dB).

## 13. Installation et utilisation du notebook

### 13.1 Prérequis

- Python 3.14 (développé et testé avec 3.14.4) ;
- un GPU NVIDIA : l'entraînement a été mené sur une RTX 5060 Ti 16 Go, en bf16 ;
- environ 80 Go d'espace disque pour les corpus.

### 13.2 Environnement Python

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

L'index PyTorch fournit la version CUDA 13.0 de `torch`. Adaptez-le à votre pilote (voir
[pytorch.org](https://pytorch.org/get-started/locally/)).

### 13.3 Données

Les scripts reprennent un téléchargement interrompu et ignorent ce qui est déjà prêt : ils peuvent être
relancés sans risque.

```bash
python download_librispeech.py   # LibriSpeech, 7 sous-ensembles (environ 60 Go d'archives)
python download_noise.py         # MUSAN, RIR simulées et réelles, ACE, WHAM! (tt), DEMAND
```

Le partage MUSAN (`data/manifests/musan/`) et le jeu de validation figé (`data/manifests/val/`) sont
versionnés : inutile de relancer `split_musan.py` et `build_val_set.py`. Ces scripts restent fournis
pour les régénérer ou créer une nouvelle version de la validation.

### 13.4 Notebook

Ouvrez `robust_asr.ipynb` avec le noyau de `.venv` et exécutez les cellules dans l'ordre :

| Section | Contenu |
|---|---|
| 1 à 7 | Configuration, tokenizer, manifestes, banques de bruit, dégradations, datasets, Log-Mel |
| 8 | Modèle, avec l'analyse de la divergence et de la QK-norm |
| 9 | Beam search joint CTC / attention |
| 10 | Correction d'erreurs optionnelle (`Config.use_corrector`) |
| 11 | Entraînement : environ 22 min par époque, 22 h pour 60 époques |
| 12 | Test sur toutes les conditions (environ 2 h 30) |
| 13 | Transcription d'un fichier audio |

Relancer `fit()` reprend à la dernière époque de `checkpoints/last.pt`. Le code d'inférence du notebook
(sections 2, 7, 8, 9 et 10) est repris tel quel dans `src/asr/` pour l'application : un test vérifie
que les deux restent identiques.

### 13.5 Tests

```bash
.venv/bin/python -m pytest tests/test_*.py
```

Les 122 tests de recherche couvrent `src/augment/` (tirage du bruit, babble, jeu de validation),
`split_musan.py` et `build_val_set.py`. Ils s'exécutent en quelques secondes et ne demandent aucun
corpus.

## 14. Résultats

WER du meilleur modèle sur la validation mélangée (`val_v2`, CTC glouton, sélection du modèle) :
**21.29 %**.

Les WER de test, par sous-ensemble et par condition, avec le beam search joint, sont produits par la
section 12 du notebook dans `checkpoints/test_results.md`.

## 15. Arborescence

```
.
├── robust_asr.ipynb          pipeline complet : modèle, entraînement, décodage, test
├── download_librispeech.py   téléchargement et vérification SHA-256 de LibriSpeech
├── download_noise.py         bruits et RIR, rééchantillonnés en 16 kHz
├── split_musan.py            partage train/validation de MUSAN
├── build_val_set.py          génération du jeu de validation figé
├── src/
│   ├── augment/              tirage du bruit, babble, relecture du jeu de validation
│   ├── asr/                  moteur d'inférence extrait du notebook (application)
│   ├── audio/                décodage et conversion de l'audio reçu (application)
│   └── runtime/              choix du périphérique CPU / GPU (application)
├── backend/                  API FastAPI, Dockerfile, dépendances CPU et GPU
├── frontend/                 interface Vue 3, configuration Nginx, Dockerfile
├── models/                   versions de modèle pour l'application (poids non versionnés)
├── scripts/                  export d'un checkpoint en version de modèle, transcription en ligne de commande
├── docker/start.sh           lancement avec détection du GPU
├── docker-compose.yml        configuration CPU ; docker-compose.gpu.yml : surcharge GPU
├── docs/ARCHITECTURE.md      audit, architecture et choix de l'application
├── tests/                    tests de recherche, unitaires, de non-régression, d'API, de bout en bout
├── data/manifests/
│   ├── musan/                partage MUSAN (versionné)
│   └── val/                  jeu de validation val_v2 (versionné)
├── checkpoints/              modèles d'entraînement (non versionnés) et résultats de test
├── requirements.txt          environnement de recherche
└── requirements-dev.txt      outils de test de l'application
```

## 16. Données et licences

Aucun corpus audio n'est redistribué dans ce dépôt : les scripts les téléchargent depuis leurs sources
officielles (OpenSLR, Zenodo, WHAM!). Seuls cinq énoncés de test de LibriSpeech servent de fixtures aux
tests de l'application (`tests/fixtures/librispeech/`, licence CC BY 4.0). Chaque corpus reste soumis à
sa propre licence, à consulter avant tout usage, en particulier commercial :

| Corpus | Source |
|---|---|
| LibriSpeech | [OpenSLR 12](https://www.openslr.org/12) |
| MUSAN | [OpenSLR 17](https://www.openslr.org/17) |
| RIR simulées et réelles | [OpenSLR 26](https://www.openslr.org/26), [OpenSLR 28](https://www.openslr.org/28) |
| ACE | [Zenodo 6257551](https://zenodo.org/records/6257551) |
| WHAM! | [wham.whisper.ai](http://wham.whisper.ai/) |
| DEMAND | [Zenodo 1227121](https://zenodo.org/records/1227121) |
