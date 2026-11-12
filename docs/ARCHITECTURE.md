# Architecture de l'application web Robust-ASR

Ce document décrit la transformation du projet de recherche Robust-ASR en application web locale :
une interface Vue.js, une API FastAPI et un moteur ASR Python autonome, le tout dockerisé. Il réunit
l'audit du dépôt de recherche, l'architecture retenue et les risques identifiés.

Principe directeur : **un monolithe modulaire local**, avec un frontend indépendant et un backend qui
encapsule un moteur ASR utilisable sans HTTP. L'application sert un seul utilisateur par machine,
sur `localhost` : aucune infrastructure multi-utilisateurs, ni Redis, ni Celery, ni Kubernetes.

## 1. Décisions validées (12 novembre 2026)

| Sujet | Décision |
|---|---|
| Dépôt | Clone du dépôt de recherche (`git clone --no-hardlinks`), branche `webapp`. Le notebook, les scripts, `src/augment/` et les tests restent à leur place, inchangés : les déplacer casserait `Path("data")`, `from src.augment …`, `import build_val_set` et `from download_noise import download`. Le remote `recherche` pointe sur le dépôt d'origine, en lecture seule (`push` désactivé). |
| Audio long | Limite stricte de 30 s dans le MVP. Une segmentation sur les silences pourra lever cette limite plus tard, sans changer l'API. |
| Stéréo | Moyenne des canaux. Pour un signal mono, ou un stéréo aux canaux identiques comme celui d'un micro de navigateur, le résultat est identique au premier canal lu par `load_audio`. |
| Checkpoint | Export allégé et vérifié bit à bit (environ 102 Mo au lieu de 306 Mo, sans l'état de l'optimiseur). Le chargeur accepte aussi le checkpoint d'entraînement complet. |

## 2. Audit du dépôt de recherche

### 2.1 Modèle

`RobustASR` (cellule 17 du notebook), 25,5 M de paramètres :

```
forme d'onde 16 kHz
  → LogMel : 80 bandes, n_fft 512, fenêtre 400, pas 160, échelle HTK, puissance 2, log(max(x, 1e-10)), CMVN
  → SpecAugment (identité en évaluation)
  → ConvFrontend : 2 Conv2d de pas 2 (temps et fréquence ÷ 4), Linear 640 → 256, LayerNorm
  → encodage positionnel sinusoïdal (6 000 positions au plus)
  → encodeur : 12 couches pré-LN, d = 256, 4 têtes, FFN 2 048, QK-norm (RMSNorm par tête)
  → tête CTC (30 symboles) et décodeur d'attention de 6 couches, avec cache incrémental
```

Le tokenizer travaille par caractère : blank (0), espace, apostrophe, A à Z, et `<sos>`/`<eos>` (29).
La sortie est donc **en anglais, en majuscules, sans ponctuation ni chiffres**.

### 2.2 Checkpoints et CMVN

- `checkpoints/best.pt` et `last.pt` ont les mêmes poids : époque d'indice 59, WER_sel de 21,29 % sur
  `val_v2` en CTC glouton. Le fichier (306 Mo) contient `model`, `optimizer`, `scheduler`, `epoch`,
  `best_wer`, `val_set`, `val_set_sha256` et `history`. Les poids seuls, tous en float32, font
  environ 102 Mo.
- `torch.load(weights_only=True)` suffit : le checkpoint ne contient que des tenseurs et des types simples.
- **Les hyperparamètres ne sont pas dans le checkpoint**, seulement dans le `Config` du notebook. D'où le
  `config.yaml` de chaque version de modèle.
- **La CMVN fait partie du `state_dict`** (`logmel.mean`, `logmel.std`) : elle a été copiée dans ces
  buffers avant l'entraînement. `data/manifests/cmvn.pt` lui est identique bit à bit. La fenêtre de Hann
  et le banc de filtres Mel, recalculés par torchaudio 2.11.1, le sont aussi.
- `checkpoints/run1_diverged/` utilise l'ancienne architecture (`in_proj_weight`, sans QK-norm) : il ne
  se charge pas sans conversion et n'est pas exposé comme version.

### 2.3 Prétraitement et décodage dans le notebook

- `load_audio` lit le premier canal en float32 avec soundfile. Il ne rééchantillonne pas
  (`transcribe` suppose du 16 kHz) et n'applique aucune normalisation (ni gain, ni composante
  continue, ni pré-accentuation).
- Le seul rééchantillonnage du projet est `torchaudio.functional.resample` avec ses paramètres par
  défaut (`convert_to_16k_mono` dans `download_noise.py`) : c'est la convention reprise ici.
- La STFT (`center=True`, padding `reflect`) **échoue en dessous de 257 échantillons**, soit 16 ms.
- `joint_beam_search` : beam de 10, poids CTC 0,3, score de préfixe CTC vectorisé (`logcumsumexp`),
  encodeur et décodeur sous autocast bf16 sur CUDA, scores CTC en float64, au plus `n_frames + 1` pas.
- `ctc_greedy` sert à la validation. `WordCorrector` est désactivé (`use_corrector=False`) et
  demanderait le vocabulaire des transcriptions d'entraînement.

### 2.4 Dépendances internes

`autocast()` lit la variable globale `device`, `ctc_greedy` lit `tok`, et `transcribe` lit `model`,
`device`, `tok` et `corrector`. Les cellules 7, 9 et 11 exigent les corpus (manifestes, banques de
bruit, MUSAN) : **le notebook ne peut pas transcrire sans les 80 Go de données.** Le chemin
d'inférence n'a besoin que de torch, torchaudio, soundfile et numpy.

### 2.5 Mesures

Mesures faites en exécutant le code exact des cellules (extrait par AST), sur une RTX 5060 Ti et sur
6 threads CPU :

| | GPU, bf16 | CPU, fp32 |
|---|---|---|
| Temps de calcul / durée audio (12 énoncés de test, 82 s) | 0,044 | 0,19 |
| Énoncé de 10 s | 0,43 s | 2,4 s |
| Pire cas (environ 34 s d'audio) | 2,3 s | 30 s |

Les transcriptions CPU et GPU sont identiques sur 12 énoncés sur 12.

| Durée de l'audio | WER |
|---|---|
| 25 à 30 s, 28 énoncés naturels | 11,7 % |
| 30 à 35 s, 16 énoncés naturels | 18,0 % |
| 50 s, énoncés concaténés | 60 % |
| 77 s, énoncés concaténés | 103 % |
| 240 s ou plus | échec : l'encodage positionnel est limité à 6 000 positions |

Le modèle n'est fiable que jusqu'à environ 30 s, la durée maximale vue à l'entraînement (29,7 s).

## 3. Arborescence

```text
Robust-ASR-WebApp/
├── robust_asr.ipynb, download_librispeech.py, download_noise.py,
│   split_musan.py, build_val_set.py, test_cuda.py        recherche, inchangés
├── requirements.txt, pytest.ini                          environnement de recherche, inchangés
├── requirements-dev.txt                                  outils de test de l'application
├── src/
│   ├── augment/                 inchangé
│   ├── asr/                     moteur extrait du notebook
│   │   ├── config.py            ModelConfig : sous-ensemble du Config du notebook
│   │   ├── tokenizer.py         CharTokenizer (cellule 5)
│   │   ├── features.py          make_pad_mask, LogMel, SpecAugment (cellules 3 et 15)
│   │   ├── model.py             autocast, RobustASR et ses blocs (cellules 3 et 17)
│   │   ├── decoding.py          joint_beam_search, ctc_greedy (cellule 21)
│   │   ├── corrector.py         WordCorrector (cellule 23), désactivé par défaut
│   │   ├── model_store.py       versions présentes dans models/, config.yaml, intégrité
│   │   └── engine.py            ASREngine : chargement unique, verrou, transcription
│   ├── audio/
│   │   ├── errors.py            erreurs audio typées, avec un code stable
│   │   ├── decoding.py          octets ou fichier → PCM float32 (soundfile, puis PyAV)
│   │   └── preprocessing.py     mono, 16 kHz, validations
│   └── runtime/
│       └── device_manager.py    détection CUDA/CPU, sonde, fallback
├── backend/
│   ├── app/
│   │   ├── __main__.py          lancement : python -m backend.app
│   │   ├── main.py              create_app(), lifespan
│   │   ├── api/routes/          transcribe, system, health, models
│   │   ├── api/dependencies.py
│   │   ├── schemas/             modèles Pydantic des réponses
│   │   ├── services/            EngineRuntime : chargement, exécuteur, garde, timeout
│   │   └── core/                réglages, erreurs, logs JSON, middlewares
│   ├── Dockerfile
│   └── requirements-base.txt, requirements-cpu.txt, requirements-gpu.txt
├── frontend/                    Vue 3, TypeScript, Vite, configuration Nginx, Dockerfile
├── models/v1/                   best.pt, cmvn.pt (non versionnés), config.yaml
├── scripts/
│   ├── export_model.py          checkpoint de recherche → models/<version>/, vérifié
│   └── transcribe.py            transcription en ligne de commande, sans FastAPI
├── tests/
│   ├── test_*.py                tests de recherche, inchangés
│   ├── fixtures/                quelques énoncés LibriSpeech (CC BY 4.0)
│   ├── unit/                    audio, runtime, configuration, moteur
│   ├── regression/              moteur comparé au code du notebook
│   └── api/                     FastAPI, moteur factice et moteur réel
├── docker/start.sh
├── docker-compose.yml, docker-compose.gpu.yml, .dockerignore, .env.example
└── docs/ARCHITECTURE.md
```

Il n'y a pas de dossier `config/` : les réglages de l'application passent par des variables
d'environnement, et la configuration d'inférence vit dans `models/<version>/config.yaml`.

## 4. Moteur ASR (`src/asr`, `src/audio`, `src/runtime`)

### 4.1 Règle d'extraction

Le code du notebook est repris **à l'identique**, docstrings et commentaires compris. Seules trois
retouches lèvent les dépendances à l'état global :

1. `autocast()` devient `autocast(device_type)`, déduit du tenseur d'entrée. Le comportement ne change
   pas : bf16 sur CUDA, désactivé ailleurs.
2. `ctc_greedy(…, blank=0)` reçoit l'indice du blank au lieu de lire le `tok` global.
3. `transcribe(path)` devient `ASREngine.transcribe_file`.

`RobustASR.forward`, la loss d'entraînement, est conservé : le module reste utilisable pour entraîner.

Le notebook ne peut pas être modifié, donc son code et `src/asr` coexistent. Pour éviter toute dérive,
le test de non-régression exécute les cellules du notebook lui-même et exige l'égalité bit à bit avec
le moteur sur CPU. Le notebook pourra importer `src.asr` le jour où cela sera décidé.

### 4.2 Interfaces

```python
# src/runtime/device_manager.py
select_device(requested: "auto" | "cuda" | "cpu" = "auto") -> DeviceInfo
place_model(model, info, check) -> DeviceInfo   # en cas d'échec CUDA : retour du même objet sur CPU

# src/audio
decode_audio(source: bytes | Path, *, max_duration_s=None) -> DecodedAudio    # (trames, canaux), fréquence d'origine
prepare_waveform(audio, *, min_duration_s, max_duration_s) -> torch.Tensor     # (n,), 16 kHz, mono, float32

# src/asr/engine.py
class ASREngine:
    @classmethod
    def load(cls, model_dir, device="auto", *, max_duration_s=None, num_threads=None) -> "ASREngine"
    def transcribe(self, waveform, sample_rate=16_000) -> Transcription   # protégé par un verrou
    def transcribe_bytes(self, data: bytes) -> Transcription
    def transcribe_file(self, path) -> Transcription
```

Utilisation sans FastAPI :

```python
from src.asr.engine import ASREngine

engine = ASREngine.load("models/v1")
print(engine.transcribe_file("enregistrement.webm").text)
```

### 4.3 Versions de modèle

`scripts/export_model.py` lit le checkpoint et la CMVN de recherche, sans les modifier, puis écrit
`models/<version>/` :

- `best.pt` : `{"model": state_dict, …}` avec la provenance, sans l'optimiseur ni l'historique ;
- `cmvn.pt` : copie à l'octet près ;
- `config.yaml` : hyperparamètres (noms du `Config` du notebook), tokenizer, décodage, limites,
  provenance et empreintes SHA-256.

Le script recharge ensuite la version exportée et compare chaque tenseur à la source. Il n'écrase
jamais une version existante.

Au chargement, le moteur vérifie les empreintes, charge les poids avec `strict=True` et exige que
`cmvn.pt` soit égal à la CMVN du checkpoint. Le checkpoint fait foi ; en cas d'écart, le chargement est
refusé avec un message clair. La variable `ASR_MODEL_VERSION` choisit la version chargée au démarrage.

### 4.4 Chaîne audio

1. **Décodage** : soundfile d'abord (WAV, FLAC, OGG Vorbis ou Opus, MP3), le même lecteur que
   `load_audio`, donc des échantillons identiques à ceux du notebook. Sinon **PyAV**, qui embarque
   FFmpeg dans sa roue, pour WebM/Opus (Chrome, Firefox) et MP4/AAC (Safari). Le décodage s'arrête
   au-delà de la durée maximale, ce qui protège des fichiers qui annoncent ou contiennent des heures
   d'audio.
2. **Mono** : moyenne des canaux.
3. **16 kHz** : `torchaudio.functional.resample`, omis si l'entrée est déjà à 16 kHz.
4. **Validations** : fichier vide, valeurs non finies, durée de 0,1 à 30 s.
5. **Aucune normalisation**, comme à l'entraînement.

### 4.5 Choix du périphérique

1. `ASR_DEVICE=cpu` : CPU.
2. `torch.version.cuda` vaut `None` (build CPU) : CPU.
3. `torch.cuda.is_available()` est faux (pas de pilote, ou GPU non exposé au conteneur) : CPU.
4. L'architecture du GPU (`sm_XY`) n'est pas couverte par `torch.cuda.get_arch_list()`, ni en binaire
   ni en PTX : CPU.
5. Sinon, une sonde exécute un petit produit matriciel bf16 et synchronise ; si elle échoue : CPU.

Le checkpoint est lu une seule fois, sur CPU. Le modèle est ensuite déplacé sur le GPU et validé par
une transcription de chauffe. En cas d'échec, le même objet revient sur CPU, sans relire le disque, et
la raison est exposée par `/api/system`. `ASR_DEVICE=cuda` sans GPU utilisable lève une erreur
explicite plutôt qu'un fallback silencieux.

## 5. Backend FastAPI

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/api/transcribe` | POST | multipart, champ `file` : transcription |
| `/api/system` | GET | périphérique, versions, limites, formats |
| `/api/health` | GET | 200 si le modèle est prêt, 503 s'il charge ou a échoué (avec la raison) |
| `/api/models` | GET | versions présentes dans `models/`, version chargée |

Réponse de transcription :

```json
{
  "transcription": "HE HOPED THERE WOULD BE STEW FOR DINNER",
  "processing_time": 0.47,
  "inference_time": 0.43,
  "preprocessing_time": 0.04,
  "audio_duration": 10.4,
  "device": "cuda",
  "model_version": "v1",
  "sample_rate": 16000,
  "input_format": "FLAC",
  "request_id": "3f2c9a1b7e4d4c10"
}
```

Toutes les erreurs ont la forme `{"error": {"code", "message", "request_id"}}`, sans trace Python :

| Code | HTTP | Cas |
|---|---|---|
| `INVALID_REQUEST` | 400 | requête mal formée, champ `file` absent |
| `EMPTY_FILE` | 400 | fichier de 0 octet |
| `FORBIDDEN_ORIGIN`, `INVALID_HOST` | 403, 400 | requête d'un autre site, en-tête `Host` inconnu |
| `FILE_TOO_LARGE` | 413 | plus de `ASR_MAX_UPLOAD_MB` (25 Mo par défaut) |
| `INVALID_AUDIO`, `EMPTY_AUDIO` | 422 | format non pris en charge, fichier corrompu, aucun échantillon |
| `AUDIO_TOO_SHORT`, `AUDIO_TOO_LONG` | 422 | hors de 0,1 à 30 s |
| `ENGINE_BUSY` | 409 | une transcription est déjà en cours |
| `MODEL_NOT_READY` | 503 | modèle en chargement ou en erreur |
| `INFERENCE_TIMEOUT` | 504 | plus de `ASR_INFERENCE_TIMEOUT_S` (120 s par défaut) |
| `INTERNAL_ERROR` | 500 | erreur inattendue : message générique, trace dans les logs seulement |

**Exécution.**

- Un seul processus Uvicorn.
- Le moteur est chargé une fois, en arrière-plan, au démarrage : `/api/health` répond « loading »
  pendant ce temps.
- Le décodage et l'inférence tournent dans un `ThreadPoolExecutor` à un seul thread : la boucle ASGI
  n'est jamais bloquée.
- Une garde renvoie immédiatement 409 si une transcription est en cours ; le moteur a en plus son
  propre verrou.
- Au-delà du timeout, l'API répond 504. Le calcul en cours ne peut pas être interrompu : la garde reste
  donc prise jusqu'à sa fin réelle, et deux inférences ne tournent jamais en parallèle.

**Sécurité.**

- En-têtes `Host` limités à `ASR_ALLOWED_HOSTS`, contre le DNS rebinding.
- Les POST portant `Sec-Fetch-Site: cross-site` ou `same-site`, ou un `Origin` étranger, sont refusés :
  un site ouvert dans le navigateur ne peut pas utiliser l'API. curl reste utilisable.
- Pas de middleware CORS, inutile puisque tout passe par la même origine.
- Taille des requêtes bornée par le backend lui-même, en plus de Nginx.
- L'audio est traité en mémoire et n'est jamais conservé. Pour le garder, l'utilisateur le télécharge
  depuis le navigateur.
- Logs JSON sur la sortie standard (`request_id`, durées, périphérique), **sans le texte transcrit**.

## 6. Frontend Vue.js

Vue 3, TypeScript et Vite, sans bibliothèque de composants, avec trois vues :

- **Transcription** : enregistrement avec MediaRecorder (type MIME négocié selon le navigateur, arrêt
  automatique avant la limite, avec un compte à rebours), import par glisser-déposer, lecteur, envoi,
  indicateur de traitement, résultat copiable, temps de traitement, durée, périphérique et version.
- **Historique** : texte seulement, stocké dans le navigateur (`localStorage`), effaçable.
- **Système** : état du modèle, périphérique, versions, limites, formats acceptés.

Le frontend ne contient aucune logique ASR. Il signale que le modèle est anglais et qu'il produit des
majuscules sans ponctuation.

## 7. Docker

- **`backend/Dockerfile`** : `python:3.14.10-slim-trixie`, en plusieurs étapes. `ARG TORCH_VARIANT=cpu|gpu`
  choisit `requirements-cpu.txt` ou `requirements-gpu.txt` : torch 2.14.1 et torchaudio 2.11.1, les
  versions de la recherche. Les roues CUDA embarquent le runtime ; seul le pilote vient de l'hôte. Les
  copies sont explicites (`src/asr`, `src/audio`, `src/runtime`, `backend/app`) : ni notebook, ni
  données, ni modèles dans l'image.
- **`frontend/Dockerfile`** : build Node, puis `nginx-unprivileged`, qui sert l'application compilée
  et relaie `/api/` vers le backend.
- **`docker-compose.yml`** (CPU) :
  - le backend est seul sur le réseau `internal` (`internal: true`, sans accès Internet) et ne publie
    aucun port ;
  - le frontend est publié sur `127.0.0.1:8080` ;
  - `models/` est monté en lecture seule ;
  - conteneurs non root, système de fichiers en lecture seule, `/tmp` en mémoire, `cap_drop: ALL`,
    `no-new-privileges`, health checks.
- **`docker-compose.gpu.yml`** : variante `gpu` de l'image (PyTorch CUDA 13.0) et réservation du GPU NVIDIA.
- **`docker/start.sh`** :
  1. vérifie Docker, son démon et Compose v2 ;
  2. vérifie le runtime NVIDIA ;
  3. teste `nvidia-smi` dans un conteneur `--gpus all` ;
  4. choisit la configuration, vérifie la présence du modèle, puis lance les conteneurs et attend
     qu'ils soient prêts.

  Le backend revérifie CUDA de lui-même, ce qui fait une double sécurité.

## 8. Tests

- **Audio** : WAV PCM16 et float, FLAC, OGG/Opus, MP3, WebM/Opus et M4A/AAC générés en mémoire ;
  stéréo ; 8, 44,1 et 48 kHz (longueur exacte, fréquence dominante préservée) ; fichier vide, tronqué,
  aléatoire ; valeurs NaN ; trop court ; trop long ; égalité avec `load_audio` du notebook.
- **Runtime** : CUDA absent du build, CUDA indisponible, architecture non couverte, sonde en échec,
  fallback sur le même objet (une seule lecture du checkpoint), `cuda` exigé sans GPU, exécution GPU
  réelle (marqueur `gpu`).
- **Non-régression** :
  - comparaison avec les cellules du notebook : features, CMVN, poids, log-probabilités CTC
    (`torch.equal` sur CPU), tokens du beam search, textes ;
  - fichier de référence versionné ;
  - CPU contre GPU, avec des tolérances adaptées au bf16.
- **API** : moteur factice (codes HTTP, taille, origine, garde 409, timeout 504, absence de trace,
  requêtes successives) et moteur réel (formats compressés, transcription).

## 9. Risques et compatibilité

| Risque | Traitement |
|---|---|
| Audio de plus de 30 s inexploitable, échec au-delà de 240 s | limite stricte ; segmentation en extension |
| Sortie anglaise en majuscules, sans ponctuation | renvoyée telle quelle, signalée dans l'interface |
| Les roues CUDA 13.0 exigent un pilote ≥ 580 | fallback CPU automatique ; autre version CUDA possible dans `requirements-gpu.txt` |
| GPU antérieurs à Turing (sm < 75) | détectés par `device_manager`, fallback CPU |
| Sur CPU, jusqu'à environ 30 s de calcul pour 30 s d'audio | indicateur de progression, timeouts de 120 s (API) et 180 s (Nginx) |
| Différences numériques bf16 / fp32 | mesurées nulles sur 12 énoncés ; tolérances dans les tests |
| Image GPU de plusieurs gigaoctets | inhérent aux bibliothèques CUDA |
| Pas de normalisation de gain à l'entraînement | contrôle automatique du gain du navigateur, aucune normalisation serveur |
| torchaudio en maintenance | usage limité à `features.py` et `preprocessing.py` |
| Micro inaccessible hors `localhost` en HTTP | documenté : le navigateur exige un contexte sécurisé |

## 10. Extensions prévues (hors MVP)

- **Mode d'évaluation de robustesse** : `ASREngine.transcribe(waveform)` accepte un signal déjà dégradé.
  `src/augment` et les fonctions `mix_at_snr` et `apply_rir` du notebook fourniront le bruit, la
  réverbération et le SNR. `/api/models` et `model_store` permettent de comparer plusieurs versions, et
  jiwer calculera le WER lorsque les références seront fournies.
- **Segmentation** des enregistrements longs sur les silences, en décodant les segments par lot dans
  `joint_beam_search`.
- **Historique côté serveur** avec SQLite, et conservation de l'audio sur demande explicite.
- **Décodage CTC glouton** (`decoding.method: ctc_greedy`), plus rapide sur CPU, déjà accepté par la
  configuration.

## 11. État de l'implémentation (12 novembre 2026)

Les phases 3 à 7 sont réalisées, chacune testée avant la suivante.

| Phase | Résultat |
|---|---|
| 3. Moteur | `src/asr`, `src/audio` et `src/runtime` extraits ; `models/v1` exporté (102 Mo, vérifié tenseur par tenseur) ; sur CPU, le moteur reproduit le notebook bit à bit (features, log-probabilités, tokens, texte) |
| 4. Backend | FastAPI avec les quatre endpoints, exécuteur à un thread, garde 409, timeout 504, erreurs JSON sans trace, logs JSON, protections Host, Origin et taille |
| 5. Frontend | Vue 3 avec les vues Transcription, Historique et Système ; enregistrement MediaRecorder, import, lecteur, résultats, erreurs |
| 6. Docker | images CPU (1,7 Go) et GPU (9,3 Go), Compose CPU et surcharge GPU, `docker/start.sh` ; les deux modes sont validés sur la RTX 5060 Ti, ainsi que le repli sur CPU de l'image GPU sans GPU |
| 7. Documentation | README (application et recherche), ce document |

**Tests.** 307 tests Python : 122 de recherche, inchangés ; 111 unitaires ; 33 de non-régression ;
41 d'API. S'y ajoutent 22 tests Vitest et un parcours de bout en bout dans Chromium, exécuté en mode CPU
puis en mode GPU. Ce parcours couvre l'import, l'enregistrement au micro simulé, l'historique, la vue
Système, l'erreur et l'affichage mobile.

**Versions retenues** : Python 3.14.10, torch 2.14.1 (+cpu ou +cu130), torchaudio 2.11.1, PyAV 19.0.2,
FastAPI 0.162.5, Uvicorn 0.54.0 ; Node 26.13.1, Vue 3.5.45, vue-router 5.5.0, Vite 8.3.8,
nginx-unprivileged 1.30.6.

**Écarts avec la proposition, et pourquoi :**

- **TypeScript 6.0.3, et non 7.0.** La version 7, native, n'expose plus l'API JavaScript dont dépend
  `vue-tsc` 3.3 (`typescript/lib/tsc`).
- **torchaudio 2.11.1 vient de PyPI dans l'image GPU.** L'index cu130 de PyTorch s'arrête à 2.11.0 ;
  c'est la roue de l'environnement de recherche.
- **`httpx2` pour le `TestClient`.** Starlette 1.x le préfère à `httpx`, désormais déprécié.
- **Une seule source de réglages.** Nginx dérive ses limites de taille et de délai des variables
  `ASR_*` du backend (`frontend/nginx/limits.envsh`).
