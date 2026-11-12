# Robust-ASR : Transformer hybride CTC / attention sur LibriSpeech

Reconnaissance vocale robuste au bruit et à la réverbération, de la forme d'onde à la transcription.
Un Transformer encodeur-décodeur, entraîné de zéro sur les 960 h de LibriSpeech avec des dégradations
réalistes (RIR, bruit, musique, babble), puis évalué sur des conditions jamais vues à l'entraînement :
RIR réelles, bruits WHAM! et DEMAND de −5 à 20 dB.

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

Tout le pipeline (données, modèle, entraînement, décodage, test) est dans le notebook
[`robust_asr.ipynb`](robust_asr.ipynb), qui documente aussi chaque choix et les mesures qui l'ont motivé.

## Sommaire

1. [Points saillants](#1-points-saillants)
2. [Répartition des données](#2-répartition-des-données)
3. [Installation](#3-installation)
4. [Utilisation](#4-utilisation)
5. [Résultats](#5-résultats)
6. [Arborescence](#6-arborescence)
7. [Données et licences](#7-données-et-licences)

---

## 1. Points saillants

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

## 2. Répartition des données

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

## 3. Installation

### 3.1 Prérequis

- Python 3.14 (développé et testé avec 3.14.4) ;
- un GPU NVIDIA : l'entraînement a été mené sur une RTX 5060 Ti 16 Go, en bf16 ;
- environ 80 Go d'espace disque pour les corpus.

### 3.2 Environnement Python

```bash
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
```

L'index PyTorch fournit la version CUDA 13.0 de `torch`. Adaptez-le à votre pilote (voir
[pytorch.org](https://pytorch.org/get-started/locally/)).

### 3.3 Données

Les scripts reprennent un téléchargement interrompu et ignorent ce qui est déjà prêt : ils peuvent être
relancés sans risque.

```bash
python download_librispeech.py   # LibriSpeech, 7 sous-ensembles (environ 60 Go d'archives)
python download_noise.py         # MUSAN, RIR simulées et réelles, ACE, WHAM! (tt), DEMAND
```

Le partage MUSAN (`data/manifests/musan/`) et le jeu de validation figé (`data/manifests/val/`) sont
versionnés : inutile de relancer `split_musan.py` et `build_val_set.py`. Ces scripts restent fournis
pour les régénérer ou créer une nouvelle version de la validation.

## 4. Utilisation

### 4.1 Notebook

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

Relancer `fit()` reprend à la dernière époque de `checkpoints/last.pt`.

### 4.2 Tests

```bash
.venv/bin/python -m pytest
```

Les 122 tests couvrent `src/augment/` (tirage du bruit, babble, jeu de validation), `split_musan.py`
et `build_val_set.py`. Ils s'exécutent en quelques secondes et ne demandent aucun corpus.

## 5. Résultats

WER du meilleur modèle sur la validation mélangée (`val_v2`, CTC glouton, sélection du modèle) :
**21.29 %**.

Les WER de test, par sous-ensemble et par condition, avec le beam search joint, sont produits par la
section 12 du notebook dans [`checkpoints/test_results.md`](checkpoints/test_results.md).

## 6. Arborescence

```
.
├── robust_asr.ipynb          pipeline complet : modèle, entraînement, décodage, test
├── download_librispeech.py   téléchargement et vérification SHA-256 de LibriSpeech
├── download_noise.py         bruits et RIR, rééchantillonnés en 16 kHz
├── split_musan.py            partage train/validation de MUSAN
├── build_val_set.py          génération du jeu de validation figé
├── src/augment/
│   ├── noise_mix.py          tirage du bruit d'entraînement en deux étages
│   ├── babble.py             bruit de babble à partir de LibriSpeech
│   └── val_set.py            relecture du jeu de validation et résumé de la grille
├── tests/                    tests pytest
├── data/manifests/
│   ├── musan/                partage MUSAN (versionné)
│   └── val/                  jeu de validation val_v2 (versionné)
├── checkpoints/              modèles (non versionnés) et résultats de test
└── requirements.txt
```

## 7. Données et licences

Aucun corpus audio n'est redistribué dans ce dépôt : les scripts les téléchargent depuis leurs sources
officielles (OpenSLR, Zenodo, WHAM!). Chacun reste soumis à sa propre licence, à consulter avant tout
usage, en particulier commercial :

| Corpus | Source |
|---|---|
| LibriSpeech | [OpenSLR 12](https://www.openslr.org/12) |
| MUSAN | [OpenSLR 17](https://www.openslr.org/17) |
| RIR simulées et réelles | [OpenSLR 26](https://www.openslr.org/26), [OpenSLR 28](https://www.openslr.org/28) |
| ACE | [Zenodo 6257551](https://zenodo.org/records/6257551) |
| WHAM! | [wham.whisper.ai](http://wham.whisper.ai/) |
| DEMAND | [Zenodo 1227121](https://zenodo.org/records/1227121) |
