"""Écrit golden_v1.json : les transcriptions du code du notebook sur les fixtures, sur CPU et sur GPU.

Ces textes sont produits par le notebook lui-même (via notebook_reference), jamais par le moteur :
ils fixent la référence que le moteur doit reproduire, même sans le notebook ni le dépôt de recherche.

    python tests/regression/make_golden.py
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from notebook_reference import build_model, load_namespace  # noqa: E402
from src.asr.model_store import load_weights, read_spec  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "librispeech"
GOLDEN = Path(__file__).parent / "golden_v1.json"


def main() -> None:
    spec = read_spec(ROOT / "models" / "v1")
    state_dict = load_weights(spec).state_dict
    references = json.loads((FIXTURES / "references.json").read_text())
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    golden = {
        "model_version": spec.version,
        "checkpoint_sha256": spec.sha256["checkpoint"],
        "torch": torch.__version__,
        "utterances": {item["id"]: {"reference": item["text"]} for item in references},
    }
    for name in devices:
        torch.backends.cudnn.benchmark = False
        namespace = load_namespace(torch.device(name))
        build_model(namespace, state_dict)
        for item in references:
            golden["utterances"][item["id"]][name] = namespace["transcribe"](str(FIXTURES / item["file"]))
    GOLDEN.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n")
    print(f"{GOLDEN} écrit ({', '.join(devices)})")


if __name__ == "__main__":
    main()
