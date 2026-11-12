"""Référence de non-régression : le code du notebook lui-même, exécuté sans les corpus.

Le notebook ne peut pas servir tel quel : ses cellules construisent les manifestes, les banques de
bruit et lancent l'entraînement. On n'en exécute donc que les définitions, extraites par AST :
imports, classes, fonctions, et les affectations de `device`, `cfg` et `tok`. Les fonctions du
notebook (`transcribe`, `joint_beam_search`, `RobustASR`…) tournent alors exactement comme dans le
notebook, avec leurs variables globales, et le moteur extrait est comparé à elles.
"""

import ast
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "robust_asr.ipynb"
KEPT_ASSIGNMENTS = {"device", "cfg", "tok"}


def notebook_definitions(path: Path = NOTEBOOK) -> list[ast.stmt]:
    """Instructions de définition de toutes les cellules de code, dans l'ordre du notebook."""
    statements = []
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                statements.append(node)
            elif isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) and target.id in KEPT_ASSIGNMENTS for target in node.targets
            ):
                statements.append(node)
    return statements


def load_namespace(device: torch.device, path: Path = NOTEBOOK) -> dict:
    """Espace de noms du notebook, avec `device` imposé et sans correcteur (Config.use_corrector=False)."""
    namespace = {"__name__": "notebook_reference"}
    for node in notebook_definitions(path):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    namespace["device"] = device
    namespace["corrector"] = None
    return namespace


def build_model(namespace: dict, state_dict: dict) -> torch.nn.Module:
    """Le modèle du notebook, chargé comme dans run_test (cellule 29), sur namespace['device']."""
    model = namespace["RobustASR"](namespace["cfg"], namespace["tok"]).to(namespace["device"])
    model.load_state_dict(state_dict)
    model.eval()
    namespace["model"] = model
    return model
