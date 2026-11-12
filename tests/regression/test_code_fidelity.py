"""Le code extrait est celui du notebook : arbres syntaxiques comparés, définition par définition.

Toute modification d'une classe ou d'une fonction, dans le notebook ou dans src/asr, fait échouer ce
test. Les trois retouches de l'extraction (autocast, blank de ctc_greedy) sont annulées par une
transformation explicite avant la comparaison : elles sont les seules différences admises.
"""

import ast
from pathlib import Path

import pytest
import torch

from notebook_reference import notebook_definitions

ROOT = Path(__file__).resolve().parents[2]
MODULES = ["tokenizer", "features", "model", "decoding", "corrector"]

VERBATIM = [
    "CharTokenizer",
    "make_pad_mask",
    "LogMel",
    "SpecAugment",
    "PositionalEncoding",
    "ConvFrontend",
    "allowed_keys",
    "rms_norm",
    "Attention",
    "feed_forward",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "AttentionDecoder",
    "cumulative_sum",
    "ctc_advance",
    "WordCorrector",
]
RETOUCHED = ["RobustASR", "joint_beam_search", "ctc_greedy"]


def definitions(nodes) -> dict[str, ast.AST]:
    return {node.name: node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


@pytest.fixture(scope="module")
def notebook() -> dict[str, ast.AST]:
    return definitions(notebook_definitions())


@pytest.fixture(scope="module")
def extracted() -> dict[str, ast.AST]:
    nodes = []
    for module in MODULES:
        nodes += ast.parse((ROOT / "src" / "asr" / f"{module}.py").read_text(encoding="utf-8")).body
    return definitions(nodes)


class UndoRetouches(ast.NodeTransformer):
    """Ramène le code extrait à la forme du notebook : autocast(), tok.blank, sans device_type."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "autocast":
            node.args, node.keywords = [], []
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        return None if targets == ["device_type"] else node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        if node.name == "ctc_greedy":
            node.args.args = [arg for arg in node.args.args if arg.arg != "blank"]
            node.args.defaults = []
            for child in ast.walk(node):
                if isinstance(child, ast.Compare):
                    child.comparators = [
                        ast.Attribute(value=ast.Name(id="tok", ctx=ast.Load()), attr="blank", ctx=ast.Load())
                        if isinstance(comparator, ast.Name) and comparator.id == "blank"
                        else comparator
                        for comparator in child.comparators
                    ]
        return node


def dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize("name", VERBATIM)
def test_definition_is_copied_verbatim(name: str, notebook: dict, extracted: dict) -> None:
    assert dump(extracted[name]) == dump(notebook[name]), f"{name} diffère du notebook"


@pytest.mark.parametrize("name", RETOUCHED)
def test_retouched_definition_differs_only_by_its_retouches(name: str, notebook: dict, extracted: dict) -> None:
    assert dump(extracted[name]) != dump(notebook[name]), f"{name} devait porter une retouche"
    undone = ast.fix_missing_locations(UndoRetouches().visit(extracted[name]))
    assert dump(undone) == dump(notebook[name]), f"{name} diffère du notebook au-delà des retouches prévues"


def test_every_inference_definition_of_the_notebook_is_checked(notebook: dict) -> None:
    for name in VERBATIM + RETOUCHED:
        assert name in notebook, f"{name} n'existe plus dans le notebook"


def test_autocast_keeps_the_notebook_behaviour() -> None:
    """bf16 sur CUDA, désactivé ailleurs, comme autocast() avec le device global du notebook."""
    from src.asr.model import autocast

    assert autocast("cpu").fast_dtype == torch.bfloat16 and not autocast("cpu")._enabled
    assert autocast("cuda").fast_dtype == torch.bfloat16 and autocast("cuda")._enabled
