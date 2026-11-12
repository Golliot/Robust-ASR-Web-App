"""Tests du choix du périphérique : chaque raison d'écarter CUDA, et le repli sur CPU sans rechargement."""

from types import SimpleNamespace

import pytest
import torch

from src.runtime import device_manager
from src.runtime.device_manager import (
    DeviceInfo,
    DeviceUnavailableError,
    arch_supported,
    cpu_info,
    place_model,
    probe_cuda,
    select_device,
)

BUILD_ARCHS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def fake_cuda_info(requested: str = "auto") -> DeviceInfo:
    return DeviceInfo(
        type="cuda",
        name="GPU factice",
        requested=requested,
        precision="bf16-autocast",
        torch_version=torch.__version__,
        cuda_version="13.0",
        cudnn_version=None,
        capability=(12, 0),
        total_memory_mb=16_000,
        num_threads=1,
    )


@pytest.fixture
def cuda_build(monkeypatch: pytest.MonkeyPatch):
    """Un build CUDA avec un GPU visible, quelle que soit la machine de test."""
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: BUILD_ARCHS)
    return monkeypatch


def set_gpu(monkeypatch: pytest.MonkeyPatch, major: int, minor: int) -> None:
    properties = SimpleNamespace(name=f"GPU sm_{major}{minor}", major=major, minor=minor, total_memory=8 * 2**30)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)


@pytest.mark.parametrize(
    ("capability", "archs", "supported"),
    [
        ((12, 0), BUILD_ARCHS, True),  # Blackwell grand public (RTX 50)
        ((8, 9), BUILD_ARCHS, True),  # Ada : binaire sm_86 compatible
        ((8, 6), ["sm_80"], True),
        ((8, 0), ["sm_86"], False),  # un binaire ne tourne pas sur une capacité inférieure
        ((6, 1), BUILD_ARCHS, False),  # Pascal : absent de ce build
        ((7, 0), BUILD_ARCHS, False),  # Volta
        ((12, 1), ["sm_90", "compute_90"], True),  # PTX compilé à la volée
        ((8, 6), ["compute_90"], False),
        ((9, 0), ["sm_90a"], True),  # variante spécifique : architecture exacte seulement
        ((10, 0), ["sm_90a"], False),
    ],
)
def test_architecture_support(capability: tuple[int, int], archs: list[str], supported: bool) -> None:
    assert arch_supported(capability, archs) is supported


def test_cpu_request_never_touches_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_manager, "probe_cuda", lambda: pytest.fail("CUDA ne doit pas être sondé"))
    info = select_device("cpu")
    assert info.type == "cpu"
    assert info.precision == "fp32"
    assert info.fallback_reason is None


def test_cpu_build_of_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", None)
    info = select_device("auto")
    assert info.type == "cpu"
    assert "sans CUDA" in info.fallback_reason
    with pytest.raises(DeviceUnavailableError, match="sans CUDA"):
        select_device("cuda")


def test_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = select_device("auto")
    assert info.type == "cpu"
    assert "CUDA indisponible" in info.fallback_reason


def test_no_visible_gpu(cuda_build: pytest.MonkeyPatch) -> None:
    cuda_build.setattr(torch.cuda, "device_count", lambda: 0)
    assert "Aucun GPU" in probe_cuda().reason


def test_unsupported_architecture(cuda_build: pytest.MonkeyPatch) -> None:
    set_gpu(cuda_build, 6, 1)
    probe = probe_cuda()
    assert not probe.usable
    assert "sm_61" in probe.reason and "n'est pas pris en charge" in probe.reason
    assert select_device("auto").fallback_reason == probe.reason


def test_cuda_initialization_error(cuda_build: pytest.MonkeyPatch) -> None:
    set_gpu(cuda_build, 12, 0)

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA error: initialization error")

    cuda_build.setattr(torch, "randn", broken)
    probe = probe_cuda()
    assert not probe.usable
    assert "initialization error" in probe.reason
    info = select_device("auto")
    assert info.type == "cpu" and "Échec du calcul de test" in info.fallback_reason


def test_unknown_request_is_refused() -> None:
    with pytest.raises(ValueError, match="Périphérique inconnu"):
        select_device("tpu")


def test_place_model_on_cpu() -> None:
    model, calls = torch.nn.Linear(4, 2), []
    info = place_model(model, cpu_info("cpu"), lambda m, device: calls.append(device.type))
    assert info.type == "cpu" and calls == ["cpu"]


def test_failure_on_gpu_moves_the_same_model_back_to_cpu() -> None:
    """Le repli ne reconstruit pas le modèle et ne relit aucun fichier : il déplace le même objet."""
    model = torch.nn.Linear(4, 2)
    weights = model.weight.detach().clone()
    calls = []

    def check(m, device):
        calls.append(device.type)
        assert m is model
        if device.type == "cuda":
            raise RuntimeError("no kernel image is available for execution on the device")

    info = place_model(model, fake_cuda_info(), check)
    assert info.type == "cpu"
    assert "no kernel image" in info.fallback_reason
    assert calls[-1] == "cpu"
    assert model.weight.device.type == "cpu"
    assert torch.equal(model.weight.detach(), weights)


def test_failure_on_required_gpu_is_an_error() -> None:
    def check(m, device):
        if device.type == "cuda":
            raise RuntimeError("out of memory")

    with pytest.raises(DeviceUnavailableError, match="out of memory"):
        place_model(torch.nn.Linear(4, 2), fake_cuda_info(requested="cuda"), check)


def test_device_info_is_serializable() -> None:
    info = fake_cuda_info().as_dict()
    assert info["capability"] == "12.0"
    assert info["type"] == "cuda"


@pytest.mark.gpu
def test_real_gpu_is_selected() -> None:
    info = select_device("auto")
    assert info.type == "cuda"
    assert info.precision == "bf16-autocast"
    assert info.fallback_reason is None
    assert info.capability is not None
