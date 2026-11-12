"""Choix du périphérique de calcul : GPU NVIDIA CUDA s'il est réellement utilisable, sinon CPU.

`select_device` vérifie, dans l'ordre :
    1. que PyTorch est un build CUDA (torch.version.cuda) ;
    2. que CUDA est disponible : pilote présent, GPU visible (conteneur lancé avec le GPU) ;
    3. que l'architecture du GPU est couverte par ce build (binaire sm_XY ou PTX compute_XY) ;
    4. qu'un petit calcul bf16, la précision du modèle sur GPU, s'exécute réellement.

`place_model` déplace ensuite le modèle, puis le valide par un calcul complet. Si cette étape échoue
sur GPU, le même objet revient sur CPU : le checkpoint n'est jamais relu, le modèle jamais reconstruit.

La demande vaut « auto » (GPU si possible, sinon CPU), « cuda » (GPU exigé : une erreur plutôt qu'un
repli silencieux) ou « cpu ».
"""

import logging
import platform
import re
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

REQUESTS = ("auto", "cuda", "cpu")
_ARCH = re.compile(r"^(sm|compute)_(\d+)([a-z]?)$")


class DeviceUnavailableError(RuntimeError):
    """Le GPU a été exigé (« cuda ») mais n'est pas utilisable."""


@dataclass(frozen=True)
class DeviceInfo:
    """Périphérique retenu, et pourquoi CUDA ne l'a pas été le cas échéant."""

    type: str  # « cuda » ou « cpu »
    name: str  # « NVIDIA GeForce RTX 5060 Ti », ou le modèle de processeur
    requested: str  # « auto », « cuda » ou « cpu »
    precision: str  # « bf16-autocast » sur GPU, « fp32 » sur CPU, comme dans le notebook
    torch_version: str
    cuda_version: str | None  # version CUDA du build de PyTorch
    cudnn_version: int | None
    capability: tuple[int, int] | None
    total_memory_mb: int | None
    num_threads: int  # threads CPU de PyTorch
    fallback_reason: str | None = None

    @property
    def torch_device(self) -> torch.device:
        return torch.device("cuda", 0) if self.type == "cuda" else torch.device("cpu")

    def as_dict(self) -> dict:
        info = asdict(self)
        info["capability"] = None if self.capability is None else f"{self.capability[0]}.{self.capability[1]}"
        return info


@dataclass(frozen=True)
class CudaProbe:
    usable: bool
    reason: str | None = None
    name: str | None = None
    capability: tuple[int, int] | None = None
    total_memory_mb: int | None = None


def cpu_name() -> str:
    """Nom du processeur (Linux : /proc/cpuinfo), à défaut celui que donne platform."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "CPU"


def arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Vrai si un code de ce build tourne sur cette capacité.

    Un binaire sm_XY tourne sur toute capacité X.Z avec Z ≥ Y (même génération). Un code PTX
    compute_XY est compilé à la volée pour toute capacité ≥ X.Y. Les variantes à suffixe (sm_90a)
    ne tournent que sur leur architecture exacte.
    """
    major, minor = capability
    for entry in arch_list:
        match = _ARCH.match(entry)
        if not match:
            continue
        kind, digits, suffix = match.groups()
        entry_major, entry_minor = int(digits[:-1]), int(digits[-1])
        if suffix:
            if (entry_major, entry_minor) == (major, minor):
                return True
        elif kind == "sm" and entry_major == major and entry_minor <= minor:
            return True
        elif kind == "compute" and (entry_major, entry_minor) <= (major, minor):
            return True
    return False


def probe_cuda(index: int = 0) -> CudaProbe:
    """Vérifie que le GPU `index` est réellement utilisable par ce build de PyTorch."""
    if torch.version.cuda is None:
        return CudaProbe(False, "PyTorch est installé sans CUDA (build CPU).")

    # is_available() peut seulement avertir (pilote trop ancien, par exemple) : le message sert de raison.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        available = torch.cuda.is_available()
    if not available:
        detail = f" ({caught[-1].message})" if caught else ""
        return CudaProbe(
            False,
            "CUDA indisponible : aucun pilote NVIDIA utilisable, ou GPU non exposé au conteneur" + detail + ".",
        )
    if torch.cuda.device_count() <= index:
        return CudaProbe(False, "Aucun GPU CUDA visible.")

    try:
        properties = torch.cuda.get_device_properties(index)
    except RuntimeError as error:
        return CudaProbe(False, f"GPU inaccessible : {error}")
    capability = (properties.major, properties.minor)
    name, memory_mb = properties.name, properties.total_memory // 2**20

    arch_list = torch.cuda.get_arch_list()
    if not arch_supported(capability, arch_list):
        return CudaProbe(
            False,
            f"{name} (sm_{capability[0]}{capability[1]}) n'est pas pris en charge par ce build de PyTorch "
            f"({', '.join(arch_list)}).",
            name,
            capability,
            memory_mb,
        )

    try:
        x = torch.randn(64, 64, device=torch.device("cuda", index))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = x @ x
        torch.cuda.synchronize(index)
        if not torch.isfinite(y.float()).all():
            raise RuntimeError("résultat non fini")
    except Exception as error:  # noqa: BLE001 - toute erreur CUDA rend le GPU inutilisable
        return CudaProbe(False, f"Échec du calcul de test sur {name} : {error}", name, capability, memory_mb)
    return CudaProbe(True, None, name, capability, memory_mb)


def _cudnn_version() -> int | None:
    try:
        return torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001 - simple information
        return None


def cpu_info(requested: str, fallback_reason: str | None = None) -> DeviceInfo:
    return DeviceInfo(
        type="cpu",
        name=cpu_name(),
        requested=requested,
        precision="fp32",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=None,
        capability=None,
        total_memory_mb=None,
        num_threads=torch.get_num_threads(),
        fallback_reason=fallback_reason,
    )


def select_device(requested: str = "auto") -> DeviceInfo:
    """Choisit le périphérique selon la demande et le matériel réellement utilisable."""
    if requested not in REQUESTS:
        raise ValueError(f"Périphérique inconnu : {requested!r} (valeurs possibles : {', '.join(REQUESTS)})")
    if requested == "cpu":
        return cpu_info(requested)

    probe = probe_cuda()
    if not probe.usable:
        if requested == "cuda":
            raise DeviceUnavailableError(f"GPU exigé, mais inutilisable : {probe.reason}")
        logger.warning("CUDA écarté, calcul sur CPU : %s", probe.reason)
        return cpu_info(requested, probe.reason)

    return DeviceInfo(
        type="cuda",
        name=probe.name,
        requested=requested,
        precision="bf16-autocast",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cudnn_version=_cudnn_version(),
        capability=probe.capability,
        total_memory_mb=probe.total_memory_mb,
        num_threads=torch.get_num_threads(),
    )


def place_model(
    model: torch.nn.Module,
    info: DeviceInfo,
    check: Callable[[torch.nn.Module, torch.device], None],
) -> DeviceInfo:
    """Place le modèle sur le périphérique choisi et le valide par `check`.

    En cas d'échec sur GPU, le même objet revient sur CPU (sans relire le checkpoint), sauf si le GPU
    a été exigé : l'erreur est alors levée.
    """
    if info.type == "cpu":
        model.to(info.torch_device)
        check(model, info.torch_device)
        return info

    try:
        model.to(info.torch_device)
        check(model, info.torch_device)
        return info
    except Exception as error:  # noqa: BLE001 - toute erreur sur GPU déclenche le repli
        model.to("cpu")
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - le contexte CUDA peut être inutilisable
            pass
        reason = f"Échec du modèle sur {info.name} : {error}"
        if info.requested == "cuda":
            raise DeviceUnavailableError(f"GPU exigé, mais inutilisable : {reason}") from error
        logger.warning("Modèle replié sur CPU : %s", reason)
        fallback = replace(cpu_info(info.requested), fallback_reason=reason)
        check(model, fallback.torch_device)
        return fallback
