'2D BlochONet：W96 时间条件系数 FNO + 正交补 corrector。\n\n输入 (kappa,beta):[B,2]；输出 [B,33,512,512] complex64，T=1。\n33 帧全部参与每次更新，时间分块仅控制显存，不改变 loss 或更新次数。\n'
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, Iterable, List, Optional, Sequence, Tuple
from itertools import product
import json
import math
import random
import warnings
import numpy as np
import torch
from torch import nn
import h5py
from contextlib import nullcontext
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# 一、常用参数与资产路径（运行前修改这一节）
MODE = "train"                 # train / infer / evaluate
CASE = "twod_harmonic"
DEVICE = "auto"
ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "configs" / f"{CASE}.json"
DATA_PATH = ROOT / "assets" / CASE / "reference_trajectories.h5"
BASIS_PATH = ROOT / "assets" / CASE / "basis.npz"
INITIALIZATION_PATH = ROOT / "assets" / CASE / "shared_initial.pt"  # 仅新训练需要
CHECKPOINT_PATH = ROOT / "results" / CASE / "best_model.pt"
SAVE_PATH = ROOT / "results" / CASE / "best_model.pt"
PARAMETERS = [1.0, 0.0]
SPLIT = "test"

COEFFICIENT_WIDTH = 96
COEFFICIENT_DEPTH = 4
COEFFICIENT_MODES = (16, 16)
TIME_EMBED_DIM = 32
TIME_FREQUENCY_CYCLES = (0.5, 12.0)
SPECTRAL_SEED = 302026
SPECTRAL_TEMPLATE_MODES = (16, 17)
CORRECTOR_WIDTH = 128
CORRECTOR_DEPTH = 4
CORRECTOR_FOURIER_FEATURES = 6
CORRECTOR_CHUNK = 4096
TIME_CHUNK = 4
UPDATES = 20000
VALIDATE_EVERY = 500
LEARNING_RATE = 1e-3
HOLD_STEPS = 2000
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MASS_WEIGHT = 0.01

# ============================================================================
# 二、必要数学工具、物理算子与固定 basis
# ============================================================================


def normalized_gaussian(x1: np.ndarray, x2: np.ndarray, x0=(math.pi, math.pi), alpha: float=5.0) -> np.ndarray:
    """函数 `normalized_gaussian`。参数：x1, x2, x0, alpha。沿用来源实现的计算、形状检查及返回值。"""
    psi = np.exp(-alpha * ((x1 - x0[0]) ** 2 + (x2 - x0[1]) ** 2)).astype(np.complex128)
    if x1.shape[0] < 2 or x2.shape[1] < 2:
        raise ValueError('grid must contain at least two points in each direction')
    dx = float(x1[1, 0] - x1[0, 0])
    dy = float(x2[0, 1] - x2[0, 0])
    mass = dx * dy * np.sum(np.abs(psi) ** 2)
    return psi / np.sqrt(mass)


def lattice_fourier_coefficient(delta_p: tuple[int, int], v0: float=0.5, eta: float=1.0) -> float:
    """函数 `lattice_fourier_coefficient`。参数：delta_p, v0, eta。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Exact Fourier coefficient V_hat(p-q) for the specified lattice."""
    d1, d2 = delta_p
    if (abs(d1), abs(d2)) in {(1, 0), (0, 1)}:
        return v0 / 2.0
    if abs(d1) == 1 and abs(d2) == 1:
        return eta * v0 / 4.0
    return 0.0


@dataclass(frozen=True)
class BlochBasis2D:
    """二维 Bloch 本征谱与本征向量数据，加载时检查数组形状与有限性。"""
    k1: np.ndarray
    k2: np.ndarray
    p_vectors: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray
    kpw: int
    n_bands: int
    v0: float
    eta: float

    def validate(self, residual_tolerance: float=1e-10, orthogonality_tolerance: float=2e-12) -> dict[str, float]:
        """函数 `validate`。参数：residual_tolerance, orthogonality_tolerance。沿用来源实现的计算、形状检查及返回值。"""
        if self.eigenvectors.shape != (len(self.k1), len(self.k2), len(self.p_vectors), self.n_bands):
            raise ValueError('eigenvector array has an inconsistent shape')
        max_orth = 0.0
        max_residual = 0.0
        for i, k1 in enumerate(self.k1):
            for j, k2 in enumerate(self.k2):
                q = self.eigenvectors[i, j]
                e = self.energies[i, j]
                gram = q.conj().T @ q
                max_orth = max(max_orth, float(np.linalg.norm(gram - np.eye(self.n_bands), ord=2)))
                h = plane_wave_hamiltonian(float(k1), float(k2), self.kpw, self.v0, self.eta)[0]
                residual = h @ q - q * e[None, :]
                scale = max(1.0, float(np.linalg.norm(h, ord=2)))
                max_residual = max(max_residual, float(np.linalg.norm(residual, ord=2) / scale))
        if max_orth > orthogonality_tolerance:
            raise ValueError(f'basis orthogonality error {max_orth:.3e} exceeds tolerance')
        if max_residual > residual_tolerance:
            raise ValueError(f'eigenpair residual {max_residual:.3e} exceeds tolerance')
        return {'max_orthogonality_error': max_orth, 'max_relative_eigen_residual': max_residual}



def plane_wave_vectors(kpw: int) -> np.ndarray:
    """函数 `plane_wave_vectors`。参数：kpw。沿用来源实现的计算、形状检查及返回值。"""
    p = np.arange(-kpw, kpw + 1, dtype=np.int64)
    p1, p2 = np.meshgrid(p, p, indexing='ij')
    return np.stack([p1.ravel(), p2.ravel()], axis=1)


def plane_wave_hamiltonian(k1: float, k2: float, kpw: int, v0: float=0.5, eta: float=1.0) -> tuple[np.ndarray, np.ndarray]:
    """函数 `plane_wave_hamiltonian`。参数：k1, k2, kpw, v0, eta。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
True coupled 2D plane-wave Hamiltonian H_pq(k)."""
    pvec = plane_wave_vectors(kpw)
    dim = len(pvec)
    h = np.zeros((dim, dim), dtype=np.float64)
    shifted = pvec.astype(np.float64) + np.array([k1, k2], dtype=np.float64)
    np.fill_diagonal(h, 0.5 * np.sum(shifted ** 2, axis=1))
    lookup = {tuple(p): i for i, p in enumerate(pvec.tolist())}
    deltas = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for i, p in enumerate(pvec):
        for d in deltas:
            q = (int(p[0] - d[0]), int(p[1] - d[1]))
            j = lookup.get(q)
            if j is not None:
                h[i, j] = lattice_fourier_coefficient(d, v0=v0, eta=eta)
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-15):
        raise RuntimeError('constructed plane-wave Hamiltonian is not Hermitian')
    return (h, pvec)


def load_basis(path: str | Path) -> BlochBasis2D:
    """读取二维 Bloch 谱，检查维度与有限性，不要求额外留档字段。"""
    with np.load(path, allow_pickle=False) as z:
        basis = BlochBasis2D(z['k1'], z['k2'], z['p_vectors'], z['energies'], z['eigenvectors'], int(z['kpw'][0]), int(z['n_bands'][0]), float(z['v0'][0]), float(z['eta'][0]))
    shape = (len(basis.k1), len(basis.k2))
    if basis.energies.shape != (*shape, basis.n_bands):
        raise ValueError('basis 的能带数组形状不一致')
    if basis.eigenvectors.shape != (*shape, len(basis.p_vectors), basis.n_bands):
        raise ValueError('basis 的本征向量数组形状不一致')
    if not np.isfinite(basis.energies).all() or not np.isfinite(basis.eigenvectors).all():
        raise ValueError('basis 含非有限谱数据')
    return basis


@dataclass
class BlockBlochTransform:
    """FFT 分块 Bloch analysis/synthesis，以及完整空间上的正交补投影。"""
    basis: BlochBasis2D
    nx: int
    ny: int
    x_min: float
    x_max: float
    eps: float
    device: torch.device | str = 'cpu'
    complex_dtype: torch.dtype = torch.complex64

    def __post_init__(self) -> None:
        """函数 `__post_init__`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        self.device = torch.device(self.device)
        self.ncx, self.ncy = (len(self.basis.k1), len(self.basis.k2))
        if self.nx % self.ncx or self.ny % self.ncy:
            raise ValueError('physical grid must be divisible by the Bloch grid')
        expected = (self.x_max - self.x_min) / (2.0 * math.pi * self.eps)
        if abs(expected - self.ncx) > 1e-10 or abs(expected - self.ncy) > 1e-10:
            raise ValueError('Bloch grid does not match the number of lattice cells')
        p = self.basis.p_vectors
        rx = np.arange(-self.ncx // 2, self.ncx // 2, dtype=np.int64)
        ry = np.arange(-self.ncy // 2, self.ncy // 2, dtype=np.int64)
        ix = np.empty((self.ncx, len(p)), dtype=np.int64)
        iy = np.empty((self.ncy, len(p)), dtype=np.int64)
        for i, r in enumerate(rx):
            modes = r + self.ncx * p[:, 0]
            if np.any(modes < -self.nx // 2) or np.any(modes > self.nx // 2 - 1):
                raise ValueError('plane-wave cutoff requests modes outside the physical FFT grid')
            ix[i] = modes % self.nx
        for j, r in enumerate(ry):
            modes = r + self.ncy * p[:, 1]
            if np.any(modes < -self.ny // 2) or np.any(modes > self.ny // 2 - 1):
                raise ValueError('plane-wave cutoff requests modes outside the physical FFT grid')
            iy[j] = modes % self.ny
        linear = ix[:, None, :] * self.ny + iy[None, :, :]
        if len(np.unique(linear)) != linear.size:
            raise ValueError('residue/plane-wave map contains duplicate physical Fourier modes')
        self.linear_indices = torch.as_tensor(linear, dtype=torch.long, device=self.device)
        self.q = torch.as_tensor(self.basis.eigenvectors, dtype=self.complex_dtype, device=self.device)
        self.energies = torch.as_tensor(self.basis.energies, dtype=torch.float64, device=self.device)
        self.dx = (self.x_max - self.x_min) / self.nx
        self.dy = (self.x_max - self.x_min) / self.ny
        nx_modes = np.fft.fftfreq(self.nx) * self.nx
        ny_modes = np.fft.fftfreq(self.ny) * self.ny
        omega_x = 2.0 * np.pi * nx_modes / (self.x_max - self.x_min)
        omega_y = 2.0 * np.pi * ny_modes / (self.x_max - self.x_min)
        phase = np.exp(-1j * (omega_x[:, None] * self.x_min + omega_y[None, :] * self.x_min))
        self.analysis_origin_phase = torch.as_tensor(phase, dtype=self.complex_dtype, device=self.device)

    @property
    def n_bands(self) -> int:
        """函数 `n_bands`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return self.basis.n_bands

    def group_fourier(self, psi: torch.Tensor) -> torch.Tensor:
        """函数 `group_fourier`。参数：psi。沿用来源实现的计算、形状检查及返回值。"""
        if psi.shape[-2:] != (self.nx, self.ny):
            raise ValueError(f'expected spatial shape {(self.nx, self.ny)}, got {tuple(psi.shape[-2:])}')
        psi = psi.to(device=self.device, dtype=self.complex_dtype)
        coeff = torch.fft.fft2(psi, norm='ortho') * math.sqrt(self.dx * self.dy)
        coeff = coeff * self.analysis_origin_phase
        flat = coeff.reshape(*coeff.shape[:-2], self.nx * self.ny)
        grouped = flat[..., self.linear_indices.reshape(-1)]
        return grouped.reshape(*coeff.shape[:-2], self.ncx, self.ncy, len(self.basis.p_vectors))

    def ungroup_fourier(self, grouped: torch.Tensor) -> torch.Tensor:
        """函数 `ungroup_fourier`。参数：grouped。沿用来源实现的计算、形状检查及返回值。"""
        expected = (self.ncx, self.ncy, len(self.basis.p_vectors))
        if grouped.shape[-3:] != expected:
            raise ValueError(f'expected grouped shape ending in {expected}')
        lead = grouped.shape[:-3]
        flat = torch.zeros(*lead, self.nx * self.ny, dtype=self.complex_dtype, device=self.device)
        index = self.linear_indices.reshape(-1)
        values = grouped.reshape(*lead, -1).to(self.complex_dtype)
        expand_shape = (1,) * len(lead) + (len(index),)
        flat = flat.scatter_add(-1, index.reshape(expand_shape).expand(*lead, -1), values)
        physical_coeff = flat.reshape(*lead, self.nx, self.ny)
        fft_coeff = physical_coeff * self.analysis_origin_phase.conj()
        return torch.fft.ifft2(fft_coeff / math.sqrt(self.dx * self.dy), norm='ortho')

    def analysis(self, psi: torch.Tensor) -> torch.Tensor:
        """函数 `analysis`。参数：psi。沿用来源实现的计算、形状检查及返回值。"""
        grouped = self.group_fourier(psi)
        return torch.einsum('...ijp,ijpm->...mij', grouped, self.q.conj())

    def synthesis(self, coefficients: torch.Tensor) -> torch.Tensor:
        """函数 `synthesis`。参数：coefficients。沿用来源实现的计算、形状检查及返回值。"""
        expected = (self.n_bands, self.ncx, self.ncy)
        if coefficients.shape[-3:] != expected:
            raise ValueError(f'expected coefficient shape ending in {expected}')
        grouped = torch.einsum('ijpm,...mij->...ijp', self.q, coefficients.to(self.complex_dtype))
        return self.ungroup_fourier(grouped)

    def project(self, psi: torch.Tensor) -> torch.Tensor:
        """函数 `project`。参数：psi。沿用来源实现的计算、形状检查及返回值。"""
        return self.synthesis(self.analysis(psi))

    def complement(self, psi: torch.Tensor) -> torch.Tensor:
        """函数 `complement`。参数：psi。沿用来源实现的计算、形状检查及返回值。"""
        return psi.to(self.complex_dtype) - self.project(psi)

    def parseval_error(self, psi: torch.Tensor) -> float:
        """函数 `parseval_error`。参数：psi。沿用来源实现的计算、形状检查及返回值。"""
        grouped = self.group_fourier(psi)
        grouped_norm = torch.sum(torch.abs(grouped) ** 2)
        projected_group = torch.zeros_like(grouped)
        projected_group.copy_(grouped)
        selected_physical_norm = self.dx * self.dy * torch.sum(torch.abs(self.ungroup_fourier(projected_group)) ** 2)
        return float(torch.abs(grouped_norm - selected_physical_norm) / torch.clamp(grouped_norm, min=1e-30))


def normalize_force(force: torch.Tensor, force_min: float=-0.6, force_max: float=0.6) -> torch.Tensor:
    """函数 `normalize_force`。参数：force, force_min, force_max。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
This package receives already normalized two-parameter external-potential inputs."""
    return force

# ============================================================================
# 三、网络结构与完整 forward
# ============================================================================


def translation_phase_features(k1: torch.Tensor, k2: torch.Tensor, x0: tuple[float, float], eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """按 float64 计算 k*x0/eps 的 sin/cos，再转网络精度；仅作为输入特征。"""
    phase = (k1.to(torch.float64) * float(x0[0]) + k2.to(torch.float64) * float(x0[1])) / float(eps)
    return (torch.sin(phase).to(torch.float32), torch.cos(phase).to(torch.float32))


class FourierFeatureMap(nn.Module):
    """对物理坐标和参数构造 Fourier 特征，保持正式 sin/cos 排列与频率。"""

    def __init__(self, input_dim: int, n_frequencies: int):
        """函数 `__init__`。参数：input_dim, n_frequencies。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_frequencies = int(n_frequencies)
        self.register_buffer('frequencies', 2.0 ** torch.arange(n_frequencies, dtype=torch.float32) * math.pi)

    @property
    def output_dim(self) -> int:
        """函数 `output_dim`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return self.input_dim * (1 + 2 * self.n_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """函数 `forward`。参数：x。沿用来源实现的计算、形状检查及返回值。"""
        angles = x[..., None] * self.frequencies
        return torch.cat([x, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1)


class PhysicalCorrector2D(nn.Module):
    """时间、二维位置、微观坐标与外势参数的 Fourier-MLP；分块计算全部空间点。"""

    def __init__(self, n_frequencies: int=6, width: int=128, depth: int=4):
        """函数 `__init__`。参数：n_frequencies, width, depth。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.features = FourierFeatureMap(7, n_frequencies)
        layers: list[nn.Module] = [nn.Linear(self.features.output_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        final = nn.Linear(width, 2)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def _evaluate_features(self, values: torch.Tensor) -> torch.Tensor:
        """函数 `_evaluate_features`。参数：values。沿用来源实现的计算、形状检查及返回值。"""
        return self.network(self.features(values))

    def forward(self, times: torch.Tensor, x: torch.Tensor, y: torch.Tensor, forces: torch.Tensor, *, eps: float, t_final: float, x_min: float, x_max: float, chunk_size: int=65536, checkpoint_chunks: bool=True) -> torch.Tensor:
        """函数 `forward`。参数：times, x, y, forces。沿用来源实现的计算、形状检查及返回值。"""
        device = forces.device
        times, x, y = (times.to(device), x.to(device), y.to(device))
        b, nt, nx, ny = (len(forces), len(times), len(x), len(y))
        macro_x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        macro_y = 2.0 * (y - x_min) / (x_max - x_min) - 1.0
        micro_x = torch.remainder(x / eps + math.pi, 2.0 * math.pi) / math.pi - 1.0
        micro_y = torch.remainder(y / eps + math.pi, 2.0 * math.pi) / math.pi - 1.0
        mx, my = torch.meshgrid(macro_x, macro_y, indexing='ij')
        ux, uy = torch.meshgrid(micro_x, micro_y, indexing='ij')
        spatial = torch.stack([mx, my, ux, uy], dim=-1).reshape(-1, 4)
        normalized_f = normalize_force(forces).to(torch.float32)
        outputs: list[torch.Tensor] = []
        for start in range(0, nx * ny, chunk_size):
            stop = min(start + chunk_size, nx * ny)
            space = spatial[start:stop]
            n = stop - start
            tpart = (times / t_final).reshape(1, nt, 1, 1).expand(b, nt, n, 1)
            spart = space.reshape(1, 1, n, 4).expand(b, nt, n, 4)
            fpart = normalized_f.reshape(b, 1, 1, 2).expand(b, nt, n, 2)
            values = torch.cat([tpart, spart, fpart], dim=-1).reshape(-1, 7)
            if checkpoint_chunks and self.training:
                value = checkpoint(self._evaluate_features, values, use_reentrant=False)
            else:
                value = self._evaluate_features(values)
            outputs.append(value.reshape(b, nt, n, 2))
        raw = torch.cat(outputs, dim=2).reshape(b, nt, nx, ny, 2)
        return torch.complex(raw[..., 0], raw[..., 1])


@dataclass
class BlochForwardDiagnostics:
    band_part: torch.Tensor
    corrector: torch.Tensor
    coefficient_tilde: torch.Tensor


class TimeEmbedding(nn.Module):
    """查询时刻的 sin/cos 嵌入；频率覆盖正式的 0.5 到 12 周期。"""

    def __init__(self, dimension=32, cycles=(0.5, 12.0)):
        """函数 `__init__`。参数：dimension, cycles。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        if dimension % 2 or not 0 < cycles[0] < cycles[1]:
            raise ValueError('Invalid Fourier time encoding')
        frequencies = torch.exp(torch.linspace(math.log(cycles[0]), math.log(cycles[1]), dimension // 2, dtype=torch.float64)).float()
        self.register_buffer('frequencies', frequencies)
        self.dimension = dimension

    def forward(self, tau):
        """函数 `forward`。参数：tau。沿用来源实现的计算、形状检查及返回值。"""
        angles = 2 * math.pi * tau[..., None] * self.frequencies
        return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)


def enforce_hermitian_planes_2d(spectrum, real_last_size):
    """函数 `enforce_hermitian_planes_2d`。参数：spectrum, real_last_size。沿用来源实现的计算、形状检查及返回值。"""
    nx = spectrum.shape[-2]
    negative_x = -torch.arange(nx, device=spectrum.device) % nx
    out = spectrum.clone()
    planes = [0] + ([real_last_size // 2] if real_last_size % 2 == 0 else [])
    for ky in planes:
        plane = spectrum[..., ky]
        out[..., ky] = 0.5 * (plane + plane.index_select(-1, negative_x).conj())
    return out


class SpectralConv2dReal(nn.Module):
    """实通道二维谱卷积，保留 Hermitian 对称平面和正式频率模板初始化。"""

    def __init__(self, in_channels, out_channels, modes, *, template_modes=(16, 17), seed=302026):
        """函数 `__init__`。参数：in_channels, out_channels, modes。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.in_channels, self.out_channels = (in_channels, out_channels)
        self.modes, self.template_modes = (tuple(modes), tuple(template_modes))
        mx, my = self.modes
        tx, ty = self.template_modes
        if min(mx, my) < 1 or mx > tx or my > ty:
            raise ValueError('Requested modes exceed the initialization template')
        scale = 1 / math.sqrt(in_channels * out_channels)
        weights = []
        for sign in range(2):
            generator = torch.Generator(device='cpu').manual_seed(seed + sign)
            template = scale * torch.randn(in_channels, out_channels, tx, ty, 2, generator=generator)
            value = template[:, :, :mx if sign == 0 else tx, :my]
            if sign:
                value = value[:, :, tx - mx:]
            weights.append(nn.Parameter(value.clone().contiguous()))
        self.weights = nn.ParameterList(weights)

    def forward(self, x):
        """函数 `forward`。参数：x。沿用来源实现的计算、形状检查及返回值。"""
        if x.ndim != 4 or torch.is_complex(x):
            raise ValueError('Spatial FNO expects real [query_batch,C,K1,K2]')
        nx, ny = x.shape[-2:]
        mx, my = self.modes
        if 2 * mx > nx or my > ny // 2 + 1:
            raise ValueError(f'Overlapping/excessive modes {self.modes} for grid {(nx, ny)}')
        transformed = torch.fft.rfft2(x, norm='ortho')
        out = torch.zeros(x.shape[0], self.out_channels, nx, ny // 2 + 1, dtype=transformed.dtype, device=x.device)
        for weight, block in zip(self.weights, (slice(0, mx), slice(nx - mx, nx))):
            w = torch.view_as_complex(weight.contiguous())
            out[:, :, block, :my] = torch.einsum('bixy,ioxy->boxy', transformed[:, :, block, :my], w)
        out = enforce_hermitian_planes_2d(out, ny)
        return torch.fft.irfft2(out, s=(nx, ny), norm='ortho')


class PointwiseLinear2d(nn.Conv2d):
    """对各网格点使用同一线性映射，不混合空间位置。"""

    def forward(self, x):
        """函数 `forward`。参数：x。沿用来源实现的计算、形状检查及返回值。"""
        if self.kernel_size != (1, 1) or self.groups != 1:
            raise ValueError('Only unit ungrouped kernels are supported')
        z = torch.matmul(self.weight[:, :, 0, 0], x.flatten(2))
        if self.bias is not None:
            z = z + self.bias[None, :, None]
        return z.reshape(x.shape[0], self.out_channels, *x.shape[2:])


class TimeConditionedFNO2d(nn.Module):
    """时间嵌入拼接输入并逐层加偏置；谱卷积/局部映射后 GELU，输出头零初始化。"""

    def __init__(self, in_channels, out_channels, width=48, depth=4, modes=(16, 16), time_embed_dim=32, spectral_seed=302026, template_modes=(16, 17)):
        """函数 `__init__`。参数：in_channels, out_channels, width, depth, modes, time_embed_dim, spectral_seed, template_modes。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.in_channels, self.out_channels = (in_channels, out_channels)
        self.width, self.depth, self.modes = (width, depth, tuple(modes))
        self.lift = PointwiseLinear2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2dReal(width, width, modes, template_modes=template_modes, seed=spectral_seed + 100 * layer) for layer in range(depth)])
        self.local = nn.ModuleList([PointwiseLinear2d(width, width, 1) for _ in range(depth)])
        self.time_bias = nn.ModuleList([nn.Linear(time_embed_dim, width) for _ in range(depth)])
        self.project1 = PointwiseLinear2d(width, 2 * width, 1)
        self.project2 = PointwiseLinear2d(2 * width, out_channels, 1)
        nn.init.zeros_(self.project2.weight)
        nn.init.zeros_(self.project2.bias)
        self.checkpoint_layers = True
        self.offload_saved_activations = True

    @staticmethod
    def _block(z, embedding, spectral, local, time_bias):
        """函数 `_block`。参数：z, embedding, spectral, local, time_bias。沿用来源实现的计算、形状检查及返回值。"""
        return F.gelu(spectral(z) + local(z) + time_bias(embedding)[:, :, None, None])

    def _head(self, z):
        """函数 `_head`。参数：z。沿用来源实现的计算、形状检查及返回值。"""
        return self.project2(F.gelu(self.project1(z)))

    def forward(self, features, embedding):
        """函数 `forward`。参数：features, embedding。沿用来源实现的计算、形状检查及返回值。"""
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError('Incorrect query input shape')
        if embedding.ndim != 2 or embedding.shape[0] != features.shape[0]:
            raise ValueError('Time embeddings must match query batch explicitly')
        # 仅训练时把保存的激活卸载到 CPU；计算公式及参数更新保持不变。
        offload = self.training and features.is_cuda and self.offload_saved_activations
        with torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext():
            z = self.lift(features)
            for spectral, local, bias in zip(self.spectral, self.local, self.time_bias):
                if self.training and self.checkpoint_layers and torch.is_grad_enabled():
                    z = checkpoint(self._block, z, embedding, spectral, local, bias, use_reentrant=False)
                else:
                    z = self._block(z, embedding, spectral, local, bias)
            if self.training and self.checkpoint_layers and torch.is_grad_enabled():
                return checkpoint(self._head, z, use_reentrant=False)
            return self._head(z)


class TimeConditionedBlochCoefficient2D(nn.Module):
    """由 c0、准动量、载波相位和外势参数预测 F_theta，系数为 c0+(t/T)F_theta。"""

    def __init__(self, n_bands, width=48, depth=4, modes=(16, 16), time_embed_dim=32, cycles=(0.5, 12), spectral_seed=302026, template_modes=(16, 17)):
        """函数 `__init__`。参数：n_bands, width, depth, modes, time_embed_dim, cycles, spectral_seed, template_modes。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.n_bands = n_bands
        self.time_embedding = TimeEmbedding(time_embed_dim, cycles)
        self.operator = TimeConditionedFNO2d(7 + time_embed_dim, 2 * n_bands, width, depth, modes, time_embed_dim, spectral_seed, template_modes)

    def increment(self, c0, times, k1, k2, forces, *, x0, eps, t_final):
        """函数 `increment`。参数：c0, times, k1, k2, forces。沿用来源实现的计算、形状检查及返回值。"""
        if c0.ndim != 4 or not torch.is_complex(c0):
            raise ValueError('c0 must be complex [B,M,K1,K2]')
        b, m, nx, ny = c0.shape
        nt = len(times)
        if m != self.n_bands or forces.shape != (b, 2):
            raise ValueError('Band or parameter shape mismatch')
        tau = times / t_final
        tt = tau[None, :, None, None].expand(b, nt, nx, ny)
        kk1 = k1[None, None, :, None].expand(b, nt, nx, ny)
        kk2 = k2[None, None, None, :].expand(b, nt, nx, ny)
        sine, cosine = translation_phase_features(kk1, kk2, x0, eps)
        f1 = forces[:, 0, None, None, None].expand(b, nt, nx, ny)
        f2 = forces[:, 1, None, None, None].expand(b, nt, nx, ny)
        original = torch.stack((tt, kk1, kk2, sine, cosine, f1, f2), dim=2).float()
        embedding = self.time_embedding(tau).to(dtype=original.dtype)
        query_e = embedding[None].expand(b, nt, -1).reshape(b * nt, -1)
        field_e = embedding[None, :, :, None, None].expand(b, nt, -1, nx, ny)
        features = torch.cat((original, field_e), dim=2).reshape(b * nt, -1, nx, ny)
        out = self.operator(features, query_e).reshape(b, nt, 2, m, nx, ny)
        return torch.complex(out[:, :, 0], out[:, :, 1]).permute(0, 2, 1, 3, 4)

    def forward(self, c0, times, k1, k2, forces, *, x0, eps, t_final):
        """函数 `forward`。参数：c0, times, k1, k2, forces。沿用来源实现的计算、形状检查及返回值。"""
        increment = self.increment(c0, times, k1, k2, forces, x0=x0, eps=eps, t_final=t_final)
        return c0[:, :, None] + (times / t_final)[None, None, :, None, None] * increment


class BlochONet2D(nn.Module):
    """2D 完整前向父类：初值分析→系数演化→exp(-i*t*E/eps)→物理重构→corrector 正交补。t=0 使用原硬初值约束。"""

    def __init__(self, transform, cfg):
        """正式 W96 时间条件网络；保持 coefficient.* 与 corrector.* 参数键。"""
        super().__init__()
        p = cfg['physics']
        self.transform = transform
        self.eps, self.x0, self.t_final = (p['eps'], tuple(p['center']), p['T'])
        self.x_min, self.x_max = p['domain']
        self.coefficient = TimeConditionedBlochCoefficient2D(transform.n_bands, COEFFICIENT_WIDTH, COEFFICIENT_DEPTH, COEFFICIENT_MODES, TIME_EMBED_DIM, TIME_FREQUENCY_CYCLES, SPECTRAL_SEED, SPECTRAL_TEMPLATE_MODES)
        self.corrector = PhysicalCorrector2D(CORRECTOR_FOURIER_FEATURES, CORRECTOR_WIDTH, CORRECTOR_DEPTH)
        self.coefficient.operator.checkpoint_layers = True
        self.coefficient.operator.offload_saved_activations = True

    def forward(self, psi0: torch.Tensor, times: torch.Tensor, x: torch.Tensor, y: torch.Tensor, forces: torch.Tensor, *, corrector_chunk_size: int=65536, checkpoint_corrector: bool=True, return_diagnostics: bool=False) -> torch.Tensor | tuple[torch.Tensor, BlochForwardDiagnostics]:
        """函数 `forward`。参数：psi0, times, x, y, forces。沿用来源实现的计算、形状检查及返回值。"""
        c0 = self.transform.analysis(psi0)
        r0 = self.transform.complement(psi0)
        k1 = torch.as_tensor(self.transform.basis.k1, dtype=torch.float32, device=psi0.device)
        k2 = torch.as_tensor(self.transform.basis.k2, dtype=torch.float32, device=psi0.device)
        ctilde = self.coefficient(c0, times, k1, k2, forces, x0=self.x0, eps=self.eps, t_final=self.t_final)
        energies = self.transform.energies.permute(2, 0, 1).to(device=psi0.device)
        # 先用 double 形成谱相位，再转回 complex64，保持正式相位精度。
        phase = torch.exp(-1j * times.to(torch.float64).reshape(1, 1, -1, 1, 1) * energies.reshape(1, *energies.shape[:1], 1, *energies.shape[1:]) / self.eps)
        physical_c = ctilde * phase.to(ctilde.dtype)
        band = self.transform.synthesis(physical_c.permute(0, 2, 1, 3, 4))
        raw = self.corrector(times, x, y, forces, eps=self.eps, t_final=self.t_final, x_min=self.x_min, x_max=self.x_max, chunk_size=corrector_chunk_size, checkpoint_chunks=checkpoint_corrector)
        raw_complement = self.transform.complement(raw)
        time_scale = (times / self.t_final).reshape(1, -1, 1, 1)
        corrector = r0[:, None] + time_scale * raw_complement
        psi = band + corrector
        zero_time = (times == 0).reshape(1, -1, 1, 1)
        # t=0 硬约束直接返回原初值，不对其它时刻事后归一化。
        psi = torch.where(zero_time, psi0[:, None].expand_as(psi), psi)
        if return_diagnostics:
            return (psi, BlochForwardDiagnostics(band, corrector, ctilde))
        return psi



# ============================================================================
# 四、数据读取、损失与评估指标
# ============================================================================


def relative_complex_l2_squared(prediction: torch.Tensor, reference: torch.Tensor, eps: float=1e-14) -> torch.Tensor:
    """函数 `relative_complex_l2_squared`。参数：prediction, reference, eps。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Mean per-sample, per-time squared relative complex L2 error."""
    if prediction.shape != reference.shape or prediction.ndim < 3:
        raise ValueError('prediction/reference shapes must match and include spatial dimensions')
    spatial_dims = tuple(range(2, prediction.ndim))
    numerator = torch.sum(torch.abs(prediction - reference) ** 2, dim=spatial_dims)
    denominator = torch.sum(torch.abs(reference) ** 2, dim=spatial_dims).clamp_min(eps)
    return torch.mean(numerator / denominator)


def reference_relative_mass_squared(prediction: torch.Tensor, reference: torch.Tensor, eps: float=1e-14) -> torch.Tensor:
    """函数 `reference_relative_mass_squared`。参数：prediction, reference, eps。沿用来源实现的计算、形状检查及返回值。"""
    spatial_dims = tuple(range(2, prediction.ndim))
    mass_prediction = torch.sum(torch.abs(prediction) ** 2, dim=spatial_dims)
    mass_reference = torch.sum(torch.abs(reference) ** 2, dim=spatial_dims)
    return torch.mean(((mass_prediction - mass_reference) / mass_reference.clamp_min(eps)) ** 2)


def training_loss(prediction: torch.Tensor, reference: torch.Tensor, mass_weight: float=0.01) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """与原实现一致的复波函数相对 L2 平方均值和参考质量误差。"""
    psi_loss = relative_complex_l2_squared(prediction, reference)
    mass_loss = reference_relative_mass_squared(prediction, reference)
    total = psi_loss + float(mass_weight) * mass_loss
    return (total, {'loss': total.detach(), 'psi_loss': psi_loss.detach(), 'mass_loss': mass_loss.detach()})


def physical_potential(x, y, xi, cfg):
    """函数 `potential`。参数：x, y, xi, cfg。沿用来源实现的计算、形状检查及返回值。"""
    kappa, beta = xi
    u, v = (x - cfg['harmonic_center'][0], y - cfg['harmonic_center'][1])
    external = 0.5 * kappa * (1.3 * u * u + 2 * beta * u * v + 0.7 * v * v)
    cx, cy = (np.cos(x / cfg['eps']), np.cos(y / cfg['eps']))
    return cfg['v0'] * (cx + cy + cfg['eta'] * cx * cy) + external


def physical_relative_timewise(a, b):
    """complex128 空间相对 L2，返回各帧误差，最终对时刻算术平均。"""
    return np.sqrt(np.sum(np.abs(a - b) ** 2, axis=(-2, -1)) / np.sum(np.abs(b) ** 2, axis=(-2, -1)))


def physical_energy(psi, V, eps, dx):
    """函数 `energy`。参数：psi, V, eps, dx。沿用来源实现的计算、形状检查及返回值。"""
    w = 2 * np.pi * np.fft.fftfreq(psi.shape[-1], d=dx)
    ft = np.fft.fft2(psi, axes=(-2, -1))
    gx = np.fft.ifft2(1j * w[:, None] * ft, axes=(-2, -1))
    gy = np.fft.ifft2(1j * w[None, :] * ft, axes=(-2, -1))
    return dx ** 2 * np.sum(0.5 * eps ** 2 * (np.abs(gx) ** 2 + np.abs(gy) ** 2) + V * np.abs(psi) ** 2, axis=(-2, -1)).real


def physical_metrics(pred, ref, xi, cfg):
    """complex128 评估波函数、密度、质量和 Hamiltonian，不作相位或幅值修正。"""
    x = np.linspace(*cfg['domain'], cfg['nx'], endpoint=False)
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, x, indexing='ij')
    V = physical_potential(xx, yy, xi, cfg)
    rp, rr = (np.abs(pred) ** 2, np.abs(ref) ** 2)
    mp, mr = (dx ** 2 * rp.sum(axis=(-2, -1)), dx ** 2 * rr.sum(axis=(-2, -1)))
    hp, hr = (physical_energy(pred, V, cfg['eps'], dx), physical_energy(ref, V, cfg['eps'], dx))
    valid = np.abs(hr) > 1e-14
    relh = np.full_like(hr, np.nan)
    relh[valid] = np.abs(hp[valid] - hr[valid]) / np.abs(hr[valid])
    series = {'psi_relative_l2': physical_relative_timewise(pred, ref), 'density_relative_l2': physical_relative_timewise(rp, rr), 'mass_relative_error': np.abs(mp - mr) / mr, 'hamiltonian_absolute_error': np.abs(hp - hr), 'hamiltonian_relative_error': relh}
    result = {}
    for key, values in series.items():
        for suffix, subset in [('all_time', values), ('exclude_t0', values[1:]), ('final', values[-1:])]:
            result[key + '.' + suffix] = float(np.mean(subset)) if np.isfinite(subset).all() else None
    result['invalid_hamiltonian_denominator_count'] = int((~valid).sum())
    return result


# 外部资产读取和旧正式权重兼容（不读取旧工程源码）


def read_configuration(path):
    """读取物理网格、参数样本、实际划分和外部文件名。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require_asset(cfg, name, path):
    """检查文件存在；网格、维度和归一化设置在数据或模型构造处检查。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'请在文件顶部设置 {name} 路径：{path}')
    return path


def load_weights(cfg, path):
    """读取纯 state_dict 或旧训练字典，网络参数始终严格加载。

    旧格式中的归一化统计和固定初值使用 NumPy 数组。这里只为这些基础
    数组类型开放受限读取，仍使用 weights_only=True，不加载自定义类。
    旧字典按任务显式取出计算所需字段，其余训练记录不进入新模型。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'请在文件顶部设置 CHECKPOINT_PATH：{path}')
    numpy_types = [np.ndarray, np.dtype, np._core.multiarray._reconstruct,
                   np._core.multiarray.scalar]
    numpy_types += [type(np.dtype(name)) for name in (
        'float32', 'float64', 'complex64', 'complex128',
        'int64', 'uint32', 'uint8', 'bool')]
    with torch.serialization.safe_globals(numpy_types):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError('权重必须是非空 state_dict 或已支持的训练字典')
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    if cfg['kind'] == 'stage1' and 'model_state' in checkpoint:
        return {'model_state': checkpoint['model_state']}
    if cfg['kind'] == '1d':
        keys = ('coeff_net_state', 'correct_net_state', 'xi_mean', 'xi_std', 'c0', 'r0')
        if all(key in checkpoint for key in keys):
            return {key: checkpoint[key] for key in keys}
    if cfg['kind'] == '2d' and 'model' in checkpoint:
        return {'model': checkpoint['model']}
    raise ValueError('权重格式与当前任务不匹配')


def runtime_device(name):
    """仅在运行主流程时选择设备；import 不初始化 CUDA。"""
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def seed_runtime(seed):
    """保持正式初始化精度：float32/complex64，不开启 AMP 或 TF32。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def grids(cfg, device):
    """空间网格不含周期右端点；保持原 float32 运算顺序，时间含 0 和 T。"""
    p = cfg['physics']
    times = torch.linspace(0, p['T'], p['nt'], device=device)
    x = p['domain'][0] + torch.arange(p['nx'], device=device, dtype=torch.float32) * ((p['domain'][1] - p['domain'][0]) / p['nx'])
    return times, x


def two_d_build(cfg, basis_path, device, weights=None, initialization=None):
    """正式初始化：新系数网络 seed202026，corrector 来自共享初始权重。

    读取已有模型时严格恢复 coefficient.* / corrector.*，不需要共享初始文件。
    basis 是外部固定算子，不会进入模型 state_dict。
    """
    p = cfg['physics']
    seed_runtime(cfg['seed'])
    basis = load_basis(basis_path)
    if len(basis.k1) != p['nk'] or len(basis.k2) != p['nk'] or basis.n_bands != p['M']:
        raise ValueError('basis 的准动量网格或带数与当前任务不匹配')
    transform = BlockBlochTransform(basis, p['nx'], p['nx'], *p['domain'], p['eps'], device=device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(202026)
        model = BlochONet2D(transform, cfg)
    if weights is not None:
        state = weights['model'] if 'model' in weights else weights
        model.load_state_dict(state, strict=True)
    else:
        initial = torch.load(require_asset(cfg, 'initialization', initialization), map_location='cpu', weights_only=True)
        model.corrector.load_state_dict(initial['corrector'], strict=True)
    return model.to(device)


def two_d_initial(cfg, device):
    """物理初值：中心 Gaussian 在完整周期网格上离散归一化，不读取参考 HDF5。"""
    p = cfg['physics']
    x = np.linspace(*p['domain'], p['nx'], endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    psi = normalized_gaussian(xx, yy, x0=tuple(p['center']), alpha=p['gaussian_alpha'])
    return torch.tensor(psi[None], dtype=torch.complex64, device=device)


def two_d_normalized(raw, cfg, device):
    """(kappa,beta) 按原范围映射至 [-1,1]：先 float64，再转网络 float32。"""
    raw = torch.as_tensor(raw, dtype=torch.float64, device=device).reshape(-1, 2)
    ranges = torch.tensor(cfg['physics']['ranges'], dtype=torch.float64, device=device)
    return (2 * (raw - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1).float()


def two_d_forward(model, psi0, times, x, normalized):
    """一次查询若干时间帧；corrector 的空间分块只是显存优化，不改变输出网格。"""
    return model(psi0, times, x, x, normalized, corrector_chunk_size=CORRECTOR_CHUNK,
                 checkpoint_corrector=True)


def two_d_prediction(model, psi0, normalized, cfg):
    """每次查询四帧，最后拼成 [B,33,512,512] 完整复波函数。"""
    times, x = grids(cfg, psi0.device)
    model.eval()
    frames = []
    for start in range(0, len(times), TIME_CHUNK):
        frames.append(two_d_forward(model, psi0, times[start:start + TIME_CHUNK], x, normalized))
    return torch.cat(frames, dim=1)


class TwoDData:
    """只读 HDF5，单次只读一条完整轨迹；实际参数表和 128/32/64 划分不重建。"""
    def __init__(self, cfg, path):
        self.h5 = h5py.File(require_asset(cfg, 'data', path), 'r')
        p = cfg['physics']
        if self.h5['psi'].shape != (len(cfg['parameters']), p['nt'], p['nx'], p['nx']):
            raise ValueError('2D 完整波函数形状不匹配')
        self.xi = np.asarray(cfg['parameters'], dtype=np.float64)

    def reference(self, index):
        """标签由 complex128 转训练 complex64，返回 CPU [1,33,Nx,Ny]。"""
        if not self.h5['complete'][index]:
            raise ValueError('该条参考轨迹不完整')
        return torch.from_numpy(np.asarray(self.h5['psi'][index], dtype=np.complex64))[None]

    def close(self):
        """训练和评估结束后关闭只读文件。"""
        self.h5.close()


# ============================================================================
# 五、训练与 validation：时间分块累计全部 33 帧梯度后更新一次
# ============================================================================
def two_d_update(model, optimizer, reference, normalized, cfg):
    """各块 loss 乘帧数/Nt，保持完整轨迹均值；只执行一次 clip 和 optimizer.step。"""
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    times, x = grids(cfg, device)
    psi0 = reference[:, 0].to(device)
    terms = {'loss': 0.0, 'psi_loss': 0.0, 'mass_loss': 0.0}
    for start in range(0, len(times), TIME_CHUNK):
        stop = min(start + TIME_CHUNK, len(times))
        ref = reference[:, start:stop].to(device)
        pred = two_d_forward(model, psi0, times[start:stop], x, normalized)
        loss, current = training_loss(pred, ref, MASS_WEIGHT)
        if not torch.isfinite(loss):
            raise FloatingPointError('训练 loss 非有限值')
        weight = (stop - start) / len(times)
        (weight * loss).backward()
        for key in terms:
            terms[key] += weight * float(current[key])
        del loss, pred, ref, current
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    return terms, float(norm)


def training_stream(indices, seed, updates):
    """按正式 RandomState 顺序逐轮洗牌；每次更新一条轨迹，划分本身不变。"""
    rng = np.random.RandomState(seed)
    pieces = []
    while sum(map(len, pieces)) < updates:
        order = np.array(indices, dtype=np.int64, copy=True)
        rng.shuffle(order)
        pieces.append(order)
    return np.concatenate(pieces)[:updates]


def two_d_lr(step):
    """前 2000 步保持 1e-3，随后余弦下降至第 20000 步的 1e-5。"""
    if step <= HOLD_STEPS:
        return LEARNING_RATE
    fraction = (step - HOLD_STEPS) / (UPDATES - HOLD_STEPS)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * fraction))


def selection(model, data, indices, cfg, device):
    """仅 validation 的逐帧复相对 L2 均值参与选模；保持原 complex128 指标。"""
    values = []
    with torch.inference_mode():
        for index in indices:
            ref = data.reference(index)
            normalized = two_d_normalized(data.xi[index], cfg, device)
            pred = two_d_prediction(model, ref[:, 0].to(device), normalized, cfg).cpu().numpy()[0].astype(np.complex128)
            truth = ref.numpy()[0].astype(np.complex128)
            values.append(float(physical_relative_timewise(pred, truth).mean()))
    return float(np.mean(values))


def train(cfg, data_path, basis_path, initialization_path, save_path, device):
    """W96/20K 正式流程；仅验证误差改善时覆盖一个纯 state_dict 文件。"""
    basis_path = require_asset(cfg, 'basis', basis_path)
    model = two_d_build(cfg, basis_path, device, initialization=initialization_path)
    data = TwoDData(cfg, data_path)
    order = training_stream(cfg['split']['train'], cfg['seed'], UPDATES)
    validation_ids = cfg['split']['validation']
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best = float('inf')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 原协议在更新前也验证一次，作为后续改善的基准。
        value = selection(model, data, validation_ids, cfg, device)
        if value < best:
            best = value
            torch.save(model.state_dict(), save_path)
        print(f'update=0 validation={value:.6g}', flush=True)
        for step, index in enumerate(order, start=1):
            for group in optimizer.param_groups:
                group['lr'] = two_d_lr(step)
            normalized = two_d_normalized(data.xi[index], cfg, device)
            terms, grad = two_d_update(model, optimizer, data.reference(index), normalized, cfg)
            if step == 1 or step % 10 == 0:
                print(f"update={step}/{UPDATES} global_id={index} loss={terms['loss']:.6g} grad={grad:.6g}", flush=True)
            if step % VALIDATE_EVERY == 0 or step == UPDATES:
                value = selection(model, data, validation_ids, cfg, device)
                if value < best:
                    best = value
                    torch.save(model.state_dict(), save_path)
                print(f'validation={value:.6g} best={best:.6g}', flush=True)
    finally:
        data.close()
    return model


# ============================================================================
# 六、加载最佳权重后推理、评估；常规运行不输出其他文件
# ============================================================================
def infer(cfg, checkpoint_path, basis_path, parameters, device):
    """不读取参考标签：由物理定义构造固定初值，返回完整 33 帧。"""
    model = two_d_build(cfg, require_asset(cfg, 'basis', basis_path), device, weights=load_weights(cfg, checkpoint_path))
    initial = two_d_initial(cfg, device)
    normalized = two_d_normalized(parameters, cfg, device)
    with torch.inference_mode():
        return two_d_prediction(model, initial, normalized, cfg)


def evaluate(cfg, checkpoint_path, data_path, basis_path, split, device):
    """对真实划分逐条计算论文波函数、密度、质量与 Hamiltonian 指标。"""
    model = two_d_build(cfg, require_asset(cfg, 'basis', basis_path), device, weights=load_weights(cfg, checkpoint_path))
    data = TwoDData(cfg, data_path)
    samples = []
    try:
        with torch.inference_mode():
            for index in cfg['split'][split]:
                initial = data.reference(index)[:, 0].to(device)
                normalized = two_d_normalized(data.xi[index], cfg, device)
                pred = two_d_prediction(model, initial, normalized, cfg).cpu().numpy()[0].astype(np.complex128)
                truth = np.asarray(data.h5['psi'][index], dtype=np.complex128)
                values = physical_metrics(pred, truth, data.xi[index], cfg['physics'])
                samples.append(values)
                print(f'global_id={index} {values}', flush=True)
        result = {name: float(np.mean([row[name] for row in samples])) if all(row[name] is not None for row in samples) else None for name in samples[0]}
        print(f'{split}（相对存储参考解）: {result}', flush=True)
        return result
    finally:
        data.close()


if __name__ == '__main__':
    configuration = read_configuration(METADATA_PATH)
    device = runtime_device(DEVICE)
    if MODE == 'train':
        train(configuration, DATA_PATH, BASIS_PATH, INITIALIZATION_PATH, SAVE_PATH, device)
    elif MODE == 'infer':
        prediction = infer(configuration, CHECKPOINT_PATH, BASIS_PATH, PARAMETERS, device)
        print(f'psi shape={tuple(prediction.shape)}; dtype={prediction.dtype}; device={prediction.device}')
    elif MODE == 'evaluate':
        evaluate(configuration, CHECKPOINT_PATH, DATA_PATH, BASIS_PATH, SPLIT, device)
    else:
        raise ValueError('MODE 必须为 train、infer 或 evaluate')
