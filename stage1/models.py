'Stage-I：学习周期 Hamiltonian 的低带子空间和 Ritz 谱。\n\n逐段阅读：参数 -> 平面波算子 -> Fourier-MLP/复 QR -> 数据/loss -> 训练 -> 推理评估。\nmu 为晶格参数，k 为准动量；Q:[B,Npw,8]，QᴴQ=I。导入本文件不开始运行。\n'
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
from torch.utils.data import Dataset, DataLoader

# 一、常用参数与资产路径（运行前修改这一节）
MODE = "train"                 # train / infer / evaluate
CASE = "stage1_mathieu"         # stage1_mathieu / stage1_rectangular / stage1_mathieu2
DEVICE = "auto"                # cpu / cuda / auto
ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "configs" / f"{CASE}.json"
DATA_PATH = ROOT / "assets" / CASE / "dataset.npz"
CHECKPOINT_PATH = ROOT / "results" / CASE / "best_model.pt"
SAVE_PATH = ROOT / "results" / CASE / "best_model.pt"
PARAMETERS = [1.0]              # Mathieu/rectangular 为一维；Mathieu2 改为 [mu1,mu2]
SPLIT = "test"                 # 评估 train / validation / test

# 正式网络和训练参数：seed 和 Npw 由各 case 的原始元数据确定。
NUM_BANDS = 8
HIDDEN_DIMS = [256, 256, 256, 256]
K_FOURIER_MODES = 8
DROPOUT = 0.0
EPOCHS = 500
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 1.0

# ============================================================================
# 二、必要数学工具、物理算子与固定 basis
# ============================================================================


@dataclass
class DataConfig:
    """晶格势与平面波截断配置。n_pw 必须为奇数；K 是学习的低带数。"""
    potential_family: str = 'mathieu'
    a_min: float = 0.5
    a_max: float = 3.0
    num_a: int = 81
    mu1_min: float = 0.5
    mu1_max: float = 1.5
    num_mu1: int = 11
    mu2_min: float = -0.3
    mu2_max: float = 0.3
    num_mu2: int = 7
    rect_A_min: float = 0.5
    rect_A_max: float = 2.0
    rect_num_A: int = 81
    rect_w_min: float = 3.141592653589793
    rect_w_max: float = 3.141592653589793
    rect_num_w: int = 1
    rect_c_min: float = 3.141592653589793
    rect_c_max: float = 3.141592653589793
    rect_num_c: int = 1
    num_k: int = 128
    n_pw: int = 129
    K: int = 8
    include_redundant_endpoint: bool = False
    save_projectors: bool = False
    label_dtype: str = 'float32'
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15


def fourier_features_k(k: torch.Tensor, num_modes: int) -> torch.Tensor:
    """把 k:[B] 展开为原始 k 与交错的 sin/cos 特征，保留原通道顺序。"""
    if k.ndim == 1:
        k = k[:, None]
    feats = [k]
    for m in range(1, num_modes + 1):
        feats.append(torch.sin(2.0 * math.pi * m * k))
        feats.append(torch.cos(2.0 * math.pi * m * k))
    return torch.cat(feats, dim=-1)


def canonicalize_qr(q: torch.Tensor, r: torch.Tensor, eps: float=1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """复数 QR 后固定 R 对角线相位；返回与正式权重相同规范下的 Q、R。"""
    diag_r = torch.diagonal(r, dim1=-2, dim2=-1)
    phase = diag_r / torch.clamp(torch.abs(diag_r), min=eps)
    phase_conj = torch.conj(phase)
    q = q * phase_conj[:, None, :]
    r = phase[:, :, None] * r
    return (q, r)


def complex_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """函数 `complex_mse`。参数：x, y。沿用来源实现的计算、形状检查及返回值。"""
    diff = x - y
    return torch.mean(diff.real ** 2 + diff.imag ** 2)


class PeriodicPotentialFamily1D:
    """周期晶格势接口：物理空间值与平面波 Fourier 系数使用同一势定义。"""
    family_name: str = 'abstract'
    mu_names: Sequence[str] = ()

    def build_mu_grid(self, config: DataConfig) -> np.ndarray:
        """函数 `build_mu_grid`。参数：config。沿用来源实现的计算、形状检查及返回值。"""
        raise NotImplementedError

    def value(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """函数 `value`。参数：y, mu。沿用来源实现的计算、形状检查及返回值。"""
        raise NotImplementedError

    def fourier_coeff_array(self, n: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """函数 `fourier_coeff_array`。参数：n, mu。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Return Vhat_n for integer offsets n."""
        raise NotImplementedError

    def fourier_coeff_array_torch(self, n: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """函数 `fourier_coeff_array_torch`。参数：n, mu。沿用来源实现的计算、形状检查及返回值。"""
        raise NotImplementedError

    def validate_mu(self, mu: np.ndarray) -> np.ndarray:
        """函数 `validate_mu`。参数：mu。沿用来源实现的计算、形状检查及返回值。"""
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        if len(self.mu_names) and mu.shape[0] != len(self.mu_names):
            raise ValueError(f"Expected parameter vector of length {len(self.mu_names)} for family '{self.family_name}', got shape {mu.shape}.")
        return mu


@dataclass
class MathieuPotentialFamily(PeriodicPotentialFamily1D):
    """Mathieu 晶格，保持原 cos 势的幅值、常数项和 Fourier 系数约定。"""
    family_name: str = 'mathieu'
    mu_names: Sequence[str] = ('a',)

    def build_mu_grid(self, config: DataConfig) -> np.ndarray:
        """函数 `build_mu_grid`。参数：config。沿用来源实现的计算、形状检查及返回值。"""
        a_values = np.linspace(config.a_min, config.a_max, config.num_a, dtype=np.float64)
        return a_values[:, None]

    def value(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """函数 `value`。参数：y, mu。沿用来源实现的计算、形状检查及返回值。"""
        mu = self.validate_mu(mu)
        return float(mu[0]) * np.cos(np.asarray(y, dtype=np.float64))

    def fourier_coeff_array(self, n: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """函数 `fourier_coeff_array`。参数：n, mu。沿用来源实现的计算、形状检查及返回值。"""
        mu = self.validate_mu(mu)
        a = float(mu[0])
        n = np.asarray(n, dtype=np.int64)
        out = np.zeros_like(n, dtype=np.complex128)
        out[n == 1] = 0.5 * a
        out[n == -1] = 0.5 * a
        return out

    def fourier_coeff_array_torch(self, n: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """函数 `fourier_coeff_array_torch`。参数：n, mu。沿用来源实现的计算、形状检查及返回值。"""
        if mu.ndim == 1:
            mu = mu[:, None]
        a = mu[:, 0:1]
        dtype = torch.complex64 if mu.dtype == torch.float32 else torch.complex128
        out = torch.zeros((mu.shape[0], n.numel()), device=mu.device, dtype=dtype)
        out[:, n == 1] = (0.5 * a).to(dtype)
        out[:, n == -1] = (0.5 * a).to(dtype)
        return out


@dataclass
class Mathieu2PotentialFamily(PeriodicPotentialFamily1D):
    """双参数 Mathieu2 晶格；mu=[mu1,mu2]，对应一阶与二阶谐波。"""
    family_name: str = 'mathieu2'
    mu_names: Sequence[str] = ('mu1', 'mu2')

    def build_mu_grid(self, config: DataConfig) -> np.ndarray:
        """函数 `build_mu_grid`。参数：config。沿用来源实现的计算、形状检查及返回值。"""
        mu1_values = np.linspace(config.mu1_min, config.mu1_max, config.num_mu1, dtype=np.float64)
        mu2_values = np.linspace(config.mu2_min, config.mu2_max, config.num_mu2, dtype=np.float64)
        return np.asarray(list(product(mu1_values, mu2_values)), dtype=np.float64)

    def value(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """函数 `value`。参数：y, mu。沿用来源实现的计算、形状检查及返回值。"""
        mu = self.validate_mu(mu)
        y = np.asarray(y, dtype=np.float64)
        return float(mu[0]) * np.cos(y) + float(mu[1]) * np.cos(2.0 * y)

    def fourier_coeff_array(self, n: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """函数 `fourier_coeff_array`。参数：n, mu。沿用来源实现的计算、形状检查及返回值。"""
        mu = self.validate_mu(mu)
        n = np.asarray(n, dtype=np.int64)
        out = np.zeros_like(n, dtype=np.complex128)
        out[n == 1] = 0.5 * float(mu[0])
        out[n == -1] = 0.5 * float(mu[0])
        out[n == 2] = 0.5 * float(mu[1])
        out[n == -2] = 0.5 * float(mu[1])
        return out

    def fourier_coeff_array_torch(self, n: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """函数 `fourier_coeff_array_torch`。参数：n, mu。沿用来源实现的计算、形状检查及返回值。"""
        if mu.ndim == 1:
            mu = mu[:, None]
        mu1 = mu[:, 0:1]
        mu2 = mu[:, 1:2]
        dtype = torch.complex64 if mu.dtype == torch.float32 else torch.complex128
        out = torch.zeros((mu.shape[0], n.numel()), device=mu.device, dtype=dtype)
        out[:, n == 1] = (0.5 * mu1).to(dtype)
        out[:, n == -1] = (0.5 * mu1).to(dtype)
        out[:, n == 2] = (0.5 * mu2).to(dtype)
        out[:, n == -2] = (0.5 * mu2).to(dtype)
        return out


@dataclass
class RectangularPotentialFamily(PeriodicPotentialFamily1D):
    """矩形晶格：区间内低势、区间外高势；周期化与 inclusive 端点约定保持原样。"""
    family_name: str = 'rectangular'
    mu_names: Sequence[str] = ('A',)
    fixed_w: float = np.pi
    fixed_c: float = np.pi

    def build_mu_grid(self, config: DataConfig) -> np.ndarray:
        """函数 `build_mu_grid`。参数：config。沿用来源实现的计算、形状检查及返回值。"""
        self.fixed_w = float(config.rect_w_min)
        self.fixed_c = float(config.rect_c_min)
        A_values = np.linspace(config.rect_A_min, config.rect_A_max, config.rect_num_A, dtype=np.float64)
        if int(config.rect_num_w) == 1 and int(config.rect_num_c) == 1:
            return A_values[:, None]
        w_values = np.linspace(config.rect_w_min, config.rect_w_max, config.rect_num_w, dtype=np.float64)
        c_values = np.linspace(config.rect_c_min, config.rect_c_max, config.rect_num_c, dtype=np.float64)
        return np.asarray(list(product(A_values, w_values, c_values)), dtype=np.float64)

    def mu_names_for_dim(self, mu_dim: int) -> Sequence[str]:
        """函数 `mu_names_for_dim`。参数：mu_dim。沿用来源实现的计算、形状检查及返回值。"""
        return ('A',) if int(mu_dim) == 1 else ('A', 'w', 'c')

    def _expand_mu(self, mu: np.ndarray) -> tuple[float, float, float]:
        """函数 `_expand_mu`。参数：mu。沿用来源实现的计算、形状检查及返回值。"""
        arr = np.asarray(mu, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return (float(arr[0]), float(self.fixed_w), float(self.fixed_c))
        if arr.size == 3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))
        raise ValueError('Rectangular potential expects canonical mu=(A,) or legacy mu=(A,w,c).')

    def validate_mu(self, mu: np.ndarray) -> np.ndarray:
        """函数 `validate_mu`。参数：mu。沿用来源实现的计算、形状检查及返回值。"""
        arr = np.asarray(mu, dtype=np.float64).reshape(-1)
        if arr.size not in {1, 3}:
            raise ValueError('Rectangular potential expects canonical mu=(A,) or legacy mu=(A,w,c).')
        return arr

    def value(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """函数 `value`。参数：y, mu。沿用来源实现的计算、形状检查及返回值。"""
        A, w, c = self._expand_mu(mu)
        y = np.mod(np.asarray(y, dtype=np.float64), 2.0 * np.pi)
        left = np.mod(c - 0.5 * w, 2.0 * np.pi)
        right = np.mod(c + 0.5 * w, 2.0 * np.pi)
        if w < 0.0 or w > 2.0 * np.pi:
            raise ValueError('Rectangular potential requires 0 <= w <= 2*pi.')
        if np.isclose(w, 0.0):
            indicator = np.zeros_like(y, dtype=np.float64)
        elif np.isclose(w, 2.0 * np.pi):
            indicator = np.ones_like(y, dtype=np.float64)
        elif left <= right:
            indicator = ((y >= left) & (y <= right)).astype(np.float64)
        else:
            indicator = ((y >= left) | (y <= right)).astype(np.float64)
        return A - A * indicator

    def fourier_coeff_array(self, n: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """函数 `fourier_coeff_array`。参数：n, mu。沿用来源实现的计算、形状检查及返回值。"""
        A, w, c = self._expand_mu(mu)
        if w < 0.0 or w > 2.0 * np.pi:
            raise ValueError('Rectangular potential requires 0 <= w <= 2*pi.')
        n = np.asarray(n, dtype=np.int64)
        out = np.zeros_like(n, dtype=np.complex128)
        zero_mask = n == 0
        out[zero_mask] = A * (1.0 - w / (2.0 * np.pi))
        nonzero = ~zero_mask
        nn = n[nonzero].astype(np.float64)
        out[nonzero] = -A * np.exp(-1j * nn * c) * np.sin(0.5 * nn * w) / (np.pi * nn)
        return out

    def fourier_coeff_array_torch(self, n: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """函数 `fourier_coeff_array_torch`。参数：n, mu。沿用来源实现的计算、形状检查及返回值。"""
        if mu.ndim == 1:
            mu = mu[:, None]
        if mu.shape[1] == 1:
            A = mu[:, 0:1]
            w = torch.full_like(A, float(self.fixed_w))
            c = torch.full_like(A, float(self.fixed_c))
        elif mu.shape[1] == 3:
            A = mu[:, 0:1]
            w = mu[:, 1:2]
            c = mu[:, 2:3]
        else:
            raise ValueError('Rectangular potential expects canonical mu=(A,) or legacy mu=(A,w,c).')
        if torch.any((w < 0.0) | (w > 2.0 * torch.pi)):
            raise ValueError('Rectangular potential requires 0 <= w <= 2*pi.')
        n = n.to(device=mu.device)
        real_dtype = mu.dtype
        dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        n_real = n.to(real_dtype)[None, :]
        out = torch.zeros((mu.shape[0], n.numel()), device=mu.device, dtype=dtype)
        zero_mask = n == 0
        out[:, zero_mask] = (A * (1.0 - w / (2.0 * torch.pi))).to(dtype)
        nonzero = ~zero_mask
        if torch.any(nonzero):
            nn = n_real[:, nonzero]
            out[:, nonzero] = (-A * torch.exp(-1j * nn * c) * torch.sin(0.5 * nn * w) / (torch.pi * nn)).to(dtype)
        return out


@dataclass
class KronigPenneyPotentialFamily(RectangularPotentialFamily):
    family_name: str = 'kronig_penney'

    def build_mu_grid(self, config: DataConfig) -> np.ndarray:
        """函数 `build_mu_grid`。参数：config。沿用来源实现的计算、形状检查及返回值。"""
        return np.asarray([[1.0, np.pi, np.pi]], dtype=np.float64)


def make_potential_family_by_name(name: str) -> PeriodicPotentialFamily1D:
    """函数 `make_potential_family_by_name`。参数：name。沿用来源实现的计算、形状检查及返回值。"""
    family = str(name).lower().replace('-', '_')
    if family == 'mathieu':
        return MathieuPotentialFamily()
    if family == 'mathieu2':
        return Mathieu2PotentialFamily()
    if family == 'rectangular':
        return RectangularPotentialFamily()
    if family in {'kp', 'kronig_penney'}:
        return KronigPenneyPotentialFamily()
    raise ValueError(f'Unknown potential family {name!r}.')


def make_potential_family(config: DataConfig | str) -> PeriodicPotentialFamily1D:
    """函数 `make_potential_family`。参数：config。沿用来源实现的计算、形状检查及返回值。"""
    if isinstance(config, str):
        return make_potential_family_by_name(config)
    return make_potential_family_by_name(config.potential_family)


@dataclass
class PlaneWaveBloch1D:
    """H(k)=diag((n+k)^2/2)+Vhat(n-m)，支持 NumPy 和批量 Torch Hamiltonian。"""
    n_pw: int
    potential_family: Optional[PeriodicPotentialFamily1D] = None
    _diff_matrix_np: np.ndarray = field(init=False, repr=False)
    _modes_np: np.ndarray = field(init=False, repr=False)
    _modes_torch: torch.Tensor = field(init=False, repr=False)
    _diff_offsets_np: np.ndarray = field(init=False, repr=False)
    _diff_offsets_torch: torch.Tensor = field(init=False, repr=False)
    _diff_index_np: np.ndarray = field(init=False, repr=False)
    _diff_index_torch: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """函数 `__post_init__`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        if self.n_pw % 2 != 1:
            raise ValueError('n_pw must be odd so the Fourier indices are symmetric around 0.')
        if self.potential_family is None:
            self.potential_family = MathieuPotentialFamily()
        half = self.n_pw // 2
        self._modes_np = np.arange(-half, half + 1, dtype=np.int64)
        self._modes_torch = torch.tensor(self._modes_np, dtype=torch.get_default_dtype())
        self._diff_matrix_np = self._modes_np[:, None] - self._modes_np[None, :]
        self._diff_offsets_np = np.arange(-(self.n_pw - 1), self.n_pw, dtype=np.int64)
        self._diff_offsets_torch = torch.tensor(self._diff_offsets_np, dtype=torch.int64)
        self._diff_index_np = self._diff_matrix_np + (self.n_pw - 1)
        self._diff_index_torch = torch.tensor(self._diff_index_np, dtype=torch.long)

    def _potential_matrix_numpy(self, mu: np.ndarray) -> np.ndarray:
        """函数 `_potential_matrix_numpy`。参数：mu。沿用来源实现的计算、形状检查及返回值。"""
        vhat = self.potential_family.fourier_coeff_array(self._diff_offsets_np, mu)
        return vhat[self._diff_index_np]

    def hamiltonian_numpy(self, mu: np.ndarray, k: float) -> np.ndarray:
        """函数 `hamiltonian_numpy`。参数：mu, k。沿用来源实现的计算、形状检查及返回值。"""
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        diag = 0.5 * (self._modes_np.astype(np.float64) + float(k)) ** 2
        h = np.diag(diag).astype(np.complex128)
        h += self._potential_matrix_numpy(mu)
        return h

    def solve_numpy(self, mu: np.ndarray, k: float) -> Tuple[np.ndarray, np.ndarray]:
        """函数 `solve_numpy`。参数：mu, k。沿用来源实现的计算、形状检查及返回值。"""
        h = self.hamiltonian_numpy(mu=mu, k=k)
        evals, evecs = np.linalg.eigh(h)
        return (evals.real.astype(np.float64), evecs.astype(np.complex128))

    def batch_hamiltonian_torch(self, mu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """函数 `batch_hamiltonian_torch`。参数：mu, k。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Assemble a batch of Hamiltonians.

        Args:
            mu: tensor of shape [B, mu_dim]
            k: tensor of shape [B]
        Returns:
            H: tensor of shape [B, N, N], complex dtype.
        """
        if mu.ndim == 1:
            mu = mu[:, None]
        if mu.ndim != 2 or k.ndim != 1 or mu.shape[0] != k.shape[0]:
            raise ValueError('mu must have shape [B, mu_dim] and k must have shape [B].')
        device = mu.device
        real_dtype = mu.dtype
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        modes = self._modes_torch.to(device=device, dtype=real_dtype)
        diag = 0.5 * (modes[None, :] + k[:, None]) ** 2
        h = torch.diag_embed(diag).to(complex_dtype)
        if hasattr(self.potential_family, 'fourier_coeff_array_torch'):
            vhat_batch = self.potential_family.fourier_coeff_array_torch(self._diff_offsets_torch.to(device=device), mu)
            pot = vhat_batch[:, self._diff_index_torch.to(device=device)]
        else:
            pot_batch = []
            for b in range(mu.shape[0]):
                mu_np = mu[b].detach().cpu().double().numpy()
                pot_batch.append(self._potential_matrix_numpy(mu_np))
            pot = torch.tensor(np.stack(pot_batch, axis=0), device=device, dtype=complex_dtype)
        return h + pot

    def basis_modes(self) -> np.ndarray:
        """函数 `basis_modes`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return self._modes_np.copy()

    @property
    def diff_offsets(self) -> np.ndarray:
        """函数 `diff_offsets`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return self._diff_offsets_np.copy()

# ============================================================================
# 三、网络结构与完整 forward
# ============================================================================


class MLP(nn.Module):
    """全连接网络，层顺序和 mlp.net.* 参数键保持正式实现。"""

    def __init__(self, in_dim: int, hidden_dims: Iterable[int], out_dim: int, dropout: float=0.0):
        """函数 `__init__`。参数：in_dim, hidden_dims, out_dim, dropout。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(din, dout))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """函数 `forward`。参数：x。沿用来源实现的计算、形状检查及返回值。"""
        return self.net(x)


class QFactorFourierMLP(nn.Module):
    """输入 mu:[B,d]、k:[B]，输出复矩阵与复 QR 正交化后的 Q:[B,Npw,K]。"""

    def __init__(self, n_pw: int, K: int, hidden_dims: Iterable[int], mu_dim: int=1, k_fourier_modes: int=8, dropout: float=0.0):
        """函数 `__init__`。参数：n_pw, K, hidden_dims, mu_dim, k_fourier_modes, dropout。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.n_pw = n_pw
        self.K = K
        self.mu_dim = mu_dim
        self.k_fourier_modes = k_fourier_modes
        in_dim = mu_dim + (1 + 2 * k_fourier_modes)
        out_dim = 2 * n_pw * K
        self.mlp = MLP(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=out_dim, dropout=dropout)

    def forward(self, mu: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """函数 `forward`。参数：mu, k。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Return raw complex factor B and orthonormalized Q.

        Args:
            mu: [B, mu_dim] or [B]
            k: [B]
        Returns:
            B: [B, N_pw, K] complex
            Q: [B, N_pw, K] complex
        """
        if mu.ndim == 1:
            mu = mu[:, None]
        if mu.ndim != 2 or k.ndim != 1 or mu.shape[0] != k.shape[0]:
            raise ValueError('mu must have shape [B, mu_dim] and k must have shape [B].')
        if mu.shape[1] != self.mu_dim:
            raise ValueError(f'Expected mu_dim={self.mu_dim}, got input with shape {mu.shape}.')
        k_feats = fourier_features_k(k, self.k_fourier_modes)
        x = torch.cat([mu, k_feats], dim=-1)
        out = self.mlp(x)
        out = out.view(-1, self.n_pw, self.K, 2)
        B = torch.complex(out[..., 0], out[..., 1])
        # 复 QR 将任意复输出映射至正交子空间；这里不能换成实数 QR。
        q, r = torch.linalg.qr(B, mode='reduced')
        q, r = canonicalize_qr(q, r)
        return (B, q)



# ============================================================================
# 四、数据读取、损失与评估指标
# ============================================================================


class BlochTensorDataset(Dataset):
    """把指定 mu 划分与全部 k 做笛卡尔积；q 为 complex64，谱为 float32。"""

    def __init__(self, data: Dict[str, np.ndarray], split: str, include_projector: bool=True, sample_mask: np.ndarray | None=None):
        """函数 `__init__`。参数：data, split, include_projector, sample_mask。沿用来源实现的计算、形状检查及返回值。"""
        split_key = {'train': 'train_mu_idx', 'val': 'val_mu_idx', 'test': 'test_mu_idx'}[split]
        mu_indices = data[split_key]
        k_values = data['k_values']
        mask = None if sample_mask is None else np.asarray(sample_mask, dtype=bool)
        if mask is not None:
            expected = (int(np.asarray(data['mu_values']).shape[0]), int(len(k_values)))
            if mask.shape != expected:
                raise ValueError(f'sample_mask must have shape {expected}, got {mask.shape}.')
        samples: List[Tuple[int, int]] = []
        for imu in mu_indices:
            for ik in range(len(k_values)):
                if mask is not None and (not bool(mask[int(imu), int(ik)])):
                    continue
                samples.append((int(imu), int(ik)))
        self.samples = samples
        self.data = data
        self.include_projector = include_projector

    def __len__(self) -> int:
        """函数 `__len__`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """函数 `__getitem__`。参数：idx。沿用来源实现的计算、形状检查及返回值。"""
        imu, ik = self.samples[idx]
        q = self.data['q_real'][imu, ik] + 1j * self.data['q_imag'][imu, ik]
        mu = self.data['mu_values'][imu]
        sample = {'mu': torch.tensor(mu, dtype=torch.float32), 'k': torch.tensor(self.data['k_values'][ik], dtype=torch.float32), 'q': torch.tensor(q, dtype=torch.complex64), 'evals': torch.tensor(self.data['evals'][imu, ik], dtype=torch.float32), 'mu_index': torch.tensor(imu, dtype=torch.long), 'k_index': torch.tensor(ik, dtype=torch.long)}
        if self.include_projector:
            if 'p_real' in self.data and 'p_imag' in self.data:
                p = self.data['p_real'][imu, ik] + 1j * self.data['p_imag'][imu, ik]
            else:
                p = q @ np.conjugate(q.T)
            sample['p'] = torch.tensor(p, dtype=torch.complex64)
        return sample


def load_dataset_npz(path: str) -> Dict[str, np.ndarray]:
    """读取原 NPZ 数组，不生成标签或重新抽样划分。"""
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


@dataclass
class LossWeights:
    """正式总损失为子空间投影损失 + Ritz 谱 MSE；QR 正交误差仅作诊断。"""
    alpha_sub: float = 1.0
    beta_orth: float = 0.0
    gamma_ritz: float = 1.0
    delta_residual: float = 0.0
    eta_energy: float = 0.0


def projector_from_q(q: torch.Tensor) -> torch.Tensor:
    """Q:[B,Npw,M] -> QQᴴ:[B,Npw,Npw]，消除带内基变换的不唯一性。"""
    return torch.matmul(q, torch.conj(q).transpose(-2, -1))


def projector_loss(pred_q: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    """预测投影与参考投影的 Frobenius 距离，保持原批量均值和归一化。"""
    pred_p = projector_from_q(pred_q)
    diff = pred_p - target_p
    return 0.5 * torch.mean(torch.sum(diff.real ** 2 + diff.imag ** 2, dim=(-2, -1)))


def energy_weighted_projector_loss(pred_q: torch.Tensor, target_p: torch.Tensor, h_batch: torch.Tensor) -> torch.Tensor:
    """函数 `energy_weighted_projector_loss`。参数：pred_q, target_p, h_batch。沿用来源实现的计算、形状检查及返回值。"""
    pred_p = projector_from_q(pred_q)
    diff = pred_p - target_p
    h_diff = torch.matmul(h_batch, diff)
    h_target = torch.matmul(h_batch, target_p)
    num = torch.sum(h_diff.real ** 2 + h_diff.imag ** 2, dim=(-2, -1))
    den = torch.clamp(torch.sum(h_target.real ** 2 + h_target.imag ** 2, dim=(-2, -1)), min=1e-12)
    return 0.5 * torch.mean(num / den)


def compute_losses(pred_q: torch.Tensor, target_p: torch.Tensor, h_batch: torch.Tensor, target_evals: torch.Tensor, weights: LossWeights) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """子空间投影 + Ritz 特征值损失；保留原运算次序，正交性和残差作为观测项。"""
    sub = projector_loss(pred_q=pred_q, target_p=target_p)
    gram = torch.matmul(torch.conj(pred_q).transpose(-2, -1), pred_q)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)[None, :, :]
    orth = complex_mse(gram, eye)
    hq = torch.matmul(h_batch, pred_q)
    hred = torch.matmul(torch.conj(pred_q).transpose(-2, -1), hq)
    pred_evals = torch.linalg.eigvalsh(hred).real
    ritz = torch.mean((pred_evals - target_evals) ** 2)
    residual_matrix = hq - torch.matmul(pred_q, hred)
    residual_ratio = torch.linalg.norm(residual_matrix, dim=(-2, -1)) / torch.clamp(torch.linalg.norm(hq, dim=(-2, -1)), min=1e-12)
    residual = torch.mean(residual_ratio)
    residual_loss = torch.mean(residual_ratio ** 2)
    energy_weighted = energy_weighted_projector_loss(pred_q=pred_q, target_p=target_p, h_batch=h_batch)
    total = weights.alpha_sub * sub + weights.gamma_ritz * ritz + weights.delta_residual * residual_loss + weights.eta_energy * energy_weighted
    terms = {'loss': total.detach(), 'sub': sub.detach(), 'orth': orth.detach(), 'ritz': ritz.detach(), 'residual': residual.detach(), 'residual_loss': residual_loss.detach(), 'energy_weighted': energy_weighted.detach()}
    return (total, terms)


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


def stage1_dataset(cfg, filename):
    """读取 NPZ，并原样使用对应 seed 的 mu 划分，不按 (mu,k) 随机重划分。"""
    data = load_dataset_npz(str(require_asset(cfg, 'data', filename)))
    for key in ('train_mu_idx', 'val_mu_idx', 'test_mu_idx'):
        seed_key = f'seed_{cfg["seed"]}_{key}'
        if seed_key in data:
            data[key] = np.asarray(data[seed_key], dtype=np.int64)
        if not np.array_equal(data[key], cfg['split'][key]):
            raise ValueError(f'原始划分发生变化：{key}')
    return data


def stage1_build(cfg, weights, device):
    """三个晶格使用相同 MLP/QR 算法；平面波维数来自对应物理元数据。"""
    physics = cfg['physics']
    data_config = DataConfig(**physics)
    dimension = 2 if physics['potential_family'] == 'mathieu2' else 1
    model = QFactorFourierMLP(physics['n_pw'], NUM_BANDS, HIDDEN_DIMS, dimension,
                             K_FOURIER_MODES, DROPOUT).to(device)
    if weights is not None:
        # 旧正式文件使用 model_state；新文件自身就是 state_dict。
        state = weights['model_state'] if 'model_state' in weights else weights
        model.load_state_dict(state, strict=True)
    operator = PlaneWaveBloch1D(physics['n_pw'], make_potential_family(data_config))
    return model, operator


def stage1_metrics(model, operator, mu, k, qref=None, eref=None):
    """Q:[B,Npw,M]，Ritz 谱:[B,M]；比较子空间投影，避免本征向量相位歧义。"""
    _, q = model(mu, k)
    h = operator.batch_hamiltonian_torch(mu, k).to(mu.device)
    reduced = q.mH @ h @ q
    eigenvalues = torch.linalg.eigvalsh(reduced).real
    identity = torch.eye(q.shape[-1], device=q.device)
    metrics = {
        'orthogonality': torch.linalg.matrix_norm(q.mH @ q - identity),
        'residual': torch.linalg.matrix_norm(h @ q - q @ reduced)
                    / torch.linalg.matrix_norm(h @ q).clamp_min(1e-12),
    }
    if qref is not None:
        metrics['projector_error'] = torch.linalg.matrix_norm(q @ q.mH - qref @ qref.mH) / math.sqrt(q.shape[-1])
    if eref is not None:
        metrics['ritz_relative_l2'] = torch.linalg.vector_norm(eigenvalues - eref, dim=-1) / torch.linalg.vector_norm(eref, dim=-1).clamp_min(1e-12)
    return q, eigenvalues, metrics


# ============================================================================
# 五、训练与 validation：只保存验证指标改善时的 best_model.pt
# ============================================================================
def epoch(model, operator, loader, optimizer, weights, device):
    """遍历一个 epoch；每批按样本数加权，避免最后一个小批改变均值定义。"""
    model.train(optimizer is not None)
    totals = {}
    count = 0
    for batch in loader:
        mu, k, qref, eigenvalues = (batch[key].to(device) for key in ('mu', 'k', 'q', 'evals'))
        target_projector = projector_from_q(qref)
        h = operator.batch_hamiltonian_torch(mu, k).to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        _, q = model(mu, k)
        loss, terms = compute_losses(q, target_projector, h, eigenvalues, weights)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + len(mu) * float(value)
        count += len(mu)
    return {key: value / count for key, value in totals.items()}


def train(cfg, data_path, save_path, device):
    """正式预算 500 epochs；test 不参与优化、调度或最佳权重选择。"""
    data = stage1_dataset(cfg, data_path)
    seed_runtime(cfg['seed'])
    model, operator = stage1_build(cfg, None, device)
    train_data = BlochTensorDataset(data, 'train', include_projector=False)
    validation_data = BlochTensorDataset(data, 'val', include_projector=False)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    weights = LossWeights(alpha_sub=1.0, gamma_ritz=1.0, delta_residual=0.0, eta_energy=0.0)
    best = float('inf')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, EPOCHS + 1):
        fit = epoch(model, operator, train_loader, optimizer, weights, device)
        with torch.no_grad():
            validation = epoch(model, operator, validation_loader, None, weights, device)
        scheduler.step(validation['loss'])
        if validation['loss'] < best:
            best = validation['loss']
            torch.save(model.state_dict(), save_path)
        print(f"epoch={step}/{EPOCHS} train={fit['loss']:.6g} validation={validation['loss']:.6g} best={best:.6g}", flush=True)
    return model


# ============================================================================
# 六、加载最佳权重后的推理和评估；不自动写预测或指标文件
# ============================================================================
def infer(cfg, checkpoint_path, parameters, device):
    """一组晶格参数返回完整 k 网格的 Q 和 Ritz 谱，不读取参考标签。"""
    model, operator = stage1_build(cfg, load_weights(cfg, checkpoint_path), device)
    model.eval()
    count = cfg['physics']['num_k']
    k = np.linspace(-0.5, 0.5 - 1 / count, count)
    raw = np.asarray(parameters).reshape(1, -1)
    mu = torch.tensor(np.repeat(raw, count, axis=0), dtype=torch.float32, device=device)
    with torch.inference_mode():
        q, eigenvalues, _ = stage1_metrics(model, operator, mu, torch.tensor(k, dtype=torch.float32, device=device))
    return q, eigenvalues


def evaluate(cfg, checkpoint_path, data_path, split, device):
    """按原 mu 划分评估所有 k；返回聚合字典，同时逐 mu 打印指标。"""
    model, operator = stage1_build(cfg, load_weights(cfg, checkpoint_path), device)
    model.eval()
    data = stage1_dataset(cfg, data_path)
    key = {'train': 'train_mu_idx', 'validation': 'val_mu_idx', 'test': 'test_mu_idx'}[split]
    k = torch.tensor(data['k_values'], dtype=torch.float32, device=device)
    samples = []
    with torch.inference_mode():
        for index in data[key]:
            mu = torch.tensor(np.repeat(data['mu_values'][index:index + 1], len(k), axis=0), dtype=torch.float32, device=device)
            qref = torch.tensor(data['q_real'][index] + 1j * data['q_imag'][index], dtype=torch.complex64, device=device)
            eref = torch.tensor(data['evals'][index], dtype=torch.float32, device=device)
            _, _, metrics = stage1_metrics(model, operator, mu, k, qref, eref)
            values = {name: float(value.mean()) for name, value in metrics.items()}
            samples.append(values)
            print(f'global_id={index} {values}', flush=True)
    result = {name: float(np.mean([row[name] for row in samples])) for name in samples[0]}
    print(f'{split}: {result}', flush=True)
    return result


if __name__ == '__main__':
    configuration = read_configuration(METADATA_PATH)
    device = runtime_device(DEVICE)
    if MODE == 'train':
        train(configuration, DATA_PATH, SAVE_PATH, device)
    elif MODE == 'infer':
        q, eigenvalues = infer(configuration, CHECKPOINT_PATH, PARAMETERS, device)
        print(f'Q shape={tuple(q.shape)}; Ritz shape={tuple(eigenvalues.shape)}; device={q.device}')
    elif MODE == 'evaluate':
        evaluate(configuration, CHECKPOINT_PATH, DATA_PATH, SPLIT, device)
    else:
        raise ValueError('MODE 必须为 train、infer 或 evaluate')
