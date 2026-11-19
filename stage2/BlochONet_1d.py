'1D BlochONet：固定初值下，从外势参数预测完整时空波函数。\n\n输入 xi:[B,d]，输出 psi:[B,11,Nx] complex64；时间帧是联合预测而非时间推进。\n八带系数 FNO + 两次硬正交补投影的物理 corrector，保持原 c0/r0 初值约束。\n'
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

# 一、常用参数与资产路径（运行前修改这一节）
MODE = "train"                 # train / infer / evaluate
CASE = "harmonic"              # harmonic / step / kp_d8 / closure_interior
DEVICE = "auto"
ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "configs" / f"{CASE}.json"
DATA_PATH = ROOT / "assets" / CASE / "trajectories.npz"
BASIS_PATH = ROOT / "assets" / CASE / "basis.npz"
CHECKPOINT_PATH = ROOT / "results" / CASE / "best_model.pt"
SAVE_PATH = ROOT / "results" / CASE / "best_model.pt"
PARAMETERS = [1.0]              # kp_d8 改为实际的八维外势系数
SPLIT = "test"

NUM_BANDS = 8
MAIN_WIDTH = 128
MAIN_DEPTH = 4
MAIN_MODES_T = 5
MAIN_MODES_K = 16
CORRECT_WIDTH = 128
CORRECT_DEPTH = 4
CORRECT_FOURIER_FREQUENCIES = 6
ACTIVATION = "tanh"
PROJECTOR_RCOND = 1e-8
UPDATES = 10000
BATCH_TRAJ = 4
EVAL_BATCH_TRAJ = 8
VALIDATE_EVERY = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LAMBDA_MASS = 0.01

# ============================================================================
# 二、必要数学工具、物理算子与固定 basis
# ============================================================================


def load_npz_dict(path: str | Path) -> Dict[str, np.ndarray]:
    """函数 `load_npz_dict`。参数：path。沿用来源实现的计算、形状检查及返回值。"""
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


@dataclass(frozen=True)
class HermitianPseudoinverseDiagnostics:
    dimension: int
    effective_rank: int
    rcond: float
    cutoff: float
    eigenvalue_min: float
    eigenvalue_min_retained: float
    eigenvalue_max: float
    condition_number: float
    pseudoinverse_residual: float


def hermitian_pseudoinverse(matrix: torch.Tensor, *, rcond: float) -> tuple[torch.Tensor, HermitianPseudoinverseDiagnostics]:
    """Hermitian Gram 矩阵先 eigh，再按固定相对阈值取逆；不会补随机列。"""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'Expected a square matrix, got {tuple(matrix.shape)}.')
    if not torch.is_complex(matrix):
        raise TypeError('Gram matrix must be complex so basis phase information is preserved.')
    rcond = float(rcond)
    if not 0.0 < rcond < 1.0:
        raise ValueError(f'rcond must lie in (0, 1), got {rcond}.')
    hermitian = 0.5 * (matrix + torch.conj(matrix).transpose(-2, -1))
    eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
    eigenvalues = eigenvalues.real
    eigenvalue_max_t = torch.clamp(torch.max(eigenvalues), min=0.0)
    cutoff_t = rcond * eigenvalue_max_t
    retained = eigenvalues > cutoff_t
    effective_rank = int(torch.count_nonzero(retained).detach().cpu())
    if effective_rank == 0:
        raise RuntimeError('Weighted basis Gram matrix has no eigenvalue above the requested rcond cutoff.')
    inverse_values = torch.where(retained, torch.reciprocal(eigenvalues), torch.zeros_like(eigenvalues))
    pseudoinverse = eigenvectors * inverse_values.unsqueeze(0) @ torch.conj(eigenvectors).transpose(-2, -1)
    residual_num = torch.linalg.matrix_norm(hermitian @ pseudoinverse @ hermitian - hermitian)
    residual_den = torch.clamp(torch.linalg.matrix_norm(hermitian), min=torch.finfo(eigenvalues.dtype).eps)
    retained_values = eigenvalues[retained]
    min_retained_t = torch.min(retained_values)
    condition_t = eigenvalue_max_t / min_retained_t
    diagnostics = HermitianPseudoinverseDiagnostics(dimension=int(matrix.shape[0]), effective_rank=effective_rank, rcond=rcond, cutoff=float(cutoff_t.detach().cpu()), eigenvalue_min=float(torch.min(eigenvalues).detach().cpu()), eigenvalue_min_retained=float(min_retained_t.detach().cpu()), eigenvalue_max=float(eigenvalue_max_t.detach().cpu()), condition_number=float(condition_t.detach().cpu()), pseudoinverse_residual=float((residual_num / residual_den).detach().cpu()))
    return (pseudoinverse, diagnostics)


def factor_weighted_basis(basis_matrix: torch.Tensor, *, weight: float, rcond: float) -> tuple[torch.Tensor, torch.Tensor, HermitianPseudoinverseDiagnostics]:
    """离散内积 W=dx I 下构造加权 Gram 和伪逆，供投影/最小二乘共用。"""
    if basis_matrix.ndim != 2:
        raise ValueError(f'basis_matrix must be two-dimensional, got {tuple(basis_matrix.shape)}.')
    if not torch.is_complex(basis_matrix):
        raise TypeError('basis_matrix must be complex.')
    gram = float(weight) * (torch.conj(basis_matrix).transpose(0, 1) @ basis_matrix)
    gram_pinv, diagnostics = hermitian_pseudoinverse(gram, rcond=rcond)
    return (gram, gram_pinv, diagnostics)


def _metadata_string(data: Dict[str, np.ndarray], key: str, default: str='') -> str:
    """函数 `_metadata_string`。参数：data, key, default。沿用来源实现的计算、形状检查及返回值。"""
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def infer_basis_source_from_metadata(data: Dict[str, np.ndarray], path: str | Path | None=None) -> str:
    """函数 `infer_basis_source_from_metadata`。参数：data, path。沿用来源实现的计算、形状检查及返回值。"""
    raw = _metadata_string(data, 'basis_source', '').strip()
    if raw:
        return raw
    created_by = _metadata_string(data, 'created_by', '') or _metadata_string(data, 'created_by_script', '') or _metadata_string(data, 'generator', '')
    if 'generate_stage1_basis.py' in created_by or 'generate_exact_bloch_basis.py' in created_by or 'stage1.scripts.generate_exact_basis' in created_by:
        return 'exact_eigensolver_missing_basis_source_metadata'
    return Path(path).name if path is not None else 'unknown'


@dataclass
class FrozenBlochBasis1D:
    """外部固定 Bloch 基：读取复 Fourier 系数和谱，在超晶胞网格上投影与重构。

Nk 是晶胞数；K_exported 是保留带数。basis 不参与学习，也不写入 state_dict。"""
    k_values: torch.Tensor
    evals: torch.Tensor
    q: torch.Tensor
    modes: torch.Tensor
    k_weights: torch.Tensor
    p_pw: Optional[torch.Tensor] = None
    basis_source: str = 'unknown'
    source_checkpoint: Optional[str] = None
    x_grid: Optional[torch.Tensor] = None
    drop_redundant_endpoint: bool = True
    supercell_num_cells: Optional[int] = None
    supercell_eps: Optional[float] = None
    _supercell_x_grid: Optional[torch.Tensor] = field(default=None, repr=False)
    _supercell_phi: Optional[torch.Tensor] = field(default=None, repr=False)
    _supercell_gram_inv: Optional[torch.Tensor] = field(default=None, repr=False)
    _supercell_gram_diagnostics: Optional[HermitianPseudoinverseDiagnostics] = field(default=None, repr=False)

    @classmethod
    def from_stage1_dataset(cls, path: str | Path, mu_index: int=0, x_grid: Optional[np.ndarray]=None, device: Optional[torch.device]=None, real_dtype: torch.dtype=torch.float32, drop_redundant_endpoint: bool=True) -> 'FrozenBlochBasis1D':
        """函数 `from_stage1_dataset`。参数：path, mu_index, x_grid, device, real_dtype, drop_redundant_endpoint。沿用来源实现的计算、形状检查及返回值。"""
        data = load_npz_dict(str(path))
        evals_raw = data['evals']
        q_raw = data['q_real'] + 1j * data['q_imag']
        evals = evals_raw[mu_index] if evals_raw.ndim == 3 else evals_raw
        q = q_raw[mu_index] if q_raw.ndim == 4 else q_raw
        k_values = data['k_values']
        modes = data['modes']
        p_pw = None
        if 'p_real' in data and 'p_imag' in data:
            p_raw = data['p_real'] + 1j * data['p_imag']
            p_pw = p_raw[mu_index] if p_raw.ndim == 4 else p_raw
        basis_source = infer_basis_source_from_metadata(data, path)
        source_checkpoint = cls._optional_string_metadata(data, 'source_checkpoint')
        actual_K = evals.shape[-1]
        if 'K' in data:
            metadata_K = int(data['K'][0]) if data['K'].ndim > 0 else int(data['K'])
            if metadata_K != actual_K:
                warnings.warn(f'metadata K = {metadata_K} != evals.shape[-1] = {actual_K}, using actual_K = {actual_K}')
        if drop_redundant_endpoint and cls._has_redundant_periodic_endpoint(k_values):
            evals = evals[:-1]
            q = q[:-1]
            k_values = k_values[:-1]
            if p_pw is not None:
                p_pw = p_pw[:-1]
        x_tensor = None if x_grid is None else torch.tensor(np.asarray(x_grid), dtype=real_dtype)
        cplx_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        return cls(k_values=torch.tensor(k_values, dtype=real_dtype, device=device), evals=torch.tensor(evals, dtype=real_dtype, device=device), q=torch.tensor(q, dtype=cplx_dtype, device=device), modes=torch.tensor(modes, dtype=torch.int64, device=device), k_weights=torch.tensor(cls._uniform_k_weights(k_values, drop_redundant_endpoint), dtype=real_dtype, device=device), p_pw=None if p_pw is None else torch.tensor(p_pw, dtype=cplx_dtype, device=device), basis_source=basis_source, source_checkpoint=source_checkpoint, x_grid=None if x_tensor is None else x_tensor.to(device=device), drop_redundant_endpoint=drop_redundant_endpoint)

    @staticmethod
    def _string_metadata(data: Dict[str, np.ndarray], key: str, default: str) -> str:
        """函数 `_string_metadata`。参数：data, key, default。沿用来源实现的计算、形状检查及返回值。"""
        if key not in data:
            return default
        value = np.asarray(data[key]).reshape(-1)[0]
        if isinstance(value, bytes):
            return value.decode('utf-8')
        return str(value)

    @staticmethod
    def _optional_string_metadata(data: Dict[str, np.ndarray], key: str) -> Optional[str]:
        """函数 `_optional_string_metadata`。参数：data, key。沿用来源实现的计算、形状检查及返回值。"""
        if key not in data:
            return None
        return FrozenBlochBasis1D._string_metadata(data, key, '')

    @staticmethod
    def _has_redundant_periodic_endpoint(k_values: np.ndarray) -> bool:
        """函数 `_has_redundant_periodic_endpoint`。参数：k_values。沿用来源实现的计算、形状检查及返回值。"""
        k_values = np.asarray(k_values, dtype=np.float64)
        return k_values.ndim == 1 and k_values.size >= 3 and np.isclose(k_values[-1] - k_values[0], 1.0, atol=1e-12)

    @staticmethod
    def _uniform_k_weights(k_values: np.ndarray, drop_redundant_endpoint: bool=False) -> np.ndarray:
        """函数 `_uniform_k_weights`。参数：k_values, drop_redundant_endpoint。沿用来源实现的计算、形状检查及返回值。"""
        k_values = np.asarray(k_values, dtype=np.float64)
        if k_values.ndim != 1 or k_values.size < 2:
            raise ValueError('k_values must be a 1D array with at least two points.')
        dk = float(k_values[1] - k_values[0])
        if drop_redundant_endpoint:
            weights = np.full_like(k_values, dk, dtype=np.float64)
        else:
            weights = np.full_like(k_values, dk, dtype=np.float64)
            weights[0] *= 0.5
            weights[-1] *= 0.5
        return weights

    @property
    def Nk(self) -> int:
        """函数 `Nk`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return int(self.k_values.numel())

    @property
    def K(self) -> int:
        """函数 `K`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return int(self.evals.shape[1])

    @property
    def n_pw(self) -> int:
        """函数 `n_pw`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return int(self.q.shape[1])

    @property
    def K_exported(self) -> int:
        """函数 `K_exported`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        return self.K

    def phase_factor(self, tau: torch.Tensor) -> torch.Tensor:
        """函数 `phase_factor`。参数：tau。沿用来源实现的计算、形状检查及返回值。"""
        if tau.ndim == 0:
            tau = tau[None]
        evals = self.evals.transpose(0, 1)[None, :, :]
        tau = tau[:, None, None].to(dtype=evals.dtype, device=evals.device)
        return torch.exp(1j * evals * tau)

    def to_interaction(self, b: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """函数 `to_interaction`。参数：b, tau。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：

        Convert to interaction picture (phase-stripped variables).
        Mathematical definition: c = exp(i E * tau) * b, where tau = t / eps.
        
        Here, b is the original Bloch coefficient C_M(t,k), and c is the
        phase-stripped coefficient \\widetilde{C}_M(t,k) = exp(i E_M(k) t / eps) C_M(t,k).
        """
        squeeze = False
        if b.ndim == 2:
            b = b[None, ...]
            squeeze = True
        phase = self.phase_factor(tau).to(dtype=b.dtype, device=b.device)[:, :b.shape[1], :]
        out = phase * b
        return out[0] if squeeze else out

    def from_interaction(self, c: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """函数 `from_interaction`。参数：c, tau。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：

        Convert from interaction picture back to original variables.
        Mathematical definition: b = exp(-i E * tau) * c, where tau = t / eps.
        
        Reconstructs original Bloch coefficient C_M(t,k) from phase-stripped
        coefficient \\widetilde{C}_M(t,k).
        """
        squeeze = False
        if c.ndim == 2:
            c = c[None, ...]
            squeeze = True
        phase = self.phase_factor(tau).to(dtype=c.dtype, device=c.device)[:, :c.shape[1], :]
        out = torch.conj(phase) * c
        return out[0] if squeeze else out

    def reconstruct_wavefunction_x(self, c: torch.Tensor, tau: torch.Tensor, x_grid: Optional[torch.Tensor]=None) -> torch.Tensor:
        """函数 `reconstruct_wavefunction_x`。参数：c, tau, x_grid。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Reconstruct on a cell-variable grid.

        This function interprets x_grid as the *cell variable* y \\in [0, 2pi),
        not the physical x-grid used in the epsilon-scaled problem.
        """
        x = self.x_grid if x_grid is None else x_grid
        if x is None:
            raise ValueError('x_grid is required for reconstruction.')
        squeeze = False
        if c.ndim == 2:
            c = c[None, ...]
            squeeze = True
        b = self.from_interaction(c, tau)
        x = x.to(device=b.device, dtype=self.k_values.dtype)
        modes = self.modes.to(device=b.device, dtype=self.k_values.dtype)
        k = self.k_values.to(device=b.device, dtype=self.k_values.dtype)
        weights = self.k_weights.to(device=b.device, dtype=self.k_values.dtype)
        q = self.q[:, :, :b.shape[1]].to(dtype=b.dtype, device=b.device)
        phase_nx = torch.exp(1j * modes[:, None] * x[None, :]) / np.sqrt(2.0 * np.pi)
        u_xkm = torch.einsum('jnm,nx->jmx', q, phase_nx)
        bloch_phase = torch.exp(1j * k[:, None] * x[None, :])
        phi_xkm = u_xkm * bloch_phase[:, None, :]
        b_jm = b.transpose(1, 2)
        psi = torch.einsum('bjm,jmx,j->bx', b_jm, phi_xkm, weights.to(dtype=b.real.dtype))
        return psi[0] if squeeze else psi

    def setup_supercell_frame(self, x_grid: torch.Tensor, eps: float, num_cells: Optional[int]=None, gram_rcond: float=1e-10) -> None:
        """函数 `setup_supercell_frame`。参数：x_grid, eps, num_cells, gram_rcond。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Build the epsilon-scaled Bloch frame on the physical domain D=[0,2pi].

        Parameters
        ----------
        x_grid : physical x-grid on [0, 2pi)
        eps    : semiclassical / lattice scale parameter; typically eps = 1 / num_cells
        num_cells : number of microscopic cells in D, must match the number of unique k-points

        Notes
        -----
        The physical Bloch mode is built from the cell mode via y = x / eps and
            phi_m^eps(x, k) = eps^{-1/2} * exp(i k y) * chi_m(y, k),  y = x/eps,
        which matches the scaling used in the BDTS paper for P_m^eps psi.  Projection
        and reconstruction both use the same frame / Gram matrix, so changing the
        global normalization requires regenerating stage-2 data and retraining.
        """
        num_cells = int(round(1.0 / float(eps))) if num_cells is None else int(num_cells)
        if num_cells != self.Nk:
            raise ValueError(f'num_cells (= {num_cells}) must match the number of unique k-points Nk (= {self.Nk}).')
        self.supercell_num_cells = num_cells
        self.supercell_eps = float(eps)
        self._supercell_x_grid = x_grid.to(device=self.q.device, dtype=self.k_values.dtype)
        y = self._supercell_x_grid / float(eps)
        y_mod = torch.remainder(y, 2.0 * np.pi)
        modes = self.modes.to(device=self.q.device, dtype=self.k_values.dtype)
        k = self.k_values.to(device=self.q.device, dtype=self.k_values.dtype)
        q = self.q.to(device=self.q.device)
        phase_nx = torch.exp(1j * modes[:, None] * y_mod[None, :]) / np.sqrt(2.0 * np.pi)
        u_xkm = torch.einsum('jnm,nx->jmx', q, phase_nx)
        bloch_phase = torch.exp(1j * k[:, None] * y[None, :])
        phi = u_xkm * bloch_phase[:, None, :] / np.sqrt(float(eps))
        dx = float((self._supercell_x_grid[1] - self._supercell_x_grid[0]).detach().cpu().item())
        phi_flat = phi.reshape(self.Nk * self.K, self._supercell_x_grid.numel())
        self._supercell_phi = phi
        basis_matrix = phi_flat.transpose(0, 1).contiguous()
        _, self._supercell_gram_inv, self._supercell_gram_diagnostics = factor_weighted_basis(basis_matrix, weight=dx, rcond=float(gram_rcond))
        diag = self._supercell_gram_diagnostics
        print(f'[stage2_basis] supercell Gram: rank={diag.effective_rank}/{diag.dimension}, cond={diag.condition_number:.3e}, pinv_residual={diag.pseudoinverse_residual:.3e}, cutoff={diag.cutoff:.3e}')

    def _require_supercell(self) -> None:
        """函数 `_require_supercell`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        if self._supercell_phi is None or self._supercell_x_grid is None or self._supercell_gram_inv is None:
            raise RuntimeError('Call setup_supercell_frame before using supercell projection/reconstruction.')

    def reconstruct_supercell_wavefunction_x(self, c: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """函数 `reconstruct_supercell_wavefunction_x`。参数：c, tau。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Reconstruct the physical wavefunction on D=[0,2pi] from interaction-picture coefficients."""
        self._require_supercell()
        squeeze = False
        if c.ndim == 2:
            c = c[None, ...]
            squeeze = True
        b = self.from_interaction(c, tau)
        m = int(b.shape[1])
        b_flat = b.transpose(1, 2).reshape(b.shape[0], self.Nk * m)
        phi_flat = self._supercell_phi[:, :m, :].reshape(self.Nk * m, self._supercell_x_grid.numel())
        psi = torch.einsum('bd,dx->bx', b_flat, phi_flat)
        return psi[0] if squeeze else psi

    def project_supercell_wavefunction_x(self, psi: torch.Tensor, tau: torch.Tensor, return_interaction: bool=True) -> torch.Tensor:
        """函数 `project_supercell_wavefunction_x`。参数：psi, tau, return_interaction。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Project a physical-domain wavefunction on D=[0,2pi] into the discrete Bloch frame."""
        self._require_supercell()
        squeeze = False
        if psi.ndim == 1:
            psi = psi[None, ...]
            squeeze = True
        phi_flat = self._supercell_phi.reshape(self.Nk * self.K, self._supercell_x_grid.numel())
        dx = float((self._supercell_x_grid[1] - self._supercell_x_grid[0]).detach().cpu().item())
        rhs = dx * torch.einsum('dx,bx->bd', torch.conj(phi_flat), psi.to(phi_flat.dtype))
        b_flat = torch.einsum('de,be->bd', self._supercell_gram_inv, rhs)
        b = b_flat.reshape(psi.shape[0], self.Nk, self.K).transpose(1, 2).contiguous()
        out = self.to_interaction(b, tau) if return_interaction else b
        return out[0] if squeeze else out

    def reconstruct_physical_wavefunction(self, c: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """函数 `reconstruct_physical_wavefunction`。参数：c, tau。沿用来源实现的计算、形状检查及返回值。"""
        if self._supercell_phi is not None:
            return self.reconstruct_supercell_wavefunction_x(c, tau)
        return self.reconstruct_wavefunction_x(c, tau)

    def tau_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """函数 `tau_from_t`。参数：t。沿用来源实现的计算、形状检查及返回值。"""
        if self.supercell_eps is None:
            raise ValueError('supercell_eps is not set. Call setup_supercell_frame(...) first or load trajectory data with x_grid/eps.')
        return t.to(dtype=self.evals.dtype, device=self.evals.device) / float(self.supercell_eps)

    def reconstruct_physical_wavefunction_t(self, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """函数 `reconstruct_physical_wavefunction_t`。参数：c, t。沿用来源实现的计算、形状检查及返回值。"""
        return self.reconstruct_physical_wavefunction(c, self.tau_from_t(t))


class LowBandProjector:
    """在完整 x 网格上构造八带采样矩阵，以加权 Gram 伪逆实现 P 与 I-P。"""

    def __init__(self, basis: FrozenBlochBasis1D, num_bands: int, x_grid: torch.Tensor, dx: float, rcond: float=1e-08, complex_dtype: Optional[torch.dtype]=None) -> None:
        """函数 `__init__`。参数：basis, num_bands, x_grid, dx, rcond, complex_dtype。沿用来源实现的计算、形状检查及返回值。"""
        self.basis = basis
        self.num_bands = int(num_bands)
        self.x_grid = x_grid.to(device=basis.q.device, dtype=basis.evals.dtype)
        self.dx = float(dx)
        self.rcond = float(rcond)
        self.complex_dtype = complex_dtype or basis.q.dtype
        if self.num_bands < 1:
            raise ValueError(f'num_bands must be positive, got {self.num_bands}.')
        if self.num_bands > basis.K_exported:
            raise ValueError(f'Requested {self.num_bands} bands, but basis has K_exported={basis.K_exported}.')
        if basis._supercell_phi is None:
            raise RuntimeError('LowBandProjector requires basis.setup_supercell_frame(...) before construction.')
        if int(basis._supercell_phi.shape[-1]) != int(self.x_grid.numel()):
            raise ValueError(f'x_grid has Nx={self.x_grid.numel()}, but basis supercell frame has Nx={basis._supercell_phi.shape[-1]}.')
        self._build_basis_matrix()

    def _build_basis_matrix(self) -> None:
        """函数 `_build_basis_matrix`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        phi = self.basis._supercell_phi[:, :self.num_bands, :].to(dtype=self.complex_dtype)
        self.basis_matrix = phi.reshape(self.basis.Nk * self.num_bands, phi.shape[-1]).transpose(0, 1).contiguous()
        self.nx = int(self.basis_matrix.shape[0])
        self.rank = int(self.basis_matrix.shape[1])
        self.gram, self.gram_inv, factor = factor_weighted_basis(self.basis_matrix, weight=self.dx, rcond=self.rcond)
        self.gram_eig_min = factor.eigenvalue_min
        self.gram_eig_max = factor.eigenvalue_max
        self.gram_condition = factor.condition_number
        self.effective_rank = factor.effective_rank
        self.gram_cutoff = factor.cutoff
        self.gram_pseudoinverse_residual = factor.pseudoinverse_residual

    def project(self, psi: torch.Tensor) -> torch.Tensor:
        """函数 `project`。参数：psi。沿用来源实现的计算、形状检查及返回值。"""
        z = psi.to(dtype=self.basis_matrix.dtype, device=self.basis_matrix.device)
        if int(z.shape[-1]) != self.nx:
            raise ValueError(f'Expected last dimension Nx={self.nx}, got {z.shape[-1]}.')
        rhs = self.dx * torch.einsum('nr,...n->...r', torch.conj(self.basis_matrix), z)
        coeff = torch.einsum('rs,...s->...r', self.gram_inv, rhs)
        return torch.einsum('nr,...r->...n', self.basis_matrix, coeff)

    def orthogonalize(self, psi: torch.Tensor) -> torch.Tensor:
        """函数 `orthogonalize`。参数：psi。沿用来源实现的计算、形状检查及返回值。"""
        return psi - self.project(psi)

    def project_low_x(self, psi: torch.Tensor) -> torch.Tensor:
        """函数 `project_low_x`。参数：psi。沿用来源实现的计算、形状检查及返回值。"""
        return self.project(psi)


MATHIEU2_ALIASES = {'mathieu2', 'mathieu_2', 'two_harmonic_mathieu'}


CORRELATED_RANDOM_KL_ALIASES = {'correlated_random_kl', 'correlated-random-kl', 'random_kl', 'periodic_random_kl'}


def int_value(data: Any, key: str, default: int) -> int:
    """函数 `int_value`。参数：data, key, default。沿用来源实现的计算、形状检查及返回值。"""
    if key not in data:
        return int(default)
    return int(np.asarray(data[key]).reshape(-1)[0])


def correlated_random_kl_lambda_values(Q: int, ell_c: float=0.25 * np.pi, lambda_mode: str='gaussian_spectral_decay') -> tuple[np.ndarray, str]:
    """函数 `correlated_random_kl_lambda_values`。参数：Q, ell_c, lambda_mode。沿用来源实现的计算、形状检查及返回值。"""
    Q = int(Q)
    if Q <= 0:
        raise ValueError(f'Q must be positive, got {Q}.')
    ell_c = float(ell_c)
    if ell_c <= 0.0:
        raise ValueError(f'ell_c must be positive, got {ell_c}.')
    mode = str(lambda_mode or 'gaussian_spectral_decay').lower()
    if mode == 'auto':
        mode = 'periodic_gaussian_bessel'
    q = np.arange(1, Q + 1, dtype=np.float64)
    if mode == 'periodic_gaussian_bessel':
        try:
            from scipy.special import iv
            z = 1.0 / ell_c ** 2
            values = np.exp(-z) * iv(q, z)
            values = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise FloatingPointError('invalid Bessel spectral coefficients')
            return (values, mode)
        except Exception:
            mode = 'gaussian_spectral_decay'
    if mode != 'gaussian_spectral_decay':
        raise ValueError(f'Unsupported lambda_mode={lambda_mode!r}.')
    return (np.exp(-0.5 * (q * ell_c) ** 2).astype(np.float64), mode)


def correlated_random_kl_normalizer(lambda_values: np.ndarray) -> float:
    """函数 `correlated_random_kl_normalizer`。参数：lambda_values。沿用来源实现的计算、形状检查及返回值。"""
    lam = np.asarray(lambda_values, dtype=np.float64).reshape(-1)
    if lam.size == 0 or np.any(lam <= 0.0):
        raise ValueError('lambda_values must be a nonempty positive vector.')
    return float(np.sqrt(1.0 / 3.0 * np.sum(lam)))


def correlated_random_kl_potential(x_grid: np.ndarray, xi: np.ndarray, ell_c: float=0.25 * np.pi, sigma_U: float=0.03, lambda_values: np.ndarray | None=None, lambda_mode: str='gaussian_spectral_decay', normalization_Z: float | None=None, subtract_mean: bool=True) -> np.ndarray:
    """函数 `correlated_random_kl_potential`。参数：x_grid, xi, ell_c, sigma_U, lambda_values, lambda_mode, normalization_Z, subtract_mean。沿用来源实现的计算、形状检查及返回值。"""
    x = np.asarray(x_grid, dtype=np.float64)
    coeff = np.asarray(xi, dtype=np.float64).reshape(-1)
    if coeff.size <= 0 or coeff.size % 2 != 0:
        raise ValueError(f'xi must have positive even size, got {coeff.size}.')
    Q = coeff.size // 2
    if lambda_values is None:
        lam, _ = correlated_random_kl_lambda_values(Q, ell_c=ell_c, lambda_mode=lambda_mode)
    else:
        lam = np.asarray(lambda_values, dtype=np.float64).reshape(-1)
        if lam.size != Q:
            raise ValueError(f'lambda_values has size {lam.size}, expected Q={Q}.')
    Z = correlated_random_kl_normalizer(lam) if normalization_Z is None else float(normalization_Z)
    if Z <= 0.0:
        raise ValueError(f'normalization_Z must be positive, got {Z}.')
    out = np.zeros_like(x, dtype=np.float64)
    for mode in range(1, Q + 1):
        scale = np.sqrt(float(lam[mode - 1]))
        a = float(coeff[2 * (mode - 1)])
        b = float(coeff[2 * (mode - 1) + 1])
        out += scale * (a * np.cos(mode * x) + b * np.sin(mode * x))
    out *= float(sigma_U) / Z
    if subtract_mean:
        out = out - float(np.mean(out))
    return out.astype(np.float64)


def mathieu_value(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """函数 `mathieu_value`。参数：y, mu。沿用来源实现的计算、形状检查及返回值。"""
    return float(np.asarray(mu, dtype=np.float64).reshape(-1)[0]) * np.cos(y)


def mathieu2_value(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """函数 `mathieu2_value`。参数：y, mu。沿用来源实现的计算、形状检查及返回值。"""
    mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    if mu_arr.size < 2:
        raise ValueError(f'mathieu2 requires [mu1, mu2], got shape {mu_arr.shape}.')
    return float(mu_arr[0]) * np.cos(y) + float(mu_arr[1]) * np.cos(2.0 * y)


def rectangular_value(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """函数 `rectangular_value`。参数：y, mu。沿用来源实现的计算、形状检查及返回值。"""
    a, width, center = map(float, np.asarray(mu, dtype=np.float64).reshape(-1))
    y = np.mod(np.asarray(y, dtype=np.float64), 2.0 * np.pi)
    left = np.mod(center - 0.5 * width, 2.0 * np.pi)
    right = np.mod(center + 0.5 * width, 2.0 * np.pi)
    if np.isclose(width, 0.0):
        indicator = np.zeros_like(y, dtype=np.float64)
    elif np.isclose(width, 2.0 * np.pi):
        indicator = np.ones_like(y, dtype=np.float64)
    elif left <= right:
        indicator = ((y >= left) & (y <= right)).astype(np.float64)
    else:
        indicator = ((y >= left) | (y <= right)).astype(np.float64)
    return a - a * indicator


def micro_mu_from_data(data: Any, traj_index: int | None=None) -> np.ndarray:
    """函数 `micro_mu_from_data`。参数：data, traj_index。沿用来源实现的计算、形状检查及返回值。"""
    if traj_index is not None and 'micro_mu_values' in data:
        return np.asarray(data['micro_mu_values'], dtype=np.float64)[int(traj_index)]
    if traj_index is not None and 'mu_values_by_traj' in data:
        return np.asarray(data['mu_values_by_traj'], dtype=np.float64)[int(traj_index)]
    if 'micro_mu' in data:
        return np.asarray(data['micro_mu'], dtype=np.float64)
    return np.array([1.0], dtype=np.float64)


def micro_potential_x_from_data(data: Any, x_grid: np.ndarray, eps: float, traj_index: int | None=None) -> np.ndarray:
    """函数 `micro_potential_x_from_data`。参数：data, x_grid, eps, traj_index。沿用来源实现的计算、形状检查及返回值。"""
    y_mod = np.mod(np.asarray(x_grid, dtype=np.float64) / float(eps), 2.0 * np.pi)
    micro_family = string_value(data, 'micro_potential_family', 'mathieu').lower()
    mu = micro_mu_from_data(data, traj_index=traj_index)
    if micro_family == 'mathieu':
        return mathieu_value(y_mod, mu).astype(np.float64)
    if micro_family in MATHIEU2_ALIASES:
        return mathieu2_value(y_mod, mu).astype(np.float64)
    if micro_family in {'rectangular', 'kronig_penney', 'kp', 'kronig-penney'}:
        return rectangular_value(y_mod, mu).astype(np.float64)
    raise ValueError(f'Unsupported micro_potential_family={micro_family!r}.')


def build_total_potential_x_from_data(data: Any, traj_index: int, x_grid: np.ndarray, eps: float) -> np.ndarray:
    """函数 `build_total_potential_x_from_data`。参数：data, traj_index, x_grid, eps。沿用来源实现的计算、形状检查及返回值。"""
    v_micro = micro_potential_x_from_data(data, x_grid, eps, traj_index=traj_index)
    u_external = external_potential_x_from_data(data, traj_index, x_grid)
    return (v_micro + u_external).astype(np.float64)


def scalar_value(data: np.lib.npyio.NpzFile, key: str, default: float) -> float:
    """函数 `scalar_value`。参数：data, key, default。沿用来源实现的计算、形状检查及返回值。"""
    if key not in data:
        return float(default)
    return float(np.asarray(data[key]).reshape(-1)[0])


def string_value(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    """函数 `string_value`。参数：data, key, default。沿用来源实现的计算、形状检查及返回值。"""
    if key not in data:
        return default
    return str(np.asarray(data[key]).reshape(-1)[0])


def mass_values(psi: torch.Tensor, dx: float) -> torch.Tensor:
    """函数 `mass_values`。参数：psi, dx。沿用来源实现的计算、形状检查及返回值。"""
    return dx * torch.sum(torch.abs(psi) ** 2, dim=-1)


def rel_l2_sq_by_time(pred: torch.Tensor, ref: torch.Tensor, dx: float, eps: float=1e-12) -> torch.Tensor:
    """函数 `rel_l2_sq_by_time`。参数：pred, ref, dx, eps。沿用来源实现的计算、形状检查及返回值。"""
    num = dx * torch.sum(torch.abs(pred - ref) ** 2, dim=-1)
    den = dx * torch.sum(torch.abs(ref) ** 2, dim=-1) + eps
    return num / den


def relative_abs_error(pred: torch.Tensor, ref: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    """函数 `relative_abs_error`。参数：pred, ref, eps。沿用来源实现的计算、形状检查及返回值。"""
    return torch.abs(pred - ref) / (torch.abs(ref) + eps)


def external_potential_x_from_data(data: Any, traj_index: int, x_grid: np.ndarray) -> np.ndarray:
    """函数 `external_potential_x_from_data`。参数：data, traj_index, x_grid。沿用来源实现的计算、形状检查及返回值。"""
    x = np.asarray(x_grid, dtype=np.float64)
    family = string_value(data, 'external_family', 'harmonic').lower()
    values_key = 'external_values' if 'external_values' in data else 'U_values'
    u_values = np.asarray(data[values_key], dtype=np.float64)
    coeff = np.asarray(u_values[int(traj_index)], dtype=np.float64).reshape(-1)
    if family == 'harmonic':
        center = scalar_value(data, 'harmonic_center', np.pi)
        return (float(coeff[0]) * (x - center) ** 2).astype(np.float64)
    if family == 'step':
        left = scalar_value(data, 'step_left', np.pi / 2.0)
        right = scalar_value(data, 'step_right', 3.0 * np.pi / 2.0)
        out = np.zeros_like(x, dtype=np.float64)
        out[(x >= left) & (x <= right)] = float(coeff[0])
        return out
    if family in CORRELATED_RANDOM_KL_ALIASES:
        d_xi = int_value(data, 'd_xi', coeff.size)
        Q = int_value(data, 'correlated_kl_Q', d_xi // 2)
        ell_c = scalar_value(data, 'ell_c', 0.25 * np.pi)
        sigma_U = scalar_value(data, 'sigma_U', scalar_value(data, 'sigma_u', 0.03))
        lambda_mode = string_value(data, 'lambda_mode', 'gaussian_spectral_decay')
        if 'lambda_values' in data:
            lam = np.asarray(data['lambda_values'], dtype=np.float64).reshape(-1)[:Q]
        else:
            lam, lambda_mode = correlated_random_kl_lambda_values(Q, ell_c=ell_c, lambda_mode=lambda_mode)
        Z = scalar_value(data, 'normalization_Z', correlated_random_kl_normalizer(lam))
        return correlated_random_kl_potential(x, coeff[:d_xi], ell_c=ell_c, sigma_U=sigma_U, lambda_values=lam, lambda_mode=lambda_mode, normalization_Z=Z, subtract_mean=True)
    raise ValueError(f'Unsupported external_family={family!r}.')

# ============================================================================
# 三、网络结构与完整 forward
# ============================================================================


def make_mlp(in_dim: int, out_dim: int, width: int, depth: int, activation: str) -> nn.Sequential:
    """函数 `make_mlp`。参数：in_dim, out_dim, width, depth, activation。沿用来源实现的计算、形状检查及返回值。"""
    if depth < 1:
        raise ValueError('depth must be at least 1.')
    if activation == 'tanh':
        act: nn.Module = nn.Tanh()
    elif activation == 'gelu':
        act = nn.GELU()
    elif activation == 'silu':
        act = nn.SiLU()
    else:
        raise ValueError(f'Unsupported activation={activation!r}.')
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(last, width))
        layers.append(act)
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def zero_initialize_last_linear(module: nn.Sequential) -> None:
    """函数 `zero_initialize_last_linear`。参数：module。沿用来源实现的计算、形状检查及返回值。"""
    for layer in reversed(module):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class FourierFeatureMap(nn.Module):
    """对物理坐标和参数构造 Fourier 特征，保持正式 sin/cos 排列与频率。"""

    def __init__(self, in_dim: int, num_frequencies: int, include_input: bool=True):
        """函数 `__init__`。参数：in_dim, num_frequencies, include_input。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        freqs = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self) -> int:
        """函数 `out_dim`。参数：无显式参数。沿用来源实现的计算、形状检查及返回值。"""
        base = self.in_dim if self.include_input else 0
        return base + 2 * self.in_dim * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """函数 `forward`。参数：x。沿用来源实现的计算、形状检查及返回值。"""
        parts = [x] if self.include_input else []
        angles = 2.0 * np.pi * x[..., None, :] * self.freqs[:, None]
        parts.extend([torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)])
        return torch.cat(parts, dim=-1)


class PhysicalCorrectorOperatorNet(nn.Module):
    """输入时间、x、微观坐标 x/eps 和归一化参数；输出实/虚两通道修正。"""

    def __init__(self, u_dim: int, t_final: float, x_length: float, eps: float, width: int=CORRECT_WIDTH, depth: int=CORRECT_DEPTH, fourier_frequencies: int=CORRECT_FOURIER_FREQUENCIES, activation: str=ACTIVATION) -> None:
        """函数 `__init__`。参数：u_dim, t_final, x_length, eps, width, depth, fourier_frequencies, activation。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.u_dim = int(u_dim)
        self.t_final = float(max(t_final, 1e-12))
        self.x_length = float(max(x_length, 1e-12))
        self.eps = float(eps)
        self.features = FourierFeatureMap(in_dim=3 + self.u_dim, num_frequencies=fourier_frequencies, include_input=True)
        self.net = make_mlp(self.features.out_dim, 2, width, depth, activation)
        zero_initialize_last_linear(self.net)

    def forward(self, t_values: torch.Tensor, x_grid: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """函数 `forward`。参数：t_values, x_grid, xi_norm。沿用来源实现的计算、形状检查及返回值。"""
        batch = int(xi_norm.shape[0])
        nt = int(t_values.numel())
        nx = int(x_grid.numel())
        t_norm = (t_values.reshape(1, nt, 1, 1) / self.t_final).expand(batch, nt, nx, 1)
        x_norm = (x_grid.reshape(1, 1, nx, 1) / self.x_length).expand(batch, nt, nx, 1)
        y_norm = torch.remainder(x_grid / self.eps, 2.0 * math.pi).reshape(1, 1, nx, 1).expand(batch, nt, nx, 1) / (2.0 * math.pi)
        xi_grid = xi_norm.reshape(batch, 1, 1, self.u_dim).expand(batch, nt, nx, self.u_dim)
        mapped = self.features(torch.cat([t_norm, x_norm, y_norm, xi_grid], dim=-1))
        raw = self.net(mapped)
        return torch.complex(raw[..., 0], raw[..., 1])


def translation_phase_features(k_values: torch.Tensor, x0: float, eps: float) -> torch.Tensor:
    """按 float64 计算 k*x0/eps 的 sin/cos，再转网络精度；仅作为输入特征。"""
    phase = torch.remainder(k_values.double() * (float(x0) / float(eps)), 2.0 * math.pi)
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1).float()


class FNO2D(nn.Module):
    """沿时间和准动量的系数 FNO；FFT 截取低模态，局部卷积补充逐点映射。"""

    def __init__(self, in_channels: int, out_channels: int, width: int, modes_t: int, modes_k: int, layers: int) -> None:
        """函数 `__init__`。参数：in_channels, out_channels, width, modes_t, modes_k, layers。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spec_layers = nn.ModuleList([SpectralConv2D(width, width, modes_t, modes_k) for _ in range(layers)])
        self.mix_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(layers)])
        self.out_proj = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """函数 `forward`。参数：x。沿用来源实现的计算、形状检查及返回值。"""
        hidden = self.in_proj(x)
        for spectral, pointwise in zip(self.spec_layers, self.mix_layers):
            hidden = torch.nn.functional.gelu(spectral(hidden) + pointwise(hidden))
        return self.out_proj(hidden)


def zero_initialize_conv(layer: nn.Conv2d) -> None:
    """函数 `zero_initialize_conv`。参数：layer。沿用来源实现的计算、形状检查及返回值。"""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class BlochCoefficientFNO(nn.Module):
    """预测 interaction-frame 系数 c(t,k)，通过 c0+t/T*delta 保留固定初值。"""

    def __init__(self, u_dim: int, num_bands: int, t_final: float, k_values: torch.Tensor, c0: torch.Tensor, x0: float, eps: float, width: int=MAIN_WIDTH, layers: int=MAIN_DEPTH, modes_t: int=MAIN_MODES_T, modes_k: int=MAIN_MODES_K) -> None:
        """函数 `__init__`。参数：u_dim, num_bands, t_final, k_values, c0, x0, eps, width, layers, modes_t, modes_k。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.u_dim = int(u_dim)
        self.num_bands = int(num_bands)
        self.t_final = float(max(t_final, 1e-12))
        physical_k = k_values.detach().clone()
        k_min = torch.min(physical_k)
        k_span = torch.max(physical_k) - k_min
        if torch.abs(k_span) < 1e-12:
            raise ValueError('Degenerate physical k grid')
        self.register_buffer('physical_k_values', physical_k)
        self.register_buffer('k_min', k_min)
        self.register_buffer('k_span', k_span)
        self.register_buffer('phase_features', translation_phase_features(physical_k, x0, eps))
        self.register_buffer('c0', c0.detach().clone())
        self.fno = FNO2D(in_channels=4 + self.u_dim, out_channels=2 * self.num_bands, width=width, modes_t=modes_t, modes_k=modes_k, layers=layers)
        zero_initialize_conv(self.fno.out_proj)

    def forward(self, t_values: torch.Tensor, k_values: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        """函数 `forward`。参数：t_values, k_values, xi_norm。沿用来源实现的计算、形状检查及返回值。"""
        batch = int(xi_norm.shape[0])
        nt = int(t_values.numel())
        nk = int(k_values.numel())
        if nk != int(self.physical_k_values.numel()):
            raise ValueError(f'Runtime Nk={nk} differs from initialized Nk={self.physical_k_values.numel()}')
        if not torch.allclose(k_values, self.physical_k_values.to(k_values), rtol=1e-06, atol=1e-07):
            raise ValueError('Runtime k grid differs from FrozenBlochBasis1D.k_values')
        t_norm = (t_values.reshape(1, nt, 1, 1) / self.t_final).expand(batch, nt, nk, 1)
        kappa = ((k_values - self.k_min) / self.k_span).reshape(1, 1, nk, 1)
        kappa = kappa.expand(batch, nt, nk, 1)
        phase = self.phase_features.to(k_values).reshape(1, 1, nk, 2).expand(batch, nt, nk, 2)
        xi_features = xi_norm.reshape(batch, 1, 1, self.u_dim).expand(batch, nt, nk, self.u_dim)
        trunk = torch.cat([t_norm, kappa, phase, xi_features], dim=-1)
        raw = self.fno(trunk.permute(0, 3, 1, 2).contiguous())
        raw = raw.permute(0, 2, 3, 1).contiguous().reshape(batch, nt, nk, self.num_bands, 2)
        delta_c = torch.complex(raw[..., 0], raw[..., 1]).permute(0, 1, 3, 2).contiguous()
        t_factor = (t_values / self.t_final).clamp(0.0, 1.0).reshape(1, nt, 1, 1)
        c_pred = self.c0.reshape(1, 1, self.num_bands, nk) + t_factor * delta_c
        return c_pred


def reconstruct_bloch_batch(basis: FrozenBlochBasis1D, c_pred: torch.Tensor, t_values: torch.Tensor) -> torch.Tensor:
    """恢复 exp(-itE/eps) 谱相位后执行 Bloch 合成，返回 [B,Nt,Nx]。"""
    batch, nt, num_bands, nk = c_pred.shape
    c_flat = c_pred.reshape(batch * nt, num_bands, nk)
    t_flat = t_values.reshape(1, nt).expand(batch, nt).reshape(batch * nt)
    psi_flat = basis.reconstruct_physical_wavefunction_t(c_flat, t_flat)
    return psi_flat.reshape(batch, nt, -1)


def forward_model(coeff_net: BlochCoefficientFNO, correct_net: PhysicalCorrectorOperatorNet, basis: FrozenBlochBasis1D, projector: LowBandProjector, t_values: torch.Tensor, k_values: torch.Tensor, x_grid: torch.Tensor, xi_norm: torch.Tensor, r0: torch.Tensor, t_final: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """完整前向：系数重构 + r0 + (t/T)*(I-P)^2 corrector，不做事后归一化。"""
    c_pred = coeff_net(t_values, k_values, xi_norm)
    psi_bloch = reconstruct_bloch_batch(basis, c_pred, t_values)
    correct_raw_delta = correct_net(t_values, x_grid, xi_norm)
    correct_delta = projector.orthogonalize(correct_raw_delta)
    correct_delta = projector.orthogonalize(correct_delta)
    t_factor = (t_values / float(max(t_final, 1e-12))).clamp(0.0, 1.0).reshape(1, -1, 1)
    r0_view = r0.to(device=psi_bloch.device, dtype=psi_bloch.dtype).reshape(1, 1, -1)
    correct_raw = r0_view + t_factor * correct_raw_delta
    correct = r0_view + t_factor * correct_delta
    psi_pred = psi_bloch + correct
    return (c_pred, psi_bloch, correct_raw, correct, psi_pred)


class SpectralConv2D(nn.Module):
    """对二维 FFT 的正/负频率块分别乘学习权重，逆变换恢复网格。"""

    def __init__(self, in_channels: int, out_channels: int, modes_t: int, modes_k: int):
        """函数 `__init__`。参数：in_channels, out_channels, modes_t, modes_k。沿用来源实现的计算、形状检查及返回值。"""
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_t = int(modes_t)
        self.modes_k = int(modes_k)
        scale = 1.0 / math.sqrt(max(1, in_channels * out_channels))
        self.weight_pos_real = nn.Parameter(scale * torch.randn(out_channels, in_channels, self.modes_t, self.modes_k))
        self.weight_pos_imag = nn.Parameter(scale * torch.randn(out_channels, in_channels, self.modes_t, self.modes_k))
        self.weight_neg_real = nn.Parameter(scale * torch.randn(out_channels, in_channels, self.modes_t, self.modes_k))
        self.weight_neg_imag = nn.Parameter(scale * torch.randn(out_channels, in_channels, self.modes_t, self.modes_k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """函数 `forward`。参数：x。沿用来源实现的计算、形状检查及返回值。"""
        batch, _, nt, nk = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(batch, self.out_channels, nt, nk // 2 + 1, dtype=x_ft.dtype, device=x.device)
        mt = min(self.modes_t, max(1, nt // 2))
        mk = min(self.modes_k, nk // 2 + 1)
        w_pos = torch.complex(self.weight_pos_real[..., :mt, :mk], self.weight_pos_imag[..., :mt, :mk])
        out_ft[:, :, :mt, :mk] = torch.einsum('bcxy,ocxy->boxy', x_ft[:, :, :mt, :mk], w_pos)
        w_neg = torch.complex(self.weight_neg_real[..., :mt, :mk], self.weight_neg_imag[..., :mt, :mk])
        out_ft[:, :, -mt:, :mk] = torch.einsum('bcxy,ocxy->boxy', x_ft[:, :, -mt:, :mk], w_neg)
        neg_t = torch.remainder(-torch.arange(nt, device=out_ft.device), nt)
        boundary_columns = (0, nk // 2) if nk % 2 == 0 else (0,)
        for column in boundary_columns:
            boundary = out_ft[..., column].clone()
            out_ft[..., column] = 0.5 * (boundary + torch.conj(boundary.index_select(-1, neg_t)))
        return torch.fft.irfft2(out_ft, s=(nt, nk), dim=(-2, -1))


class BlochONet1D(nn.Module):
    """外势参数 -> 完整复波函数 [B,11,Nx]。

    coeff 预测 interaction-frame 的八带系数，correct 预测物理空间修正。
    固定初值拆为低带 c0 和正交残差 r0；c0、r0、参数均值/标准差均为
    buffer，随 state_dict 保存。外部 basis/Gram 矩阵不写入权重文件。
    请在目标设备上构造模型，以保持外部 Bloch 变换与网络位于同一设备。
    """
    def __init__(self, cfg, basis_path, device, state=None, training_data=None):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        g = cfg['grid']
        p = cfg['physics']
        self.eps = g['eps']
        self.dx = g['length'] / g['nx']
        self.x_np = g['x_min'] + np.arange(g['nx'], dtype=np.float64) * self.dx
        self.register_buffer('x', torch.tensor(self.x_np, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer('t', torch.tensor(g['times'], dtype=torch.float32, device=device), persistent=False)
        self.tf = float(g['times'][-1])
        self.basis = FrozenBlochBasis1D.from_stage1_dataset(basis_path, mu_index=0,
            real_dtype=torch.float32, drop_redundant_endpoint=True, device=device)
        if self.basis.Nk != g['num_cells'] or self.basis.K_exported < NUM_BANDS:
            raise ValueError('basis 的 Nk/带数不匹配')
        self.basis.setup_supercell_frame(self.x, eps=self.eps, num_cells=g['num_cells'])
        self.register_buffer('k', self.basis.k_values.to(device=device, dtype=torch.float32), persistent=False)
        self.projector = LowBandProjector(self.basis, NUM_BANDS, self.x, self.dx,
            rcond=PROJECTOR_RCOND, complex_dtype=torch.complex64)

        if state is None:
            if training_data is None:
                raise ValueError('新训练需要完整数据；已有权重推理不需要标签')
            data = training_data
            raw = np.asarray(data['U_values'], dtype=np.float64).reshape(len(data['U_values']), -1)
            ids = np.asarray(cfg['split']['train_indices'], dtype=np.int64)
            mean = raw[ids].mean(0)
            std = raw[ids].std(0)
            std = np.where(std < 1e-8, 1.0, std)
            all_initial = np.asarray(data['psi_real'][:, 0]) + 1j * np.asarray(data['psi_imag'][:, 0])
            validate_fixed_initial_state(all_initial, p['initial_condition_tol'], 'psi0')
            psi0 = torch.tensor(all_initial[0], dtype=torch.complex64, device=device)
            with torch.no_grad():
                seed_c = self.basis.project_supercell_wavefunction_x(psi0[None], tau=self.t[:1] * 0, return_interaction=True)[0, :NUM_BANDS]
                c0, r0, leakage, error = refined_initial_decomposition(self.basis, self.projector, psi0, seed_c, self.t[:1], self.dx)
            if leakage > p['initial_projector_tol'] or error > max(p['initial_condition_tol'], 2 * torch.finfo(torch.float32).eps):
                raise ValueError('固定初值分解不满足正式数值约束')
        else:
            mean = state['mean']
            std = state['std']
            c0 = state['coeff.c0'].to(device)
            r0 = state['r0'].to(device)
        mean = torch.as_tensor(mean, dtype=torch.float64, device=device)
        std = torch.as_tensor(std, dtype=torch.float64, device=device)
        # 正式统计只由真正使用的 train128 计算；不能掺入训练池剩余六条。
        if not np.array_equal(mean.cpu().numpy(), np.asarray(p['xi_mean'])) or not np.array_equal(std.cpu().numpy(), np.asarray(p['xi_std'])):
            raise ValueError('归一化统计与原始 train128 不一致')
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clone())
        self.register_buffer('r0', torch.as_tensor(r0, dtype=torch.complex64, device=device).clone())
        self.coeff = BlochCoefficientFNO(u_dim=p['d_xi'], num_bands=NUM_BANDS, t_final=self.tf,
            k_values=self.k, c0=c0, x0=p['x0'], eps=self.eps, width=MAIN_WIDTH,
            layers=MAIN_DEPTH, modes_t=MAIN_MODES_T, modes_k=MAIN_MODES_K).to(device)
        self.correct = PhysicalCorrectorOperatorNet(u_dim=p['d_xi'], t_final=self.tf,
            x_length=g['length'], eps=self.eps, width=CORRECT_WIDTH, depth=CORRECT_DEPTH,
            fourier_frequencies=CORRECT_FOURIER_FREQUENCIES, activation=ACTIVATION).to(device)
        if state is not None:
            self.load_state_dict(state, strict=True)

    def forward(self, parameters):
        """float64 归一化再转 float32；联合输出 11 帧，不是逐步时间积分。"""
        raw = torch.as_tensor(parameters, dtype=torch.float64, device=self.mean.device).reshape(-1, len(self.mean))
        normalized = ((raw - self.mean) / self.std).float()
        # forward_model 内部保留两次正交补投影，以及 c0/r0 初值约束。
        return forward_model(self.coeff, self.correct, self.basis, self.projector,
                             self.t, self.k, self.x, normalized, self.r0, self.tf)[-1]


def one_d_build(cfg, basis_path, device, weights=None, training_data=None):
    """旧正式字典 -> 新 nn.Module state_dict 的显式、一对一转换。

    coeff_net_state[k] -> coeff.k；correct_net_state[k] -> correct.k。
    原来的 xi_mean/xi_std/r0 -> mean/std/r0 buffer，c0 已包含在 coeff.c0。
    兼容早期整理版的附加留档字段；网络参数仍严格加载，不改变数值精度。
    """
    state = weights
    if weights is not None and 'coeff_net_state' in weights:
        if not torch.equal(torch.as_tensor(weights['c0']), weights['coeff_net_state']['c0'].cpu()):
            raise ValueError('旧权重中的两个 c0 副本不一致')
        state = {'coeff.' + key: value for key, value in weights['coeff_net_state'].items()}
        state.update({'correct.' + key: value for key, value in weights['correct_net_state'].items()})
        state['mean'] = torch.as_tensor(weights['xi_mean'], dtype=torch.float64)
        state['std'] = torch.as_tensor(weights['xi_std'], dtype=torch.float64)
        state['r0'] = torch.as_tensor(weights['r0'], dtype=torch.complex64)
    elif weights is not None:
        # 兼容早期整理版中不参与波函数计算的附加留档 buffer。
        state = weights.copy()
        state.pop('asset_identity', None)
    return BlochONet1D(cfg, basis_path, device, state=state, training_data=training_data)



# ============================================================================
# 四、数据读取、损失与评估指标
# ============================================================================


def ecm_spectral_derivative_x(psi: torch.Tensor, dx: float) -> torch.Tensor:
    """函数 `spectral_derivative_x`。参数：psi, dx。沿用来源实现的计算、形状检查及返回值。"""
    n = int(psi.shape[-1])
    xi = 2.0 * np.pi * torch.fft.fftfreq(n, d=float(dx), device=psi.device, dtype=psi.real.dtype)
    shape = [1] * psi.ndim
    shape[-1] = n
    return torch.fft.ifft(1j * xi.reshape(shape) * torch.fft.fft(psi, dim=-1), dim=-1)


def metric__check_complex_btx(value: torch.Tensor, name: str) -> None:
    """函数 `_check_complex_btx`。参数：value, name。沿用来源实现的计算、形状检查及返回值。"""
    if value.ndim != 3:
        raise ValueError(f'{name} must have shape [batch, time, x], got {tuple(value.shape)}.')
    if not torch.is_complex(value):
        raise TypeError(f'{name} must be a complex tensor.')


def metric__check_pair(pred: torch.Tensor, ref: torch.Tensor) -> None:
    """函数 `_check_pair`。参数：pred, ref。沿用来源实现的计算、形状检查及返回值。"""
    metric__check_complex_btx(pred, 'pred')
    metric__check_complex_btx(ref, 'ref')
    if pred.shape != ref.shape:
        raise ValueError(f'pred/ref shapes differ: {tuple(pred.shape)} != {tuple(ref.shape)}.')


def metric__time_mean(values: torch.Tensor, *, exclude_t0: bool) -> torch.Tensor:
    """函数 `_time_mean`。参数：values。沿用来源实现的计算、形状检查及返回值。"""
    if values.ndim != 2:
        raise ValueError(f'Expected [batch, time], got {tuple(values.shape)}.')
    if exclude_t0:
        if values.shape[1] <= 1:
            return torch.full((values.shape[0],), float('nan'), dtype=values.dtype, device=values.device)
        values = values[:, 1:]
    return torch.mean(values, dim=1)


def metric_relative_l2_by_time(pred: torch.Tensor, ref: torch.Tensor, dx: float, *, denom_eps: float=1e-12) -> torch.Tensor:
    """空间复相对 L2 -> [B,Nt]；不对齐全局相位，不重新归一化。"""
    metric__check_pair(pred, ref)
    num = float(dx) * torch.sum(torch.abs(pred - ref).square(), dim=-1)
    den = float(dx) * torch.sum(torch.abs(ref).square(), dim=-1)
    return torch.sqrt(torch.clamp(num, min=0.0)) / (torch.sqrt(torch.clamp(den, min=0.0)) + float(denom_eps))


def metric_real_relative_l2_by_time(pred: torch.Tensor, ref: torch.Tensor, dx: float, *, denom_eps: float=1e-12) -> torch.Tensor:
    """函数 `real_relative_l2_by_time`。参数：pred, ref, dx。沿用来源实现的计算、形状检查及返回值。"""
    if pred.ndim != 3 or ref.ndim != 3 or pred.shape != ref.shape:
        raise ValueError('real pred/ref must share shape [batch, time, x].')
    num = float(dx) * torch.sum((pred - ref).square(), dim=-1)
    den = float(dx) * torch.sum(ref.square(), dim=-1)
    return torch.sqrt(torch.clamp(num, min=0.0)) / (torch.sqrt(torch.clamp(den, min=0.0)) + float(denom_eps))


def metric_full_spacetime_relative_l2(pred: torch.Tensor, ref: torch.Tensor, dx: float, *, denom_eps: float=1e-12) -> torch.Tensor:
    """函数 `full_spacetime_relative_l2`。参数：pred, ref, dx。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Appendix metric: one time-space relative norm per trajectory."""
    if pred.ndim != 3 or ref.ndim != 3 or pred.shape != ref.shape:
        raise ValueError('pred/ref must share shape [batch, time, x].')
    reduce_dims = (1, 2)
    if torch.is_complex(pred) != torch.is_complex(ref):
        raise TypeError('pred/ref must either both be complex or both be real.')
    diff_sq = torch.abs(pred - ref).square() if torch.is_complex(pred) else (pred - ref).square()
    ref_sq = torch.abs(ref).square() if torch.is_complex(ref) else ref.square()
    num = float(dx) * torch.sum(diff_sq, dim=reduce_dims)
    den = float(dx) * torch.sum(ref_sq, dim=reduce_dims)
    return torch.sqrt(torch.clamp(num, min=0.0)) / (torch.sqrt(torch.clamp(den, min=0.0)) + float(denom_eps))


def metric_mass_curve(psi: torch.Tensor, dx: float) -> torch.Tensor:
    """函数 `mass_curve`。参数：psi, dx。沿用来源实现的计算、形状检查及返回值。"""
    metric__check_complex_btx(psi, 'psi')
    return float(dx) * torch.sum(torch.abs(psi).square(), dim=-1)


def metric_current_density(psi: torch.Tensor, eps: float, dx: float) -> torch.Tensor:
    """函数 `current_density`。参数：psi, eps, dx。沿用来源实现的计算、形状检查及返回值。"""
    metric__check_complex_btx(psi, 'psi')
    psi_x = ecm_spectral_derivative_x(psi, dx)
    return float(eps) * torch.imag(torch.conj(psi) * psi_x)


def metric_energy_curve(psi: torch.Tensor, potential_x: torch.Tensor, eps: float, dx: float, *, scaled: bool) -> torch.Tensor:
    """函数 `energy_curve`。参数：psi, potential_x, eps, dx。沿用来源实现的计算、形状检查及返回值。"""
    metric__check_complex_btx(psi, 'psi')
    potential = potential_x.to(device=psi.device, dtype=psi.real.dtype)
    if potential.ndim == 1:
        potential = potential[None, :]
    if potential.ndim != 2 or potential.shape[-1] != psi.shape[-1]:
        raise ValueError(f'potential_x must have shape [batch, x] or [x], got {tuple(potential.shape)}.')
    if potential.shape[0] not in (1, psi.shape[0]):
        raise ValueError('potential_x batch dimension must be 1 or match psi.')
    psi_x = ecm_spectral_derivative_x(psi, dx)
    density = torch.abs(psi).square()
    if scaled:
        kinetic = 0.5 * float(eps) * torch.abs(psi_x).square()
        potential_term = potential[:, None, :] / float(eps) * density
    else:
        kinetic = 0.5 * float(eps) ** 2 * torch.abs(psi_x).square()
        potential_term = potential[:, None, :] * density
    return float(dx) * torch.sum(kinetic + potential_term, dim=-1).real


def metric__relative_abs_by_time(pred: torch.Tensor, ref: torch.Tensor, *, denom_eps: float) -> torch.Tensor:
    """函数 `_relative_abs_by_time`。参数：pred, ref。沿用来源实现的计算、形状检查及返回值。"""
    return torch.abs(pred - ref) / (torch.abs(ref) + float(denom_eps))


def metric__curve_fields(prefix: str, curve: torch.Tensor) -> dict[str, torch.Tensor]:
    """函数 `_curve_fields`。参数：prefix, curve。沿用来源实现的计算、形状检查及返回值。"""
    return {f'{prefix}_mean': metric__time_mean(curve, exclude_t0=False), f'{prefix}_mean_excluding_t0': metric__time_mean(curve, exclude_t0=True), f'{prefix}_final': curve[:, -1]}


def metric_metrics_per_trajectory(pred: torch.Tensor, ref: torch.Tensor, potential_x: torch.Tensor, eps: float, dx: float, *, denom_eps: float=1e-12) -> dict[str, torch.Tensor]:
    """函数 `metrics_per_trajectory`。参数：pred, ref, potential_x, eps, dx。沿用来源实现的计算、形状检查及返回值。

原始接口与公式说明：
Canonical paper metrics, with every value retained per trajectory."""
    metric__check_pair(pred, ref)
    psi_rel = metric_relative_l2_by_time(pred, ref, dx, denom_eps=denom_eps)
    rho_pred = torch.abs(pred).square()
    rho_ref = torch.abs(ref).square()
    density_rel = metric_real_relative_l2_by_time(rho_pred, rho_ref, dx, denom_eps=denom_eps)
    mass_pred = metric_mass_curve(pred, dx)
    mass_ref = metric_mass_curve(ref, dx).detach()
    mass_error = metric__relative_abs_by_time(mass_pred, mass_ref, denom_eps=denom_eps)
    mass_drift = metric__relative_abs_by_time(mass_pred, mass_pred[:, :1], denom_eps=denom_eps)
    current_pred = metric_current_density(pred, eps, dx)
    current_ref = metric_current_density(ref, eps, dx).detach()
    current_rel = metric_real_relative_l2_by_time(current_pred, current_ref, dx, denom_eps=denom_eps)
    energy_pred = metric_energy_curve(pred, potential_x, eps, dx, scaled=True)
    energy_ref = metric_energy_curve(ref, potential_x, eps, dx, scaled=True).detach()
    energy_error = metric__relative_abs_by_time(energy_pred, energy_ref, denom_eps=denom_eps)
    energy_drift = metric__relative_abs_by_time(energy_pred, energy_pred[:, :1], denom_eps=denom_eps)
    hamiltonian_pred = metric_energy_curve(pred, potential_x, eps, dx, scaled=False)
    hamiltonian_ref = metric_energy_curve(ref, potential_x, eps, dx, scaled=False).detach()
    hamiltonian_error = metric__relative_abs_by_time(hamiltonian_pred, hamiltonian_ref, denom_eps=denom_eps)
    hamiltonian_drift = metric__relative_abs_by_time(hamiltonian_pred, hamiltonian_pred[:, :1], denom_eps=denom_eps)
    out: dict[str, torch.Tensor] = {'initial_psi_rel_l2': psi_rel[:, 0], 'initial_density_rel_l2': density_rel[:, 0], 'psi_rel_l2_full_spacetime': metric_full_spacetime_relative_l2(pred, ref, dx, denom_eps=denom_eps), 'density_rel_l2_full_spacetime': metric_full_spacetime_relative_l2(rho_pred, rho_ref, dx, denom_eps=denom_eps), 'mass_rel_error_max': torch.max(mass_error, dim=1).values, 'mass_rel_drift_max': torch.max(mass_drift, dim=1).values, 'energy_rel_drift_max': torch.max(energy_drift, dim=1).values, 'energy_scaled_rel_drift_max': torch.max(energy_drift, dim=1).values, 'hamiltonian_rel_drift_max': torch.max(hamiltonian_drift, dim=1).values}
    out.update(metric__curve_fields('psi_rel_l2', psi_rel))
    out.update(metric__curve_fields('density_rel_l2', density_rel))
    out.update(metric__curve_fields('mass_rel_error', mass_error))
    out.update(metric__curve_fields('mass_rel_drift', mass_drift))
    out.update(metric__curve_fields('current_rel_l2', current_rel))
    out.update(metric__curve_fields('energy_rel_error', energy_error))
    out.update(metric__curve_fields('energy_rel_drift', energy_drift))
    out.update(metric__curve_fields('energy_scaled_rel_error', energy_error))
    out.update(metric__curve_fields('energy_scaled_rel_drift', energy_drift))
    out.update(metric__curve_fields('hamiltonian_rel_error', hamiltonian_error))
    out.update(metric__curve_fields('hamiltonian_rel_drift', hamiltonian_drift))
    return out


def metric_aggregate_metrics(pred: torch.Tensor, ref: torch.Tensor, potential_x: torch.Tensor, eps: float, dx: float, *, denom_eps: float=1e-12) -> dict[str, float]:
    """逐轨迹/逐帧论文指标，再作算术平均；最大值字段保留总体最大值。"""
    per_traj = metric_metrics_per_trajectory(pred, ref, potential_x, eps, dx, denom_eps=denom_eps)
    out: dict[str, float] = {}
    for key, values in per_traj.items():
        reduced = torch.max(values) if key.endswith('_max') else torch.mean(values)
        out[key] = float(reduced.detach().cpu())
    return out


def compute_training_loss(psi_pred: torch.Tensor, psi_ref: torch.Tensor, *, dx: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """逐帧复相对 L2 的平方均值 + 0.01*参考质量相对误差平方均值。"""
    data_loss = torch.mean(rel_l2_sq_by_time(psi_pred, psi_ref, dx))
    mass_ref = mass_values(psi_ref, dx).detach()
    mass_pred = mass_values(psi_pred, dx)
    mass_loss = torch.mean(relative_abs_error(mass_pred, mass_ref).square())
    total = data_loss + LAMBDA_MASS * mass_loss
    return (total, {'complex_l2': data_loss, 'mass_reference': mass_loss})


def refined_initial_decomposition(basis: FrozenBlochBasis1D, projector: LowBandProjector, psi0: torch.Tensor, c0: torch.Tensor, t0: torch.Tensor, dx: float, iterations: int=5) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """按原五轮加权最小二乘精化 c0，计算正交初值残差及泄漏/重构误差。"""
    current_c = c0.clone()
    for _ in range(iterations):
        psi_bloch0 = reconstruct_bloch_batch(basis, current_c.reshape(1, 1, NUM_BANDS, basis.Nk), t0)[0, 0]
        residual = psi0 - psi_bloch0
        rhs = projector.dx * torch.einsum('nr,n->r', torch.conj(projector.basis_matrix), residual.to(projector.basis_matrix))
        delta_flat = torch.einsum('rs,s->r', projector.gram_inv, rhs)
        delta_c = delta_flat.reshape(basis.Nk, NUM_BANDS).transpose(0, 1).contiguous()
        current_c = current_c + delta_c.to(current_c)
    psi_bloch0 = reconstruct_bloch_batch(basis, current_c.reshape(1, 1, NUM_BANDS, basis.Nk), t0)[0, 0]
    residual = psi0 - psi_bloch0
    for _ in range(1):
        residual = projector.orthogonalize(residual)
    low = projector.project_low_x(residual)
    residual_norm = torch.sqrt(dx * torch.sum(torch.abs(residual).square()))
    low_norm = torch.sqrt(dx * torch.sum(torch.abs(low).square()))
    leakage = float((low_norm / (residual_norm + 1e-12)).detach().cpu())
    ic_error = float(torch.max(torch.abs(psi_bloch0 + residual - psi0)).detach().cpu())
    return (current_c, residual, leakage, ic_error)


def validate_fixed_initial_state(initial_values: np.ndarray, tolerance: float, label: str) -> float:
    """检查所有轨迹在 t=0 的完整初值相同，避免把变化初值误当模型常量。"""
    values = np.asarray(initial_values)
    if values.shape[0] < 1:
        raise ValueError(f'{label} has no trajectories')
    spread = float(np.max(np.abs(values - values[0:1])))
    if spread > float(tolerance):
        raise ValueError(f'Fixed initial-state contract failed for {label}: spread={spread:.3e} > {tolerance:.3e}')
    return spread


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


def one_d_data(cfg, filename):
    """完整复轨迹 NPZ；沿用小型元数据中的真实划分和原始保存网格。"""
    with np.load(require_asset(cfg, 'data', filename), allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    groups = [np.asarray(cfg['split'][key], dtype=np.int64) for key in ('train_indices', 'val_indices', 'test_indices')]
    if len(groups[0]) != 128 or any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('实际 train128/validation/test 划分有误')
    if data['x_grid'].shape != (cfg['grid']['nx'],) or not np.allclose(data['t_values'], cfg['grid']['times'], rtol=0, atol=1e-14):
        raise ValueError('完整空间或时间网格不匹配')
    return data


def selection(model, data, indices):
    """正式选模指标：逐帧空间复相对 L2，再对时间、validation 轨迹取算术平均。"""
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), EVAL_BATCH_TRAJ):
            ids = np.asarray(indices[start:start + EVAL_BATCH_TRAJ], dtype=int)
            pred = model(np.asarray(data['U_values'])[ids])
            ref = torch.tensor(data['psi_real'][ids] + 1j * data['psi_imag'][ids], dtype=torch.complex64, device=model.device)
            errors = metric_relative_l2_by_time(pred, ref, model.dx)
            values.extend(errors.mean(dim=1).cpu().tolist())
    return float(np.mean(values))


# ============================================================================
# 五、训练：10000 次更新，validation 选模，不写末态或优化器状态
# ============================================================================
def train(cfg, data_path, basis_path, save_path, device):
    """每步随机取四条 train 轨迹，完整 11 帧计算原损失；余弦学习率不改预算。"""
    data = one_d_data(cfg, data_path)
    basis_path = require_asset(cfg, 'basis', basis_path)
    seed_runtime(cfg['seed'])
    model = one_d_build(cfg, basis_path, device, training_data=data)
    train_ids = np.asarray(cfg['split']['train_indices'], dtype=int)
    validation_ids = np.asarray(cfg['split']['val_indices'], dtype=int)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES, eta_min=LEARNING_RATE * 0.05)
    rng = np.random.default_rng(cfg['seed'])
    best = float('inf')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, UPDATES + 1):
        model.train()
        ids = rng.choice(train_ids, size=BATCH_TRAJ, replace=False)
        optimizer.zero_grad(set_to_none=True)
        pred = model(np.asarray(data['U_values'])[ids])
        ref = torch.tensor(data['psi_real'][ids] + 1j * data['psi_imag'][ids], dtype=torch.complex64, device=device)
        loss, _ = compute_training_loss(pred, ref, dx=model.dx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % VALIDATE_EVERY == 0 or step == UPDATES:
            value = selection(model, data, validation_ids)
            if value < best:
                best = value
                torch.save(model.state_dict(), save_path)
            print(f'update={step}/{UPDATES} loss={float(loss.detach()):.6g} validation={value:.6g} best={best:.6g}', flush=True)
    return model


# ============================================================================
# 六、加载最佳权重后的完整推理与独立评估
# ============================================================================
def infer(cfg, checkpoint_path, basis_path, parameters, device):
    """一组或多组外势参数 -> [B,11,Nx]，只读取权重、basis 和小型元数据。"""
    weights = load_weights(cfg, checkpoint_path)
    model = one_d_build(cfg, require_asset(cfg, 'basis', basis_path), device, weights=weights)
    model.eval()
    with torch.inference_mode():
        return model(parameters)


def evaluate(cfg, checkpoint_path, data_path, basis_path, split, device):
    """复用论文指标；complex128 计算误差，不相位对齐，不事后归一化。"""
    model = one_d_build(cfg, require_asset(cfg, 'basis', basis_path), device, weights=load_weights(cfg, checkpoint_path))
    model.eval()
    data = one_d_data(cfg, data_path)
    key = {'train': 'train_indices', 'validation': 'val_indices', 'test': 'test_indices'}[split]
    samples = []
    with torch.inference_mode():
        for index in cfg['split'][key]:
            parameters = np.asarray(data['U_values'][index]).reshape(1, -1)
            pred = model(parameters).to(torch.complex128)
            ref = torch.tensor((data['psi_real'][index] + 1j * data['psi_imag'][index])[None], dtype=torch.complex128, device=device)
            potential = build_total_potential_x_from_data(data, index, model.x_np, model.eps)
            values = metric_aggregate_metrics(pred, ref, torch.tensor(potential, dtype=torch.float64, device=device), model.eps, model.dx)
            row = {name: float(value) for name, value in values.items()}
            samples.append(row)
            print(f'global_id={index} {row}', flush=True)
    result = {name: max(row[name] for row in samples) if name.endswith('_max') else float(np.mean([row[name] for row in samples])) for name in samples[0]}
    print(f'{split}（相对存储参考解）: {result}', flush=True)
    return result


if __name__ == '__main__':
    configuration = read_configuration(METADATA_PATH)
    device = runtime_device(DEVICE)
    if MODE == 'train':
        train(configuration, DATA_PATH, BASIS_PATH, SAVE_PATH, device)
    elif MODE == 'infer':
        prediction = infer(configuration, CHECKPOINT_PATH, BASIS_PATH, PARAMETERS, device)
        print(f'psi shape={tuple(prediction.shape)}; dtype={prediction.dtype}; device={prediction.device}')
    elif MODE == 'evaluate':
        evaluate(configuration, CHECKPOINT_PATH, DATA_PATH, BASIS_PATH, SPLIT, device)
    else:
        raise ValueError('MODE 必须为 train、infer 或 evaluate')
