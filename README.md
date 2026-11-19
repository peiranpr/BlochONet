# BlochONet

**面向周期晶格势半经典薛定谔方程的两阶段算子学习框架。**

BlochONet 求解包含周期晶格势的半经典薛定谔方程：

$$
i\varepsilon\partial_t\psi
=\left[-\frac{\varepsilon^2}{2}\Delta
+V\!\left(\frac{x}{\varepsilon};\mu\right)
+U(x;\xi)\right]\psi,
\qquad \psi(0,x)=\psi_0(x).
$$

其中，$\varepsilon$ 为半经典参数，$V$ 为周期晶格势，$\mu$ 表示晶格参数，$U$ 为由参数 $\xi$ 决定的外势。小 $\varepsilon$ 引起的快速振荡与多尺度结构，使直接学习波函数具有较大难度。

BlochONet 将问题分为两个阶段。**Stage-I** 学习由动能与周期晶格势构成的 Bloch Hamiltonian 的低能子空间，并通过子空间内的 Ritz 问题获得能带信息，得到 Bloch spectral data。**Stage-II** 学习外势参数到解函数的映射，利用这些谱数据构造 Bloch branch 与 corrector，两支网络联合训练，以谱结构描述主要振荡分量，并补充低带表示未覆盖的分量，从而增强网络对该半经典问题的表征能力。

## 方法实现（Implementations in this repository）

```text
Stage-I：周期晶格参数 μ、准动量 k
    │
    ├─ 构造动能 + 周期晶格势的 Bloch Hamiltonian
    ├─ 子空间网络与复数 QR 正交化
    └─ 子空间投影损失 + Ritz 谱损失
    │
    ▼
Bloch spectral data：低带子空间、能带与匹配的 Bloch basis
    │
    ▼
Stage-II：外势参数 ξ、固定初值及 Bloch spectral data
    │
    ├─ Bloch branch
    │    系数网络 → 谱相位恢复 → Bloch 重构
    │
    └─ Corrector
         物理空间网络 → 正交补投影 → 初值约束
    │
    ▼
两支网络联合训练，输出 ψ = ψ_Bloch + ψ_corrector
```
更详细的代码导读见 [INTRO.md](INTRO.md)。

## 仓库结构（Repository Structure）

```text
BlochONet/
├── stage1/
│   ├── __init__.py
│   └── models.py            # Stage-I 模型、数据读取、训练、推理与评估
├── stage2/
│   ├── __init__.py
│   ├── BlochONet_1d.py       # 完整 1D Stage-II
│   └── BlochONet_2d.py       # 完整 2D Stage-II
├── configs/                 # 物理网格、参数样本、数据划分等
├── requirements.txt
├── README.md
└── INTRO.md                 # 各主脚本的计算流程与阅读顺序
```

运行时可将外部资产放在 `assets/<case>/`，得到的最佳模型默认保存到 `results/<case>/best_model.pt`。这两个目录您可按需准备或创建。

## 安装（Installation）

使用 Python 3.11 或以上版本，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

主要依赖见 [requirements.txt](requirements.txt)。训练建议使用支持 CUDA 的 PyTorch；设备通过主脚本顶部的 `DEVICE = "auto"`、`"cuda"` 或 `"cpu"` 选择。

## 数据生成与准备（Data Generation）

完整的数据准备流程分为：

1. **Stage-I reference**：在指定晶格参数与准动量网格上构造平面波 Hamiltonian，数值求解低能特征子空间与特征值（fourier展开），保存用于训练的谱数据。
2. **Stage-II basis**：为选定晶格准备相应的 Bloch basis 和能带数据，通过传统方法求解本征问题即可，详细算法过程请见论文的附录。
3. **Stage-II reference**：固定物理网格、初值、外势参数与保存时刻，用参考求解器（精细步长的TSSP方法）生成完整复波函数标签，并保留实际 train/validation/test 划分。

## 训练（Training）

在对应主脚本顶部设置运行模式、算例及资产路径，例如一维 harmonic：

```python
MODE = "train"
CASE = "harmonic"
DEVICE = "auto"
# 在下方 DATA_PATH、BASIS_PATH 中填写实际文件路径。
```

然后在仓库根目录执行所需任务：

```bash
# Stage-I
python stage1/models.py

# 一维 Stage-II
python stage2/BlochONet_1d.py

# 二维 Stage-II
python stage2/BlochONet_2d.py
```

网络与训练超参数直接列在各主脚本顶部，默认训练设置为：

| 实现 | 训练预算 | Batch | 验证与选模 |
|---|---|---|---|
| Stage-I | 500 epochs | 512 个 `(μ,k)` 样本 | 每个 epoch 验证，以 validation loss 选模 |
| Stage-II 1D | 10000 次更新 | 4 条完整轨迹 | 按 validation 复波函数相对 L2 均值选模 |
| Stage-II 2D | 20000 次更新 | 1 条完整轨迹 | 每 500 次更新验证，更新前也验证一次 |

训练仅在 validation 指标改善时覆盖一个 **`best_model.pt`**，内容为标准 `state_dict`。训练进度打印到终端，目前吗没有保存末态模型、优化器状态、历史或预测文件。

## 推理（Inference）

将所选主脚本顶部改为 `MODE = "infer"`，设置 `CHECKPOINT_PATH`、`PARAMETERS`，Stage-II 还需设置 `BASIS_PATH`，然后执行同一个主脚本。

| 任务 | `PARAMETERS` 的含义 | 返回结果 |
|---|---|---|
| Stage-I Mathieu / 矩形晶格 | 一个晶格幅值 | 完整 k 网格上的子空间 `Q` 和 Ritz 谱 |
| Stage-I Mathieu2 | `[mu1, mu2]` | 完整 k 网格上的子空间 `Q` 和 Ritz 谱 |
| 1D harmonic / step / closure | 一个外势参数 | `[B, 11, Nx]` 复波函数 |
| 1D `kp_d8` | 八维外势系数 | `[B, 11, 16384]` 复波函数 |
| 2D harmonic | `[kappa, beta]` | `[B, 33, 512, 512]` 复波函数 |

也可以在 Python 中直接取得结果张量：

```python
import torch
from stage2 import BlochONet_1d as bloch

cfg = bloch.read_configuration("configs/harmonic.json")
psi = bloch.infer(
    cfg,
    checkpoint_path="path/to/best_model.pt",
    basis_path="path/to/basis.npz",
    parameters=[1.0],
    device=torch.device("cuda"),
)
print(psi.shape)  # [1, 11, 16384]
```

示例中，一维算例的保存时刻为 `0, 0.02, ..., 0.2`，二维为 ε=1/32、T=1，共 33 帧，网络直接查询这些时间点。

## 评估（Evaluation）

在所选主脚本顶部设置：

```python
MODE = "evaluate"
SPLIT = "test"       # 也可选择 "train" 或 "validation"
```

填写数据、权重及所需 basis 路径，执行对应主脚本即可打印逐样本和聚合指标。Stage-I 评估子空间、Ritz 谱和本征残差；Stage-II 评估复波函数、密度、质量及 Hamiltonian 等指标。

Stage-II 的主波函数指标为每个保存时刻的空间复相对 L2 误差，再对全部时刻和所选轨迹取算术平均：

$$
E_\psi=\frac{1}{N_{\mathrm{traj}}N_t}
\sum_{j=1}^{N_{\mathrm{traj}}}\sum_{n=0}^{N_t-1}
\frac{\|\widehat\psi_j(t_n)-\psi_j(t_n)\|_{L^2}}
{\|\psi_j(t_n)\|_{L^2}+\delta}.
$$

## 测试（Tests）

安装依赖后，可先在仓库根目录执行不需要数据的导入检查：

```bash
python -c "import stage1.models, stage2.BlochONet_1d, stage2.BlochONet_2d"
```

模型测试使用上述 `MODE = "evaluate"`、`SPLIT = "test"` 流程。harmonic/原 step 使用 train128/validation29/test29；KP、学习基闭环与二维算例使用 train128/validation32/test64。Stage-I 按晶格参数划分为 train1024/validation128/test256，各自覆盖完整 k 网格。
