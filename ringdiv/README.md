# RingDiv: Molecular Generation Evaluation Toolkit

**RingDiv** is a lightweight, modular Python package designed for evaluating molecular generation models. It provides a comprehensive suite of metrics ranging from basic chemical properties to advanced topological and curvature-based descriptors.

Originally developed as the evaluation engine for the HGR project, RingDiv is now a standalone package that can be easily integrated into any molecular generation workflow.

## 📦 Installation

### From Source (Editable Mode)
If you are developing within the HGR project monorepo, it is recommended to install in editable mode:

```bash
cd /path/to/HGR
pip install -e ringdiv/
```

### Dependencies
RingDiv relies on standard chemistry and ML libraries:
- `rdkit`
- `fcd_torch`
- `networkx`
- `numpy`, `scipy`, `pandas`
- `requests`, `tqdm` (for dataset downloading)

Optional dependencies (only needed when you enable specific metrics):
- **Curvature**: `gudhi`, `POT` (import name: `ot`)
- **NSPDK**: `scikit-learn`, `joblib`, `dill`


## 🚀 Quick Start

### 1. Calculating Metrics
The core API is `get_all_metrics`. It accepts a list of generated molecules (RDKit objects or SMILES) and a reference test set.

```python
from rdkit import Chem
import ringdiv

# 1. Prepare your data
gen_smiles = ["CCO", "c1ccccc1", ...]
ref_smiles = ["CCO", "CCCl", ...] # Reference test set

gen_mols = [Chem.MolFromSmiles(s) for s in gen_smiles if s]

# 2. Compute metrics
# You can specify which metrics to compute via the 'metrics' list
results = ringdiv.get_all_metrics(
    gen=gen_mols, 
    test_smiles=ref_smiles, 
    train_smiles=ref_smiles, # Optional, for Novelty benchmarks
    metrics=['validity', 'unique', 'fcd', 'fragments', 'scaffolds'],
    use_cache=False
)

print(results)
# Output: {'Validity': 1.0, 'FCD': 0.5, ...}
```

### 2. Loading Datasets
RingDiv includes built-in support for downloading benchmarks (RingDiv, RingDiv300k).

```python
import ringdiv

# Ensure raw files exist (download if enabled by your setup) and get paths
paths = ringdiv.ensure_dataset("ringdiv300k")
print(paths)

# Load SMILES and split by the provided test indices
train_smiles, test_smiles = ringdiv.load_train_test_smiles("ringdiv300k")
print(len(train_smiles), len(test_smiles))
```

### 3. Supported Metrics

| Metric Category | Indicators | Valid Values |
| :--- | :--- | :--- |
| **Basic** | Validity, Uniqueness, Novelty | 0.0 - 1.0 |
| **Distribution** | FCD (Fréchet ChemNet Distance) | Lower is better |
| **Chemistry** | SNN (Nearest Neighbor), Frag (Fragment), Scaf (Scaffold) | Higher/Sim is better |
| **Properties** | LogP, QED, SA (Synthetic Accessibility), Weight | Wasserstein Dist. |
| **Graph Kernel** | NSPDK (Neighborhood Subgraph Pairwise Distance Kernel) | Chemical Sim. |
| **Curvature** | Ollivier-Ricci, Forman, Balanced Forman | Graph Topology |


## 🔌 Public API（对外接口）

RingDiv 当前在顶层 `ringdiv` 包中对外暴露的主要接口如下（推荐从这里使用，而不是深入子模块）：

- `ringdiv.get_all_metrics(gen, test_smiles=None, train_smiles=None, metrics=None, ...)`
  - **用途**：统一评估入口，按 `metrics` 选择计算 Validity/Uniqueness/Novelty/FCD/NSPDK/Curvature 等
- `ringdiv.ensure_dataset(dataset_name, ...)`
  - **用途**：准备/检查数据集文件（必要时下载），返回 raw 文件路径信息
- `ringdiv.read_test_idx(test_idx_npz_path) -> (test_idx, meta)`
  - **用途**：读取 `*_test_idx.npz`，返回 test 索引与元信息（`n_total/seed/test_ratio/...`）
- `ringdiv.load_train_test_smiles(dataset_name, ...) -> (train_smiles, test_smiles)`
  - **用途**：从 `*_property.csv` 读取 SMILES，并用 `*_test_idx.npz` 划分 train/test
- `ringdiv.load_full_smiles(dataset_name, ...) -> full_smiles`
  - **用途**：从 `*_property.csv` 读取全量 SMILES（保持 csv 行顺序）



## 📁 Project Structure

```text
ringdiv/
├── dataio/         # Dataset loaders (RingDiv, RingDiv300k) with auto-download
├── metrics/        # Core metric implementations (KL, FCD, NSPDK, Curvature...)
├── statics/        # Static resources (fragment scores, filter lists)
└── utils/          # Internal utilities
```
