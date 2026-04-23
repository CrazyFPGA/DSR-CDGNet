# DSR-CDGNet: Dynamic Spatio-Temporal Reasoning and Confidence-Driven Gating Model

> 🌐 **Language / 语言**: [English](./README_EN.md) ｜ [简体中文](./README.md)

---

## 📢 Paper Status

> **This paper is currently under review.**
>
> This work is a collaboration between the author and **Dr. Ricardo Emanuel Vaz Vargas** from **Petrobras (Petróleo Brasileiro S.A.)**, Brazil.
>
> Once accepted, the full citation (BibTeX), journal / conference name, DOI, and link to the paper will be updated here. Stay tuned.

```
Title : Dynamic Spatio-Temporal Reasoning and Confidence-Driven Gating
        Model for Multi-Sensor Fault Detection in Oil Wells
Status: Under Review (to be updated)
Cite  : (to be updated upon acceptance)
Link  : (to be updated upon acceptance)
```

---

## 📖 Overview

Multi-sensor measurement systems in oil wells face two key challenges for fault detection: **existing methods fail to fully exploit the spatial topological information among sensors**, and **detection decisions lack confidence quantification**, which limits the effective utilization of measurement information.

To address these issues, we propose **DSR-CDGNet** (Hierarchical Spatio-temporal Reasoning with Dynamic Inference Gating Network), whose main contributions are:

1. **Node-integrity maintenance strategy** — balances measurement data quality and topological completeness.
2. **Data preprocessing pipeline** — extracts feature sequences carrying both *state* and *trend* information from raw signals, achieving dimensionality reduction and enhancement of multi-source measurements.
3. **Physical graph topology** — for the first time in this problem, a sensor-layout-based physical graph structure is introduced.
4. **Centrality-Guided Graph Convolution (CGC)** — models the spatial relationships among sensors.
5. **Dual-stream attention with iterative refinement** — strengthens system-level temporal features through iterative reasoning.
6. **Confidence-driven dynamic gating (Dynamic ITR)** — ensures detection confidence while enabling efficient inference.

Using the **3W dataset**, a real multi-sensor production dataset from offshore oil wells in Brazil, we evaluate the model on two representative fault categories with markedly different evolution rates — **fast abrupt faults** and **long-term slow-varying faults**.

### 🏆 Key Results

| Metric | Fast Abrupt Fault | Long-term Slow Fault |
|--------|:---:|:---:|
| F1-Score | Superior | **0.97** |
| False Negative Rate (FNR) | Superior | **0.02** |
| Inference FLOPs Reduction | \(20\%\sim25\%\) | \(20\%\sim25\%\) |

---

## 📁 Project Structure

```
DSR-CDGNet/
├── main.py                  # Main entry: training, validation, testing, analysis
├── model.py                 # DSR-CDGNet model definition
├── data_preprocessing.py    # Preprocessing, sequence generation, DataLoaders
├── data_analysis.py         # Data-quality report generator
├── graph_utils.py           # Physical graph construction, PageRank analysis
├── config_loader.py         # YAML config loader with task-adaptive hyperparams
├── config.yaml              # Global configuration file
└── environment.yml          # Conda environment specification
```

### File-by-file description

| File | Role |
|------|------|
| **`main.py`** | Main program entry. Handles random-seed setup, DataLoader construction, model initialization, training / validation loop with early stopping, best-model checkpointing, test-set evaluation, FLOPs and inference-latency profiling, confusion matrix, permutation feature importance, PageRank visualization, robustness analysis (AWGN / impulse noise), and the final consolidated report. |
| **`model.py`** | Defines the **DSR-CDGNet** model and its components: `AdaptiveGraphLearner` (GRU-driven adaptive graph learner), `CGCConv` (centrality-guided graph convolution), and the iterative-reasoning blocks (Dynamic ITR / Dynamic ITR_CE / Single ITR). Also ships with ablation variants (GCN / GAT / SAGE / APPNP / GIN / ChebConv) and temporal baselines (GRU / MLP / LSTM / Transformer). |
| **`data_preprocessing.py`** | End-to-end data pipeline: frozen-variable removal, positive/negative-class alignment, missing-value cleaning, ASR + PAA downsampling (mean / slope features), optional VMD decomposition with shared-attention reconstruction, one-hot encoding of well state, sliding-window sequence generation, training-set-fit standardization, and sequence-level train / val / test DataLoaders. |
| **`data_analysis.py`** | Generates a detailed per-folder / per-well data-quality report (non-missing counts, frozen-variable analysis, variance checks) and exports it to Excel for traceability. |
| **`graph_utils.py`** | Builds the predefined physical adjacency matrix \(A_{\text{physical}}\) from `graph_definitions` in `config.yaml`; analyzes PageRank centrality of the final fused graph \(A_{\text{final}} = A_{\text{physical}} + \alpha \cdot A_{\text{learned}}\); provides graph visualization utilities. |
| **`config_loader.py`** | Loads `config.yaml` and auto-adjusts the batch size and learning rate based on the selected `TARGET_FAULT_CLASS` (2 or 8). |
| **`config.yaml`** | Central hub for all configurable options: paths, mode switches, hyperparameters, graph definitions, GCN / ITR module settings, ASR / PAA / VMD parameters, and robustness-test parameters. |
| **`environment.yml`** | Full conda environment specification for reproducing the experimental setup. |

---

## 🧪 Dataset

- **Name**: **3W dataset**, publicly released by Petrobras
- **Source**: Real multi-sensor production data from Brazilian offshore oil wells
- **Fault categories used in this work**:
  - **Fast abrupt fault**: class 2 (e.g., Spurious DHSV closure)
  - **Long-term slow fault**: class 8 (e.g., Hydrate formation in production line)

> ⚠️ The raw data is **not** included in this repository. Please download it from the [official 3W dataset repository](https://github.com/petrobras/3W) and set the extracted path in `ROOT_DATA_PATH` inside `config.yaml`.

---

## ⚙️ Environment & Dependencies

The project is developed with **Python 3.8** and **PyTorch 2.2.2 (CUDA 12.1)**. The full list is in `environment.yml`; core dependencies are:

| Library | Version |
|---------|---------|
| python | 3.8.20 |
| torch | 2.2.2+cu121 |
| torch-geometric | 2.6.1 |
| torch-sparse | 0.6.18+pt22cu121 |
| torchvision | 0.17.2+cu121 |
| networkx | 3.0 |
| numpy | 1.24.1 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| scipy | 1.10.1 |
| matplotlib | 3.7.5 |
| seaborn | 0.13.2 |
| vmdpy | 0.2 |
| thop | 0.1.1 |
| PyYAML | 6.0.2 |
| joblib | 1.4.2 |

### Create the environment

```bash
# Option 1: directly from environment.yml (recommended)
conda env create -f environment.yml
conda activate gnn

# Option 2: manual install
conda create -n gnn python=3.8
conda activate gnn
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.6.1
pip install torch-sparse==0.6.18+pt22cu121 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install networkx==3.0 numpy==1.24.1 pandas==2.0.3 scikit-learn==1.3.2 \
            scipy==1.10.1 matplotlib==3.7.5 seaborn==0.13.2 vmdpy==0.2 \
            thop==0.1.1.post2209072238 pyyaml joblib openpyxl
```

---

## 🚀 How to Use

### 1. Prepare the dataset

Download the data from the [official 3W repository](https://github.com/petrobras/3W) and point `config.yaml` to its location:

```yaml
ROOT_DATA_PATH: '/your/path/to/3W/dataset'
```

### 2. Edit the configuration

Open `config.yaml`. Key options:

```yaml
# Target fault type (2 = fast abrupt, 8 = long-term slow)
TARGET_FAULT_CLASS: 2

# HGC-STP module (spatial)
hgc_stp_module:
  gcn_type: 'CGC'          # CGC / GCN / GAT / APPNP / GIN / NONE
  k_knn: 3                 # KNN neighbors in the adaptive graph
  alpha_initial: 0.5       # Initial fusion weight α (learnable)
  gcn_hidden_dim: 16
  gcn_layers: 3

# ITR module (temporal + dynamic reasoning)
itr_module:
  itr_type: 'Dynamic_ITR'  # Dynamic_ITR / Dynamic_ITR_CE / Single_ITR / GRU / MLP / LSTM / Transformer
  enable_early_exit: false
  early_exit_threshold: 0.95
  num_iterations: 5
  nhead: 4
  lambda_initial: 0.2
  lambda_final: 0.5

# Training hyperparameters
hyperparameters:
  sequence_length: 90
  epochs: 500
  EARLY_STOPPING_PATIENCE: 5
  DETERMINISTIC_MODE: true
```

> 💡 `config_loader.py` automatically selects the `batch size` and `learning rate` based on `TARGET_FAULT_CLASS` (class 2 → 256 @ 0.001, class 8 → 1024 @ 0.002).

### 3. Run training

```bash
python main.py
```

### 4. Outputs

All results are stored under `results_<timestamp>/<experiment_name>_<timestamp>/`:

- `best_model.pth` — best-F1 checkpoint on the validation set
- `config.yaml` — backup of the configuration used for the run
- `final_evaluation_report.txt` — consolidated report (Accuracy / F1 / MCC / FNR / FPR / FLOPs / latency / top-10 PageRank nodes)
- `confusion_matrix_test.svg` — test-set confusion matrix
- `loss_convergence_*.svg` — training / validation loss curves
- `confidence_evolution_heatmap.svg` — confidence-score heatmap across iterations (Dynamic ITR)
- `iterative_reasoning_evolution.svg` — evolution of iterative reasoning depth over epochs
- `confidence_kde_test_correct.svg` — KDE of confidences on correctly predicted test samples
- `pagerank_centrality_final_graph.png` — PageRank visualization of the final fused graph
- `feature_importance_barchart.svg` — permutation feature-importance barchart
- `plot_data/` — raw CSV sources behind every figure for easy reproducibility

### 5. Ablations & baselines

Switching the `gcn_type` / `itr_type` fields in `config.yaml` is all you need for ablation. Examples:

```yaml
# Disable the spatial graph module (temporal only)
hgc_stp_module:
  gcn_type: 'NONE'

# Use a standard Transformer encoder as baseline
itr_module:
  itr_type: 'Transformer'
```

---

## 📐 Method Overview

The core forward pass of DSR-CDGNet can be summarized as:

\[
\mathbf{A}_{\text{final}} = \mathbf{A}_{\text{physical}} + \alpha \cdot \mathbf{A}_{\text{learned}}
\]

\[
\mathbf{h}_i^{(l+1)} = \sigma \Bigg( \sum_{j \in \mathcal{N}(i)} e_{ij} \cdot \text{PR}(j) \cdot \mathbf{W}^{(l)} \mathbf{h}_j^{(l)} + \mathbf{W}^{(l)} \mathbf{h}_i^{(l)} \Bigg)
\]

where \(\text{PR}(j)\) is the PageRank centrality score of node \(j\), and \(e_{ij}\) is the edge weight.

Then \(N=5\) iterative refinement blocks with confidence gating are applied, with the composite loss:

\[
\mathcal{L} = \sum_{k=1}^{N} C_k \cdot \mathcal{L}_{\text{CE}}^{(k)} + \lambda \cdot \mathcal{L}_{\text{penalty}}
\]

where \(C_k\) is the normalized confidence gate at iteration \(k\), and \(\lambda\) is scheduled in a stepwise manner between \(\lambda_{\text{initial}}\) and \(\lambda_{\text{final}}\).

---

## 📄 Citation

The following placeholder will be updated once the paper is accepted:

```bibtex
@unpublished{hsrdignet2026,
  title   = {Dynamic Spatio-Temporal Reasoning and Confidence-Driven Gating Model for Multi-Sensor Fault Detection in Oil Wells},
  author  = {Fei Cao and Ricardo Emanuel Vaz Vargas},
  note    = {Manuscript under review},
  year    = {2026}
}
```

---

## 🙏 Acknowledgments

- We gratefully thank **Petrobras** for publicly releasing the **3W dataset**, which provides invaluable real-world industrial data for fault detection research in the oil and gas sector.
- We thank **Dr. Ricardo Emanuel Vaz Vargas** (Petrobras) for his critical support on domain knowledge, data interpretation, and method design throughout this work.

---

## 📬 Contact

For questions, suggestions, or collaborations, please open an Issue or reach out by email via caofei2nuc@gmail.com.

