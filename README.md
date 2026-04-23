# DSR-CDGNet：动态时空推理与置信驱动门控模型

> 🌐 **语言 / Language**：[简体中文](./README.md) ｜ [English](./README_EN.md)

---

## 📢 论文状态

> **本论文当前正在投稿中（Under Review）。**
>
> 本工作得到了**巴西国家石油公司（Petrobras）** 的 **Ricardo Emanuel Vaz Vargas** 博士的指导，他同时也是本文的合著者。
>
> 论文一经接收，本仓库将立即补充完整的引用信息（BibTeX）、期刊/会议名称、DOI 以及原文链接。敬请关注更新。

```
Title : Dynamic Spatio-Temporal Reasoning and Confidence-Driven Gating
        Model for Multi-Sensor Fault Detection in Oil Wells
Status: Under Review (待补充)
Cite  : 待补充
Link  : 待补充
```

---

## 📖 项目简介

油井多传感器测量系统在故障检测中面临两大挑战：**现有方法未能充分利用传感器间的空间拓扑信息**，且**检测决策缺乏置信度量化**，制约了测量信息的有效利用。

为此，本项目提出了 **DSR-CDGNet**（Dynamic spatio-temporal reasoning and confidence-driven gating model，动态时空推理与置信驱动门控模型），主要贡献包括：

1. **节点完整性维护策略**：权衡测量数据质量与拓扑完整性。
2. **数据预处理机制**：从原始信号中提取包含"状态"与"趋势"的特征序列，实现多源测量数据的降维与增强。
3. **物理图拓扑结构**：首次在油井故障检测问题中引入基于传感器布局的物理图拓扑。
4. **中心性引导图卷积单元**：建模传感器间的空间关系。
5. **双流注意力机制**：在时间维度上通过迭代精炼强化系统级时序测量特征的表达。
6. **置信驱动的动态门控机制**：在保障检测置信度的同时实现高效推理。

基于源自**巴西深海油井现场的真实多传感器生产数据集 3W**，选取演变速率差异显著的两类典型故障——**快速突变型故障**与**长时缓变型故障**——进行评估。

### 🏆 核心结果

| 指标 | 快速突变型故障 | 长时缓变型故障 |
|------|:---:|:---:|
| F1-Score | 优越 | **0.97** |
| 漏报率 (FNR) | 优越 | **0.02** |
| 推理计算量减少 | 20%-25% | 20%-25% |

---

## 📁 项目结构与文件说明

```
DSR-CDGNet/
├── main.py                  # 主入口：训练、验证、测试与可视化分析
├── model.py                 # DSR-CDGNet 核心模型定义
├── data_preprocessing.py    # 数据预处理、序列生成、标准化、DataLoader 构建
├── data_analysis.py         # 数据质量报告生成
├── graph_utils.py           # 物理图构建、PageRank 中心性分析与可视化
├── config_loader.py         # YAML 配置加载器（按任务自动调参）
├── config.yaml              # 全局配置文件（超参数、图拓扑定义等）
└── environment.yml          # Conda 环境依赖定义
```

### 各文件功能详解

| 文件 | 作用 |
|------|------|
| **`main.py`** | 程序主入口。负责随机种子设置、数据加载器构建、模型初始化、训练与验证循环、早停、最佳模型保存、测试集评估、FLOPs 与推理延迟度量、混淆矩阵、置换重要性（Permutation Importance）、PageRank 可视化、鲁棒性（AWGN / 脉冲噪声）分析，并生成最终综合报告。 |
| **`model.py`** | 定义 **DSR-CDGNet** 主模型以及 `AdaptiveGraphLearner`（GRU 驱动的自适应图学习器）、`CGCConv`（中心性引导图卷积）、迭代推理块（Dynamic ITR / Dynamic ITR_CE / Single ITR）等组件。同时内置多种 GCN 变体（GCN / GAT / APPNP / GIN / ChebConv）用于消融对比，以及 GRU / MLP / LSTM / Transformer 等时序骨干用于对照实验。 |
| **`data_preprocessing.py`** | 实现完整数据流水线：冻结变量剔除、正负类对齐、缺失值清洗、ASR+PAA 降采样（均值/斜率特征）、可选 VMD 分解与共享注意力重构、状态独热编码、滑动窗口序列生成、基于训练集参数的标准化，以及面向序列的训练/验证/测试 DataLoader 构建。 |
| **`data_analysis.py`** | 对原始数据集按文件夹与井号生成详细的数据质量报告（非缺失样本统计、冻结变量分析、方差检验），并导出为 Excel 便于追溯。 |
| **`graph_utils.py`** | 根据 `config.yaml` 的 `graph_definitions` 构建预定义物理邻接矩阵 \(A_{\text{physical}}\)；训练过程中分析最终融合图 \(A_{\text{final}} = A_{\text{physical}} + \alpha \cdot A_{\text{learned}}\) 的 PageRank 中心性；提供图结构可视化工具。 |
| **`config_loader.py`** | 加载 `config.yaml`，并根据 `TARGET_FAULT_CLASS`（2 或 8）自动切换对应的 batch size 与学习率组合。 |
| **`config.yaml`** | 所有可配置参数的集中入口：路径、模式开关、超参数、图定义、GCN / ITR 模块参数、ASR/PAA/VMD 参数、鲁棒性测试参数等。 |
| **`environment.yml`** | Conda 环境完整依赖清单，可直接用于复现实验环境。 |

---

## 🧪 数据集

- **名称**：**3W 数据集**（由巴西国家石油公司 Petrobras 公开发布）
- **来源**：巴西深海油井真实多传感器生产数据
- **故障类型**（本项目关注）：
  - **快速突变型故障**：class 2（例如 Spurious DHSV closure）
  - **长时缓变型故障**：class 8（例如 Hydrate formation in production line）

---

## ⚙️ 环境依赖

本项目基于 **Python 3.8** 与 **PyTorch 2.2.2 (CUDA 12.1)** 开发。完整依赖见 `environment.yml`，核心依赖如下：

| 库 | 版本 |
|-----|------|
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

### 创建环境

```bash
# 方式 1：直接从 environment.yml 创建（推荐）
conda env create -f environment.yml
conda activate gnn

# 方式 2：手动创建
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

## 🚀 如何使用

### 1. 准备数据集

将实际数据路径填入 `config.yaml`：

```yaml
ROOT_DATA_PATH: '/your/path/to/3W/dataset'
```

### 2. 修改配置文件

打开 `config.yaml`，关键参数如下：

```yaml
# 目标故障类型（2 = 快速突变型，8 = 长时缓变型）
TARGET_FAULT_CLASS: 2

# HGC-STP 模块（空间维度）
hgc_stp_module:
  gcn_type: 'CGC'          # 可选：CGC / GCN / GAT / APPNP / GIN / NONE
  k_knn: 3                 # 自适应图 KNN 近邻数
  alpha_initial: 0.5       # 图融合权重 α 初始值（可学习）
  gcn_hidden_dim: 16
  gcn_layers: 3

# ITR 模块（时间维度 + 动态推理）
itr_module:
  itr_type: 'Dynamic_ITR'  # Dynamic_ITR / Dynamic_ITR_CE / Single_ITR / GRU / MLP / LSTM / Transformer
  enable_early_exit: false
  early_exit_threshold: 0.95
  num_iterations: 5
  nhead: 4
  lambda_initial: 0.2
  lambda_final: 0.5

# 训练超参数
hyperparameters:
  sequence_length: 90
  epochs: 500
  EARLY_STOPPING_PATIENCE: 5
  DETERMINISTIC_MODE: true
```

> 💡 `config_loader.py` 会根据 `TARGET_FAULT_CLASS` 自动设置对应任务的 `batch size` 与 `learning rate`（class 2 → 256 @ 0.001，class 8 → 1024 @ 0.002）。

### 3. 启动训练

```bash
python main.py
```

### 4. 查看输出

所有输出保存在 `results_<timestamp>/<experiment_name>_<timestamp>/` 目录下，包括：

- `best_model.pth` — 验证集 F1 最优的模型权重
- `config.yaml` — 本次实验使用的配置备份
- `final_evaluation_report.txt` — 综合性能报告（准确率 / F1 / MCC / FNR / FPR / FLOPs / 延迟 / PageRank Top-10）
- `confusion_matrix_test.svg` — 测试集混淆矩阵
- `loss_convergence_*.svg` — 损失收敛曲线
- `confidence_evolution_heatmap.svg` — 置信度演进热力图（Dynamic ITR）
- `iterative_reasoning_evolution.svg` — 迭代推理深度演化图
- `confidence_kde_test_correct.svg` — 测试集正确样本置信度 KDE 分布
- `pagerank_centrality_final_graph.png` — 最终融合图的 PageRank 可视化
- `feature_importance_barchart.svg` — 置换特征重要性柱状图
- `plot_data/` — 所有可视化所用的源数据（CSV），便于复现绘图

### 5. 消融与对比

切换 `config.yaml` 中的 `gcn_type` / `itr_type` 字段即可完成图模块或推理模块的消融实验。例如：

```yaml
# 禁用空间图模块（仅时序）
hgc_stp_module:
  gcn_type: 'NONE'

# 使用标准 Transformer 对照
itr_module:
  itr_type: 'Transformer'
```

---

## 📄 引用

论文接收后将更新本节，暂时可引用为：

```bibtex
@unpublished{DSR-CDGNet2026,
  title   = {Dynamic Spatio-Temporal Reasoning and Confidence-Driven Gating Model for Multi-Sensor Fault Detection in Oil Wells},
  author  = {Fei Cao and Ricardo Emanuel Vaz Vargas},
  note    = {Manuscript under review},
  year    = {2026}
}
```

---

## 🙏 致谢

- 感谢 **巴西国家石油公司 Petrobras** 公开 **3W 数据集**，为油气行业故障检测研究提供了宝贵的真实工业数据。
- 感谢合作者 **Ricardo Emanuel Vaz Vargas 博士**（Petrobras）在领域知识、数据理解与方法设计上给予的关键支持。

---

## 📬 联系方式

如有任何问题、建议或合作意向，欢迎通过 Issues 或邮件与我（caofei2nuc@gmail.com）联系。

