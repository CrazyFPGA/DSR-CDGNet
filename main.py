import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import gaussian_kde
from pathlib import Path
import shutil
import glob
import random
from sklearn.metrics import (f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, matthews_corrcoef)
from collections import defaultdict
from config_loader import config
from data_preprocessing import get_dataloaders
from graph_utils import create_predefined_adj, analyze_and_visualize_pagerank
from model import HSR_DIGNET
import logging
# 设置日志级别为 INFO
logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed, deterministic=True):
    # 优化cudnn设置以解决性能瓶颈
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 对于高性能GPU，允许非确定性算法以换取速度，benchmark=True能让cuDNN自动寻找最高效的算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

def count_parameters(model):
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from sklearn.metrics import confusion_matrix
import numpy as np


def calculate_fnr_fpr(y_true, y_pred):
    """
    专门计算二元分类下的漏报率 (FNR) 和误报率 (FPR)。

    假设标签定义：
    0: 负类 (Negative) -> 正常 (Normal)
    1: 正类 (Positive) -> 故障 (Fault)

    Returns:
        fnr (float): 漏报率 (False Negative Rate)
        fpr (float): 误报率 (False Positive Rate)
    """
    # 强制使用 labels=[0, 1] 确保混淆矩阵始终是 2x2，即使测试集中缺少某类样本也不会报错
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 解包混淆矩阵
    # [TN, FP]
    # [FN, TP]
    tn, fp, fn, tp = cm.ravel()

    # 1. 计算 FNR (漏报率)
    # 分母：所有真实的故障样本 (TP + FN)
    # 分子：被预测为正常的故障样本 (FN)
    total_real_faults = tp + fn
    if total_real_faults > 0:
        fnr = fn / total_real_faults
    else:
        fnr = 0.0  # 如果没有故障样本，漏报率为0

    # 2. 计算 FPR (误报率/虚警率)
    # 分母：所有真实的正常样本 (TN + FP)
    # 分子：被预测为故障的正常样本 (FP)
    total_real_normal = tn + fp
    if total_real_normal > 0:
        fpr = fp / total_real_normal
    else:
        fpr = 0.0  # 如果没有正常样本，误报率为0

    return fnr, fpr

def calculate_flops(model, x_sample, state_sample, device):
    """
    计算模型的 FLOPs (浮点运算次数)
    使用 thop 库进行统计
    注意：计算的是单次前向传播（batch_size=1）的 FLOPs
    返回：(flops_g, flops_m) 元组，单位为 G 和 M
    """
    try:
        import warnings
        import os
        import sys
        from contextlib import redirect_stderr
        
        # 抑制 thop 库内部多进程清理时的警告
        # 这是一个已知问题，不影响功能，发生在程序退出时的清理阶段
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from thop import profile
            
            model_to_profile = model.module if isinstance(model, nn.DataParallel) else model
            model_to_profile.eval()
            
            # 确保输入在正确的设备上
            x_sample = x_sample.to(device)
            state_sample = state_sample.to(device)
            
            with torch.no_grad():
                # 使用 redirect_stderr 抑制 multiprocessing 清理时的错误输出
                with open(os.devnull, 'w') as devnull:
                    with redirect_stderr(devnull):
                        flops, params = profile(model_to_profile, inputs=(x_sample, state_sample), verbose=False)
            
            # 转换为 GFLOPs (G) 和 MFLOPs (M)
            flops_g = flops / 1e9
            flops_m = flops / 1e6
            return flops_g, flops_m
    except ImportError:
        logging.warning("  [Warning] thop library not installed. FLOPs calculation skipped. Install with: pip install thop")
        return None, None
    except Exception as e:
        # 忽略 multiprocessing 清理时的 OSError（目录非空错误）
        # 这是 thop 库内部使用多进程时的已知问题，不影响功能
        if isinstance(e, OSError) and "Directory not empty" in str(e):
            # 如果 FLOPs 计算成功但清理失败，尝试从异常中恢复
            # 但实际上这种情况很少发生，因为错误发生在清理阶段
            logging.debug(f"  [Debug] Multiprocessing cleanup warning (can be ignored): {e}")
            return None, None
        logging.warning(f"  [Warning] FLOPs calculation failed: {e}")
        return None, None

def measure_inference_latency(model, x_sample, state_sample, device, num_warmup=10, num_runs=100):
    """
    测量单样本推理延迟 (ms/sample)
    注意：使用 batch_size=1 进行测量，即单样本推理时间
    num_warmup: 预热次数，避免首次运行的开销
    num_runs: 实际测量次数
    """
    model_to_test = model.module if isinstance(model, nn.DataParallel) else model
    model_to_test.eval()
    
    # 确保输入在正确的设备上
    x_sample = x_sample.to(device)
    state_sample = state_sample.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_to_test(x_sample, state_sample)
    
    # 同步 GPU（如果使用 GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 实际测量
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            _ = model_to_test(x_sample, state_sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000  # 转换为毫秒
            latencies.append(latency_ms)
    
    # 返回平均延迟（毫秒/样本）
    avg_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)
    
    return avg_latency_ms, std_latency_ms

def evaluate_for_f1(model, loader, device):
    """
    辅助函数：在给定loader上评估模型，并返回 Macro F1-Score。
    用于计算特征重要性分析的基线F1。
    """
    model_to_eval = model.module if isinstance(model, nn.DataParallel) else model
    model_to_eval.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for x_batch, state_batch, y_batch, *__ in loader:
            x_batch = x_batch.to(device)
            state_batch = state_batch.to(device)
            # 适配 HSR-DIGNET 的输出
            final_logits, _ = model_to_eval(x_batch, state_batch)
            # 处理 final_logits 可能是列表的情况（如 GRU/MLP 模式）
            if isinstance(final_logits, list):
                final_logits = final_logits[0]  # 取列表中的第一个（也是唯一的）元素
            # 使用最后一次迭代的预测
            final_preds = torch.argmax(final_logits, dim=1)
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)

def calculate_feature_importance(model, loader, physical_node_names, graph_definitions, baseline_f1, cfg):
    """使用置换重要性计算特征贡献度。"""
    logging.info(f"\n--- [Feature Importance Calculation] ---")
    logging.info(f" - Metric: Macro F1-Score Drop")
    repeats = cfg['hyperparameters']['PERMUTATION_REPEATS']  # 从config读取重复次数, 默认为10
    logging.info(f" - Repetitions per node: {repeats}")

    num_features_per_node = 2
    model_to_eval = model.module if isinstance(model, nn.DataParallel) else model

    # 从 graph_definitions 提取场景映射
    # 这的node_to_scene节点是所有的节点，为的是提供场景名字，例如：A:生产通道、B:生产通道
    node_to_scene = {}
    scenes = graph_definitions.get("scenes", [])
    for scene_name in scenes:
        scene_info = graph_definitions.get(scene_name, {})
        for node in scene_info.get("nodes", []):
            node_to_scene[node] = scene_name

    # 创建一个特殊的字典。这个字典用来存储每个特征的重要性分数。
    # defaultdict(list) 是一个很方便的工具，当我们尝试向一个不存在的键（featurename）添加值时，它会自动为这个键创建一个空的列表[]，避免了代码出错。
    importances_dict = defaultdict(list)
    model_to_eval.eval()

    with torch.no_grad():
        # 这里用的是实际的特征feature，没有虚拟节点了，缺失值处理之后、保存下来的实际变量
        # i指定就是第几个维度，实际就是第几个特征
        for i, node_name in enumerate(physical_node_names):
            logging.info(f"  - Permuting node {i + 1}/{len(physical_node_names)}: '{node_name}'...")

            # 确定此节点在 x_batch 特征维度中对应的列索引
            feature_idx_start = i * num_features_per_node
            feature_idx_end = (i + 1) * num_features_per_node
            for n in range(repeats):  # 执行 N 次重复
                # 先清空两个列表，用来收集在“搞乱”数据后，模型的新预测结果和对应的真实标签。
                all_labels_perm, all_preds_perm = [], []
                for x_batch, state_batch, y_batch, *__ in loader:
                    x_batch_permuted = x_batch.clone()
                    # 创建一个随机的排列,目的是特征值与原本的标签错开，数据依旧是之前的数据，只是打乱了对应关系
                    perm = torch.randperm(x_batch_permuted.size(0))
                    # 遍历此节点对应的所有特征 (例如 PAA-mean, PAA-slope)
                    for feat_idx in range(feature_idx_start, feature_idx_end):
                        if feat_idx < x_batch_permuted.shape[2]:  # 安全检查
                            # 对该节点的每个特征执行相同的置换
                            x_batch_permuted[:, :, feat_idx] = x_batch_permuted[perm, :, feat_idx]

                    x_batch_permuted = x_batch_permuted.to(DEVICE)
                    state_batch = state_batch.to(DEVICE)
                    # intermediate_logits的尾速是[256,2],样本数量、两类标签的分数
                    final_logits, _ = model_to_eval(x_batch_permuted, state_batch)
                    # 处理 final_logits 可能是列表的情况（如 GRU/MLP 模式）
                    if isinstance(final_logits, list):
                        final_logits = final_logits[0]  # 取列表中的第一个（也是唯一的）元素
                    # 3）在最后一个维度上取 argmax（对类别维做 argmax）
                    # dim=-1：表示最后一个维度，对于二维张量来说就是第1维（维度编号从0开始）
                    # argmax()‌：返回指定维度上最大值的索引，就是比较哪个类别的概率大了；从模型输出的类别分数里选出预测的类别 ID（比如 0/1）。
                    final_preds = final_logits.argmax(dim=-1) # shape: [batch]

                    all_labels_perm.extend(y_batch.numpy())
                    all_preds_perm.extend(final_preds.cpu().numpy())
                # 计算置换后的F1
                permuted_f1 = f1_score(all_labels_perm, all_preds_perm, average='macro', zero_division=0)
                # F1下降 = 基线 - 置换后
                importance_drop = baseline_f1 - permuted_f1
                # 因为要计算好多次，记录下来每一次的值
                importances_dict[node_name].append(importance_drop)
    # 整理结果
    results = []
    for node_name, drops in importances_dict.items():
        results.append({
            'Variable (Node)': node_name,
            'Scene': node_to_scene.get(node_name, 'N/A'),
            'Mean Importance (F1 Drop)': np.mean(drops),
            'Std Dev': np.std(drops)
        })

    importance_df = pd.DataFrame(results).sort_values(by='Mean Importance (F1 Drop)', ascending=False)
    logging.info(" - Feature importance calculation complete.")

    logging.info(" - Top 5 Important Nodes:")
    logging.info(importance_df.head(6).to_string(index=False))

    return importance_df

def calculate_dynamic_gating_loss(intermediate_logits, confidence_scores, labels, criterion, lambda_penalty):
    """
    计算动态门控损失 L_total = L_main + L_penalty
    """
    # 迭代次数
    num_iterations = len(intermediate_logits)
    main_loss = torch.tensor(0.0, device=labels.device)
    penalty_loss = torch.tensor(0.0, device=labels.device)
    raw_loss_k_means = []

    # L_main = Σ (c_k * L_k)
    # 权重化的精度损失
    for k in range(num_iterations):
        # logits_k 的形状是 [batch_size, num_classes]，label的形状是 [batch_size]
        logits_k = intermediate_logits[k]
        confidence_k = confidence_scores[k]
        # criterion(reduction='none')返回一个 [batch_size] 的损失向量
        # 计算当前迭代的分类损失 (每个样本一个损失值)
        loss_k = criterion(logits_k, labels)
        # 想同步看到loss的变化
        raw_loss_k_means.append(loss_k.mean().item())
        # 使用 .mean() 对批次内的加权损失进行平均
        # 使用 .squeeze() 明确维度匹配
        # 用置信度加权，然后对批次求平均，最后累加到 main_loss
        main_loss += (confidence_k.squeeze() * loss_k).mean()

    # L_penalty = λ * Σ (k * (1 - c_k))
    # 惩罚“不确定性”，而不是“确定性”，以鼓励置信度分数上升。
    penalty_sum = torch.tensor(0.0, device=labels.device)
    for k in range(num_iterations):
        # 惩罚 (1 - confidence)，即不确定性
        # k+1 是迭代步数 (1, 2, 3...)
        penalty_sum += ((k + 1) * (1 - confidence_scores[k])).mean()
    # 应用惩罚系数 lambda
    penalty_loss += lambda_penalty * penalty_sum
    # 3. 计算总损失
    total_loss = main_loss + penalty_loss
    # 4. 返回三元组：总损失（用于反向传播），以及 L_main 和 L_penalty 的数值（用于日志和绘图）
    main_loss_val = main_loss.item()
    penalty_loss_val = penalty_loss.item()

    return total_loss, main_loss_val, penalty_loss_val, raw_loss_k_means


import pandas as pd
import os


def save_plot_data(data_dict, output_dir, filename):
    """
    通用函数：将字典数据保存为CSV文件
    data_dict: { 'Column1': [data...], 'Column2': [data...] }
    """
    save_path = os.path.join(output_dir, 'plot_data')
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, filename)

    try:
        df = pd.DataFrame(data_dict)
        df.to_csv(file_path, index=False)
        print(f"  [Data Saved] Raw plot data saved to: {file_path}")
    except Exception as e:
        print(f"  [Warning] Failed to save plot data for {filename}: {e}")

def add_awgn_noise(x_batch, level):
    """
    在批次张量上添加加性高斯白噪声 (AWGN)。
    level 含义：噪声标准差 = level * 每个样本自身标准差。
    x_batch: [B, T, F]
    """
    if level <= 0:
        return x_batch
    with torch.no_grad():
        # 按样本计算整体标准差，避免不同特征尺度差异过大
        B = x_batch.size(0)
        x_flat = x_batch.view(B, -1)
        std_per_sample = x_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-8
        noise_std = level * std_per_sample  # [B,1]
        noise = torch.randn_like(x_batch) * noise_std.view(B, 1, 1)
        return x_batch + noise


def add_impulse_noise(x_batch, density, magnitude_factor):
    """
    在批次张量上添加瞬态脉冲干扰。
    density 含义：被污染的时间点比例 (0~1)。
    magnitude_factor 含义：脉冲幅度 = magnitude_factor * 每个样本自身标准差。
    设计：在随机的时间步 t 上，对该时间步所有特征施加同符号的尖峰。
    """
    if density <= 0:
        return x_batch
    with torch.no_grad():
        B, T, F = x_batch.size()
        x_flat = x_batch.view(B, -1)
        std_per_sample = x_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-8  # [B,1]
        # 为每个样本、每个时间步生成是否被击中的掩码
        hit_mask = torch.rand(B, T, device=x_batch.device) < density  # [B,T]
        # 为每个样本、每个时间步生成脉冲符号 {-1, +1}
        signs = torch.randint(low=0, high=2, size=(B, T), device=x_batch.device, dtype=torch.float32)
        signs = signs * 2 - 1  # 0/1 -> -1/+1
        # 脉冲幅度（与样本 std 成正比），广播到时间维，再广播到特征维
        spike_amp = magnitude_factor * std_per_sample  # [B,1]
        spike_amp = spike_amp.expand(B, T)  # [B,T]
        spikes_scalar = signs * spike_amp * hit_mask.float()  # [B,T]
        spikes = spikes_scalar.unsqueeze(-1).expand(B, T, F)  # [B,T,F]
        return x_batch + spikes

def robustness_evaluation(model, test_loader, device, cfg, output_dir):
    """
    在测试集上进行鲁棒性分析：
    - AWGN：不同噪声强度水平
    - 瞬态脉冲干扰：不同密度水平
    所有结果保存为 CSV，不影响原始测试流程。
    """
    rob_cfg = cfg.get('robustness_test', {})
    if not rob_cfg.get('enable', False):
        return

    logging.info("\n--- [Robustness Evaluation on Test Set] ---")

    gaussian_levels = rob_cfg.get('gaussian_levels', [0.1, 0.3, 0.5, 0.7, 1.0])
    impulse_levels = rob_cfg.get('impulse_levels', [0.01, 0.03, 0.05, 0.10, 0.20])
    impulse_mag = float(rob_cfg.get('impulse_magnitude_factor', 5.0))
    include_clean = bool(rob_cfg.get('include_clean_baseline', True))

    results_rows = []

    def _evaluate_with_noise(noise_type, level):
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_batch, state_batch, y_batch, *rest in test_loader:
                x_batch = x_batch.to(device)
                state_batch = state_batch.to(device)
                y_true = y_batch.numpy()

                if noise_type == 'gaussian':
                    x_noisy = add_awgn_noise(x_batch, level)
                elif noise_type == 'impulse':
                    x_noisy = add_impulse_noise(x_batch, level, impulse_mag)
                else:
                    x_noisy = x_batch

                logits, _ = model(x_noisy, state_batch)
                if isinstance(logits, list):
                    logits = logits[0]
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y_true)

        all_true_np = np.array(all_true)
        all_preds_np = np.array(all_preds)
        acc = accuracy_score(all_true_np, all_preds_np)
        f1_macro = f1_score(all_true_np, all_preds_np, average='macro', zero_division=0)
        f1_weighted = f1_score(all_true_np, all_preds_np, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(all_true_np, all_preds_np)
        fnr, fpr = calculate_fnr_fpr(all_true_np, all_preds_np)

        results_rows.append({
            'noise_type': noise_type,
            'level': level,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': mcc,
            'fnr': fnr,
            'fpr': fpr
        })

        logging.info(f"  [Robustness] noise={noise_type}, level={level:.3f} | "
                     f"Acc={acc:.4f}, F1_macro={f1_macro:.4f}, MCC={mcc:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}")

    # 可选：添加干净基线（方便直接对比）
    if include_clean:
        _evaluate_with_noise('clean', 0.0)

    # AWGN
    for lv in gaussian_levels:
        _evaluate_with_noise('gaussian', float(lv))

    # Transient impulse
    for lv in impulse_levels:
        _evaluate_with_noise('impulse', float(lv))

    # 保存 CSV
    plot_data_dir = os.path.join(output_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)
    df_results = pd.DataFrame(results_rows)
    csv_path = os.path.join(plot_data_dir, 'data_robustness_test_noise.csv')
    df_results.to_csv(csv_path, index=False)
    logging.info(f"  [Data Saved] Robustness evaluation results saved to: {csv_path}")

    # 生成简要的折线图（分别对 AWGN 与 脉冲干扰），用于快速复现图 1(a/b/c/d) 风格
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'mcc', 'fnr', 'fpr']
    colors = sns.color_palette("tab10", n_colors=len(metrics))

    def _plot_noise(noise_type, filename):
        df_noise = df_results[df_results['noise_type'] == noise_type]
        if df_noise.empty:
            return
        plt.figure(figsize=(10, 6))
        for m, c in zip(metrics, colors):
            plt.plot(df_noise['level'], df_noise[m], marker='o', label=m, color=c)
        plt.xlabel('Noise Level')
        plt.ylabel('Metric Value')
        plt.title(f'Robustness under {noise_type}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        logging.info(f"  [Plot Saved] {out_path}")

    _plot_noise('gaussian', 'robustness_awgn_metrics.svg')
    _plot_noise('impulse', 'robustness_impulse_metrics.svg')


def main(cfg, seed=None, run_id=None):
    import os
    # 设置环境变量以减少 multiprocessing 清理时的警告
    # 这是 thop 库使用多进程时的已知问题，不影响功能
    os.environ.setdefault('MP_METHOD', 'spawn')
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在程序开始时重置峰值显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    if seed is not None:
        set_seed(seed, cfg['hyperparameters']['DETERMINISTIC_MODE'])
    else:
        set_seed(config['hyperparameters'].get('RANDOM_SEED', 42), cfg['hyperparameters']['DETERMINISTIC_MODE'])

    # 提高在A100等高性能GPU上的数值稳定性
    # 不使用A100默认的混合精度加速，精度改变、对复现结果不利
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    program_start_time = time.time()
    output_dir = None

    # 1. 获取 HGC 和 ITR 的类型
    gcn_type = cfg['hgc_stp_module'].get('gcn_type', 'CGC').upper()
    itr_type = cfg['itr_module'].get('itr_type', 'Dynamic_ITR')
    # 2. 构建描述性的子目录名
    # 示例: CGC-Dynamic_ITR, GAT-Dynamic_ITR, CGC-GRU, CGC-MLP
    experiment_name = f"{gcn_type}-{itr_type}"
    # 如果是重复实验，添加 run_id
    if run_id is not None:
        experiment_name += f"_run{run_id}"

    # 是否为调试模式，刚开始使用的，为了低性能的机器，后面不用了
    if cfg['RESUME_TRAINING']:
        logging.info("--- [System] Attempting to resume training from the latest checkpoint... ---")
        history_dirs = sorted(glob.glob(f"{cfg['OUTPUT_DIR_NAME']}_*"))
        if history_dirs:
            latest_dir = Path(history_dirs[-1])
            potential_checkpoint = latest_dir / cfg['CHECKPOINT_FILE']
            if potential_checkpoint.exists():
                output_dir = latest_dir
                logging.info(f" - Found latest checkpoint. Will resume in: '{latest_dir}'")
                checkpoint_to_load = torch.load(potential_checkpoint)
            else:
                logging.info(
                    f" - [Warning] Latest directory '{latest_dir}' does not contain a checkpoint file. Starting a new run.")
        else:
            logging.info(" - [Warning] No previous result directories found. Starting a new run.")

    if output_dir is None:
        logging.info("--- [System] Starting a new training run... ---")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_dir = Path(cfg["OUTPUT_DIR_NAME"])
        # 一级目录：base_dir / timestamp
        root_dir = base_dir.parent / f"{base_dir.name}_{timestamp}"

        output_dir = root_dir / f"{experiment_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 如果一开始就传进来了 output_dir，就保证它是 Path 并创建
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy('config.yaml', output_dir / 'config.yaml')
    logging.info(f" - All outputs will be saved to: {output_dir}")
    logging.info(f" - Configuration file backed up to '{output_dir / 'config.yaml'}'")

    best_model_path = output_dir / cfg['BEST_MODEL_FILE']
    # ASR模块在内嵌函数中
    # feature_cols 是二维后的特征，数量：2*N
    # physical_node_names是最终真实的特征，数量：N
    # int_to_eventid_map是之前为了分析时间窗口敏感度的变量，现在没有用处
    # num_state_features为井的状态，如果是只有1个状态，那相当于没有，置num_state_features=0
    train_loader, val_loader, test_loader, new_feature_cols, physical_node_names, num_classes, num_state_features, int_to_eventid_map = get_dataloaders(
        cfg, output_dir)

    ASR_runtime = time.time() - program_start_time
    logging.info(f"\n\n[ASR Program Runtime]\n- {ASR_runtime:.2f} seconds ({ASR_runtime / 60:.2f} minutes)\n")

    # 这里是物理节点的数量
    num_nodes = len(physical_node_names)
    # 每个物理节点的输入特征维度：
    # - 启用 PAA 时：PAA 特征是均值+斜率 → 2 维
    # - 关闭 PAA 时：直接使用原始传感器 → 1 维
    # 为了自动适配不同模式，这里根据“总特征维度 / 节点数”自动推断每个节点的特征维度
    if num_nodes == 0:
        raise ValueError("num_nodes is 0, please check physical_node_names/preprocessing.")
    num_paa_features = len(new_feature_cols) // num_nodes
    # --- 2. 物理图构建 ---
    graph_definitions = cfg.get('graph_definitions', {})
    # A_physical是预先定义的邻间矩阵
    A_physical = create_predefined_adj(graph_definitions, physical_node_names).to(DEVICE)

    # --- 3. 模型、损失函数、优化器、调度器初始化 ---
    logging.info("--- [Step 3] Initializing model and training components... ---")
    # num_state_features为井的状态，如果是只有1个状态，那相当于没有，置num_state_features=0
    model = HSR_DIGNET(
        config=cfg,
        num_nodes=num_nodes,
        num_paa_features=num_paa_features,
        num_state_features=num_state_features,
        num_classes=num_classes,
        graph_definitions=cfg.get('graph_definitions', {}),
        physical_node_names=physical_node_names
    ).to(DEVICE)

    # 计算并记录模型参数量
    total_params = count_parameters(model)
    logging.info(f"Model initialized with {total_params:,} trainable parameters.")

    # reduction='none'很重要，因为我们要手动加权
    # 您的 criterion 必须被初始化为返回一个向量（即每个样本一个损失值），而不是一个平均值。
    criterion = nn.CrossEntropyLoss(reduction='none')
    # 专门为验证和测试创建一个标准的损失函数
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['hyperparameters']['learning_rate']))
    epochs = int(cfg['hyperparameters']['epochs'])

    # 初始化 lambda 调度参数
    lambda_initial = cfg['itr_module'].get('lambda_initial', 0.1)
    lambda_final = cfg['itr_module'].get('lambda_final', 0.5)

    # 检查config中是否有USE_LR_SCHEDULER，并且为True
    if cfg.get('USE_LR_SCHEDULER', True):
        scheduler = StepLR(optimizer,
                           step_size=cfg.get('LR_SCHEDULER_STEP_SIZE', 10),
                           gamma=cfg.get('LR_SCHEDULER_GAMMA', 0.1))
        logging.info(
            f"--- [System] StepLR scheduler enabled (step_size={scheduler.step_size}, gamma={scheduler.gamma}) ---")
    else:
        # 备份使用余弦退火，这个是好比跑完10个epochs，从0.1-0.01平均开了，不一定好；主要事迹使用了早停，这个就更不行了，嘴壶几个epcho很慢
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        logging.info(f"--- [System] CosineAnnealingLR scheduler enabled (T_max={epochs}) ---")

    # logging.info(model)
    #logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # --- [NEW & CRITICAL] 4. 训练前构建图并进行分析 ---
    logging.info("\n--- [Step 4] Performing pre-training graph construction and analysis... ---")
    # 如果数据量太大，这里需求GPU太大满足不了，class8需求200G的GPU
    # x_full_train 是整个训练集的输入张量；通常形状是 [Total_Samples, Seq_Len, Num_Nodes * Features]；
    # x_full_train = train_loader.dataset.tensors[0].to(DEVICE)

    # 实际对后面的训练无用，这里的目的是提前看一下分数和图
    # model是一个对象。图结构和PageRank是这个模型对象的内部状态；通过调用_build_and_cache_graph()更新模型的内部状态
    if gcn_type != 'NONE':
        model._build_and_cache_graph(A_physical, train_loader, output_dir, physical_node_names)
        # 分析并可视化此时的初始最终图 A_final
        # analyze_and_visualize_pagerank(model.cached_A_final, physical_node_names, output_dir)
        logging.info("--- Initial graph analysis complete. Starting training... ---")
    else:
        logging.info("--- [Ablation] gcn_type == 'NONE', skip initial graph construction and analysis. ---")

    epochs_no_improve = 0
    early_stopping_patience = cfg['hyperparameters']['EARLY_STOPPING_PATIENCE']
    # --- 5. 训练与验证循环 ---
    # 最佳模型以验证集的F1指标为准
    best_val_f1 = 0.0
    # 用于绘图，迭代块两部分损失变化的可视化
    all_val_exit_iters_draw = []
    val_avg_exit_iters_per_epoch = []
    train_main_losses_per_epoch = []
    train_penalty_losses_per_epoch = []
    # 用于绘图：Dynamic_ITR 模式下，每个 epoch 的“纯 CE”损失（训练 & 验证）
    dynamic_itr_train_ce_losses_per_epoch = []
    dynamic_itr_val_ce_losses_per_epoch = []

    # 用于绘图，k步迭代置信度分数变化
    all_epochs_confidence_data = []
    
    # 用于绘图，其他模式（Single_ITR, GRU, MLP）的损失收敛
    train_losses_per_epoch = []
    val_losses_per_epoch = []

    torch.autograd.set_detect_anomaly(True)

    # 目的：验证跑的时间效率，记录纯训练+验证的开始时间
    training_and_val_start_time = time.time()

    for epoch in range(epochs):
        if gcn_type != 'NONE':
            logging.info(f"\n===== Epoch {epoch + 1}/{epochs} Updating graph structure... =====")
        else:
            logging.info(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        epoch_start_time = time.time()
        # 在每个epoch开始时，重新构建图，获取A_learned参数，以反映可学习参数的更新
        if gcn_type != 'NONE':
            model._build_and_cache_graph(A_physical, train_loader, output_dir, physical_node_names)

        # lambda 调度：仅在 Dynamic_ITR 模式下需要
        if itr_type == 'Dynamic_ITR':
            # 每 3 个 epoch 增加 0.05 的阶梯式 lambda
            # epoch 是从 0 开始计数：
            #  epoch=0,1  -> step_idx = 0
            #  epoch=2,3  -> step_idx = 1
            #  epoch=4,5  -> step_idx = 2
            step_idx = epoch  # 每 2 个 epoch 算一个台阶
            step_size = 0.05  # 每个台阶加多少
            current_lambda = lambda_initial + step_idx * step_size
            # 限制最大不超过 lambda_final
            current_lambda = min(current_lambda, lambda_final)
            logging.info(f"Current lambda = {current_lambda:.4f}")

        # --- 训练 ---
        model.train()

        total_train_loss = 0
        # 仅在 Dynamic_ITR 下使用：累计当前 epoch 内所有 batch 的 raw CE 均值
        epoch_ce_losses = []
        train_preds, train_true = [], []
        # 用于绘图，迭代块两部分损失变化的可视化
        epoch_main_losses = []
        epoch_penalty_losses = []

        # 用于绘图，k步迭代置信度分数变化
        epoch_confidence_scores = [[] for _ in range(cfg['itr_module']['num_iterations'])]  # 假设 num_iterations=5

        # DataLoader每次吐出一个x_batch，其形状是 [Batch_Size, 90, Num_PAA_Features] = [128,90,12]。这个x_batch里包含了Batch_Size个这样的完整序列;y_batch形状是 [Batch_Size, 0]
        # y_batch标签是创建时间会话时候用的最后一个时刻的标签，但是要注意，x和y的位置不在一起
        # 从train_loader中按照批次取出来sensors, _, labels3个内容，原本有5个内容：sensors, states, labels，时间，井号
        for batch_idx, (x_batch, state_batch, y_batch, *_) in enumerate(train_loader):
            # 井的状态没有的话，这里的state_batch为[512,0]
            x_batch, state_batch, y_batch = x_batch.to(DEVICE), state_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            # 消融实验，动态门控
            if itr_type == 'Dynamic_ITR':
                # forward不再需要传入图结构，因为它会使用缓存
                intermediate_logits, confidence_scores = model(x_batch, state_batch)
                # 用于绘图，k步迭代置信度分数变化
                for k in range(cfg['itr_module']['num_iterations']):
                    epoch_confidence_scores[k].append(confidence_scores[k].mean().item())
                #loss = calculate_dynamic_gating_loss(intermediate_logits, confidence_scores, y_batch, criterion, lambda_penalty)
                loss, main_loss_val, penalty_loss_val, raw_loss_k_means = calculate_dynamic_gating_loss(
                    intermediate_logits, confidence_scores, y_batch, criterion, current_lambda
                )
                # 收集当前批次的损失值
                epoch_main_losses.append(main_loss_val)
                epoch_penalty_losses.append(penalty_loss_val)
                # raw_loss_k_means 是一个长度为 num_iterations 的列表，表示该 batch 内每个迭代步的 CE 均值
                # 这里我们取其平均值，作为该 batch 的“纯 CE”指标，用于后续按 epoch 求平均
                if isinstance(raw_loss_k_means, (list, tuple)) and len(raw_loss_k_means) > 0:
                    epoch_ce_losses.append(float(np.mean(raw_loss_k_means)))

            # Dynamic_ITR_CE: 迭代5次，但只使用最后一次迭代的logits计算单一CELoss
            elif itr_type == 'Dynamic_ITR_CE':
                # forward不再需要传入图结构，因为它会使用缓存
                # 返回格式: [final_logits], confidence_scores（用于可视化）
                intermediate_logits, confidence_scores = model(x_batch, state_batch)
                final_logits = intermediate_logits[-1]
                # 用于绘图，k步迭代置信度分数变化
                if confidence_scores is not None:
                    for k in range(cfg['itr_module']['num_iterations']):
                        epoch_confidence_scores[k].append(confidence_scores[k].mean().item())
                # 使用单一的交叉熵损失
                loss_vector = criterion(final_logits, y_batch)
                loss = loss_vector.mean()

            # 标准交叉熵损失的消融实验
            else:
                # 从model的 forward 中返回了不同模式的结果。不同结果使用的都是交叉熵损失函数
                intermediate_logits, _ = model(x_batch, state_batch)
                final_logits = intermediate_logits[-1]
                #loss = criterion(final_logits, y_batch)
                # criterion 返回的是一个损失向量，我们需要手动求平均值
                # 原代码 (错误): loss = criterion(final_logits, y_batch)
                loss_vector = criterion(final_logits, y_batch)
                loss = loss_vector.mean()  # <--- 新增的 .mean() 操作

            # 用于计算后面的平均损失的
            # .item()：将单元素张量转换为Python标量数值
            total_train_loss += loss.item()

            # 使用最后一次迭代的预测作为训练过程中的指标
            # 它的作用是作为一个简单、稳定的指示器，告诉您模型的完整推理能力是否在稳步提升。而不是最优的，刚开始可能有错误的高置信度，此时会误导后续动作
            final_preds = torch.argmax(intermediate_logits[-1], dim=1)
            train_preds.extend(final_preds.cpu().numpy())
            train_true.extend(y_batch.cpu().numpy())

            # retain_graph=True会导致计算图被保留在内存中，这会增加内存消耗。
            # 但在本使用场景中，这是完全可以接受且正确的。因为在下一次循环迭代时，当模型进行新的前向传播时，旧的图会被释放，PyTorch会构建一个全新的图。所以它不会导致内存无限增长。
            loss.backward(retain_graph=True)
            # loss.backward()
            # 在用优化器更新参数之前，先把所有参数的梯度‘量一下’，如果太大就整体缩小到合适范围，防止一步更新太猛导致梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()

            if (batch_idx + 1) % cfg['hyperparameters']['PROGRESS_REPORT_FREQ'] == 0 or (batch_idx + 1) == len(train_loader):
                if itr_type == 'Dynamic_ITR':
                    # 计算此批次的平均置信度
                    avg_confs_batch = [c.mean().item() for c in confidence_scores]
                    avg_confs_str = ", ".join([f"Iter {k + 1}: {c:.3f}" for k, c in enumerate(avg_confs_batch)])
                    logging.info(
                        f"  Epoch [{epoch + 1:02d}/{cfg['hyperparameters']['epochs']}], Batch [{batch_idx + 1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
                    logging.info(f"    -> Batch Avg Confidences: [{avg_confs_str}]")
                    logging.info(f"    -> 5次迭代CEloss的均值: [{raw_loss_k_means}]")
                else:
                    logging.info(
                        f"  Epoch [{epoch + 1:02d}/{cfg['hyperparameters']['epochs']}], Batch [{batch_idx + 1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
        # 消融实验，ture为动态门控，flase为交叉熵函数
        if itr_type == 'Dynamic_ITR':
            # 计算并保存当前 epoch 的平均损失
            avg_main_loss = np.mean(epoch_main_losses)
            avg_penalty_loss = np.mean(epoch_penalty_losses)
            train_main_losses_per_epoch.append(avg_main_loss)
            train_penalty_losses_per_epoch.append(avg_penalty_loss)

            # 用于绘图，k步迭代置信度分数变化
            avg_epoch_confs = [np.mean(scores) for scores in epoch_confidence_scores]
            all_epochs_confidence_data.append(avg_epoch_confs)

            # 计算并保存当前 epoch 的“纯 CE”训练损失（来自 raw_loss_k_means）
            if len(epoch_ce_losses) > 0:
                avg_epoch_ce_loss = float(np.mean(epoch_ce_losses))
            else:
                avg_epoch_ce_loss = 0.0
            dynamic_itr_train_ce_losses_per_epoch.append(avg_epoch_ce_loss)

        # Dynamic_ITR_CE: 也保存置信度数据用于可视化
        if itr_type == 'Dynamic_ITR_CE':
            # 用于绘图，k步迭代置信度分数变化
            avg_epoch_confs = [np.mean(scores) for scores in epoch_confidence_scores]
            all_epochs_confidence_data.append(avg_epoch_confs)

        avg_train_loss = total_train_loss / len(train_loader)
        train_f1 = f1_score(train_true, train_preds, average='macro', zero_division=0)

        # --- 验证 ---
        model.eval()
        total_val_loss = 0
        val_preds, val_true = [], []
        all_val_exit_iters = []
        all_val_exit_iters_this_epoch = []
        # 在每个epoch开始验证前，清空当前epoch的列表
        all_val_exit_iters_this_epoch.clear()
        with torch.no_grad():
            # 用于在 Dynamic_ITR 模式下累计当前 epoch 的验证 CE 损失
            epoch_val_ce_losses = []
            for i, (x_batch, state_batch, y_batch, *_) in enumerate(val_loader):
                x_batch, state_batch, y_batch = x_batch.to(DEVICE), state_batch.to(DEVICE), y_batch.to(DEVICE)
                # [修改] model(x_batch) 现在会根据评估模式返回不同内容，早停或者只使用CELOSS已经在model的forward中写好
                # final_logits 可能是一个张量（Dynamic_ITR）或列表（GRU/MLP等）
                final_logits, exit_iters = model(x_batch, state_batch)
                # 处理 final_logits 可能是列表的情况（如 GRU/MLP 模式）
                if isinstance(final_logits, list):
                    final_logits = final_logits[0]  # 取列表中的第一个（也是唯一的）元素
                # 计算验证损失（纯 CE）：
                # 1. eval_criterion 是 nn.CrossEntropyLoss()，返回一个标量
                # 2. 这里的 loss 即为该 batch 的平均 CE 损失
                loss = eval_criterion(final_logits, y_batch)
                total_val_loss += loss.item()

                # 在 Dynamic_ITR 模式下，额外记录验证集的“纯 CE”损失（按 epoch 求平均）
                if itr_type == 'Dynamic_ITR':
                    epoch_val_ce_losses.append(loss.item())

                # 基于 final_logits 进行预测
                final_preds = torch.argmax(final_logits, dim=1)
                val_preds.extend(final_preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
                if exit_iters is not None:
                    # 将当前批次的退出迭代次数收集起来
                    all_val_exit_iters_this_epoch.extend(exit_iters.cpu().numpy())
        # 每个epoch结束后，保存 epoch_exit_iters，用于绘图
        all_val_exit_iters_draw.append(all_val_exit_iters_this_epoch.copy())
        if itr_type in ['Dynamic_ITR', 'Dynamic_ITR_CE']:
            avg_exit_iter_epoch = float(np.mean(all_val_exit_iters_this_epoch)) if all_val_exit_iters_this_epoch else float('nan')
            val_avg_exit_iters_per_epoch.append(avg_exit_iter_epoch)

        avg_val_loss = total_val_loss / len(val_loader)
        # Dynamic_ITR 模式下，按 epoch 保存验证 CE 损失
        if itr_type == 'Dynamic_ITR':
            if len(epoch_val_ce_losses) > 0:
                avg_epoch_val_ce_loss = float(np.mean(epoch_val_ce_losses))
            else:
                avg_epoch_val_ce_loss = 0.0
            dynamic_itr_val_ce_losses_per_epoch.append(avg_epoch_val_ce_loss)
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        val_acc = accuracy_score(val_true, val_preds)
        
        # 记录所有模式的训练和验证损失用于绘图
        train_losses_per_epoch.append(avg_train_loss)
        val_losses_per_epoch.append(avg_val_loss)

        # 计算并打印验证集的提前退出统计信息
        if itr_type == 'Dynamic_ITR' and all_val_exit_iters:
            # 使用 np.bincount 快速统计每个迭代次数的样本数
            # bincount 的索引从0开始，所以我们创建一个从0到最大迭代数的完整数组
            iter_counts = np.bincount(all_val_exit_iters, minlength=cfg['itr_module']['num_iterations'] + 1)
            exit_summary_str = "  - Val Early Exit Stats: "
            for iter_num in range(1, len(iter_counts)):
                exit_summary_str += f"[Iter {iter_num}: {iter_counts[iter_num]} samples] "
            avg_exit_iter = np.mean(all_val_exit_iters)
            exit_summary_str += f"| Avg Exit Iter: {avg_exit_iter:.2f}"
            logging.info(exit_summary_str)

        logging.info(
            f"Epoch [{epoch + 1:02d}/{cfg['hyperparameters']['epochs']}] COMPLETE. Summary | Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f} | "
                     f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}, Time: {time.time() - epoch_start_time:.2f}s")
        # 应用 StepLR 学习率调度
        scheduler.step()
        logging.info(f"  - LR scheduler step. New LR: {scheduler.get_last_lr()[0]:.8f}")

        # 早停机制与模型保存逻辑 (基于val_f1)
        # 我们现在只关心 val_f1。Loss只用于观察。
        # 1. 检查 F1 分数是否有提升
        if val_f1 > best_val_f1:
            logging.info(f"--- Validation F1 improved ({best_val_f1:.4f} --> {val_f1:.4f}). Saving best model... ---")
            best_val_f1 = val_f1
            # 保存 F1 分数最高的模型
            torch.save(model.state_dict(), best_model_path)
            # 因为 F1 提升了，所以重置早停计数器
            epochs_no_improve = 0
        else:
            # 如果 F1 没有提升，增加早停计数器
            epochs_no_improve += 1
            logging.info(
                f"  Validation F1 did not improve. Best F1: {best_val_f1:.4f}. Patience: {epochs_no_improve}/{early_stopping_patience}")
        # 2. 检查是否触发早停
        if epochs_no_improve >= early_stopping_patience:
            logging.info(
                f"--- [Early Stopping] Triggered after {epoch + 1} epochs because Val F1 did not improve. ---")
            break
    training_and_val_runtime = time.time() - training_and_val_start_time
    logging.info("--- Training finished ---")
    # 目的：为了验证效率
    logging.info(f"  - 消融实验：Total Training + Validation Time: {training_and_val_runtime:.2f}s")

    if itr_type == 'Dynamic_ITR' or itr_type == 'Dynamic_ITR_CE':
        logging.info("--- 开始绘制：迭代推理k与置信度分数的演变情况-置信度演进热力图---")
        # 训练结束后
        confidence_data_np = np.array(all_epochs_confidence_data).T  # 转置，让行为k，列为epoch
        num_iterations = confidence_data_np.shape[0]  # 从数据形状获取迭代次数
        # 从收集的数据中安全地获取 epoch 数量
        num_epochs = len(all_epochs_confidence_data)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            confidence_data_np,
            annot=True, fmt=".2f", cmap="viridis",
            # num_epochs 指的是模型实际运行的总轮数
            xticklabels=range(1, num_epochs + 1),
            yticklabels=[f'Iter {k + 1}' for k in range(num_iterations)],
            ax=ax
        )
        ax.set_title('Average Confidence Score Evolution per Iteration', fontsize=16, weight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Iteration (k)', fontsize=12)
        heatmap_path = os.path.join(output_dir, 'confidence_evolution_heatmap.svg')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close(fig)
        logging.info("--- 绘制完成：置信度演进热力图---")

        # 2. 保存置信度演变数据 (对应 confidence_evolution_heatmap)
        # 假设 all_epochs_confidence_data 是一个 list of lists
        # 结构: [[epoch1_iter1, epoch1_iter2...], [epoch2_iter1...]]
        # 我们将其转换为 DataFrame，每一列代表一次迭代
        conf_data = {}
        conf_data['Epoch'] = list(range(1, len(all_epochs_confidence_data) + 1))
        num_iters = len(all_epochs_confidence_data[0]) if all_epochs_confidence_data else 0

        for k in range(num_iters):
            # 获取每一轮迭代在所有 epoch 上的数据
            iter_k_scores = [epoch_scores[k] for epoch_scores in all_epochs_confidence_data]
            conf_data[f'Iteration_{k + 1}'] = iter_k_scores

        save_plot_data(conf_data, output_dir, 'data_confidence_evolution.csv')

        logging.info("--- 开始绘制：验证期间迭代推理深度的演变---")
        # 假设 all_val_exit_iters_draw 是一个列表的列表
        # [[ep0_iter1, ep0_iter2,...], [ep1_iter1, ep1_iter2,...], ...]
        # 1. 转换数据为适合绘图的 DataFrame
        num_epochs = len(all_val_exit_iters_draw)
        num_iterations = 5  # 假设最多5次迭代
        data_for_plot = []
        for epoch in range(num_epochs):
            counts = np.bincount(all_val_exit_iters_draw[epoch], minlength=num_iterations + 1)[1:]  # 统计1-5的次数
            percentages = counts / counts.sum() * 100
            for i in range(num_iterations):
                data_for_plot.append({'Epoch': epoch + 1, 'Iteration': f'Iter {i + 1}', 'Percentage': percentages[i]})
        df_plot = pd.DataFrame(data_for_plot)
        df_pivot = df_plot.pivot(index='Epoch', columns='Iteration', values='Percentage')
        # ==========================================
        # [新增] 保存绘图源数据到 CSV
        # ==========================================
        try:
            # 1. 确保保存目录存在
            plot_data_dir = os.path.join(output_dir, 'plot_data')
            os.makedirs(plot_data_dir, exist_ok=True)

            # 2. 定义保存路径
            csv_path = os.path.join(plot_data_dir, 'data_iterative_reasoning_evolution.csv')

            # 3. 保存 CSV
            # df_pivot 的 index 是 'Epoch'，保存时会自动作为第一列，非常完美
            df_pivot.to_csv(csv_path, index=True)

            logging.info(f"  [Data Saved] Iterative evolution data saved to: {csv_path}")
        except Exception as e:
            logging.warning(f"  [Warning] Failed to save iterative evolution data: {e}")
        # 2. 绘图
        sns.set_theme(style="white", font="Times New Roman")
        fig, ax = plt.subplots(figsize=(10, 6))
        # 使用专业色板
        colors = sns.color_palette("viridis_r", n_colors=num_iterations)
        df_pivot.plot(kind='area', stacked=True, ax=ax, color=colors, linewidth=0.5, alpha=0.8)
        ax.set_title('Evolution of Iterative Reasoning Depth during Validation', fontsize=16, weight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Percentage of Decisions (%)', fontsize=12)
        ax.legend(title='Exit Iteration', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylim(0, 100)
        ax.set_xlim(1, num_epochs)
        # 让 X 轴刻度是整数
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        sns.despine()  # 移除顶部和右侧的轴线
        iterative_path = os.path.join(output_dir, 'iterative_reasoning_evolution.svg')
        plt.tight_layout()
        plt.savefig(iterative_path, dpi=300)
        plt.close(fig) # 然后关闭这个图形对象，释放内存
        logging.info("--- 绘制完成：验证期间迭代推理深度的演变---")

        # 验证集平均退出迭代次数随 Epoch 变化
        if val_avg_exit_iters_per_epoch:
            valid_points = [(idx + 1, v) for idx, v in enumerate(val_avg_exit_iters_per_epoch) if not np.isnan(v)]
            if valid_points:
                epochs_avg, avg_exits = zip(*valid_points)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(epochs_avg, avg_exits, marker='o', color='tab:purple', linewidth=2)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Average Exit Iteration (Validation)', fontsize=12)
                ax.set_title('Average Exit Iteration vs Epoch (Validation)', fontsize=16, weight='bold')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(True, alpha=0.3)
                avg_exit_path = os.path.join(output_dir, 'avg_exit_iteration_per_epoch.svg')
                plt.tight_layout()
                plt.savefig(avg_exit_path, dpi=300)
                plt.close(fig)
                avg_exit_data = {
                    'Epoch': list(epochs_avg),
                    'Avg_Exit_Iteration': list(avg_exits)
                }
                save_plot_data(avg_exit_data, output_dir, 'data_avg_exit_iteration_per_epoch.csv')

        # 绘制动态门控趋势图
        logging.info("\n--- 开始绘制：动态门控趋势图，两部分损失的变化趋势... ---")
        # 假设 train_main_losses_per_epoch 和 train_penalty_losses_per_epoch 是两个列表
        epochs = range(1, len(train_main_losses_per_epoch) + 1)

        sns.set_theme(style="whitegrid", font="Times New Roman")
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 绘制 L_main
        color1 = 'tab:blue'
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Main Loss (L_main)', color=color1, fontsize=12)
        ax1.plot(epochs, train_main_losses_per_epoch, color=color1, linestyle='-', label='Main Loss')
        ax1.tick_params(axis='y', labelcolor=color1)

        # 创建共享X轴的第二个Y轴
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Penalty Loss (L_penalty)', color=color2, fontsize=12)
        ax2.plot(epochs, train_penalty_losses_per_epoch, color=color2, linestyle='--', label='Penalty Loss')
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.set_title('Dynamic Gating Loss Components during Training', fontsize=16, weight='bold')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()  # 调整布局防止标签重叠

        # 添加统一的图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        dynamic_path = os.path.join(output_dir, 'dynamic_loss_components.svg')
        plt.savefig(dynamic_path, dpi=300)
        plt.close(fig) # 然后关闭这个图形对象，释放内存
        logging.info("\n--- 绘制完成：动态门控趋势图，两部分损失的变化趋势... ---")

        # 1. 保存损失函数数据 (对应 dynamic_loss_components 图)
        loss_data = {
            'Epoch': list(range(1, len(train_main_losses_per_epoch) + 1)),
            'Main_Loss': train_main_losses_per_epoch,
            'Penalty_Loss': train_penalty_losses_per_epoch
        }
        save_plot_data(loss_data, output_dir, 'data_dynamic_loss_components.csv')
        
        # 绘制训练和验证损失收敛图（Dynamic_ITR模式） - 原有图（主损失 & 验证总损失）
        logging.info("\n--- 开始绘制：Dynamic_ITR模式下的训练/验证损失收敛图（主损失+验证总损失）... ---")
        epochs_loss = range(1, len(train_losses_per_epoch) + 1)
        
        sns.set_theme(style="whitegrid", font="Times New Roman")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制训练损失和验证损失（原逻辑保持不变）
        ax.plot(epochs_loss, train_losses_per_epoch, color='tab:blue', linestyle='-', label='Train Main Loss (L_k*C_k)', linewidth=2)
        ax.plot(epochs_loss, val_losses_per_epoch, color='tab:red', linestyle='--', label='Validation Loss (Cross Entropy)', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Convergence during Training (Dynamic_ITR Mode)', fontsize=16, weight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        loss_convergence_path = os.path.join(output_dir, 'loss_convergence_dynamic_itr.svg')
        plt.savefig(loss_convergence_path, dpi=300)
        plt.close(fig)
        logging.info("--- 绘制完成：Dynamic_ITR模式下的训练/验证损失收敛图（主损失+验证总损失） ---")
        
        # 保存训练/验证损失数据到CSV（原来的文件保持不变）
        loss_convergence_data = {
            'Epoch': list(range(1, len(train_losses_per_epoch) + 1)),
            'Train_Loss': train_losses_per_epoch,
            'Validation_Loss': val_losses_per_epoch
        }
        save_plot_data(loss_convergence_data, output_dir, 'data_loss_convergence_dynamic_itr.csv')

        # 新增：绘制 Dynamic_ITR 模式下“纯 CE”训练/验证损失收敛图
        logging.info("\n--- 开始绘制：Dynamic_ITR模式下的训练/验证 Cross Entropy Loss 收敛图... ---")
        epochs_ce = range(1, len(dynamic_itr_train_ce_losses_per_epoch) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs_ce, dynamic_itr_train_ce_losses_per_epoch, color='tab:blue', linestyle='-', label='Train CE Loss (avg over k)', linewidth=2)
        ax.plot(epochs_ce, dynamic_itr_val_ce_losses_per_epoch, color='tab:red', linestyle='--', label='Validation CE Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Cross Entropy Loss', fontsize=12)
        ax.set_title('Pure CE Loss Convergence during Training (Dynamic_ITR Mode)', fontsize=16, weight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        ce_loss_convergence_path = os.path.join(output_dir, 'loss_convergence_dynamic_itr_ce_components.svg')
        plt.savefig(ce_loss_convergence_path, dpi=300)
        plt.close(fig)
        logging.info("--- 绘制完成：Dynamic_ITR模式下的训练/验证 Cross Entropy Loss 收敛图 ---")

        # 保存“纯 CE”训练/验证损失数据到 CSV
        ce_loss_convergence_data = {
            'Epoch': list(range(1, len(dynamic_itr_train_ce_losses_per_epoch) + 1)),
            'Train_CE_Loss': dynamic_itr_train_ce_losses_per_epoch,
            'Validation_CE_Loss': dynamic_itr_val_ce_losses_per_epoch
        }
        save_plot_data(ce_loss_convergence_data, output_dir, 'data_loss_convergence_dynamic_itr_ce_components.csv')

        # 组合函数动态变化图（训练集）
        if train_main_losses_per_epoch and train_penalty_losses_per_epoch and dynamic_itr_train_ce_losses_per_epoch and all_epochs_confidence_data:
            combined_loss_per_epoch = [m + p for m, p in zip(train_main_losses_per_epoch, train_penalty_losses_per_epoch)]
            avg_conf_per_epoch = [float(np.mean(epoch_scores)) for epoch_scores in all_epochs_confidence_data]
            min_len = min(len(combined_loss_per_epoch), len(dynamic_itr_train_ce_losses_per_epoch), len(avg_conf_per_epoch))
            epochs_comb = list(range(1, min_len + 1))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs_comb, avg_conf_per_epoch[:min_len], color='tab:green', marker='o', label='Avg Confidence c_k')
            ax.plot(epochs_comb, dynamic_itr_train_ce_losses_per_epoch[:min_len], color='tab:orange', marker='s', label='Avg CE Loss l_k')
            ax.plot(epochs_comb, combined_loss_per_epoch[:min_len], color='tab:blue', marker='^', label='Combined Loss L_total')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Dynamic Combination Components during Training (Dynamic_ITR)', fontsize=16, weight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            combo_path = os.path.join(output_dir, 'dynamic_combination_components.svg')
            plt.savefig(combo_path, dpi=300)
            plt.close(fig)

            combo_data = {
                'Epoch': epochs_comb,
                'Avg_Confidence_c_k': avg_conf_per_epoch[:min_len],
                'Avg_CE_Loss_l_k': dynamic_itr_train_ce_losses_per_epoch[:min_len],
                'Combined_Loss_L_total': combined_loss_per_epoch[:min_len]
            }
            save_plot_data(combo_data, output_dir, 'data_dynamic_combination_components.csv')
    
    # 为其他模式（Dynamic_ITR_CE, Single_ITR, GRU, MLP, LSTM, Transformer）绘制损失收敛图
    elif itr_type in ['Dynamic_ITR_CE', 'Single_ITR', 'GRU', 'MLP', 'LSTM', 'Transformer']:
        logging.info(f"\n--- 开始绘制：{itr_type}模式下的损失收敛图... ---")
        epochs = range(1, len(train_losses_per_epoch) + 1)
        
        sns.set_theme(style="whitegrid", font="Times New Roman")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制训练损失和验证损失
        ax.plot(epochs, train_losses_per_epoch, color='tab:blue', linestyle='-', label='Train Loss (Cross Entropy)', linewidth=2)
        ax.plot(epochs, val_losses_per_epoch, color='tab:red', linestyle='--', label='Validation Loss (Cross Entropy)', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Cross Entropy Loss', fontsize=12)
        ax.set_title(f'Loss Convergence during Training ({itr_type} Mode)', fontsize=16, weight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        loss_convergence_path = os.path.join(output_dir, f'loss_convergence_{itr_type.lower()}.svg')
        plt.savefig(loss_convergence_path, dpi=300)
        plt.close(fig)
        logging.info(f"--- 绘制完成：{itr_type}模式下的损失收敛图 ---")
        
        # 保存损失数据到CSV
        loss_data = {
            'Epoch': list(range(1, len(train_losses_per_epoch) + 1)),
            'Train_Loss': train_losses_per_epoch,
            'Validation_Loss': val_losses_per_epoch
        }
        save_plot_data(loss_data, output_dir, f'data_loss_convergence_{itr_type.lower()}.csv')

    # --- 6. 在测试集上评估最佳模型 ---
    logging.info("\n--- [Step 6] Evaluating on Test Set... ---")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    model.eval()
    # 使用最佳模型运行 PageRank 分析（仅在启用图模块时）
    if gcn_type != 'NONE':
        logging.info(" - Running graph analysis on best model...")
        # 需要训练数据来构建数据驱动的图，因为模型的训练的图，都是基于训练数据的，这里的目的就是想体现最终的图是怎样的
        model._build_and_cache_graph(A_physical, train_loader, output_dir, physical_node_names)
        # 运行PageRank分析并获取Series (用于报告)
        pagerank_series = analyze_and_visualize_pagerank(
            model.cached_A_final,
            physical_node_names,
            output_dir
        )
        # 为报告准备PageRank字符串
        if pagerank_series is not None:
            top_10_pagerank_str = pagerank_series.head(10).to_string()
        else:
            top_10_pagerank_str = "N/A (Graph has no edges)"
        logging.info(" - Best model graph analysis complete.")
    else:
        pagerank_series = None
        top_10_pagerank_str = "Graph-based analysis disabled (gcn_type == 'NONE')."

    test_preds, test_true = [], []
    test_correct_confidences = []
    # 为测试集收集退出迭代次数
    all_test_exit_iters = []
    with torch.no_grad():
        for i, (x_batch, state_batch, y_batch, *_) in enumerate(test_loader):
            x_batch, state_batch = x_batch.to(DEVICE), state_batch.to(DEVICE)
            # 保留一份 CPU 版标签用于后续 numpy 统计，同时构造与预测同设备的张量用于比较
            y_batch_cpu = y_batch.numpy()
            y_batch_device = y_batch.to(DEVICE)
            if itr_type in ['Dynamic_ITR', 'Dynamic_ITR_CE']:
                final_logits, exit_iters, iter_confidences = model.eval_forward_with_confidence(x_batch, state_batch)
            else:
                final_logits, exit_iters = model(x_batch, state_batch)
                iter_confidences = None
            # 处理 final_logits 可能是列表的情况（如 GRU/MLP 模式）
            if isinstance(final_logits, list):
                final_logits = final_logits[0]  # 取列表中的第一个（也是唯一的）元素
            final_preds = torch.argmax(final_logits, dim=1)

            test_preds.extend(final_preds.cpu().numpy())
            test_true.extend(y_batch_cpu)
            # BCE模式的时候返回的是NONE
            if exit_iters is not None:
                all_test_exit_iters.extend(exit_iters.cpu().numpy())
            if itr_type in ['Dynamic_ITR', 'Dynamic_ITR_CE'] and iter_confidences is not None and len(iter_confidences) > 0:
                correct_mask = final_preds.eq(y_batch_device)
                if correct_mask.any():
                    exit_iters_safe = exit_iters if exit_iters is not None else torch.full((final_preds.shape[0],), len(iter_confidences), device=final_preds.device, dtype=torch.long)
                    selected_conf = []
                    for idx in range(final_preds.shape[0]):
                        if correct_mask[idx]:
                            iter_idx = int(exit_iters_safe[idx].item()) - 1
                            iter_idx = max(0, min(iter_idx, len(iter_confidences) - 1))
                            conf_val = iter_confidences[iter_idx][idx].item()
                            selected_conf.append(conf_val)
                    test_correct_confidences.extend(selected_conf)

    # 测试集正确样本的置信度分布（KDE，迭代推理置信度）
    if itr_type in ['Dynamic_ITR', 'Dynamic_ITR_CE'] and test_correct_confidences:
        sns.set_theme(style="whitegrid", font="Times New Roman")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(test_correct_confidences, fill=True, color='tab:blue', bw_adjust=0.6, ax=ax)
        # 计算与绘图一致的 KDE 网格与密度，便于后续复现曲线
        conf_array = np.asarray(test_correct_confidences)
        kde_estimator = gaussian_kde(conf_array)
        kde_estimator.set_bandwidth(kde_estimator.factor * 0.6)  # 与 bw_adjust 保持一致
        kde_x = np.linspace(conf_array.min(), conf_array.max(), 200)
        kde_density = kde_estimator(kde_x)
        ax.set_xlabel('Iteration Confidence (Correct Predictions)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Confidence Distribution on Test Set (Correct Samples)', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        kde_path = os.path.join(output_dir, 'confidence_kde_test_correct.svg')
        plt.savefig(kde_path, dpi=300)
        plt.close(fig)
        # 原始置信度列表
        save_plot_data({'Confidence': test_correct_confidences}, output_dir, 'data_confidence_kde_test_correct.csv')
        # KDE 曲线数据（x 轴与密度），方便直接重绘
        save_plot_data(
            {
                'KDE_X': kde_x.tolist(),
                'KDE_Density': kde_density.tolist(),
                'BW_Adjust': [0.6] * len(kde_x)
            },
            output_dir,
            'data_confidence_kde_test_correct_curve.csv'
        )

    # --- [新增] 7. 生成可视化与综合报告 ---
    logging.info("\n--- [Step 7] Generating Visualizations and  Report... ---")

    # 定义类别名称
    if num_classes == 2:
        target_names = ['Normal', 'Fault']
    else:
        target_names = [f'Class {i}' for i in range(num_classes)]

    # 绘制混淆矩阵
    cm = confusion_matrix(test_true, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    # 去掉网格线
    disp.ax_.grid(False)
    for spine in disp.ax_.spines.values():
        spine.set_visible(False)
    plt.title(f'Confusion Matrix (Test Set)')
    cm_path = os.path.join(output_dir, 'confusion_matrix_test.svg')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logging.info(f" - Confusion Matrix saved to '{cm_path}'")

    # 将 numpy array 转为 DataFrame 保存
    # 建议加上列名和行名（类别名称），方便看
    class_names = [f'Class_{i}' for i in range(num_classes)]  # 或者使用您真实的类别名列表

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # 这里我们需要手动保存，因为 save_plot_data 接收的是字典
    save_path = os.path.join(output_dir, 'plot_data')
    os.makedirs(save_path, exist_ok=True)
    cm_file_path = os.path.join(save_path, 'data_confusion_matrix.csv')
    cm_df.to_csv(cm_file_path, index=True)  # 保留索引，因为索引是真实标签
    print(f"  [Data Saved] Confusion matrix data saved to: {cm_file_path}")

    # 运行置换重要性分析
    logging.info("\n--- [Running Feature Importance Analysis] ---")
    # 1. 获取基线F1分数
    baseline_f1_test = f1_score(test_true, test_preds, average='macro', zero_division=0)
    logging.info(f" - Baseline Macro F1 on Test Set: {baseline_f1_test:.4f}")

    # 2. 运行分析 (使用 test_loader)
    importance_df = calculate_feature_importance(
        model,
        test_loader,
        physical_node_names,
        cfg.get('graph_definitions', {}),
        baseline_f1_test,
        cfg  # 传递整个config以便读取 PERMUTATION_REPEATS
    )
    plot_data_dir = os.path.join(output_dir, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)

    # 2. 直接保存为 CSV
    csv_path = os.path.join(plot_data_dir, 'data_feature_importance.csv')

    # 注意：通常 importance_df 的索引(index)是特征名称，所以 index=True
    importance_df.to_csv(csv_path, index=True)
    logging.info(f"  [Data Saved] Feature importance data saved to: {csv_path}")

    # 3. 绘制特征重要性条形图
    # 阈值设置得较低 (0.001)，以防 F1-Drop 值普遍较小
    important_plot_df = importance_df[importance_df['Mean Importance (F1 Drop)'] > 0.001].copy()
    if not important_plot_df.empty:
        important_plot_df = important_plot_df.sort_values(by='Mean Importance (F1 Drop)', ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(6, len(important_plot_df) * 0.4)))

        ax.barh(important_plot_df['Variable (Node)'], important_plot_df['Mean Importance (F1 Drop)'],
                xerr=important_plot_df['Std Dev'], capsize=5, color='skyblue', ecolor='gray',
                label='Mean Importance')

        ax.errorbar(important_plot_df['Mean Importance (F1 Drop)'], important_plot_df['Variable (Node)'],
                    xerr=important_plot_df['Std Dev'], fmt='none', ecolor='gray', capsize=5,
                    label=f"Std Dev ({cfg.get('PERMUTATION_REPEATS', 10)} repeats)")

        ax.set_xlabel('Mean Importance (Macro F1 Drop)')
        ax.set_title('Feature Importance (Contribution > 0.001)')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        importance_plot_path = os.path.join(output_dir, 'feature_importance_barchart.svg')
        plt.savefig(importance_plot_path, dpi=300)
        plt.close()
        logging.info(f" - Feature importance bar chart saved to '{importance_plot_path}'")
    else:
        logging.info(" - No features found with importance > 0.001. Skipping bar chart.")

    # 获取测试集详细指标
    test_f1_macro = f1_score(test_true, test_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    test_acc = accuracy_score(test_true, test_preds)
    test_mcc = matthews_corrcoef(test_true, test_preds)
    test_fnr, test_fpr = calculate_fnr_fpr(test_true, test_preds)
    report_str = classification_report(test_true, test_preds, target_names=target_names, zero_division=0, digits=4)
    
    # 打印所有指标
    logging.info("\n--- [Test Set Performance Metrics] ---")
    logging.info(f"  - Accuracy: {test_acc:.4f}")
    logging.info(f"  - F1-Score (Macro): {test_f1_macro:.4f}")
    logging.info(f"  - F1-Score (Weighted): {test_f1_weighted:.4f}")
    logging.info(f"  - Matthews Correlation Coefficient (MCC): {test_mcc:.4f}")
    logging.info(f"  - False Negative Rate (FNR): {test_fnr:.4f}")
    logging.info(f"  - False Positive Rate (FPR): {test_fpr:.4f}")
    logging.info("\n--- [Classification Report] ---")
    print(report_str)
    
    # --- 计算 FLOPs 和推理延迟 ---
    logging.info("\n--- [Computational Metrics] ---")
    # 获取一个样本用于 FLOPs 和延迟计算（batch_size=1）
    sample_batch = next(iter(test_loader))
    x_sample, state_sample = sample_batch[0][0:1].to(DEVICE), sample_batch[1][0:1].to(DEVICE)  # batch_size=1
    
    # 计算 FLOPs（单次前向传播，batch_size=1）
    # 注意：thop 库在程序退出时可能产生 multiprocessing 清理警告（如 "Directory not empty"），
    # 这是 thop 库内部使用多进程时的已知问题，不影响功能，可以安全忽略
    logging.info(" - Calculating FLOPs (single forward pass, batch_size=1)...")
    flops_g, flops_m = calculate_flops(model, x_sample, state_sample, DEVICE)
    if flops_g is not None:
        # 根据数值大小选择合适的单位显示
        if flops_g >= 0.001:
            logging.info(f"  - FLOPs: {flops_g:.4f} G ({flops_m:.2f} M) (per sample)")
        else:
            logging.info(f"  - FLOPs: {flops_m:.2f} M ({flops_g:.6f} G) (per sample)")
    else:
        logging.info("  - FLOPs: N/A (thop library not available)")
    
    # 测量推理延迟（batch_size=1）
    logging.info(" - Measuring inference latency (batch_size=1)...")
    avg_latency_ms, std_latency_ms = measure_inference_latency(model, x_sample, state_sample, DEVICE)
    logging.info(f"  - Inference Latency: {avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms/sample")

    # 在最终报告之前，打印测试集的提前退出统计信息
    if all_test_exit_iters:
        iter_counts = np.bincount(all_test_exit_iters, minlength=cfg['itr_module']['num_iterations'] + 1)
        exit_summary_str = "\n[Test Set Early Exit Stats]\n"
        for iter_num in range(1, len(iter_counts)):
            exit_summary_str += f"- Exited at Iteration {iter_num}: {iter_counts[iter_num]} samples\n"
        avg_exit_iter = np.mean(all_test_exit_iters)
        exit_summary_str += f"- Average Exit Iteration: {avg_exit_iter:.2f}\n"
        logging.info(exit_summary_str)

    # 获取峰值GPU显存占用
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024
        memory_report = f"\n[Computational Efficiency]\n- Peak GPU Memory Usage: {peak_memory_mb:.2f} MB\n"
    else:
        peak_memory_mb = 0
        memory_report = "\n[Computational Efficiency]\n- GPU not available, memory usage not tracked.\n"
    logging.info(memory_report)
    # 整合所有报告内容
    # 准备 FLOPs 和延迟的字符串表示
    if flops_g is not None:
        if flops_g >= 0.001:
            flops_str = f"{flops_g:.4f} G ({flops_m:.2f} M)"
        else:
            flops_str = f"{flops_m:.2f} M ({flops_g:.6f} G)"
    else:
        flops_str = "N/A (thop library not available)"
    latency_str = f"{avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms/sample"
    
    final_text_report = f"""
    ================================================
    FINAL PERFORMANCE EVALUATION
    ================================================
        
    [Model Architecture]
    - Trainable Parameters: {total_params:,}

    [Overall Model Performance (Test Set)]
    - Accuracy: {test_acc:.4f}
    - F1-Score (Macro): {test_f1_macro:.4f}
    - F1-Score (Weighted): {test_f1_weighted:.4f}
    - Matthews Correlation Coefficient (MCC): {test_mcc:.4f}
    - False Negative Rate (FNR): {test_fnr:.4f}
    - False Positive Rate (FPR): {test_fpr:.4f}

    [Computational Efficiency]
    - FLOPs: {flops_str} (single forward pass, batch_size=1)
    - Inference Latency: {latency_str} (batch_size=1)

    [Full Classification Report (Test Set)]
    {report_str}

    ================================================
    GRAPH ANALYSIS (from Best Model)
    ================================================

    [Top 10 Nodes by PageRank Centrality]
    {top_10_pagerank_str}

    ================================================
    FEATURE IMPORTANCE (Permutation Method)
    ================================================
    (Based on Macro F1-Score Drop, {cfg.get('PERMUTATION_REPEATS', 10)} repeats)

    {importance_df.to_string()}
    """
    # 将报告写入文件
    report_path = os.path.join(output_dir, 'final_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_text_report)

    logging.info(f"\n--- Consolidated analysis report saved to '{report_path}' ---")

    # 计算总运行时间
    total_runtime = time.time() - program_start_time
    logging.info(f"\n\n[Total Program Runtime]\n- {total_runtime:.2f} seconds ({total_runtime / 60:.2f} minutes)\n")

    # 将总时间追加到报告末尾
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n[Total Program Runtime]\n- {total_runtime:.2f} seconds ({total_runtime / 60:.2f} minutes)\n")
        f.write(f"\n[Training + Validation Runtime]\n- {training_and_val_runtime:.2f} seconds\n")
        f.write(memory_report)

    # 如果是被 run_experiments.py 调用，就返回结果
    if run_id is not None:
        return test_f1_macro, test_acc  # 返回你关心的指标

    # 仅在独立运行时执行鲁棒性分析，避免批量实验时额外开销
    robustness_evaluation(model, test_loader, DEVICE, cfg, output_dir)

if __name__ == '__main__':
    import os
    # 设置环境变量以减少 thop 库使用多进程时的清理警告
    # 注意：这可能会在程序退出时产生 "Directory not empty" 警告，但不影响功能
    # 这是 thop 库内部使用 multiprocessing 时的已知问题
    os.environ.setdefault('MP_METHOD', 'spawn')
    
    # 因为 main 函数的 seed 和 run_id 参数是可选的,故不需要改变
    main(config)
