import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sklearn
from packaging.version import parse as version_parse
import re
from data_analysis import generate_data_quality_report
import matplotlib.pyplot as plt
import random
from config_loader import config
from vmdpy import VMD
import os
from scipy.stats import trim_mean, linregress
import copy
# 用于局部冻结变量
RTOL = 1e-6
ATOL = 1e-6

def align_and_report(df, feature_cols, class_name):
    """基于dropna对齐数据并打印详细报告"""
    print(f"\n--- [Data Alignment for {class_name} Class] ---")
    initial_count = len(df)
    if initial_count == 0:
        print(" - No data to align.")
        return pd.DataFrame(columns=df.columns)
    # 创建一个新的、干净的数据框（aligned_df），其中只包含在所有“关键特征列、feature_cols”上都没有缺失值的行。
    # 在数据预处理这一步，例如对于文件夹2，因为缺失得太多，所以数据都被删除掉了，故没有WELL-2-1这个id
    # 处理缺失值的函数：dropna
    # 先处理了冻结值，然后调用了该函数来删除了缺失值
    aligned_df = df.dropna(subset=feature_cols).copy()
    final_count = len(aligned_df)

    print(f" - Initial samples: {initial_count}")
    print(f" - Samples after dropping rows with any NaN in selected features: {final_count}")
    print(f" - Removed {initial_count - final_count} incomplete samples to ensure data integrity.")
    return aligned_df

def _is_constant(arr: np.ndarray) -> bool:
    """忽略 NaN 的常数判定。"""
    if arr.size == 0:
        return True
    mask = ~np.isnan(arr)
    if not np.any(mask):
        return True  # 全 NaN 也当作常数列用于冻结判定
    arrv = arr[mask]
    if arrv.size <= 1:
        return True
    return np.allclose(arrv, arrv[0], rtol=RTOL, atol=ATOL)

def _stable_variance(arr: np.ndarray, ddof=0) -> float:
    """常数/全NaN -> 0，否则用 nanvar。"""
    if _is_constant(arr):
        return 0.0
    return float(np.nanvar(arr, ddof=ddof))

def analyze_frozen_by_well(
    full_df: pd.DataFrame,
    feature_cols: list,
    variance_threshold: float,
    well_id_col: str = 'well_id'
):
    """
    仅做分析与决策，不做任何改动。
    规则：
      1) 变量在所有井都冻结 且 冻结值一致 -> 列入 cols_to_drop
      2) 部分井冻结：比较“冻结行数”与“正常行数”
         - 冻结行数 > 正常行数 -> 列入 cols_to_drop
         - 否则 -> 列入 fill_plan（仅记录需要填充的井，暂不计算/执行填充）
    返回:
      decisions = {
        "cols_to_drop": [col, ...],
        "fill_plan": [{"col": col, "frozen_wells": [...]} , ...],
        "frozen_map": {col: {well_id: True/False}},
        "well_sizes": {well_id: row_count},
      }
    """
    well_sizes = full_df.groupby(well_id_col).size().to_dict()

    # 记录每变量在各井是否冻结
    frozen_map = {col: {} for col in feature_cols}
    # 冻结值（用于“所有井都冻结且一致”的判断）
    const_value_map = {col: {} for col in feature_cols}

    # —— 逐井计算冻结状态 —— #
    for well_id, df_well in full_df.groupby(well_id_col):
        for col in feature_cols:
            vals = df_well[col].to_numpy()
            # 你的要求：先判断常数/全NaN -> var=0，否则再 nanvar
            if _is_constant(vals):
                var = 0.0
            else:
                var = np.nanvar(vals, ddof=0)
            is_frozen = (var < variance_threshold)
            frozen_map[col][well_id] = is_frozen
            # 记录冻结常数（用于全井一致性判定）
            if is_frozen:
                # 提取“常数值”：非 NaN 的第一个即可
                mask = ~np.isnan(vals)
                const_val = float(vals[mask][0]) if np.any(mask) else np.nan
                const_value_map[col][well_id] = const_val
            else:
                const_value_map[col][well_id] = None

    # —— 每变量的全局决策 —— #
    cols_to_drop = []
    fill_plan = []  # 只记录需要填充的列和井；不执行填充、也不计算填充值

    for col in feature_cols:
        flags = frozen_map[col]       # {well_id: bool}
        consts = const_value_map[col] # {well_id: const or None}
        wells = list(flags.keys())
        frozen_wells = [w for w in wells if flags[w]]
        normal_wells = [w for w in wells if not flags[w]]

        # 情况 A：所有井都冻结
        if len(normal_wells) == 0:
            cols_to_drop.append(col)
            continue
            # 若都冻结但常数不一致，视为“部分冻结”继续走下面逻辑

        # 情况 B：部分井冻结、部分井正常
        if frozen_wells and normal_wells:
            frozen_rows = int(sum(well_sizes[w] for w in frozen_wells))
            normal_rows = int(sum(well_sizes[w] for w in normal_wells))
            if frozen_rows > normal_rows:
                cols_to_drop.append(col)
            else:
                fill_plan.append({"col": col, "frozen_wells": frozen_wells, "normal_wells": normal_wells})

        # 情况 C：没有冻结井 -> 不处理

    decisions = {
        "cols_to_drop": cols_to_drop,
        "fill_plan": fill_plan,
        "frozen_map": frozen_map,
        "well_sizes": well_sizes,
    }
    return decisions

def apply_drop(full_df: pd.DataFrame, decisions: dict) -> pd.DataFrame:
    """执行列删除；不做填充。"""
    cols_to_drop = decisions.get("cols_to_drop", [])
    if not cols_to_drop:
        return full_df.copy()
    return full_df.drop(columns=cols_to_drop, errors='ignore')

def apply_fill_plan(
    df: pd.DataFrame,
    decisions: dict,
    well_id_col: str = 'well_id'
) -> pd.DataFrame:
    """
    按 analyze_frozen_by_well 产出的 fill_plan 执行填充。
    填充值 = 所有“正常井”该列的整体均值（忽略 NaN）。
    """
    out = df.copy()
    for item in decisions.get("fill_plan", []):
        col = item["col"]
        frozen_wells = item["frozen_wells"]
        normal_wells = item["normal_wells"]

        normal_mask = out[well_id_col].isin(normal_wells)
        fill_value = int(out.loc[normal_mask, col].mean(skipna=True))
        # 只填充“冻结井”的整列（按你的规则，不做逐行对齐）
        frozen_mask = out[well_id_col].isin(frozen_wells)
        out.loc[frozen_mask, col] = fill_value
    return out

def load_and_preprocess_data(cfg, output_dir):
    print(f"--- [Preprocessing Step 1: Data Loading from Target Folder] ---")
    target_folder_path = Path(cfg['ROOT_DATA_PATH']) / str(cfg['TARGET_FAULT_CLASS'])
    if not target_folder_path.is_dir():
        raise FileNotFoundError(f"Target fault folder not found: {target_folder_path}")
    # 大的列表，包含了时间戳和井号，因为一个井也有很多独立的文件
    df_list_with_id = []
    file_list = sorted(target_folder_path.glob('*.parquet'))
    # 用于统计每个文件的持续时间（仅当TARGET_FAULT_CLASS为8时）
    file_durations = []
    
    for file_index, file_path in enumerate(file_list):
        try:
            df_single = pd.read_parquet(file_path)
            well_match = re.search(r'WELL-(\d+)', file_path.stem)
            base_well_id = f"WELL-{int(well_match.group(1))}" if well_match else f"UNKNOWN_{file_path.stem}"
            # 【核心修改】创建唯一的、人类可读的事件ID
            # 格式为: "WELL-X-Y"，其中Y是文件的序号,这里的file_index是整个文件夹下的序号，并非是单独井类别的序号，目的仅做区别
            event_id_str = f"{base_well_id}-{file_index}"
            df_single['well_id'] = event_id_str
            if 'timestamp' not in df_single.columns:
                df_single.reset_index(inplace=True)
                df_single.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # 计算当前文件中故障标签的持续时间（仅当TARGET_FAULT_CLASS为2或8时）
            if cfg.get('TARGET_FAULT_CLASS') == 8 or cfg.get('TARGET_FAULT_CLASS') == 2:
                # 确定目标故障标签：故障2对应102，故障8对应108
                target_fault_label = 100 + cfg.get('TARGET_FAULT_CLASS')
                
                # 检查文件中是否存在该故障标签
                if 'class' in df_single.columns:
                    fault_mask = df_single['class'] == target_fault_label
                    fault_rows = df_single[fault_mask]
                    
                    if len(fault_rows) > 0:
                        # 获取故障标签对应的时间戳
                        fault_timestamps = fault_rows['timestamp']
                        
                        # 确保timestamp是数值类型或datetime类型
                        if pd.api.types.is_datetime64_any_dtype(fault_timestamps):
                            # 如果是datetime类型，计算第一个到最后一个的时间差
                            first_occurrence = fault_timestamps.min()
                            last_occurrence = fault_timestamps.max()
                            duration = (last_occurrence - first_occurrence).total_seconds()
                        else:
                            # 如果是数值类型，直接相减
                            first_occurrence = fault_timestamps.min()
                            last_occurrence = fault_timestamps.max()
                            duration = float(last_occurrence - first_occurrence)
                        
                        file_durations.append(duration)
                    # 如果文件中没有该故障标签，跳过该文件（不添加到file_durations）
            
            # 将每个单独的井合并在一起
            df_list_with_id.append(df_single)
        except Exception as e:
            print(f" [Warning] Could not read file {file_path}: {e}")

    if not df_list_with_id: raise ValueError(f"No valid .parquet files found in {target_folder_path}")
    
    # 当TARGET_FAULT_CLASS为2或8时，计算并打印故障标签的平均持续时间
    target_class = cfg.get('TARGET_FAULT_CLASS')
    if (target_class == 8 or target_class == 2) and file_durations:
        target_fault_label = 100 + target_class
        avg_duration = sum(file_durations) / len(file_durations)
        print(f"\n--- [Class {target_class} 故障标签{target_fault_label}持续时间统计] ---")
        print(f"  总文件数（包含故障标签{target_fault_label}）: {len(file_durations)}")
        print(f"  平均持续时间: {avg_duration:.2f} 秒 ({avg_duration/60:.2f} 分钟, {avg_duration/3600:.2f} 小时)")
        print(f"  最短持续时间: {min(file_durations):.2f} 秒 ({min(file_durations)/60:.2f} 分钟, {min(file_durations)/3600:.2f} 小时)")
        print(f"  最长持续时间: {max(file_durations):.2f} 秒 ({max(file_durations)/60:.2f} 分钟, {max(file_durations)/3600:.2f} 小时)")
        print("=" * 50)

    full_df = pd.concat(df_list_with_id, ignore_index=True)
    # 在Pandas的DataFrame对象中创建一个名为'source_folder'的新列，多分类时候有用；目前单分类、原文件都是一样的、故无用
    # 这里保留是为了多文件夹下分门别类统计、汇报数据
    full_df['source_folder'] = str(cfg['TARGET_FAULT_CLASS'])

    print(" - Data loaded with native precision.")
    # source_folder的想法，后期数据来源多了，可以知道从哪个文件夹中来的，如果是单分类，都是一个文件夹中的话，删除。
    # 单独将传感器变量提取出来
    all_feature_cols = [col for col in full_df.columns if
                        col not in ['timestamp', 'class', 'well_id', 'state', 'source_folder']]

    if cfg['GENERATE_DATA_REPORT']:
        generate_data_quality_report(full_df.copy(), all_feature_cols, output_dir, cfg)

    # 1、根据样本数量选取保留特征变量；2、冻结变量删除；3、缺失样本删除。原则：优先看样本的数量，先有数据、再看数据是什么情况
    print(f"\n--- [Preprocessing Step 2: Dynamic Feature Selection] ---")
    # 故障的子集fault_subset_df：不为0、且不为空，就是某个具体的故障
    fault_subset_df = full_df[(full_df['class'] != 0) & (full_df['class'].notna())]
    # non_missing_counts=（特征A，数量）（特征B，数量）
    non_missing_counts = fault_subset_df[all_feature_cols].notna().sum()
    # max_count=目前样本数量最大值
    max_count = non_missing_counts.max() if not non_missing_counts.empty else 0
    selection_threshold = max_count * cfg['FEATURE_SELECTION_THRESHOLD']
    # 通过样本数量来筛选特征，即选择保留哪些变量特征
    feature_cols = non_missing_counts[non_missing_counts >= selection_threshold].index.tolist()
    if not feature_cols:
        raise ValueError("No features selected. The dataset may be too sparse or the threshold is too high.")

    dropped_features = [col for col in all_feature_cols if col not in feature_cols]

    print(f" - In fault class (all non-zero), max non-missing count is {max_count}.")
    print(
        f" - Selection threshold set to {selection_threshold:.0f} ( > {cfg['FEATURE_SELECTION_THRESHOLD'] * 100}% of max).")
    print(f" - Selected {len(feature_cols)} features: {feature_cols}")
    print(f" - Dropped {len(dropped_features)} features: {dropped_features}")

    print("\n--- [Preprocessing Step 2.1: Globally Removing Frozen Variables] ---")
    # 之前的逻辑是：找出在所有正常事件中都“偷懒”的特征，将这两份名单合并，一次性“开除”所有上榜的特征。
    # 更改为：在当前文件夹下，所有的文件中，某个特征变量的值变动小于方差1e-6，不需要分别判断正负类了。原因：深度学习使用正和负类的数据是一起训练的，变量需要全局一致。
    # 后期发现，有这么几种情况的冻结表现：A、对于故障1：ESTADO-DHSV：0/1；B、P-JUS-CKGL：well-1冻结，其他文件有变动的值；C、P-PDG：各个well冻结，但是值不一样
    # 对A是正常，但是值不变，也没意义；B也不能删除，为了样本，但是得考虑标准化的事情；C这种实际可以删除，但是需要复杂判断。当时的想法是为了尽可能多保留一些节点

    # 冻结策略：
    # 若某变量在所有井都“冻结”，且冻结值一致 → 直接删除该变量；
    # 若某变量在部分井冻结、部分井正常 → 比较样本行数：
    # 冻结样本行数 > 正常样本行数 → 删除该变量；
    # 否则 → 保留变量，并把“冻结井”的该变量用所有正常井的均值进行填充。
    decisions = analyze_frozen_by_well(
        full_df=full_df,
        feature_cols=feature_cols,
        variance_threshold=config['VARIANCE_THRESHOLD'],
        well_id_col='well_id'
    )
    print("     建议删除的变量：", decisions["cols_to_drop"])
    print("     建议填充计划（示例前2项）：", decisions["fill_plan"][:2])

    # 删除所有共同的冻结变量；在清洗完数据之后，再执行填充，否则会把完全的一张Nan表错误填充了
    df_after_drop = apply_drop(full_df, decisions)
    if decisions["cols_to_drop"]:
        original_feature_count = len(feature_cols)
        # This is the "removal" step: filter the main feature_cols list.
        feature_cols = [col for col in feature_cols if col not in decisions["cols_to_drop"]]
        print(f" - Found {len(decisions['cols_to_drop'])} globally frozen variables (variance < {cfg['VARIANCE_THRESHOLD']}).")
        print(f" - Number of features for modeling reduced from {original_feature_count} to {len(feature_cols)}.")
        print(f" - Removed variables: {sorted(decisions['cols_to_drop'])}")
        print(f" - 最终被送入训练的特征变量数量为: {len(feature_cols)}，分别为 {sorted(feature_cols)}")
    else:
        print(" - No globally frozen variables found to remove.")
        print(f" - 最终被送入训练的特征变量数量为: {len(feature_cols)}，分别为 {sorted(feature_cols)}")

    # 数据报告+处理冻结变量之后，使用列名称的方式组合df
    # 这里没有包含source_folder这个类别，source_folder的作用仅仅是为了报表、分门别类
    base_cols = ['timestamp', 'well_id', 'class', 'state']
    # retained_cols=4个头标志+各种特征，well_id，通过时间区分了相同井下的事件
    retained_cols = base_cols + feature_cols
    df_after_drop = df_after_drop[[col for col in retained_cols if col in df_after_drop.columns]].copy()

    print("\n--- [Preprocessing Step 3: Data Class Assignment] ---")
    df_after_drop['class'] = pd.to_numeric(df_after_drop['class'], errors='coerce')
    # 正类 101 1
    positive_class_df = df_after_drop[df_after_drop['class'].notna() & (df_after_drop['class'] != 0)].copy()
    # 负类 0
    negative_class_df = df_after_drop[df_after_drop['class'] == 0].copy()
    # 这里的class瞬态还没有被合并，依然保持着101，class的名字也没有改变，最初的代表0 101 1的标签
    # align_and_report函数，里面删除了缺失值
    aligned_pos_df = align_and_report(positive_class_df, feature_cols, "Positive")
    aligned_neg_df = align_and_report(negative_class_df, feature_cols, "Negative")

    # 创建一个名为 'order_class' 的新列，通过已经分好的正类、负类，统一设置0和1；目的是：为了整合101变为1，将瞬态也作为故障的一种
    # order_class：A、最终提供num_classes返回值，给到model参数，说明是几分类（目前是2分类）
    # 这里将其他类别的故障形式也表示为了1，为的是后续处理方便
    aligned_pos_df.loc[:, 'order_class'] = 1
    # 目前数据集3和4只有正类class=3/4，没有class=0的文件夹，故直接创建并赋值会出错，判断校正、不创建
    if not aligned_neg_df.empty:
        aligned_neg_df.loc[:, 'order_class'] = 0
    else:
        # DataFrame是空的
        print(f" - 警告：该数据集没有负类 class = {cfg['TARGET_FAULT_CLASS']} 文件夹，没有数据可以处理!")
    # 这里将正负类的数据连接到了一起 working_df，而且已经处理了冻结值、缺失值
    working_df = pd.concat([aligned_pos_df, aligned_neg_df], ignore_index=True).copy()
    if working_df.empty:
        raise ValueError("The dataset is empty after alignment. Please check data quality or preprocessing parameters.")

    # 在清洗完数据之后，再执行填充，否则会把完全的一张Nan表错误填充了
    # working_df是完全清洗好的数据
    working_df = apply_fill_plan(working_df, decisions, well_id_col='well_id')
    print("\n--- 部分冻结的变量数据填充完毕 ---")

    # 为了后续的创建图所用真实的物理节点
    # 注意：physical_node_names 始终使用“物理变量名”，与是否启用 PAA 无关
    # - 启用 PAA 时：physical_node_names 是原始变量名，feature_cols 变为 xxx_mean/xxx_slope
    # - 关闭 PAA 时：physical_node_names 与 feature_cols 一致（每个节点 1 维特征）
    physical_node_names = feature_cols.copy()

    # === 是否启用 ASR+PAA 的开关 ===
    asr_cfg = cfg.get('asr', {})
    use_paa = asr_cfg.get('enable_paa', True)

    if use_paa:
        print("\n--- [Preprocessing Step 3.1: Adaptive Signal Refinement & PAA] ---")
        visualization_path_asr = os.path.join(output_dir, 'ASR_Visualization')
        # 直接使用还未标准化的数据去做，里面含有well_id，特征等处理好的数据
        # 返回来的feature_cols是二维的向量，_mean、_slope
        working_df, feature_cols = adaptive_signal_refinement(working_df, feature_cols, cfg, visualization_path_asr)
        if working_df is None:
            print("ASR/PAA module did not produce any features. Exiting.")
            exit()
        print("\n### Module 1 (ASR+PAA) Verification Step Finished. ###")
    else:
        # 完全跳过 ASR/PAA，保留对齐+缺失值处理后的原始传感器序列
        # 不做 10 倍降维，也不生成 mean/slope，后续时序窗口在 create_sequences 中处理
        print("\n--- [Preprocessing Step 3.1: Skip ASR/PAA] ---")
        print(" - Using raw (cleaned & aligned) sensor features without PAA (no downsampling / mean / slope).")

    print(f"\n--- [Preprocessing Step 4:  Feature Encoding] ---")
    # 注意：标准化已移至数据集划分之后，防止数据泄露
    # 标准化将在 get_dataloaders 函数中，在数据集划分之后进行
    num_state_features = 0
    if 'state' in working_df.columns:
        # 预处理，填充Unknown新类型、字符串
        working_df.loc[:, 'state'] = working_df['state'].astype('object').fillna('Unknown').astype(str)
        unique_states = working_df['state'].unique()
        # 如果只有一个状态，那无需独热编码
        if len(unique_states) >= 2:
            print(f" - Found {len(unique_states)} unique states. Performing one-hot encoding...")

            # 如果为两个状态，则删除A，B=1-A，若为1个那创建1个，3个则为3个
            drop_strategy = 'first' if len(unique_states) == 2 else None
            use_sparse_output = version_parse(sklearn.__version__) >= version_parse("1.2")
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=use_sparse_output,
                                    drop=drop_strategy) if use_sparse_output else OneHotEncoder(handle_unknown='ignore',
                                                                                                sparse=False,
                                                                                                drop=drop_strategy)
            # 执行编码并生成新特征，fit_transform转换
            state_features_encoded = encoder.fit_transform(working_df[['state']])
            categories = encoder.categories_[0]
            state_feature_names = [f"state_{cat}" for cat in (categories[1:] if drop_strategy == 'first' else categories)]
            num_state_features = state_features_encoded.shape[1]
            state_features_df = pd.DataFrame(
                state_features_encoded.toarray() if use_sparse_output and hasattr(state_features_encoded,
                                                                                  'toarray') else state_features_encoded,
                index=working_df.index, columns=state_feature_names)
            # 将state转换为数字，放到了列的最后边，合并一个大的df
            working_df = pd.concat([working_df.drop(columns=['state']), state_features_df], axis=1)
            print(f" - 'state' column one-hot encoded.")
        else:
            # 如果唯一状态数少于2，则该列为常数，直接丢弃
            print(
                f" - Only found {len(unique_states)} unique state(s). The 'state' column is constant and will be dropped.")
            working_df = working_df.drop(columns=['state'])
            num_state_features = 0  # 确保该值为0
    print("\nPreprocessing complete.")
    #return working_df, unscaled_working_df, feature_cols, len(working_df['order_class'].unique()), num_state_features

    return working_df, feature_cols, physical_node_names, len(working_df['order_class'].unique()), num_state_features

def create_sequences(df, feature_cols, cfg):
    """v4.2: 现在额外返回每个序列的时间戳和 well_id"""
    print(f"\n--- [Preprocessing Step 5: Sequence Generation] ---")
    # 转化时间格式，为'timestamp'的列，从其原始格式（通常是文本字符串或数字）转换为Pandas专用的、功能强大的“日期时间对象”（datetime object）。
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # 先分组、再排序。对整个DataFrame的行进行重新排序，并返回一个排序后的新DataFrame。
    # 因为有的井的时间相同，分属不同的井，有的相同的井的不同文件的时间也有跨度，故实际以单独的文件作为划分，这个表示就是well-1-0
    df = df.sort_values(by=['well_id', 'timestamp'])
    state_feature_cols = [col for col in df.columns if 'state_' in col]
    num_state_features = len(state_feature_cols)

    all_sequences = []
    all_timestamps = []
    all_well_ids = []
    # 按照井号区分（相当于单独的excel文件了）后，well_group是一个小的df
    for well_id, well_group in df.groupby('well_id'):
        # 判断这个差值是否大于1秒。如果大于1秒，说明数据在这里中断了，我们得到了一个True；否则是False。
        # 判断会话是否中断、断层，存储的是ture和false
        # 目前因为PAA的原哥，这里的插值变为了10
        time_diff = well_group['timestamp'].diff() > pd.Timedelta(seconds=10)
        # 根据上面找到的“断层”，为每一个连续的数据片段赋予一个唯一的“会话ID”。
        # 这是一个巧妙的技巧。它会把True当作1，False当作0来累加。当数据连续时（全是False），累加和不变；一旦遇到一个True（数据断层），累加和就加1，从而生成了一个新的`session_idsession_id。
        # 这样做是为了确保我们接下来创建的滑动窗口，不会错误地跨越一个巨大的时间鸿沟（比如一个停机维护了2小时的数据断层）
        well_group['session_id'] = time_diff.cumsum()

        for session_id, session_group in well_group.groupby('session_id'):
            # 一个简单的合理性检查。cfg['SEQUENCE_LENGTH']是您设定的窗口大小（例如60个点）。如果一个连续的数据片段连一个完整的窗口都凑不出来，那它就没有处理的价值。
            if len(session_group) < cfg['hyperparameters']['sequence_length']:
                continue
            # 分为四个干净的、准备好被快速切片的NumPy数组：sensor_data（传感器特征），state_data（状态特征），labels（标签），和`timestampstimestamps（时间戳）。
            sensor_data = session_group[feature_cols].values
            state_data = session_group[state_feature_cols].values if num_state_features > 0 else None
            # order_class代表了正常0，瞬态+确定代表了1
            labels = session_group['order_class'].values
            timestamps = session_group['timestamp'].values
            # 计算出一共可以滑动多少次
            for i in range(len(session_group) - cfg['hyperparameters']['sequence_length'] + 1):
                # 这是实际的“切片”动作。它从sensor_data这个NumPy数组中，切出了从起始位置i开始、长度为SEQUENCE_LENGTH的一段数据。
                sequence_sensors = sensor_data[i: i + cfg['hyperparameters']['sequence_length']]
                sequence_states = state_data[i + cfg['hyperparameters']['sequence_length'] - 1] if state_data is not None else np.array([])

                # 作用：为刚刚切出的那个数据窗口，找到它对应的状态和标签。
                # 逻辑：一个序列的标签（它是否是故障）是由这个序列结束时刻的状态决定的。i + cfg['SEQUENCE_LENGTH'] - 1个索引，精确地指向了当前窗口的最后一个数据点。索引都是从0开始，索引=个数-1
                sequence_label = labels[i + cfg['hyperparameters']['sequence_length'] - 1]
                # all_sequences: 存放了所有模型的输入和标签。
                # all_timestamps: 记录了每个样本的结束时间。
                # all_well_ids: 记录了每个样本来自哪一口井。
                all_sequences.append((sequence_sensors, sequence_states, sequence_label))

                all_timestamps.append(timestamps[i + cfg['hyperparameters']['sequence_length'] - 1])
                all_well_ids.append(well_id)

    if not all_sequences:
        raise ValueError("No sequences were created. This might be due to a short dataset or a large SEQUENCE_LENGTH.")

    X_sensors_list, X_states_list, y_list = zip(*all_sequences)

    total_sequences = len(y_list)
    y_np = np.array(y_list)
    # 这里最终的标签实际是1 2 3，并不是一直为1
    positive_sequences = np.sum(y_np == 1)
    negative_sequences = np.sum(y_np == 0)

    print(" - Sequence generation complete.")
    print(f"   - Total sequences created: {total_sequences}")
    print(f"   - Positive (Fault) sequences: {positive_sequences}")
    print(f"   - Negative (Normal) sequences: {negative_sequences}")

    X_sensors_np = np.array(X_sensors_list).astype(np.float32)
    X_states_np = np.array(X_states_list).astype(np.float32) if X_states_list and X_states_list[
        0].size > 0 else np.array([])

    timestamps_np = np.array(all_timestamps)
    well_ids_np = np.array(all_well_ids)

    return X_sensors_np, X_states_np, y_np, timestamps_np, well_ids_np

def seed_worker(worker_id):
    """
    Sets the random seed for each DataLoader worker.
    This is essential for reproducibility when num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(cfg, output_dir):
    # working_df是预处理后的数据（尚未标准化，标准化将在数据集划分之后进行）
    # num_state_features为井的状态，如果是只有1个状态，那相当于没有，置num_state_features=0
    scaled_df, feature_cols, physical_node_names, num_classes, num_state_features = load_and_preprocess_data(cfg, output_dir)
    # X_sensors只包含了输入数据，y是标签，X_states是独热编码、井的状态，只有一种状态的话就是NaN
    X_sensors, X_states, y, timestamps, well_ids = create_sequences(scaled_df, feature_cols, cfg)

    # np.unique(well_ids): 这是NumPy库的一个函数，它的作用是接收一个数组，然后找出其中所有不重复的元素，并将它们排序后返回。
    unique_well_ids_list = np.unique(well_ids)
    # 创建正向映射: 建立一个从整数到类别文本的映射（`int_to_wellid_mapint_to_wellid_map），方便人类阅读结果。
    int_to_wellid_map = {i: well for i, well in enumerate(unique_well_ids_list)}
    # 建立一个从类别文本到整数的映射（`wellid_towellid_to_int_map），方便机器处理数据。
    wellid_to_int_map = {well: i for i, well in int_to_wellid_map.items()}

    if cfg['DEBUG_MODE']:
        num_samples_to_use = min(50000, len(y))
        if num_samples_to_use < len(y):
            debug_indices, _ = train_test_split(np.arange(len(y)), train_size=num_samples_to_use,
                                                random_state=cfg['RANDOM_SEED'], stratify=y)
            X_sensors, y = X_sensors[debug_indices], y[debug_indices]
            X_states = X_states[debug_indices] if X_states.size > 0 else X_states
            timestamps = timestamps[debug_indices]
            well_ids = well_ids[debug_indices]

    print("\n--- [Preprocessing Step 6: Data Split & DataLoader Creation] ---")
    # 目的: 这样做更高效。只对这个轻量级的索引数组进行切分，切分完成后，再用得到的索引去原始数据中提取相应的部分。而不是直接对大的X_sensors做操作
    indices = np.arange(len(y))
    # 将indices分裂为train_val_indices和test_indices两部分
    # test_size=0.2: 明确指定了测试集test_indices应占总数据的20%
    # train_val_indices是训练验证池，这里占据80%
    # 要确保`train_train_val_indices和test_indices这两个集合中，故障样本（标签为1）和正常样本（标签为0）的比例与原始总数据集y中的比例是完全相同的。这可以防止因为随机切分导致某个集合中故障样本过多或过少的情况。
    train_val_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=cfg['RANDOM_SEED'],
                                                       stratify=y)
    y_train_val = y[train_val_indices]
    # 验证集再从train_val_indices是训练验证池中划分20%，即val_indices=80*20=16%，train_indices=64%
    # stratify=y_train_val，现在样本数量变了，比例只能从更新后的取
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=cfg['RANDOM_SEED'],
                                                  stratify=y_train_val)

    y_train_labels, y_val_labels, y_test_labels = y[train_indices], y[val_indices], y[test_indices]

    # === 标准化：在数据集划分之后进行，防止数据泄露 ===
    print("\n--- [Preprocessing Step 6.1: Feature Standardization (After Split)] ---")
    # X_sensors的形状是 [num_sequences, sequence_length, num_features]
    # 需要重塑为2D数组 [num_sequences * sequence_length, num_features] 以进行标准化
    num_sequences, sequence_length, num_features = X_sensors.shape
    X_sensors_2d = X_sensors.reshape(-1, num_features)  # [num_sequences * sequence_length, num_features]
    
    # 只在训练集上计算标准化参数（fit）
    # train_indices是序列索引，需要转换为2D数组的索引
    train_indices_2d = []
    for seq_idx in train_indices:
        start_idx = seq_idx * sequence_length
        end_idx = start_idx + sequence_length
        train_indices_2d.extend(range(start_idx, end_idx))
    train_indices_2d = np.array(train_indices_2d)
    
    scaler = StandardScaler()
    scaler.fit(X_sensors_2d[train_indices_2d])
    print(" - Standardization parameters computed on training set only.")
    
    # 用同一组参数变换所有数据
    X_sensors_2d_scaled = scaler.transform(X_sensors_2d)
    X_sensors = X_sensors_2d_scaled.reshape(num_sequences, sequence_length, num_features)
    print(" - All datasets (train/val/test) standardized using training set parameters.")

    def create_loader(selected_indices, shuffle, cfg):
        X_sens = torch.from_numpy(X_sensors[selected_indices]).float()
        y_labs = torch.from_numpy(y[selected_indices]).long()

        ts = torch.from_numpy(timestamps[selected_indices].astype('int64'))

        w_ids_int = [wellid_to_int_map[wid] for wid in well_ids[selected_indices]]
        w_ids = torch.tensor(w_ids_int, dtype=torch.long)
        # 这里是判断有无井的状态
        # TensorDataset(...)，这是PyTorch提供的一个便捷类，用于创建数据集。它的要求是，所有传入的张量，在第一个维度（也就是样本数量）上的长度必须完全相同。
        # 这里判断了井的状态是否为单一的，要进行独热编码
        # 要注意x和y的位置，不是紧挨在一起的
        if X_states.size > 0:
            X_stat = torch.from_numpy(X_states[selected_indices]).float()
            # 最终数据集中的顺序
            dataset = TensorDataset(X_sens, X_stat, y_labs, ts, w_ids)
        else:
            # 即使没有井的状态，也预留了状态的位置
            dataset = TensorDataset(X_sens, torch.empty(len(X_sens), 0), y_labs, ts, w_ids)

        batch_size = cfg['hyperparameters']['HIGH_PERFORMANCE_BATCH_SIZE'] if cfg['HIGH_PERFORMANCE_MODE'] else cfg['hyperparameters']['BASE_BATCH_SIZE']
        num_workers = 16 if cfg['HIGH_PERFORMANCE_MODE'] else 0
        # 当且仅当需要打乱数据顺序时（shuffle = True），创建一个可控的、可复现的随机数生成器。
        generator = torch.Generator().manual_seed(cfg['RANDOM_SEED']) if shuffle else None

        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          pin_memory=True,
                          num_workers=num_workers,
                          worker_init_fn=seed_worker,
                          generator=generator)
    # 创建分数据集的时候，调用create_loader就判断过井的状态，故已经包含井的状态
    train_loader = create_loader(train_indices, shuffle=True, cfg=cfg)
    val_loader = create_loader(val_indices, shuffle=False, cfg=cfg)
    test_loader = create_loader(test_indices, shuffle=False, cfg=cfg)

    print("\n--- [ Dataset Statistics ] ---")
    print(f" - Total Sequences: {len(y)}")
    print(f" - Training Set:   {len(train_indices)} sequences ({len(train_indices) / len(y):.0%})")
    print(
        f"   - Positive: {np.sum(y_train_labels == 1)}, Negative: {np.sum(y_train_labels == 0)}, Batches: {len(train_loader)}")
    print(f" - Validation Set: {len(val_indices)} sequences ({len(val_indices) / len(y):.0%})")
    print(
        f"   - Positive: {np.sum(y_val_labels == 1)}, Negative: {np.sum(y_val_labels == 0)}, Batches: {len(val_loader)}")
    print(f" - Test Set:       {len(test_indices)} sequences ({len(test_indices) / len(y):.0%})")
    print(
        f"   - Positive: {np.sum(y_test_labels == 1)}, Negative: {np.sum(y_test_labels == 0)}, Batches: {len(test_loader)}")
    print(f"------------------------------------")

    return train_loader, val_loader, test_loader, feature_cols, physical_node_names, num_classes, num_state_features, int_to_wellid_map
    #return train_loader, val_loader, test_loader, unscaled_df, feature_cols, num_classes, num_state_features, int_to_wellid_map

class SharedModalAttention(nn.Module):
    """
    一个简单的共享MLP，用于计算每个模态的注意力权重。
    """
    def __init__(self, num_imfs, hidden_dim):
        super(SharedModalAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(num_imfs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_imfs),
            nn.Softmax(dim=-1)
        )
        # num_imfs是局部变量，不能直接在forward中使用，得保存在self中作为类属性，其他地方使用self.xxx方式调用
        self.num_imfs = num_imfs

    def forward(self, imfs_tensor):
        # imfs_tensor shape: [Num_Features, K, Seq_Len]
        # 我们基于每个模态在时间上的平均绝对值来计算其重要性
        # 实际的计算式，将每个模态中的值，取绝对值求平均；然后将这几个值去训练，值自己可以对比大小
        modal_importance = torch.mean(torch.abs(imfs_tensor), dim=2)  # Shape: [Num_Features, K]
        # 针对几个模态，计算出来几个权重值
        weights = self.attention_net(modal_importance)  # Shape: [Num_Features, K]

        # 将权重广播回整个时间序列，即与权重相乘
        # weights.view(...): [Num_Features, K, 1]
        # imfs_tensor: [Num_Features, K, Seq_Len]
        weighted_imfs = imfs_tensor * weights.view(-1, self.num_imfs, 1)

        # 加权求和重构信号
        reconstructed_signal = torch.sum(weighted_imfs, dim=1)  # Shape: [Num_Features, Seq_Len]
        return reconstructed_signal, weights

def add_state_background(ax, class_labels, colors):
    for i in range(len(class_labels) - 1):
        state = class_labels.iloc[i]
        color = 'white'
        if state == 0:
            color = colors['normal']
        elif state > 100:
            color = colors['transient']
        elif state > 0:
            color = colors['fault']
        ax.axvspan(class_labels.index[i], class_labels.index[i + 1], facecolor=color, alpha=0.3)

# df_input是前面原始的数据集合，所有的标签都有
def adaptive_signal_refinement(df_input, feature_cols, config, visualization_path):
    # 缓存/离线文件开关：仅在启用 VMD 时才允许使用“预计算加载”模式
    asr_config = config.get('asr', {})
    enable_vmd = asr_config.get('enable_vmd', True)
    # 使用加载或者离线模式，都是在基于VMD使能的情况下
    use_precomputed = config.get('USE_PRECOMPUTED_ASR', False) and enable_vmd

    target_fault_class = config.get('TARGET_FAULT_CLASS', 'unknown')
    base_dir = Path("ASR_dataset")
    precomputed_dir = base_dir / str(target_fault_class)  # 结果: ASR_dataset/8

    if use_precomputed:
        print("\n--- [ASR 模块 - 目前使用从本地加载ASR结果模式] ---")

        if not precomputed_dir.exists():
            raise FileNotFoundError(f"预计算目录 '{precomputed_dir}' 不存在。请先以计算模式运行。")

        # 1. 读取所有保存的 parquet 文件
        feature_files = sorted(precomputed_dir.glob("*_features.parquet"))
        if not feature_files:
            raise FileNotFoundError(f"在 '{precomputed_dir}' 中未找到任何 .parquet 特征文件。")

        print(f"发现 {len(feature_files)} 个预计算的井数据文件。正在加载...")

        # 合并所有 DataFrame
        df_list = [pd.read_parquet(f) for f in feature_files]
        df_paa_features = pd.concat(df_list, ignore_index=True)

        # 2. 读取并打印所有日志文件
        log_files = sorted(precomputed_dir.glob("*_log.txt"))
        print("\n--- [加载的日志] ---")
        for log_file in log_files:
            well_id_from_fname = log_file.name.replace('_log.txt', '')
            print(f"\n===== 井 {well_id_from_fname} 的处理日志 =====")
            with open(log_file, 'r', encoding='utf-8') as f:
                print(f.read())
            print("==========================================\n")
    else:  # 计算模式
        print("\n--- [ASR 模块 - 目前使用计算模式] ---")
        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path)

        asr_config = config['asr']
        # 当 VMD 关闭时，不再落盘生成/加载离线 ASR 文件，直接在线计算并继续后续流程
        persist_asr_cache = bool(asr_config.get('enable_vmd', True))
        state_colors = asr_config['state_colors']
        alpha, tau, DC, init, tol = float(asr_config['vmd_alpha']), float(asr_config['vmd_tau']), bool(
            asr_config['vmd_DC']), int(asr_config['vmd_init']), float(asr_config['vmd_tol'])
        paa_window_size = int(asr_config['paa_window_size'])
        vmd_K_by_type = asr_config['vmd_K_by_type']
        node_attributes = config.get('graph_definitions', {}).get('node_attributes', {})

        # 打印结果，获取well_id列的唯一值
        print("调试作用，well_id列的唯一值：")
        for i, well_id in enumerate(df_input['well_id'].unique(), 1):
            print(f"{i}. {well_id}")
        if persist_asr_cache:
            # 在离线加载模式下才会使用到的数据分批次筛选
            # 手动筛选，防止复制粘贴出错
            df_input = df_input[df_input['well_id'].isin(config.get('WELL_SUBSET_LIST', []))]

        # 经过测试，设置为任务量的2倍，占用总体CPU的50%比较合理
        task_jobs = len(df_input['well_id'].unique()) * 2

        # --- 初始化共享注意力网络 (对所有well共享) ---
        # 我们需要找到最大的K值来初始化网络
        max_k = max(v for v in vmd_K_by_type.values() if isinstance(v, int) and v > 0)

        # df_well属于该 well_id 的子 DataFrame，即按照well_id分类后的子集合，创建的新集合
        # df_input是前面原始的数据集合，所有的标签都有
        # [(well_id, df_well), ...]
        grouped = list(df_input.groupby('well_id'))
        # 使用enumerate(grouped)遍历分组数据，同时获取索引idx和油井数据(well_id, df_well)
        # 将每个油井处理所需的所有参数打包成元组
        tasks = [
            (
                idx,
                well_id,
                df_well,
                feature_cols,
                asr_config,
                state_colors,
                alpha, tau, DC, init, tol,
                paa_window_size,
                vmd_K_by_type,
                node_attributes,
                max_k,
                visualization_path,
                precomputed_dir,
                persist_asr_cache
            )
            for idx, (well_id, df_well) in enumerate(grouped)
        ]
        # 原理：通过减少预先加载到内存中的任务数量，来平滑内存峰值，防止因任务瞬间集中提交而导致的内存崩溃。
        # 内存保护，对于任务8，最大是5；如果再遇到大的数据量，需要动态调整。
        results = Parallel(task_jobs)(
        #results = Parallel(task_jobs, pre_dispatch = asr_config['pre_dispatch'])(
            # 将任务元组解包为函数的位置参数
            delayed(_process_single_well_for_asr)(*task)
            for task in tasks
        )
        # results 是一个 list: [(idx, well_id, all_paa_window_data), ...]
        # 为了保持和原来相同的顺序（按 groupby 后的顺序），按照每个元素的第0个位置（索引）进行排序
        results_sorted = sorted(results, key=lambda x: x[0])
        successful_results = [res for res in results_sorted if res[2] is not None]
        if not successful_results:
            raise RuntimeError("ASR 计算对所有井都失败了，请检查日志。")

        # 构建 df_paa_features：
        # - 若 persist_asr_cache=True：子进程返回 parquet 路径，主进程从文件加载
        # - 若 persist_asr_cache=False：子进程直接返回 DataFrame，主进程直接拼接
        if persist_asr_cache:
            df_list = [pd.read_parquet(res[2]) for res in successful_results]
        else:
            df_list = [res[2] for res in successful_results]
        df_paa_features = pd.concat(df_list, ignore_index=True)

        # 打印日志
        print("\n--- [计算过程中的日志] ---")
        for idx, well_id, file_path, log_text in results_sorted:
            print(f"\n===== 井 {well_id} (idx={idx}) 的处理日志 =====")
            print(log_text)
            print("==========================================\n")

        # 注意：旧逻辑在此处强制退出，用于“只生成离线文件”。
        # 现在默认继续执行后续训练流程；若仍需要“只生成缓存然后退出”，建议单独加一个专用开关控制。
    # --- 后续逻辑保持不变 (无论加载还是计算，都需要执行) ---
    # 使用pd.concat()将多个DataFrame垂直堆叠合并，ignore_index=True：重置索引，创建连续的整数索引
    # 最终生成包含所有油井特征的统一DataFrame
    # df_paa_features = pd.concat(df_list, ignore_index=True)
    # 要排除的列集合
    exclude_all_cols = {'well_id', 'timestamp', 'class', 'state', 'order_class'}
    # sensor_features是二维特征：均值+斜率
    sensor_features = [col for col in df_paa_features.columns if col not in exclude_all_cols]

    return df_paa_features, sensor_features

def _process_single_well_for_asr(
            idx,
            well_id,
            df_well,
            feature_cols,
            asr_config,
            state_colors,
            alpha, tau, DC, init, tol,
            paa_window_size,
            vmd_K_by_type,
            node_attributes,
            max_k,
            visualization_path,
            precomputed_dir,
            persist_asr_cache: bool):
    logs = []
    logs.append(f"并行处理井工作：任务[{idx}] 开始处理井 {well_id}")
    # 定义保存路径（可选）
    df_save_path = None
    log_save_path = None
    if persist_asr_cache:
        save_dir = precomputed_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        df_save_path = save_dir / f"{well_id}_features.parquet"
        log_save_path = save_dir / f"{well_id}_log.txt"

    # 用于存储每个窗口聚合后的信息
    all_paa_window_data = []
    # 在子进程里构建 attention_model（如果你还保留“伪训练”）
    attention_model = SharedModalAttention(max_k, asr_config['attention_hidden_dim'])
    optimizer = torch.optim.Adam(attention_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 之前的循环是 for well_id, df_well in df_input.groupby('well_id'):
    logs.append(f"\n[ASR] Processing data for well_id: '{well_id}'")
    well_viz_path = os.path.join(visualization_path, f'well_{well_id}')
    if not os.path.exists(well_viz_path):
        os.makedirs(well_viz_path)

    df_features = df_well[feature_cols]
    class_labels = df_well['class']

    # 前期经过数据清洗，实际已经没有冻结值了；但是存在人为填充的恒定值，这个不需要分解k模态
    variances = df_well[feature_cols].var(ddof=0)
    # 处理策略：vmd设置：k=0；PAA：均值取开头的，斜率=0
    frozen_vars = variances[variances < config['VARIANCE_THRESHOLD']].index.tolist()

    enable_vmd = asr_config.get('enable_vmd', True)
    # 创建重构后的信号，以特征的行数为索引，后续两个df的操作都是基于此索引，不会乱套
    df_reconstructed_well = pd.DataFrame(index=df_features.index)

    for col in feature_cols:
        # signal：对于每个传感器特征的实际值
        signal = df_features[col].values

        # 如果启用VMD，执行复杂的分解和重构
        if enable_vmd:
            original_len = len(signal)
            sensor_type = node_attributes.get(col, {}).get('type', 'default')
            if col in frozen_vars:
                K = 0
            else:
                K = int(vmd_K_by_type.get(sensor_type, 0))

            # 提前给重建信号赋值之前的原本的信号，否则后面会出现VMD不分解的信号赋值为NAN、计算错误
            reconstructed_signal = signal
            # 奈奎斯特采样定理的启发 (Inspired by Nyquist Theorem)：虽然不完全等同，但这个思想类似。为了描述一个完整的振荡（比如一个正弦波），你至少需要两个采样点（一个波峰，一个波谷）。因此，为了在数据中可靠地识别出K个独立的振荡模式，一个非常保守的、经验性的下限就是需要 2 * K 个数据点。
            if K > 0 and original_len > 2 * K:
                try:
                    # 扩展的长度为原长度的10%
                    extension_len = int(original_len * 0.1)
                    if extension_len > 1:
                        # np.flip是翻转，为了拼接处更加平滑
                        left_ext = np.flip(signal[1:extension_len + 1])
                        right_ext = np.flip(signal[-extension_len - 1:-1])
                        # 取10%然后左右拼接
                        extended_signal = np.concatenate([left_ext, signal, right_ext])
                    else:
                        extended_signal = signal

                    u, _, _ = VMD(extended_signal, alpha, tau, K, DC, init, tol)
                    if extension_len > 1:
                        # 裁剪出来原始长度，将之前填充的边界去掉
                        # u的结果，k是5的话，会生成5行数值，代表了5个模态
                        u = u[:, extension_len: extension_len + original_len]

                    if u.shape[1] != original_len:
                        u = np.pad(u, ((0, 0), (0, original_len - u.shape[1])), mode='edge')

                    # --- 使用注意力网络重构 ---
                    # 2. 如果K值小于max_k, 进行填充，目的：匹配网络
                    imfs_tensor = torch.from_numpy(u).float()
                    if K < max_k:
                        # torch.zeros全部设置为0，imfs_tensor.shape[1]：信号的长度；占位符，为了后面的神经网络统一处理
                        padding = torch.zeros(max_k - K, imfs_tensor.shape[1])
                        imfs_tensor = torch.cat([imfs_tensor, padding], dim=0)

                    # 3. (伪训练) 让网络学习如何最好地重构原始信号
                    target_signal_tensor = torch.from_numpy(signal).float()
                    best_r2 = -1e9
                    best_state_dict = None
                    attention_model.train()
                    for epoch in range(asr_config['vmd_epochs']):  # 自监督训练
                        # 前向传播一次
                        optimizer.zero_grad()
                        # 注意力网络输入 [K, Seq_Len] -> [1, K, Seq_Len]，目的：目前只有单样本，模型需要有batch维度来训练，暂时填充1，unsqueeze(0)：第一维填充1
                        recon_signal_tensor, _ = attention_model(imfs_tensor.unsqueeze(0))
                        # squeeze(0)，只有当第 0 维大小为 1，去掉第 0 维；用途：从 batch=1 的输出里去掉 batch 维
                        loss = criterion(recon_signal_tensor.squeeze(0), target_signal_tensor)

                        # ===== 计算 R2 =====
                        # mse 就是 loss，本身已经是标量 tensor
                        mse = loss.detach()  # 不参与反向传播
                        # 目标信号的能量（方差比例因子），detach 防止梯度回传
                        energy = target_signal_tensor.pow(2).mean().detach() + 1e-8
                        # R2 = 1 - MSE / E[x^2]
                        r2 = 1.0 - mse / energy
                        # 记录日志（只记一条就够了）
                        logs.append(
                            f'VMD Epoch [{epoch + 1}/{asr_config["vmd_epochs"]}], '
                            f'MSE: {mse.item():.6f}, R2: {r2.item():.6f}'
                        )
                        # 反向传播 & 更新参数
                        loss.backward()
                        optimizer.step()

                        # ===== 保存当前最优模型（按 R2 最大）=====
                        if r2.item() > best_r2:
                            best_r2 = r2.item()
                            best_state_dict = copy.deepcopy(attention_model.state_dict())

                    # ===== 训练结束后，恢复到 R2 最好的那一次参数 =====
                    if best_state_dict is not None:
                        attention_model.load_state_dict(best_state_dict)
                    # 4. 得到最终重构信号
                    # 为了用最终学到的参数、在评估模式下，得到稳定且可保存 / 可视化的最终重构与注意力权重，之前在for循环中的不易保存，造成内存爆炸
                    attention_model.eval()
                    with torch.no_grad():
                        final_recon_tensor, final_weights = attention_model(imfs_tensor.unsqueeze(0))
                    reconstructed_signal = final_recon_tensor.squeeze(0).detach().numpy()
                    logs.append(f"    - '{col}' Attention weights (top {K}): {final_weights[0, :K].detach().numpy()}")

                    # --- VMD Plotting ---
                    # 执行这里的时候，层级关系是，某口井中的某个特征
                    if asr_config.get('enable_plot', False):
                        logs.append(f"    - VMD Plotting for {col}")
                        plt.figure(figsize=(14, 10))
                        plt.suptitle(f'VMD Decomposition: {col} @ well {well_id} (K={K})', fontsize=16)
                        plt.subplot(K + 1, 1, 1)
                        plt.plot(df_features.index.to_numpy(), signal, color='darkorange')
                        plt.title('Original Signal')
                        plt.grid(True)
                        for k in range(K):
                            plt.subplot(K + 1, 1, k + 2)
                            plt.plot(df_features.index.to_numpy(), u[k, :])
                            plt.title(f'IMF {k + 1}')
                            plt.grid(True)
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plot_path = os.path.join(well_viz_path, f'vmd_decomposition_{col}.png')
                        # logs.append(f"    - Saving VMD decomposition plot to: {plot_path}")
                        plt.savefig(plot_path)
                        plt.close()

                except Exception as e:
                    import traceback
                    logs.append(f"    VMD or Attention failed for '{col}': {e}\n{traceback.format_exc()}")
                    logs.append("    Using original signal as fallback.")
                    reconstructed_signal = signal  # 确保失败时回退
        # 如果不启用VMD，直接使用原始信号
        else:
            reconstructed_signal = signal
            logs.append(f"    - VMD is disabled for '{col}'. Using original signal for PAA.")

        # VMD信号重构时，只对特征这一列的数值做工作
        # 最终将所有的特征都汇聚到 df_reconstructed_well
        df_reconstructed_well[col] = reconstructed_signal
    logs.append(f"\n    - VMD decomposition Finished for all features in '{well_id}'\n")
    # 重构信号可视化 (上原信号 + 背景、下重构子图)
    # 这里的feature_cols包含了没有VMD分解的信号
    # 画图时间久，调试先去掉
    if asr_config.get('enable_plot', False):
        for col in feature_cols:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
            fig.suptitle(f'Original vs. Reconstructed: {col} @ well {well_id}', fontsize=16)
            # 可视化: 添加状态背景,只在原始信号上添加，其他图不添加
            add_state_background(ax1, class_labels, state_colors)
            # 横坐标：x 轴 = df_features.index
            ax1.plot(df_features.index.to_numpy(), df_features[col], label='Original Signal', color='darkorange')

            ax1.set_title('Original Signal'); ax1.legend(loc='upper right'); ax1.grid(True)

            # 如果没用VMD，ax2中的图会和ax1一样
            # 横坐标：x 轴 = df_reconstructed_well.index
            ax2.plot(df_reconstructed_well.index.to_numpy(), df_reconstructed_well[col], label='Reconstructed Signal' if enable_vmd else 'Original Signal (Bypassed)',
                     color='blue')
            ax2.set_title('Reconstructed Signal'); ax2.legend(loc='upper right'); ax2.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = os.path.join(well_viz_path, f'reconstruction_compare_{col}.png')
            # logs.append(f"    - Saving reconstruction plot to: {plot_path}")
            plt.savefig(plot_path)
            plt.close()

    # PAA 特征提取与可视化
    logs.append(f"    - Performing PAA and label aggregation for well '{well_id}'...")
    # 判断数据量是否充足？这个continue跳过的是当前的整个well_id
    # df_reconstructed_well是前面创建的针对某个井的的数据
    # 数据不够的话，直接返回原本的函数
    if len(df_reconstructed_well) < paa_window_size:
        logs.append(f"    - Skipping PAA for well '{well_id}': not enough data！！！ Please check！！！")
        return None
    # 共多少个窗口num_windows
    num_windows = len(df_reconstructed_well) // paa_window_size
    # 获取非特征列的名称，用于后续的标签聚合,例如：Timestamp、well_id、class、state、order_class
    label_cols = [col for col in df_well.columns if col not in feature_cols]
    for i in range(num_windows):
        # --- 提取当前窗口的数据 ---
        # 根据循环变量i和窗口大小，计算出当前窗口在原始DataFrame中的起始和结束索引。
        # .iloc[] 是Pandas中基于整数位置进行切片的方法。
        # 例如，当 i=0, 窗口是 [0:9]；当 i=1, 窗口是 [10:20]，依此类推。
        window_start_idx = i * paa_window_size
        window_end_idx = (i + 1) * paa_window_size
        # [0: 9]时间步的数据（仅仅有特征列的）
        window_recon_data = df_reconstructed_well.iloc[window_start_idx:window_end_idx]
        # [0: 9]时间步的数据（全部状态+特征），这里是从df_well切片的
        # window_recon_data与window_label_data是可以对的上的，一一对应的，时间也对,即原始数据切片出来的
        window_label_data = df_well.iloc[window_start_idx:window_end_idx]

        # --- 1. 初始化当前窗口的数据字典 ---
        current_window_dict = {}

        # --- 2. 处理时间戳和 well_id ---
        # 每个切片内，时间戳取原始数据的第一个起始位置
        # 因为timestamp和timestamp的取证铁树，所以放到开头赋值
        current_window_dict['timestamp'] = window_label_data['timestamp'].iloc[0]  # 取窗口起始时间戳
        current_window_dict['well_id'] = well_id

        # --- 3. 计算 PAA 特征 (均值和斜率) ---
        # (NaN值处理逻辑保留)
        window_cleaned = window_recon_data.ffill().bfill()
        if window_cleaned.isnull().values.any():
            logs.append(f"    - data contain NaN, please check！！！")
            window_cleaned = window_cleaned.fillna(0)
        # 单独处理冻结值
        for item in feature_cols:
            # 如果是 frozen_vars 中的变量
            if item in frozen_vars:
                mean_val = window_cleaned[item].iloc[paa_window_size//2]
                slope_val = 0.0
            else:
                # 计算 trimmed mean
                mean_val = trim_mean(window_cleaned[item], proportiontocut=0.1)
                # 计算斜率
                x = np.arange(paa_window_size)
                y = window_cleaned[item].values
                if np.all(y == y[0]):  # 全相等，避免linregress报错
                    slope_val = 0.0
                else:
                    slope_val, _, _, _, _ = linregress(x, y)
            # 存入结果字典
            current_window_dict[f"{item}_mean"] = mean_val
            current_window_dict[f"{item}_slope"] = slope_val
        # --- 4. 聚合标签列 (取多数票) ---
        # 这里手动去除timestamp和well_id，前面已经处理过
        # 实际就是class、井的状态state、order_class
        exclude_values = {'timestamp', 'well_id'}
        for col in [x for x in label_cols if x not in exclude_values]:
            if col in window_label_data.columns:
                # .mode() 返回出现次数最多的值，可能返回多个（如果票数一样多）
                # .iloc[0] 确保我们只取第一个，保证结果唯一
                mode_value = window_label_data[col].mode()
                if not mode_value.empty:
                    current_window_dict[col] = mode_value.iloc[0]
                else:  # 如果窗口内全是NaN，给一个默认值
                    current_window_dict[col] = 0

        # --- 5. 将处理完的窗口数据存入总列表 ---
        # all_paa_window_data就是一口井的所有、最终数据
        all_paa_window_data.append(current_window_dict)

    if not all_paa_window_data:
        logs.append("ASR/PAA Error: No windows were processed.")
        log_text = "\n".join(logs)
        # 即使失败，也保存日志（仅在持久化模式）
        if persist_asr_cache and log_save_path is not None:
            with open(log_save_path, 'w', encoding='utf-8') as f:
                f.write(log_text)
        return idx, well_id, None, log_text  # 返回None表示失败
    # --- [FINAL STEP] 创建最终的PAA特征DataFrame ---
    df_paa_features = pd.DataFrame(all_paa_window_data)
    logs.append("\n--- [ASR 模块] 单井处理完成。 ---")
    log_text = "\n".join(logs)
    if persist_asr_cache:
        logs.append("--- 正在保存结果到本地缓存（ASR_dataset）... ---")
        log_text = "\n".join(logs)
        try:
            # 保存 DataFrame
            df_paa_features.to_parquet(df_save_path)
            # 保存日志
            with open(log_save_path, 'w', encoding='utf-8') as f:
                f.write(log_text)
        except Exception as e:
            error_log = f"!!! 保存结果失败 for well {well_id}: {e} !!!"
            print(error_log)
            log_text += f"\n{error_log}"
            return idx, well_id, None, log_text

        # 子进程返回文件路径，主进程再加载
        return idx, well_id, str(df_save_path), log_text
    else:
        # 无需落盘：直接把 DataFrame 返回给主进程
        return idx, well_id, df_paa_features, log_text