import pandas as pd
import numpy as np
from functools import reduce
import re

def generate_data_quality_report(df, all_feature_cols, output_path, cfg):
    """接收输出路径以保存报告"""
    print("\n" + "="*80)
    print(" " * 25 + "INITIAL DATA QUALITY REPORT")
    print("="*80)

    report_total_stats, report_non_missing, report_frozen_vars = [], pd.DataFrame(), pd.DataFrame(index=all_feature_cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    unique_folders = sorted(df['source_folder'].unique(), key=int)

    for folder in unique_folders:
        print(f"\n{'='*15} Analyzing [Source Folder: {folder}] {'='*15}")
        folder_df = df[df['source_folder'] == str(folder)]

        # 1. Excel报告数据计算: 总览统计
        total_samples = len(folder_df)
        class_0_count = (folder_df['class'] == 0).sum()
        class_not_0_and_not_null_count = (folder_df['class'] != 0) & folder_df['class'].notna()
        class_null_count = folder_df['class'].isnull().sum()
        report_total_stats.append({
            'Folder': f"From folder '{folder}'", 'Total Samples': total_samples,
            'class=0': class_0_count, 'class!=0 and not empty': class_not_0_and_not_null_count.sum(),
            'class is empty': class_null_count
        })

        # 定义要分析的子集
        subsets_to_process = [
            (f"Folder_{folder}_class_0", folder_df[folder_df['class'] == 0])
        ]
        if int(folder) != 0:
            subsets_to_process.append(
                (f"Folder_{folder}_class_not_0", folder_df[(folder_df['class'] != 0) & (folder_df['class'].notna())])
            )

        for name, sub_df in subsets_to_process:
            print(f"\n  -> Subset: {name} | Total Rows: {len(sub_df)}")
            if sub_df.empty:
                print("     - No data to analyze in this subset.")
                continue

            # 2. Excel报告数据计算: 非缺失样本统计
            non_missing_counts = sub_df[all_feature_cols].notna().sum()
            non_missing_counts.name = name
            report_non_missing = pd.concat([report_non_missing, non_missing_counts], axis=1)

            # 3. Excel报告数据计算: 高级冻结变量分析
            unique_wells_in_subset = sub_df['well_id'].unique()
            frozen_by_well = [
                set(well_df[all_feature_cols].var(ddof=0)[lambda v: v < cfg['VARIANCE_THRESHOLD']].index)
                for _, well_df in sub_df.groupby('well_id') if not well_df.empty
            ]

            if frozen_by_well:
                common_frozen = reduce(set.intersection, frozen_by_well)
                is_frozen_mask = report_frozen_vars.index.isin(common_frozen)
                report_frozen_vars[name] = np.where(is_frozen_mask, '共同冻结', '')
            else:
                report_frozen_vars[name] = ''

            # 详细的控制台日志
            print("    [Console-Detailed Analysis]")
            for col in all_feature_cols:
                if col not in sub_df.columns: continue
                print(f"\n      -> Variable: '{col}'")
                try:
                    sorted_wells = sorted(unique_wells_in_subset, key=lambda x: int(re.search(r'\d+', str(x)).group()))
                except (AttributeError, ValueError):
                    sorted_wells = sorted(unique_wells_in_subset) # 如果well_id不是数字格式，则按字符串排序

                for well_id in sorted_wells:
                    well_sub_df = sub_df[sub_df['well_id'] == well_id]
                    if well_sub_df.empty: continue

                    col_series = well_sub_df[col]
                    missing_count = col_series.isnull().sum()
                    missing_percentage = missing_count / len(col_series) if not col_series.empty else 0
                    variance = col_series.var(ddof=0) if not col_series.empty else 0
                    is_frozen_str = " (Frozen)" if variance < cfg['VARIANCE_THRESHOLD'] else ""
                    print(f"        - Well ID: {str(well_id).ljust(15)} | Missing: {str(missing_count).rjust(7)} ({missing_percentage:7.2%}) | Variance: {variance:.4e}{is_frozen_str}")

    print("\n--- [Report Generation] ---")
    try:
        excel_path = output_path / 'data_quality_report.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(report_total_stats).set_index('Folder').to_excel(writer, sheet_name='Total_Statistics')
            report_non_missing.to_excel(writer, sheet_name='Non_Missing_Counts')
            report_frozen_vars.to_excel(writer, sheet_name='Frozen_Variables_Analysis')
        print(f" - Successfully generated report to '{excel_path}'")
    except Exception as e:
        print(f" [ERROR] Failed to generate Excel report: {e}. Please ensure 'openpyxl' is installed.")

    print("\n" + "="*80); print(" " * 27 + "END OF QUALITY REPORT"); print("="*80 + "\n")
