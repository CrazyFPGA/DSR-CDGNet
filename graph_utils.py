import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch

def create_predefined_adj(graph_definitions, physical_node_names):
    """根据 config 创建预定义的邻接矩阵 A_physical。"""
    num_nodes = len(physical_node_names)
    node_to_idx = {name: i for i, name in enumerate(physical_node_names)}

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # 添加所有边
    scenes = graph_definitions.get("scenes", [])
    all_edges = []
    for scene_name in scenes:
        scene_info = graph_definitions.get(scene_name, {})
        all_edges.extend(scene_info.get("edges", []))
    # 额外加上桥接的关系
    all_edges.extend(graph_definitions.get("inter_scene_edges", []))

    for u, v in all_edges:
        if u in node_to_idx and v in node_to_idx:
            G.add_edge(node_to_idx[u], node_to_idx[v])
        else:
            # 添加一个警告，以防config中的节点名在数据中不存在
            if u not in node_to_idx: print(f"Node '{u}' from config not found in physical nodes list.")
            if v not in node_to_idx: print(f"Node '{v}' from config not found in physical nodes list.")
    adj = nx.to_numpy_array(G, nodelist=range(num_nodes))
    print(f" - A_physical created with shape {adj.shape} and {G.number_of_edges()} edges.")
    return torch.from_numpy(adj).float()

def analyze_and_visualize_pagerank(adj_matrix, all_node_names, output_dir):
    """根据最终的邻接矩阵 A_final 分析并可视化PageRank。"""
    print("\n--- [Graph Util] Analyzing PageRank Centrality of A_final ---")

    # [CRITICAL FIX] 在转换为numpy之前，必须先detach()
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    # 根据邻接矩阵创建了一个图
    G = nx.from_numpy_array(adj_matrix)
    if G.number_of_edges() == 0:
        print(" - Graph has no edges, skipping PageRank analysis.")
        return None
    # 计算PageRank
    # 它返回一个字典，其中：键(key)是图中的节点编号（例如0, 1, 2, ...）。值(value)是该节点对应的PageRank浮点数分数。
    pagerank_scores = nx.pagerank(G)
    pagerank_series = pd.Series({all_node_names[i]: score for i, score in pagerank_scores.items()}).sort_values(
        ascending=False)

    print(" - Top 10 nodes by PageRank score:")
    print(pagerank_series.head(10).to_string())

    plt.figure(figsize=(12, max(6, len(all_node_names) // 4)))
    sns.barplot(x=pagerank_series.values, y=pagerank_series.index, hue=pagerank_series.index, palette="viridis",
                legend=False)
    plt.title('PageRank Centrality of Final Graph', fontsize=16)
    plt.xlabel('PageRank Score')
    plt.ylabel('Node Name')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'pagerank_centrality_final_graph.png')
    plt.savefig(plot_path)
    plt.close()
    print(f" - PageRank visualization saved to: {plot_path}")

    # [4. 保存原始数据]
    # 确保保存路径存在
    data_save_dir = os.path.join(output_dir, 'plot_data')
    os.makedirs(data_save_dir, exist_ok=True)

    # --- [修正点 1] 保存节点数据 ---
    # pagerank_scores 是字典，不能用 .detach()
    # 我们需要保证分数和 all_node_names 的顺序一一对应
    # 假设 all_node_names 的索引 i 对应图中的节点 i
    pr_values_list = [pagerank_scores.get(i, 0.0) for i in range(len(all_node_names))]

    node_data = {
        'Node_Name': all_node_names,
        'PageRank_Score': pr_values_list
    }
    pd.DataFrame(node_data).to_csv(os.path.join(data_save_dir, 'data_graph_nodes.csv'), index=False)

    # --- [修正点 2] 保存边数据 ---
    # adj_matrix 在函数开头已经被转为 numpy 了，直接使用即可，不需要 .detach()
    adj_np = adj_matrix

    # 过滤掉权重极小的边以减小文件体积
    rows, cols = np.where(adj_np > 1e-4)
    weights = adj_np[rows, cols]

    # 将索引转换为节点名称
    source_nodes = [all_node_names[r] for r in rows]
    target_nodes = [all_node_names[c] for c in cols]

    edge_data = {
        'Source': source_nodes,
        'Target': target_nodes,
        'Weight': weights
    }
    pd.DataFrame(edge_data).to_csv(os.path.join(data_save_dir, 'data_graph_edges.csv'), index=False)

    print(f"  [Data Saved] Graph node and edge data saved to: {data_save_dir}")

    return pagerank_series

def visualize_graph(adj_matrix, node_names, title, output_dir, filename):
    """
    一个通用的图可视化函数，可以根据邻接矩阵绘制带权重的图。
    """
    print(f"\n--- [Graph Util] Visualizing Graph: {title} ---")

    if isinstance(adj_matrix, torch.Tensor):
        # .detach() 以防万一，.cpu() 移到CPU, .numpy() 转为numpy数组
        adj_matrix = adj_matrix.detach().cpu().numpy()

    # 只保留大于一个非常小的阈值的边，避免画出几乎为0的边
    adj_matrix[np.abs(adj_matrix) < 1e-4] = 0

    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    plt.figure(figsize=(20, 20))

    # 使用 Kamada-Kawai 布局，对于中小型图效果较好
    try:
        pos = nx.kamada_kawai_layout(G)
    except nx.NetworkXError:  # 如果图不连通，回退到 spring_layout
        print(" - Warning: Graph is not connected. Falling back to spring_layout.")
        pos = nx.spring_layout(G, k=0.5, iterations=50)

    # 节点标签
    labels = {i: name for i, name in enumerate(node_names)}

    # 边的权重决定了边的宽度和透明度
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for u, v, d in edges]

    if edge_weights:
        max_weight = max(edge_weights) if edge_weights else 1.0
        min_weight = min(edge_weights) if edge_weights else 0.0

        # 将权重归一化到 [0.5, 5.0] 作为边的宽度
        widths = [0.5 + 4.5 * (w - min_weight) / (max_weight - min_weight + 1e-6) for w in edge_weights]
        # 将权重归一化到 [0.3, 1.0] 作为边的透明度
        edge_alphas = [0.3 + 0.7 * (w - min_weight) / (max_weight - min_weight + 1e-6) for w in edge_weights]
    else:
        widths = 1.5
        edge_alphas = 0.7

    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2500, alpha=0.95)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=edge_alphas, edge_color='gray', arrows=True, arrowstyle='->',
                           arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

    plt.title(title, fontsize=24, weight='bold')
    plt.box(False)  # 去掉边框

    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Graph visualization saved to: {plot_path}")