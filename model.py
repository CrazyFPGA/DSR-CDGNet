import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, SAGEConv, APPNP, GINConv, ChebConv
from torch_geometric.utils import dense_to_sparse, add_self_loops, get_laplacian, degree
import networkx as nx
import logging
import numpy as np
from graph_utils import visualize_graph
import pandas as pd
import os

class AdaptiveGraphLearner(nn.Module):
    """
    通过GRU学习节点嵌入，并生成数据驱动的自适应邻接矩阵。
    """
    def __init__(self, num_nodes, input_dim, embed_dim, k):
        super(AdaptiveGraphLearner, self).__init__()
        self.k = k
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        # Node Embedder: 使用GRU来捕捉每个节点的时间序列动态
        # 这是一个 nn.GRU 模块，是 HSR_DIGNET 模型的一部分。它的权重是可学习的参数。
        # 当您训练循环中调用 loss.backward() 和 optimizer.step() 时，优化器会更新这个 GRU 的权重。
        self.node_embedder = nn.GRU(input_size=input_dim, hidden_size=embed_dim, batch_first=True)
        logging.info(f"Initialized AdaptiveGraphLearner with k={k} and node_embed_dim={embed_dim}.")

    # forward 方法现在接收一个 data_loader
    def forward(self, data_loader, device):
        # 这个函数的目标是：遍历 data_loader，分批次计算节点的时间序列，
        # 然后将它们送入 GRU，最后聚合所有批次的输出来得到最终的节点嵌入。
        self.node_embedder.to(device)  # 确保模型在正确的设备上
        # 1. 分批次处理数据以生成节点嵌入
        all_node_embeddings = []
        with torch.no_grad():  # 在这个聚合阶段不计算梯度
            for x_batch, _, _, *__ in data_loader:
                x_batch = x_batch.to(device)

                # --- 与您原 forward 逻辑类似，但作用于 batch ---
                # x_batch shape: [Batch, Seq_Len, Num_Nodes * Features]
                x = x_batch.view(x_batch.shape[0], x_batch.shape[1], self.num_nodes, self.input_dim)
                # -> [Num_Nodes, Batch, Seq_Len, Features]
                x_node_series = x.permute(2, 0, 1, 3)
                # -> [Num_Nodes, Batch * Seq_Len, Features]
                x_node_series = x_node_series.reshape(self.num_nodes, -1, self.input_dim)

                _, node_embeddings_batch = self.node_embedder(x_node_series)
                node_embeddings_batch = node_embeddings_batch.squeeze(0)  # Shape: [Num_Nodes, embed_dim]

                all_node_embeddings.append(node_embeddings_batch.cpu())  # 移到CPU以节省GPU内存
        # 2. 聚合所有批次的节点嵌入
        # 最简单的方式是取平均
        if not all_node_embeddings:
            raise ValueError("No data processed to generate node embeddings.")

        # 将所有批次的嵌入堆叠起来，然后在批次维度上求平均
        # [Num_Batches, Num_Nodes, embed_dim] -> [Num_Nodes, embed_dim]
        node_embeddings = torch.stack(all_node_embeddings).mean(dim=0).to(device)

        # 3. 计算余弦相似度 (Similarity Calculator)
        # F.normalize(tensor, p=2, dim=1)会对输入张量沿指定维度做Lp范数归一化。常用p = 2，就是L2归一化（即把向量长度归一到1）。
        node_embeddings_norm = F.normalize(node_embeddings, p=2, dim=1)
        # 这里的归一化必须做，否则后续计算出的similarity_matrix不是余弦相似度，而只是简单的点积
        # 余弦相似度矩阵
        similarity_matrix = torch.matmul(node_embeddings_norm, node_embeddings_norm.T)

        # 4. kNN稀疏化 (Sparsification Unit)
        # 找到每行中第k大的值作为阈值
        top_k = torch.topk(similarity_matrix, k=self.k, dim=1).values
        kth_values = top_k[:, -1].view(-1, 1)

        # 创建一个掩码，只有大于等于阈值的位置为1
        mask = (similarity_matrix >= kth_values).float()

        # 将邻接矩阵与掩码相乘，得到稀疏的自适应邻接矩阵 A_learned
        # 同时保留相似度作为权重
        A_learned = similarity_matrix * mask
        return A_learned

class APPNPWrapper(nn.Module):
    """
    APPNP包装类：将线性层和APPNP传播层组合在一起
    """
    def __init__(self, in_channels, out_channels, K=10, alpha=0.1):
        super(APPNPWrapper, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha)
    
    def forward(self, x, edge_index, edge_weight=None):
        # 先通过线性层进行特征变换
        x = self.lin(x)
        # 然后进行APPNP传播（APPNP不使用edge_weight，它使用图结构）
        return self.prop(x, edge_index)

class CGCConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # 这里的node_dim=0会影响self.propagate(edge_index, x=x, edge_weight=edge_weight, pagerank_j=pagerank_j)
        # 中的x、pagerank_j的维度，node_dim=0会让self._set_size检查第0维的个数是否等于图特征节点的个数
        # 如果node_dim=1要看x和pagerank_j的维度是否正确
        super(CGCConv, self).__init__(aggr='add', node_dim=0)
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, pagerank):
        # pagerank_scores: [Num_Nodes]
        # edge_index: [2, Num_Edges]
        # 我们需要的是每条边的"源节点"（邻居节点 j）的PageRank分数
        # 在PyG的定义中，x_j的j是源节点，所以对应edge_index[0]
        # pagerank = pagerank_scores[edge_index[0]].unsqueeze(-1)  # Shape: [Num_Edges, 1]

        # 调用propagate，它会自动调用message, aggregate, update
        # edge_index = [2,19]，即源节点j和目标节点i的对应关系
        # x = [几个物理特征节点数,paa 2 + 场景 4] = [6,6]
        # pagerank_scores，每个节点的分数，只有1维！！！
        # 考虑后，pagerank_scores加上了unsqueeze(-1)，变成了[Num_Edges, 1]，因为super(CGCConv, self).__init__(aggr='add', node_dim=0)中设置了node_dim=0，故不影响
        # 但是加上了unsqueeze(-1)，变成了[Num_Edges, 1]会报错，先使用1维的吧
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, pagerank=pagerank)

    def message(self, x_j, edge_weight, pagerank_j):
        # x_j: 邻居节点的特征, shape [Num_Edges, in_channels]
        # edge_weight: 边的权重, shape [Num_Edges]
        # pagerank_j: 邻居节点的PageRank, shape [Num_Edges, 1]

        # 核心逻辑: 消息 = 边权重 * 邻居特征 * 邻居PageRank
        # edge_weight.view(-1, 1) 将其形状变为 [Num_Edges, 1] 以便广播
        # 这里必须 pagerank_j.squeeze(-1)，因为传过来的 pagerank_j 是一维的，这是的相乘操作是二维的：
        # 如果 pagerank_j 的原始形状为(n, )，那么 pagerank_j.unsqueeze(-1) 的结果形状将变为(n, 1)
        return edge_weight.view(-1, 1) * x_j * pagerank_j.unsqueeze(-1)

    def update(self, aggr_out, x):
        # aggr_out: 聚合后的邻居信息, shape [Num_Nodes, in_channels]
        # x: 节点自身的特征
        # 我们将聚合信息与节点自身信息相加（类似GCN的skip connection），然后进行线性变换
        return self.lin(aggr_out + x)

class HSR_DIGNET(nn.Module):
    # physical_node_names为物理节点；num_nodes为feature_cols为物理节点的数量，没有二维；num_paa_features=2
    def __init__(self, config, num_nodes, num_paa_features, num_state_features, num_classes, graph_definitions, physical_node_names):
        super(HSR_DIGNET, self).__init__()
        hgc_cfg = config['hgc_stp_module']
        itr_cfg = config['itr_module']  # 新增ITR配置

        self.num_nodes = num_nodes
        self.physical_node_names = physical_node_names
        self.num_paa_features = num_paa_features  # PAA特征维度 (mean+slope=2)
        self.num_state_features = num_state_features
        self.graph_definitions = graph_definitions
        self.seq_len = config['hyperparameters']['sequence_length']
        self.num_classes = num_classes

        # --- 消融实验开关 ---
        # 统一成大写，方便在配置中写 'None' / 'none'
        self.gcn_type = hgc_cfg.get('gcn_type', 'CGC')
        self.gcn_type = self.gcn_type.upper()
        self.itr_type = itr_cfg.get('itr_type', 'Dynamic_ITR')

        self.enable_early_exit = itr_cfg.get('enable_early_exit')
        self.early_exit_threshold = itr_cfg.get('early_exit_threshold')

        logging.info(f"--- Model Configuration ---")
        logging.info(f"  - HGC GCN Type: {self.gcn_type}")
        logging.info(f"  - ITR Type: {self.itr_type}")
        logging.info(f"  - Early Exit Enabled: {self.enable_early_exit}")

        # --- 缓存和工具 ---
        self.cached_A_learned = None
        self.cached_pagerank = None
        self.cached_A_final = None  # 注意：我们在_build_and_cache_graph中也缓存这个
        self.A_physical = None
        self.node_to_idx = {name: i for i, name in enumerate(physical_node_names)}
        self._build_call_count = 0

        # --- 子模块 2: 图相关模块（可关闭） ---
        if self.gcn_type != 'NONE':
            # --- 子模块 2.1: 混合图构建 (Hybrid Graph Construction) ---,构建自适应邻间矩阵
            self.adaptive_learner = AdaptiveGraphLearner(
                num_nodes=num_nodes,
                input_dim=self.num_paa_features,
                embed_dim=hgc_cfg['node_embed_dim'],
                k=hgc_cfg['k_knn']
            )
            # 图融合权重alpha，设为可学习的参数
            self.alpha = nn.Parameter(torch.tensor(hgc_cfg['alpha_initial']))

            # --- 子模块 2.2: 空间传播 (Spatio-Temporal Propagation) ---
            # 1. 场景感知嵌入层
            self.scene_embedding = nn.Embedding(
                len(self.graph_definitions['scenes']), hgc_cfg['scene_embed_dim']
            )

            # 2. 中心性引导图卷积网络
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            # 为残差连接准备投影层，因为第一层GCN的输入为4，输出位16，残差连接不能直接相加，故需要变换映射
            self.residual_projections = nn.ModuleList()
            # 第一层的输入维度 = PAA特征维度2 + 场景嵌入维度4 + 独热
            current_dim = self.num_paa_features + self.num_state_features + hgc_cfg['scene_embed_dim']
            hidden_dim = hgc_cfg['gcn_hidden_dim']

            # 残差连接必须自己添加，torch_geometric.nn.MessagePassing是一个基础的模块，没有自动实现
            for i in range(hgc_cfg['gcn_layers']):
                # --- 选择卷积层 ---
                if self.gcn_type == 'CGC':
                    conv = CGCConv(current_dim, hidden_dim)
                    output_dim = hidden_dim
                elif self.gcn_type == 'GCN':
                    conv = GCNConv(current_dim, hidden_dim)
                    output_dim = hidden_dim
                elif self.gcn_type == 'GAT':
                    heads = hgc_cfg.get('gat_heads', 4)
                    # 对于GAT，输出维度是 hidden_dim * heads
                    conv = GATConv(current_dim, hidden_dim, heads=heads, dropout=0.1)
                    output_dim = hidden_dim * heads
                elif self.gcn_type == 'SAGE':
                    conv = SAGEConv(current_dim, hidden_dim, aggr='mean')
                    output_dim = hidden_dim
                elif self.gcn_type == 'APPNP':
                    # APPNP: Approximate Personalized Propagation of Neural Predictions
                    # 使用K=10步传播，alpha=0.1作为默认值
                    K = hgc_cfg.get('appnp_K', 10)
                    alpha = hgc_cfg.get('appnp_alpha', 0.1)
                    # 使用包装类将线性层和APPNP传播层组合
                    conv = APPNPWrapper(current_dim, hidden_dim, K=K, alpha=alpha)
                    output_dim = hidden_dim
                elif self.gcn_type == 'GIN':
                    # GIN: Graph Isomorphism Network
                    # 需要MLP作为参数，使用Sequential构建
                    mlp = nn.Sequential(
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                    conv = GINConv(mlp, train_eps=True)
                    output_dim = hidden_dim
                else:
                    raise ValueError(f"Unknown gcn_type: {self.gcn_type}")

                # a. 构建主路径的卷积层 (使用当前的 current_dim)
                self.convs.append(conv)
                # b. 构建主路径的归一化层
                self.norms.append(nn.LayerNorm(output_dim))
                # c. 构建快捷路径的投影层 (同样使用当前的 current_dim 来判断)
                if current_dim != output_dim:
                    # 如果维度不匹配，添加一个线性层用于投影
                    self.residual_projections.append(nn.Linear(current_dim, output_dim))
                else:
                    # 如果维度匹配，则添加一个"什么都不做"的恒等层
                    self.residual_projections.append(nn.Identity())
                # d. 更新 current_dim，为下一层的构建做准备
                # 这一步至关重要，它确保了下一次循环时，current_dim 是正确的
                current_dim = output_dim
            self.hgc_output_dim = current_dim
            # 无需额外投影
            self.no_graph_projection = nn.Identity()
        else:
            # 无图模式：直接使用 PAA (+ 可选 state) 作为特征输入 ITR
            # 原始特征维 = PAA特征维 + 井状态维（如果有）
            self.adaptive_learner = None
            self.alpha = None
            self.scene_embedding = None
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.residual_projections = nn.ModuleList()
            raw_dim = self.num_paa_features + self.num_state_features
            # 为了兼容 MultiheadAttention 的 nhead 要求，这里将无图特征先线性投影到一个合适的维度
            # 默认使用与图卷积相同的隐藏维度 gcn_hidden_dim（通常能被 nhead 整除）
            no_graph_hidden_dim = hgc_cfg.get('no_graph_hidden_dim', hgc_cfg['gcn_hidden_dim'])
            self.no_graph_projection = nn.Linear(raw_dim, no_graph_hidden_dim)
            self.hgc_output_dim = no_graph_hidden_dim

        # --- 子模块 3: ITR ---
        # 自己的动态迭代；固定次数的迭代
        if self.itr_type == 'Dynamic_ITR' or self.itr_type == 'Single_ITR' or self.itr_type == 'Dynamic_ITR_CE':
            self.itr_module = IterativeTemporalReasoning(
                d_model=self.hgc_output_dim,
                nhead=itr_cfg['nhead'],
                dim_feedforward=self.hgc_output_dim * 2,
                dropout=itr_cfg['dropout'],
                num_iterations=itr_cfg['num_iterations']
            )
            # 对于 ITR，分类器的输入是展平后的时间序列
            feature_dim_for_classifier = self.hgc_output_dim * self.seq_len
        elif self.itr_type == 'GRU':
            gru_hidden_dim = itr_cfg.get('gru_hidden_dim', 128)
            # GRU 的输入是聚合后的特征
            self.gru_module = nn.GRU(
                input_size=self.hgc_output_dim, hidden_size=gru_hidden_dim,
                num_layers=1, batch_first=True
            )
            feature_dim_for_classifier = gru_hidden_dim
        elif self.itr_type == 'LSTM':
            lstm_hidden_dim = itr_cfg.get('lstm_hidden_dim', 128)
            # LSTM 的输入同样是聚合后的特征
            self.lstm_module = nn.LSTM(
                input_size=self.hgc_output_dim, hidden_size=lstm_hidden_dim,
                num_layers=1, batch_first=True
            )
            feature_dim_for_classifier = lstm_hidden_dim
        elif self.itr_type == 'MLP':
            mlp_hidden_dim = itr_cfg.get('mlp_hidden_dim', 128)
            # MLP 的输入是聚合后的特征
            self.mlp_module = nn.Sequential(
                nn.Linear(self.hgc_output_dim, mlp_hidden_dim),
                nn.ReLU(), nn.Dropout(0.3),
            )
            feature_dim_for_classifier = mlp_hidden_dim
        elif self.itr_type == 'Transformer':
            transformer_hidden_dim = itr_cfg.get('transformer_hidden_dim', self.hgc_output_dim)
            transformer_num_layers = itr_cfg.get('transformer_num_layers', 2)
            transformer_nhead = itr_cfg.get('transformer_nhead', 4)
            transformer_dropout = itr_cfg.get('transformer_dropout', 0.1)
            dim_feedforward = itr_cfg.get('transformer_dim_feedforward', transformer_hidden_dim * 2)
            
            # Transformer Encoder Layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_hidden_dim,
                nhead=transformer_nhead,
                dim_feedforward=dim_feedforward,
                dropout=transformer_dropout,
                batch_first=True
            )
            self.transformer_module = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_num_layers
            )
            # 如果hidden_dim与hgc_output_dim不同，需要投影层
            if transformer_hidden_dim != self.hgc_output_dim:
                self.transformer_projection = nn.Linear(self.hgc_output_dim, transformer_hidden_dim)
            else:
                self.transformer_projection = nn.Identity()
            
            # Transformer输出最后一个时间步的特征
            feature_dim_for_classifier = transformer_hidden_dim
        # --- 3. 分类器 ---
        # 注意: shared_classifier 现在移到了这里，以接收正确的输入维度
        # 我们根据不同的 ITR 类型，将其设计为不同的结构
        if self.itr_type == 'Dynamic_ITR' or self.itr_type == 'Single_ITR' or self.itr_type == 'Dynamic_ITR_CE':
            self.shared_classifier = nn.Linear(feature_dim_for_classifier, num_classes)
        else:  # For GRU, LSTM, MLP and Transformer, a simpler classifier might be better
            self.shared_classifier = nn.Sequential(
                nn.Linear(feature_dim_for_classifier, feature_dim_for_classifier // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dim_for_classifier // 2, num_classes)
            )

    def _build_and_cache_graph(self, A_physical, train_loader, output_dir, physical_node_names):
        """内部函数，用于构建图结构并缓存PageRank。"""
        # 无图模式：直接跳过图构建与PageRank
        if self.gcn_type == 'NONE':
            print("gcn_type == 'NONE': skip _build_and_cache_graph (no graph is used in this ablation).")
            self.A_physical = None
            self.cached_A_learned = None
            self.cached_pagerank = None
            self.cached_A_final = None
            return
        # 增加并打印调用次数
        self._build_call_count += 1
        print("Building/Updating and caching graph structure...")
        # 保存 A_physical 供 forward 使用
        self.A_physical = A_physical
        # 1. 自适应图学习
        # 构建图的方式不变，以固定的相似度计算单元、稀疏化单元参数去构建，但是实际上，这两个单元的参数本来就不应该被改变,也没有可以被改变的参数；为了保证稳定，这里加上no_grad
        # 构建图的方式不变，但是节点嵌入的参数更新，所以构建的自适应矩阵在变化
        # 只是要知道：这是从“输入表征变化”的角度出发的设计，不是为了让图本身参与反向传播，所以本身就应该在 no_grad 下跑。

        # 1. 重型计算：自适应图学习 (GRU部分)
        # 这里只做一次，结果缓存下来，符合你的“Epoch更新”意图
        with torch.no_grad(): # 图学习过程不参与主模型的梯度回传
            # 图构建 <- 邻间矩阵 <- 相似度计算+稀疏化
            device = A_physical.device
            # 计算 A_learned 并缓存它，而不是缓存 A_final
            self.cached_A_learned = self.adaptive_learner(train_loader, device)
        # 2. 图融合
        # 使用ReLU确保alpha为非负，softplus
        #A_final = A_physical + F.relu(self.alpha) * A_learned
        #self.cached_A_final = A_final

        # 打印 A_final 的关键信息以观察变化
        with torch.no_grad(): # 确保打印操作不影响梯度计算
            # 为了看一眼现在的图长什么样，临时算一下
            # 计算临时的 A_final 用于分析
            # 注意：temp_alpha 和 temp_A_final 都是不带梯度的
            temp_alpha = F.softplus(self.alpha).detach()
            temp_A_final = A_physical + temp_alpha * self.cached_A_learned
            # 提取左上角 5x5 子矩阵进行观察
            sub_matrix = temp_A_final[:5, :5]
            # 计算所有非零元素的均值，这是一个很好的全局变化指标
            non_zero_elements = temp_A_final[temp_A_final > 0]
            if len(non_zero_elements) > 0:
                mean_of_non_zeros = non_zero_elements.mean().item()
            else:
                mean_of_non_zeros = 0.0

            print("  - A_final (top-left 5x5 corner):")
            # 使用 numpy 格式化打印，更美观
            print(np.round(sub_matrix.cpu().numpy(), 4))
            print(f"  - Mean of non-zero elements in A_final: {mean_of_non_zeros:.6f}")
            # --- [新增] “三部曲”可视化 ---

        # 只在第一次构建图时进行可视化，避免每个epoch都画图
        # 这个图没什么用，价值不大，不需要
        '''
        if self._build_call_count == 1:
            print("\n--- [Graph Visualization] Generating graph evolution plots... ---")
            # 1. 可视化原始物理图
            visualize_graph(
                adj_matrix=A_physical,
                node_names=physical_node_names,
                title="1. Initial Physical Graph (A_physical)",
                output_dir=output_dir,
                filename="graph_vis_1_physical.png"
            )

            # 2. 可视化学习到的自适应图 (乘以alpha后)
            learned_graph_weighted = temp_alpha * self.cached_A_learned
            visualize_graph(
                adj_matrix=learned_graph_weighted,
                node_names=physical_node_names,
                title=f"2. Learned Adaptive Graph (alpha * A_learned, alpha={temp_alpha.item():.4f})",
                output_dir=output_dir,
                filename="graph_vis_2_learned.png"
            )

            # 3. 可视化最终的混合图
            visualize_graph(
                adj_matrix=temp_A_final,
                node_names=physical_node_names,
                title="3. Final Fused Graph (A_final)",
                output_dir=output_dir,
                filename="graph_vis_3_final.png"
            )
        '''
        # 3. PageRank计算
        # detach()以避免梯度流经PageRank计算过程
        # A_final.detach()会创建一个与A_final共享数据、但与计算图完全断开的新张量。
        # 如果不断开的话：当loss.backward()发生时，autograd引擎追溯到pagerank这一步时，它会发现这是一条“断头路”。
        # 它不知道如何对networkx.pagerank()这个非PyTorch函数求导，于是程序就会立刻抛出一个RuntimeError，告诉你梯度无法计算。
        # create_using=nx.DiGraph 因为相似度不一定对称
        G = nx.from_numpy_array(temp_A_final.cpu().numpy(), create_using=nx.DiGraph)
        pagerank_dict = nx.pagerank(G)

        self.cached_pagerank = torch.tensor(
            [pagerank_dict.get(i, 0) for i in range(self.num_nodes)],
            device=A_physical.device
        ).float()
        self.cached_A_final = temp_A_final.detach()
        # 打印日志...
        print(f"  - Fusion alpha (current): {temp_alpha.item():.4f}")
        print("Graph learned part (GRU) cached. Fusion will happen dynamically.")

    def hgc_stp_forward(self, x_batch, state_batch):
        """
        使用大图批处理技术高效地进行图卷积。
        """
        if self.gcn_type == 'NONE':
            raise RuntimeError("hgc_stp_forward should not be called when gcn_type == 'NONE'.")
        # x_batch shape: [Batch, Seq_Len, Num_Nodes * Features]
        # state_batch shape: [Batch, Num_State_Features]
        device = x_batch.device

        # 在这里进行实时的轻量级融合
        # 1. 获取当前的 alpha (现在可以使用 Softplus 了)
        #  虽然 alpha 每个 batch 都在变，但这里的计算量微乎其微
        current_alpha = F.softplus(self.alpha)

        # 2. 合成 A_final
        #    A_physical 是预设的，cached_A_learned 是 Epoch 开始时算好的
        #    这样既利用了缓存，又保证了梯度计算的合法性
        A_final = self.A_physical.to(device) + current_alpha * self.cached_A_learned.to(device)

        # 3. 转换稀疏矩阵 (PyG GCN 需要)
        # edge_index：谁和谁有关系； edge_weight：多大的关系
        edge_index, edge_weight = dense_to_sparse(A_final)

        # 4. 获取 PageRank (PageRank 我们还是沿用 epoch 开始时算的那个，因为它的计算不可导且慢)
        pagerank_scores = self.cached_pagerank.to(device)

        # --- 2. 时空传播 ---
        # a. 准备输入数据
        batch_size, seq_len, _ = x_batch.shape
        # 重构之后，x = [128,6,90,2]，6为特征数量，2为PAA特征数
        x = x_batch.view(batch_size, seq_len, self.num_nodes, self.num_paa_features).permute(0, 2, 1, 3)  # -> [B, N, S, F_paa]

        # [注入独热编码的 state 特征]
        # num_state_features为井的状态，如果是只有1个状态，那相当于没有，前面已经置num_state_features=0
        if self.num_state_features > 0:
            # state 特征对于一个序列来说是静态的，需要广播到所有节点和所有时间步
            # 1. 增加维度以匹配 x: [B, F_state] -> [B, 1, 1, F_state]
            state_features_expanded = state_batch.unsqueeze(1).unsqueeze(1)
            # 2. 广播到所有节点和时间步: [B, 1, 1, F_state] -> [B, N, S, F_state]
            state_features_expanded = state_features_expanded.expand(-1, self.num_nodes, seq_len, -1)
            # 3. 沿特征维度拼接
            # x 变为 [B, N, S, F_paa + F_state]
            x = torch.cat([x, state_features_expanded], dim=-1)
        # [注入场景感知嵌入]
        if self.graph_definitions and self.graph_definitions.get('scenes'):
            scene_cfg = self.graph_definitions
            scene_embeds = torch.zeros(self.num_nodes, self.scene_embedding.embedding_dim, device=device)
            # 1. 遍历 "scenes" 列表
            #    self.scene_embedding 是一个 nn.Embedding(3, 4) 的矩阵，有3行，对应3个场景
            #    i 的值会是 0, 1, 2
            for i, scene_name in enumerate(scene_cfg['scenes']):
                # 3. 找到该场景下的所有节点
                #    scene_cfg[scene_name] 就是 scene_cfg["production_channel"]
                #    scene_cfg[scene_name]['nodes'] 就是 "production_channel" 下的节点列表
                # 4. 将节点名转换为索引
                indices = [self.node_to_idx[node] for node in scene_cfg[scene_name]['nodes'] if
                           node in self.node_to_idx]
                if indices:
                    indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                    # 2. 获取该场景对应的嵌入向量
                    #    当i=0时, embeds是嵌入矩阵的第0行
                    # 可学习的权重矩阵
                    embeds = self.scene_embedding(torch.tensor(i, device=device, dtype=torch.long))
                    # 5. 将嵌入向量赋予这些节点的对应位置
                    # ... (scatter_add_ 或直接索引赋值)
                    scene_embeds.index_add_(0, indices_tensor, embeds.unsqueeze(0).expand(len(indices), -1))
            # 广播场景嵌入到批次和时间步维度
            # scene_embeds shape: [N, F_scene] -> [1, N, 1, F_scene] -> [B, N, S, F_scene]
            scene_embeds_expanded = scene_embeds.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, seq_len, -1)
            # 拼接所有特征: PAA + State + Scene
            x = torch.cat([x, scene_embeds_expanded], dim=-1)  # x shape: [B, N, S, F_paa + F_state + F_scene]

            # --- 2. 构建大图批处理所需的数据结构 ---
            # x shape: [B, N, S, F_total]
            B, N, S, F_in = x.shape
            BS = B * S

            # a. 展平节点特征: [B, N, S, F_in] -> [B, S, N, F_in] -> [BS, N, F_in] -> [BS*N, F_in]
            x_flat = x.permute(0, 2, 1, 3).reshape(BS * N, F_in)

            # b. 扩展边索引 (edge_index) 和边权重 (edge_weight)
            E = edge_index.size(1)
            # 计算偏移量: [0, N, 2N, ..., (BS-1)*N]
            offset = torch.arange(0, BS, device=device).view(-1, 1) * N
            # [2, E] -> [1, 2, E] -> [BS, 2, E]
            edge_index_expanded = edge_index.unsqueeze(0).expand(BS, -1, -1)
            # [BS, 2, E] + [BS, 1, 1] -> [BS, 2, E]
            edge_index_expanded = edge_index_expanded + offset.view(-1, 1, 1)
            # [BS, 2, E] -> [2, BS*E]
            edge_index_expanded = edge_index_expanded.view(2, BS * E)
            # 权重直接重复BS次
            edge_weight_expanded = edge_weight.repeat(BS)
            # c. 扩展 PageRank 分数
            pagerank_expanded = pagerank_scores.repeat(BS)  # [N] -> [BS*N]

            # --- 3. 在大图上执行多层图卷积 ---
            x_processed_flat = x_flat  # GCN 的初始输入
            # 循环遍历模型中定义的每一层图卷积网络
            # 实际的调用网络在init中，不在泽丽，这里只负责谁使用什么权重
            for conv_layer, norm_layer, proj_layer in zip(self.convs, self.norms, self.residual_projections):
                residual = x_processed_flat  # 当前层的输入，用于残差连接
                # --- 根据不同的卷积类型，使用不同的参数调用 ---
                if self.gcn_type == 'CGC':
                    # 自己创新的模型：使用图结构、权重和中心性
                    x_after_conv = conv_layer(
                        x_processed_flat,
                        edge_index_expanded,
                        # edge_weight_expanded代表边的权重； pagerank_expanded是节点的权重，两个层面
                        edge_weight_expanded,  # 确保CGCConv接收edge_weight
                        pagerank_expanded
                    )
                elif self.gcn_type == 'GCN':
                    # GCN：使用图结构和权重
                    x_after_conv = conv_layer(x_processed_flat, edge_index_expanded, edge_weight=edge_weight_expanded)
                elif self.gcn_type == 'GAT' or self.gcn_type == 'SAGE':
                    # GATConv 通常不直接使用数值权重，它自己学习注意力权重
                    x_after_conv = conv_layer(x_processed_flat, edge_index_expanded)
                elif self.gcn_type == 'APPNP':
                    # APPNP: 使用包装类，不需要edge_weight
                    x_after_conv = conv_layer(x_processed_flat, edge_index_expanded)
                elif self.gcn_type == 'GIN':
                    # GIN: Graph Isomorphism Network，不需要edge_weight
                    x_after_conv = conv_layer(x_processed_flat, edge_index_expanded)
                else:  # 兜底，以防万一
                    raise ValueError(f"Unknown gcn_type in hgc_stp_forward: {self.gcn_type}")
                # 应用层归一化
                x_main = norm_layer(x_after_conv)
                # 处理残差连接：如果维度不匹配，进行投影
                residual_proj = proj_layer(residual)
                # ReLU激活并加入残差
                x_processed_flat = F.relu(x_main + residual_proj)

            # --- 4. 恢复数据形状 ---
            # [BS*N, H] -> [B, S, N, H] -> [B, N, S, H]
            x_out = x_processed_flat.view(B, S, N, -1).permute(0, 2, 1, 3)
        return x_out

    def forward(self, x_batch, state_batch):
        """
        forward方法现在只负责使用已有的图进行推理。
        它不再接收 A_physical 或 x_full_train。
        """
        # --- 1. HGC-STP / 无图特征提取 ---
        if self.gcn_type == 'NONE':
            # 无图模式：直接将 PAA (+ 可选 state) 变换成 [B, N, S, H] 作为 ITR 输入
            batch_size, seq_len, _ = x_batch.shape
            x_hgc = x_batch.view(batch_size, seq_len, self.num_nodes, self.num_paa_features).permute(0, 2, 1, 3)  # [B,N,S,F_paa]
            if self.num_state_features > 0:
                state_features_expanded = state_batch.unsqueeze(1).unsqueeze(1).expand(-1, self.num_nodes, seq_len, -1)
                x_hgc = torch.cat([x_hgc, state_features_expanded], dim=-1)  # [B,N,S,F_paa+F_state]
            # 线性投影到 hgc_output_dim，使其与 ITR 的 d_model/nhead 兼容
            x_hgc = self.no_graph_projection(x_hgc)  # [B,N,S,H]
        else:
            # 这意味着图结构（A_final和pagerank）在每个epoch的第一个训练批次被构建一次，然后在该epoch的剩余训练和整个验证过程中保持不变并被复用。这既能让图结构随训练演进，又保证了计算效率。
            # model.train() → 将self.training设为True
            if self.A_physical is None or self.cached_A_learned is None:
                raise RuntimeError("Graph not built. Call _build_and_cache_graph first.")
            # --- HGC-STP 特征提取 ---
            x_hgc = self.hgc_stp_forward(x_batch, state_batch)  # -> [B, N, S, H]

        # --- 2. 根据 ITR 类型选择后续路径 ---
        # [路径A: 您的ITR模块 (Dynamic 或 Fixed)]
        if self.itr_type == 'Dynamic_ITR' or self.itr_type == 'Dynamic_ITR_CE':
            # 节点信息聚合层,将单独的传感器都聚合在了一起，准备输送给迭代推理模块
            x_itr_in = x_hgc.mean(dim=1)  # 聚合节点 -> [B, S, H_hgc]

            # 训练模式
            # `self.training` 是 nn.Module 自带的属性，model.train()时为True, model.eval()时为False
            # 因为动态门控，训练跑满，验证提前退出，故区分了训练与验证
            if self.training:
                # Dynamic_ITR_CE: 迭代5次，但只使用最后一次迭代的logits计算单一CELoss
                if self.itr_type == 'Dynamic_ITR_CE':
                    # 手动展开迭代过程，以便计算每次迭代的置信度分数
                    context_stream = x_itr_in.clone()
                    reasoning_stream = x_itr_in.clone()
                    confidence_scores = []
                    
                    for i in range(self.itr_module.num_iterations):
                        context_stream, reasoning_stream = self.itr_module.reasoning_blocks[i](context_stream, reasoning_stream)
                        # 计算置信度分数（用于可视化，不用于损失计算）
                        gate_input = reasoning_stream[:, -1, :]
                        confidence = torch.sigmoid(self.itr_module.dynamic_gate(gate_input))
                        confidence_scores.append(confidence)
                    
                    # 使用最后一次迭代的结果计算logits
                    stream_flat = reasoning_stream.reshape(reasoning_stream.shape[0], -1)
                    final_logits = self.shared_classifier(stream_flat)
                    # 返回格式: [final_logits], confidence_scores（用于可视化）
                    return [final_logits], confidence_scores
                # Dynamic_ITR: 动态门控模式下的训练模式，全部跑满
                # 模块三的输出（intermediate_streams）依然是一个列表，其中每个元素的形状都是 [Batch_Size, 90, Hidden_Dim]。它保留了完整的时间序列信息，只是对表示进行了精炼
                intermediate_streams, confidence_scores = self.itr_module(x_itr_in)
                # --- 4. 模块四: 动态决策 (获取中间预测) ---
                # intermediate_logits（一个列表，每个元素形状为[Batch_Size, Num_Classes]）
                intermediate_logits = []
                for stream in intermediate_streams:
                    # 将整个时间序列的表示展平送入分类器
                    # reshape(..., -1) 这个操作，将整个长度为90的时间序列的所有信息**"压平"成一个单一的、超长的高维向量**。
                    # 分类器是在这个融合了90个时间点所有信息的向量上进行操作的。
                    # 因此，对于每一个样本（即每一个长度为90的序列），分类器只输出一个预测结果（logits）。
                    stream_flat = stream.reshape(stream.shape[0], -1)
                    # [Batch_Size, Num_Classes]
                    logits = self.shared_classifier(stream_flat)
                    intermediate_logits.append(logits)
                return intermediate_logits, confidence_scores
            # 评估模式：提前退出和跑满5次两种
            else:
                # ITR模块的逻辑需要在这里手动展开以实现提前退出
                # [路径3.1: 评估完整DGS模型]
                if self.enable_early_exit:
                    context_stream = x_itr_in.clone()
                    reasoning_stream = x_itr_in.clone()

                    batch_size = x_batch.shape[0]
                    # 初始化用于存储最终结果的张量
                    final_logits = torch.zeros(batch_size, self.num_classes, device=x_itr_in.device)
                    # 跟踪哪些样本已经完成推理
                    finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=x_itr_in.device)
                    # 创建一个张量来记录每个样本的退出迭代次数
                    # 初始化为0，表示尚未退出
                    exit_iterations = torch.zeros(batch_size, dtype=torch.long, device=x_itr_in.device)

                    last_iter_logits = None  # 用于存储最后一次的logits
                    for i in range(self.itr_module.num_iterations):
                        # 调用 ITR 模块的内部组件
                        context_stream, reasoning_stream = self.itr_module.reasoning_blocks[i](context_stream, reasoning_stream)
                        # 计算置信度
                        gate_input = reasoning_stream[:, -1, :]
                        confidence = torch.sigmoid(self.itr_module.dynamic_gate(gate_input))
                        # 计算 logits
                        stream_flat = reasoning_stream.reshape(reasoning_stream.shape[0], -1)
                        logits = self.shared_classifier(stream_flat)
                        last_iter_logits = logits  # 始终更新

                        # 找出本次迭代中，首次达到阈值的样本
                        newly_finished_mask = (confidence.squeeze(-1) >= self.early_exit_threshold) & (~finished_mask)

                        if newly_finished_mask.any():
                            final_logits[newly_finished_mask] = logits[newly_finished_mask]
                            finished_mask |= newly_finished_mask
                            exit_iterations[newly_finished_mask] = i + 1

                        # 如果所有样本都已完成，则中断整个循环
                        if finished_mask.all():
                            break

                    # 处理直到最后都未达到阈值的样本
                    unfinished_mask = ~finished_mask
                    if unfinished_mask.any():
                        # 对于这些样本，我们采用最后一次迭代的结果
                        # `logits` 变量此时保存的就是最后一次迭代的结果
                        final_logits[unfinished_mask] = logits[unfinished_mask]
                        # [修改] 对于未提前退出的样本，其退出迭代次数为总迭代次数
                        exit_iterations[unfinished_mask] = self.itr_module.num_iterations

                    # 评估时，返回一个确定的 logits 张量和 None
                    return final_logits, exit_iterations

                # --- 路径 3.1.2: 禁用提前退出 (跑满所有迭代) ---
                else:  # not self.enable_early_exit
                    # 调用固定深度5次的前向传播
                    final_stream = self.itr_module.forward_fixed_depth(x_itr_in)
                    stream_flat = final_stream.reshape(final_stream.shape[0], -1)
                    final_logits = self.shared_classifier(stream_flat)

                    # 在这种模式下，所有样本的退出次数都是最大迭代次数
                    batch_size = x_batch.shape[0]
                    exit_iterations = torch.full((batch_size,), self.itr_module.num_iterations, dtype=torch.long,
                                                 device=x_batch.device)
                    return final_logits, exit_iterations
        # --- 路径 3.2: 评估 Dynamic_ITR_CE, Single_ITR, GRU, MLP 模型 ---
        # Dynamic_ITR_CE: 迭代5次，但评估时和训练时一样，只使用最后一次迭代的结果
        # 但在验证时，为了收集退出迭代次数用于可视化，也需要实现提前退出逻辑
        elif self.itr_type == 'Dynamic_ITR_CE':
            x_itr_in = x_hgc.mean(dim=1)
            # 如果启用了提前退出，需要手动展开迭代过程以收集退出迭代次数
            if self.enable_early_exit:
                context_stream = x_itr_in.clone()
                reasoning_stream = x_itr_in.clone()

                batch_size = x_batch.shape[0]
                # 初始化用于存储最终结果的张量
                final_logits = torch.zeros(batch_size, self.num_classes, device=x_itr_in.device)
                # 跟踪哪些样本已经完成推理
                finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=x_itr_in.device)
                # 创建一个张量来记录每个样本的退出迭代次数
                exit_iterations = torch.zeros(batch_size, dtype=torch.long, device=x_itr_in.device)

                for i in range(self.itr_module.num_iterations):
                    # 调用 ITR 模块的内部组件
                    context_stream, reasoning_stream = self.itr_module.reasoning_blocks[i](context_stream, reasoning_stream)
                    # 计算置信度
                    gate_input = reasoning_stream[:, -1, :]
                    confidence = torch.sigmoid(self.itr_module.dynamic_gate(gate_input))
                    # 计算 logits
                    stream_flat = reasoning_stream.reshape(reasoning_stream.shape[0], -1)
                    logits = self.shared_classifier(stream_flat)

                    # 找出本次迭代中，首次达到阈值的样本
                    newly_finished_mask = (confidence.squeeze(-1) >= self.early_exit_threshold) & (~finished_mask)

                    if newly_finished_mask.any():
                        final_logits[newly_finished_mask] = logits[newly_finished_mask]
                        finished_mask |= newly_finished_mask
                        exit_iterations[newly_finished_mask] = i + 1

                    # 如果所有样本都已完成，则中断整个循环
                    if finished_mask.all():
                        break

                # 处理直到最后都未达到阈值的样本
                unfinished_mask = ~finished_mask
                if unfinished_mask.any():
                    # 对于这些样本，我们采用最后一次迭代的结果
                    final_logits[unfinished_mask] = logits[unfinished_mask]
                    exit_iterations[unfinished_mask] = self.itr_module.num_iterations

                return final_logits, exit_iterations
            else:
                # 禁用提前退出时，调用固定深度5次的前向传播
                final_stream = self.itr_module.forward_fixed_depth(x_itr_in)
                stream_flat = final_stream.reshape(final_stream.shape[0], -1)
                final_logits = self.shared_classifier(stream_flat)
                # 返回格式与 Single_ITR 一致
                batch_size = x_batch.shape[0]
            exit_iterations = torch.full((batch_size,), self.itr_module.num_iterations, dtype=torch.long,
                                         device=x_batch.device)
            return final_logits, exit_iterations
        # --- 路径 3.3: 评估 Single_ITR, GRU, MLP 模型 ---
        # 对于这些消融模型，评估行为与训练行为一致
        # 它们没有“提前退出”的概念
        # Single_ITR为训练阶段只跑一次推理，使用CEloss做损失函数，用于对比跑一次CE和5次迭代的区别
        elif self.itr_type == 'Single_ITR':
            x_itr_in = x_hgc.mean(dim=1)
            final_stream = self.itr_module.forward_single_step(x_itr_in)
            stream_flat = final_stream.reshape(final_stream.shape[0], -1)
            final_logits = self.shared_classifier(stream_flat)
        elif self.itr_type == 'GRU':
            x_agg = x_hgc.mean(dim=(1, 2))
            gru_in = x_agg.unsqueeze(1)
            output, _ = self.gru_module(gru_in)
            final_features = output.squeeze(1)
            final_logits = self.shared_classifier(final_features)
        elif self.itr_type == 'LSTM':
            x_agg = x_hgc.mean(dim=(1, 2))
            lstm_in = x_agg.unsqueeze(1)
            output, _ = self.lstm_module(lstm_in)
            final_features = output.squeeze(1)
            final_logits = self.shared_classifier(final_features)
        elif self.itr_type == 'MLP':
            x_agg = x_hgc.mean(dim=(1, 2))
            final_features = self.mlp_module(x_agg)
            final_logits = self.shared_classifier(final_features)
        elif self.itr_type == 'Transformer':
            # 聚合节点维度，保留时间序列维度: [B, N, S, H] -> [B, S, H]
            x_seq = x_hgc.mean(dim=1)  # [B, S, H]
            # 投影到transformer的hidden_dim（如果需要）
            x_seq = self.transformer_projection(x_seq)  # [B, S, transformer_hidden_dim]
            # Transformer编码: [B, S, H] -> [B, S, H]
            transformer_out = self.transformer_module(x_seq)  # [B, S, transformer_hidden_dim]
            # 取最后一个时间步的特征
            final_features = transformer_out[:, -1, :]  # [B, transformer_hidden_dim]
            final_logits = self.shared_classifier(final_features)
        else:
            print(f"  - 不存在此ITR模式，请检查！！！")
            return None
        # 对于这些模型，exit_iters 总是 None
        return [final_logits], None

    def eval_forward_with_confidence(self, x_batch, state_batch):
        """
        仅在评估阶段使用，用于同时返回迭代置信度序列。
        适用于 Dynamic_ITR / Dynamic_ITR_CE，其它模式退化为普通 forward。
        返回: final_logits, exit_iterations(or None), confidence_scores(list[Tensor])
        """
        # --- 1. HGC-STP / 无图特征提取 ---
        if self.gcn_type == 'NONE':
            batch_size, seq_len, _ = x_batch.shape
            x_hgc = x_batch.view(batch_size, seq_len, self.num_nodes, self.num_paa_features).permute(0, 2, 1, 3)
            if self.num_state_features > 0:
                state_features_expanded = state_batch.unsqueeze(1).unsqueeze(1).expand(-1, self.num_nodes, seq_len, -1)
                x_hgc = torch.cat([x_hgc, state_features_expanded], dim=-1)
            x_hgc = self.no_graph_projection(x_hgc)
        else:
            if self.A_physical is None or self.cached_A_learned is None:
                raise RuntimeError("Graph not built. Call _build_and_cache_graph first.")
            x_hgc = self.hgc_stp_forward(x_batch, state_batch)  # -> [B, N, S, H]

        # 仅对 Dynamic_ITR / Dynamic_ITR_CE 收集迭代置信度
        if self.itr_type not in ['Dynamic_ITR', 'Dynamic_ITR_CE']:
            logits, exit_iters = self.forward(x_batch, state_batch)
            return logits, exit_iters, None

        x_itr_in = x_hgc.mean(dim=1)
        confidence_scores = []

        if self.enable_early_exit:
            context_stream = x_itr_in.clone()
            reasoning_stream = x_itr_in.clone()
            batch_size = x_batch.shape[0]
            final_logits = torch.zeros(batch_size, self.num_classes, device=x_itr_in.device)
            finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=x_itr_in.device)
            exit_iterations = torch.zeros(batch_size, dtype=torch.long, device=x_itr_in.device)
            last_logits = None

            for i in range(self.itr_module.num_iterations):
                context_stream, reasoning_stream = self.itr_module.reasoning_blocks[i](context_stream, reasoning_stream)
                gate_input = reasoning_stream[:, -1, :]
                confidence = torch.sigmoid(self.itr_module.dynamic_gate(gate_input))
                confidence_scores.append(confidence)

                stream_flat = reasoning_stream.reshape(reasoning_stream.shape[0], -1)
                logits = self.shared_classifier(stream_flat)
                last_logits = logits

                newly_finished_mask = (confidence.squeeze(-1) >= self.early_exit_threshold) & (~finished_mask)
                if newly_finished_mask.any():
                    final_logits[newly_finished_mask] = logits[newly_finished_mask]
                    finished_mask |= newly_finished_mask
                    exit_iterations[newly_finished_mask] = i + 1

                if finished_mask.all():
                    break

            unfinished_mask = ~finished_mask
            if unfinished_mask.any():
                final_logits[unfinished_mask] = last_logits[unfinished_mask]
                exit_iterations[unfinished_mask] = self.itr_module.num_iterations

            return final_logits, exit_iterations, confidence_scores

        # 禁用提前退出：跑满迭代并记录置信度
        context_stream = x_itr_in.clone()
        reasoning_stream = x_itr_in.clone()
        for i in range(self.itr_module.num_iterations):
            context_stream, reasoning_stream = self.itr_module.reasoning_blocks[i](context_stream, reasoning_stream)
            gate_input = reasoning_stream[:, -1, :]
            confidence = torch.sigmoid(self.itr_module.dynamic_gate(gate_input))
            confidence_scores.append(confidence)
        stream_flat = reasoning_stream.reshape(reasoning_stream.shape[0], -1)
        final_logits = self.shared_classifier(stream_flat)
        exit_iterations = torch.full((x_batch.shape[0],), self.itr_module.num_iterations, dtype=torch.long, device=x_batch.device)
        return final_logits, exit_iterations, confidence_scores
# ==============================================================================
# 迭代式时序推理模块 (Iterative Temporal Reasoning, ITR)
# ==============================================================================
class ReasoningBlock(nn.Module):
    """ITR模块中的一个迭代推理块"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # 交叉注意力单元
        self.cross_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 前馈网络
        self.ffn1 = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        self.ffn2 = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        # 层归一化和残差
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_stream, reasoning_stream):
        # 1. 交叉注意力 1 (Q=Context, K/V=Reasoning)
        residual = context_stream
        attn_out, _ = self.cross_attn_1(query=context_stream, key=reasoning_stream, value=reasoning_stream)
        context_stream = self.norm1(residual + self.dropout(attn_out))
        # FFN
        residual = context_stream
        context_stream = self.norm2(residual + self.dropout(self.ffn1(context_stream)))

        # 2. 交叉注意力 2 (Q=Reasoning, K/V=Updated Context)
        residual = reasoning_stream
        attn_out, _ = self.cross_attn_2(query=reasoning_stream, key=context_stream, value=context_stream)
        reasoning_stream = self.norm3(residual + self.dropout(attn_out))
        # FFN
        residual = reasoning_stream
        reasoning_stream = self.norm4(residual + self.dropout(self.ffn2(reasoning_stream)))

        return context_stream, reasoning_stream

class IterativeTemporalReasoning(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_iterations):
        super().__init__()
        self.num_iterations = num_iterations
        # 模型结构上等价于一个多层堆叠的可重复推理层。
        self.reasoning_blocks = nn.ModuleList([
            ReasoningBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_iterations)
        ])
        # 动态推理门
        # 将每个样本的隐状态[H]转化为一个标量置信度值，配合 sigmoid 输出[0, 1]区间的动态权重
        self.dynamic_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x shape: [B, S, H]

        # 双流初始化
        # .clone() 的作用是创建数据副本，防止对原 x 的原地修改影响后续计算
        context_stream = x.clone()
        reasoning_stream = x.clone()

        intermediate_reasoning_streams = []
        confidence_scores = []

        for i in range(self.num_iterations):
            context_stream, reasoning_stream = self.reasoning_blocks[i](context_stream, reasoning_stream)
            # 使用推理流的最后一个时间步的表示来计算置信度
            gate_input = reasoning_stream[:, -1, :]
            confidence = torch.sigmoid(self.dynamic_gate(gate_input))

            intermediate_reasoning_streams.append(reasoning_stream)
            confidence_scores.append(confidence)

        return intermediate_reasoning_streams, confidence_scores

    def forward_single_step(self, x):
        """
        一个专门用于CE消融实验的新方法，只执行一次推理步骤。
        这代表了一个标准的、非迭代的强基线。
        """
        # x shape: [B, S, H]
        context_stream = x.clone()
        reasoning_stream = x.clone()

        # 只调用第一个推理块，执行一次推理
        _, final_stream = self.reasoning_blocks[0](context_stream, reasoning_stream)
        # 直接返回这次推理的结果
        return final_stream

    def forward_fixed_depth(self, x):
        context_stream = x.clone()
        reasoning_stream = x.clone()
        for i in range(self.num_iterations):
            context_stream, reasoning_stream = self.reasoning_blocks[i](context_stream, reasoning_stream)
        return reasoning_stream