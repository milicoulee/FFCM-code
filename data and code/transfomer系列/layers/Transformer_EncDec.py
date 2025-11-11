import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x
   
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        """
        x: [B, N, F], 输入特征 (Batch, 节点数, 特征维度)
        adj: [B, N, N], 邻接矩阵 (Batch, 节点数, 节点数)
        """
        # 调试信息：输出邻接矩阵 adj 的形状
        # print(f"adj shape: {adj.shape}")

        # 邻接矩阵归一化 (加 I 矩阵，确保自环)
        I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)  # [1, N, N]
        adj = adj + I
        D = torch.diag_embed(torch.sum(adj, dim=-1).pow(-0.5))  # 计算度矩阵 D^-0.5
        adj_normalized = torch.bmm(torch.bmm(D, adj), D)  # D^-0.5 * A * D^-0.5

        # 图卷积计算
        out = torch.bmm(adj_normalized, x)  # [B, N, F]
        out = self.linear(out)             # [B, N, out_features]
        return out
    
class DeepGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_layers=3):
        super(DeepGCNLayer, self).__init__()
        self.layers = nn.ModuleList([
            GCNLayer(in_features if i == 0 else out_features, out_features)
            for i in range(num_layers)
        ])

    def forward(self, x, adj):
        """
        x: [B, N, F] 输入特征 (Batch, 节点数, 特征维度)
        adj: [B, N, N] 邻接矩阵
        """
        for gcn in self.layers:
            x = gcn(x, adj) + x  # 跳跃连接
        return x

#%%FNN+GCN1
# class EncoderLayer(nn.Module):  
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention

#         # GCN 部分
#         self.gcn1 = GCNLayer(d_model, d_ff)  # 第一层 GCN
#         self.gcn2 = GCNLayer(d_ff, d_model) # 第二层 GCN

#         # FNN 部分 (使用 Conv1d 实现)
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

#         # 归一化与 Dropout
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         # 注意力机制
#         new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
#         x = x + self.dropout(new_x)

#         # 图卷积部分 (GCN)
#         adj = attn.mean(dim=1)  # 合并多头邻接矩阵为 [B, L, L]
#         y_gcn = x
#         y_gcn = self.dropout(self.activation(self.gcn1(y_gcn, adj)))  # 第一层 GCN
#         y_gcn = self.dropout(self.gcn2(y_gcn, adj))                   # 第二层 GCN

#         # FNN 部分 (Conv1d 实现)
#         y_ffn = x.transpose(-1, 1)  # 转换为 [B, D, L]
#         y_ffn = self.dropout(self.activation(self.conv1(y_ffn)))
#         y_ffn = self.dropout(self.conv2(y_ffn)).transpose(-1, 1)  # 转换回 [B, L, D]

#         # 融合 GCN 和 FNN
#         y = y_gcn + y_ffn  # 直接相加或其他融合方式

#         # 残差连接与归一化
#         return self.norm3(x + y), attn

#%%FNN+deepGCN
# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, num_layers=3, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention

#         # Deep GCN 部分
#         self.deep_gcn = DeepGCNLayer(d_model, d_ff, num_layers=num_layers)

#         # FNN 部分 (Conv1d 实现)
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

#         # 归一化与 Dropout
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     # def forward(self, x, attn_mask=None, tau=None, delta=None):
#     #     # 注意力机制
#     #     new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)

#     #     # 邻接矩阵处理
#     #     adj = attn.mean(dim=1)  # 合并多头邻接矩阵为 [B, L, L]

#     #     # Deep GCN 部分
#     #     y_gcn = self.deep_gcn(x, adj)  # 使用 Deep GCN 处理

#     #     # FNN 部分 (Conv1d 实现)
#     #     y_ffn = x.transpose(-1, 1)  # 转换为 [B, D, L]
#     #     y_ffn = self.dropout(self.activation(self.conv1(y_ffn)))
#     #     y_ffn = self.dropout(self.conv2(y_ffn)).transpose(-1, 1)  # 转换回 [B, L, D]

#     #     # 融合 Deep GCN 和 FNN
#     #     y = y_gcn + y_ffn  # 可以尝试其他融合方式，例如拼接或加权融合

#     #     # 残差连接与归一化
#     #     x = x + self.dropout(new_x)  # 加入注意力的残差
#     #     return self.norm3(x + y), attn
    
#     def forward(self, x, attn_mask=None, tau=None, delta=None): ## 单独使用GCN 4层
#         """
#         x: [B, L, D], 输入特征 (Batch, 序列长度, 特征维度)
#         attn_mask: 注意力掩码
#         """
#         # 注意力机制，获得注意力输出和邻接矩阵
#         new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
#         adj = attn.mean(dim=1)  # 合并多头邻接矩阵为 [B, L, L]

#         # Deep GCN 部分
#         y_gcn = self.deep_gcn(x, adj)  # GCN 输出 [B, L, D]

#         # 残差连接与归一化
#         x = x + self.dropout(new_x)  # 注意力残差连接
#         return self.norm2(x + y_gcn), attn  # GCN 融合残差连接后的输出

#%%FNN+GAT
# from torch_geometric.nn import GATConv
# import torch.nn.functional as F

# class GATModule(nn.Module):
#     def __init__(self, d_model, d_ff, heads=4, dropout=0.1):
#         super(GATModule, self).__init__()
#         self.gat1 = GATConv(d_model, d_ff // heads, heads=heads, concat=True, dropout=dropout)
#         self.gat2 = GATConv(d_ff, d_model // heads, heads=heads, concat=True, dropout=dropout)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, edge_index):
#         """
#         x: [B, N, D], 输入特征 (Batch, 节点数, 特征维度)
#         edge_index: [2, E], 边索引，用于 GAT
#         """
#         B, N, D = x.size()

#         # 将 [B, N, D] 展平为 [B * N, D]
#         x = x.view(-1, D)

#         # 复制边索引以匹配批次大小
#         edge_index = edge_index.repeat(B, 1)

#         # GAT 前向传播
#         x = self.gat1(x, edge_index)  # 第一层 GAT
#         x = F.relu(x)
#         x = self.dropout(self.gat2(x, edge_index))  # 第二层 GAT

#         # 恢复形状为 [B, N, D]
#         return x.view(B, N, D)
    
# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, heads=4, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model

#         self.attention = attention
#         self.gat = GATModule(d_model, d_ff, heads=heads, dropout=dropout)

#         # FNN 部分
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

#         # 残差连接和归一化
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, edge_index, attn_mask=None, tau=None, delta=None):
#         """
#         x: [B, N, D], 输入特征 (Batch, 节点数, 特征维度)
#         edge_index: [2, E], 图的边索引，用于 GAT
#         """
#         # 注意力机制部分
#         new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
#         x = x + self.dropout(new_x)  # 残差连接

#         # GAT 部分
#         y_gat = self.gat(x, edge_index)  # [B, N, D]

#         # FNN 部分
#         y_ffn = x.transpose(-1, 1)  # [B, N, D] -> [B, D, N]
#         y_ffn = self.dropout(self.activation(self.conv1(y_ffn)))
#         y_ffn = self.dropout(self.conv2(y_ffn)).transpose(-1, 1)  # [B, D, N] -> [B, N, D]

#         # 融合 GAT 和 FNN
#         y = y_gat + y_ffn  # 可以尝试其他融合方式

#         # 残差连接与归一化
#         x = self.norm1(x + y_gat)  # GAT 的残差连接
#         y = self.norm2(y + y_ffn)  # FNN 的残差连接
#         return self.norm3(x + y), attn

#%% DepthwiseSeparableConv
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.1, activation='relu'):
#         super(DepthwiseSeparableConv, self).__init__()
#         # 深度卷积（每个通道独立卷积）
#         self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
#                                    padding=padding, groups=in_channels, bias=False)
#         # 逐点卷积（线性组合深度卷积的输出）
#         self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.activation(x)
#         x = self.dropout(x)
#         x = self.pointwise(x)
#         return x

# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
        
#         # 使用深度可分离卷积替换原有的1x1卷积
#         self.conv1 = DepthwiseSeparableConv(d_model, d_ff, kernel_size=3, padding=1, dropout=dropout, activation=activation)
#         self.conv2 = DepthwiseSeparableConv(d_ff, d_model, kernel_size=3, padding=1, dropout=dropout, activation=activation)
        
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None, tau=None, delta=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask,
#             tau=tau, delta=delta
#         )
#         x = x + self.dropout(new_x)
#         y = x = self.norm1(x)

#         # 注意这里与原代码不同的是：conv1/conv2需要输入为 (B, C, L) 的格式 
#         # 原代码 transpose(-1, 1) 适用于 Conv1d 输入格式
#         y = y.transpose(-1, 1)   # (B, L, D) -> (B, D, L)
#         y = self.conv1(y)
#         y = self.conv2(y)
#         y = y.transpose(-1, 1)   # (B, D, L) -> (B, L, D)

#         return self.norm2(x + y), attn


#%% FNN
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        
        # print(x.shape)
        # print(y.transpose(-1, 1).shape)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn





class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
