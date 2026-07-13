"""
TransformerDock — 蛋白质-蛋白质对接评分函数
基于 DeepDock (Nature MI 2021) 改造，加入 Transformer 组件。

架构：
  SurfaceEncoder (共享权重)
    → 3x MetaLayer (局部几何)
    → N x TransformerResBlock (中远程注意力)
    → GlobalSelfAttention (全链上下文)
  GeoBiasedCrossAttention (双向跨链融合 + 3D距离偏置)
  MDN Head (混合密度网络，预测距离分布)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_mean
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MetaLayer, TransformerConv

NO_INTERFACE_SCORE = -1e6


# ─────────────────────────────────────────────────────────────
# 基础组件：EdgeModel / NodeModel（MetaLayer 用）
# ─────────────────────────────────────────────────────────────

class EdgeModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ELU(),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        return self.mlp(torch.cat([src, dest, edge_attr], dim=1))


class NodeModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ELU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ELU(),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        agg = scatter_mean(
            self.mlp1(torch.cat([x[row], edge_attr], dim=1)),
            col, dim=0, dim_size=x.size(0)
        )
        return self.mlp2(torch.cat([x, agg], dim=1))


# ─────────────────────────────────────────────────────────────
# TransformerResBlock
# 用 TransformerConv 替换 MetaLayer，保留残差结构
# ─────────────────────────────────────────────────────────────

class TransformerResBlock(nn.Module):
    """
    图 Transformer 残差块。
    TransformerConv 对每条边计算注意力权重：
        α_ij = softmax_j( (W_Q x_i)^T (W_K x_j + W_E e_ij) / √d )
        x_i' = Σ_j α_ij · W_V x_j
    """
    def __init__(self, in_channels, heads=4, dropout_rate=0.15):
        super().__init__()
        assert in_channels % heads == 0, "in_channels 必须能被 heads 整除"
        head_dim = in_channels // heads

        self.norm1_x = nn.LayerNorm(in_channels)
        self.norm1_e = nn.LayerNorm(in_channels)

        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=head_dim,
            heads=heads,
            edge_dim=in_channels,
            dropout=dropout_rate,
            concat=True,
            beta=True,
        )

        # 边特征手动更新（TransformerConv 不更新边）
        self.edge_update = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels),
            nn.LayerNorm(in_channels),
            nn.ELU(),
        )

        self.norm2_x = nn.LayerNorm(in_channels)
        self.norm2_e = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 节点更新（Pre-LN + 残差）
        h = self.norm1_x(x)
        h = self.conv(h, edge_index, self.norm1_e(edge_attr))
        data.x = x + self.dropout(h)

        # 边更新（Pre-LN + 残差）
        row, col = edge_index
        h_e = self.edge_update(
            torch.cat([self.norm2_e(edge_attr), data.x[row], data.x[col]], dim=1)
        )
        data.edge_attr = edge_attr + self.dropout(h_e)

        return data


# ─────────────────────────────────────────────────────────────
# GlobalSelfAttention
# 对 dense batch 做标准 Transformer Self-Attention
# ─────────────────────────────────────────────────────────────

class GlobalSelfAttention(nn.Module):
    """
    让每个表面节点感知整条链的全局信息。
    输入/输出均为 dense batch [B, N_max, C]。
    """
    def __init__(self, hidden_dim=128, nhead=8, num_layers=2, dropout_rate=0.15):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,   # Pre-LN，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x_dense, mask):
        """
        x_dense : [B, N, C]
        mask     : [B, N]  True = 有效节点
        return   : [B, N, C]
        """
        return self.encoder(x_dense, src_key_padding_mask=~mask)


# ─────────────────────────────────────────────────────────────
# GeoBiasedCrossAttention
# 双向 Cross-Attention + 3D 几何距离偏置
# ─────────────────────────────────────────────────────────────

class GeoBiasedCrossAttention(nn.Module):
    """
    A→B 和 B→A 双向 Cross-Attention。
    注意力分数加入 RBF 编码的 3D 距离偏置，使模型天然偏向空间近邻。

    输出：
        h_a_new  : [B, N_a, C]  A 被 B 更新后的特征
        h_b_new  : [B, N_b, C]  B 被 A 更新后的特征
        pair_feat: [M, C]       有效节点对的融合特征（M = |valid pairs|）
        pair_mask: [B, N_a, N_b]
    """
    def __init__(self, hidden_dim=128, nhead=8, n_rbf=16,
                 rbf_max=20.0, dropout_rate=0.15):
        super().__init__()
        assert hidden_dim % nhead == 0
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead
        self.scale = self.head_dim ** -0.5

        # A→B
        self.q_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # B→A
        self.q_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # RBF 距离编码 → 每个 head 的标量偏置
        centers = torch.linspace(0, rbf_max, n_rbf)
        self.register_buffer('rbf_centers', centers)
        self.register_buffer('rbf_width', torch.ones(n_rbf) * (rbf_max / n_rbf))
        self.dist_bias_proj = nn.Linear(n_rbf, nhead, bias=False)

        # 输出投影 + LayerNorm（残差后）
        self.out_a = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.out_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

        # pair 特征 MLP
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
        )

        self.dropout = nn.Dropout(p=dropout_rate)

    def _rbf(self, dist_mat):
        """dist_mat [B,Na,Nb] → bias [B,Na,Nb,nhead]"""
        d = dist_mat.unsqueeze(-1)                                   # [B,Na,Nb,1]
        c = self.rbf_centers.view(1, 1, 1, -1)
        w = self.rbf_width.view(1, 1, 1, -1)
        rbf = torch.exp(-((d - c) ** 2) / (2 * w ** 2))             # [B,Na,Nb,n_rbf]
        return self.dist_bias_proj(rbf)                              # [B,Na,Nb,nhead]

    def _cross_attn(self, Q, K, V, kv_mask, dist_bias=None):
        """
        Q  : [B, Nq, C]
        K,V: [B, Nkv, C]
        kv_mask: [B, Nkv]  True=有效
        dist_bias: [B, Nq, Nkv, nhead] 或 None
        return: [B, Nq, C]
        """
        B, Nq, C = Q.shape
        Nkv = K.shape[1]
        H, D = self.nhead, self.head_dim

        Q = Q.view(B, Nq,  H, D).permute(0, 2, 1, 3)   # [B,H,Nq,D]
        K = K.view(B, Nkv, H, D).permute(0, 2, 1, 3)   # [B,H,Nkv,D]
        V = V.view(B, Nkv, H, D).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale     # [B,H,Nq,Nkv]

        if dist_bias is not None:
            attn = attn + dist_bias.permute(0, 3, 1, 2)              # [B,H,Nq,Nkv]

        if kv_mask is not None:
            pad = (~kv_mask).unsqueeze(1).unsqueeze(2)               # [B,1,1,Nkv]
            attn = attn.masked_fill(pad, -1e9)

        attn = F.softmax(attn, dim=-1)
        if kv_mask is not None:
            attn = attn.masked_fill(pad, 0.0)
            denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            attn = attn / denom
        attn = self.dropout(torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0))
        out = torch.matmul(attn, V)                                   # [B,H,Nq,D]
        return out.permute(0, 2, 1, 3).reshape(B, Nq, C)

    def forward(self, h_a, h_b, a_mask, b_mask, dist_mat=None, pair_dist_cutoff=None):
        bias_ab = self._rbf(dist_mat) if dist_mat is not None else None
        bias_ba = self._rbf(dist_mat.transpose(1, 2)) if dist_mat is not None else None

        # A→B
        ctx_a = self._cross_attn(self.q_a(h_a), self.k_b(h_b), self.v_b(h_b),
                                  b_mask, bias_ab)
        h_a_new = self.out_a(h_a + ctx_a)

        # B→A
        ctx_b = self._cross_attn(self.q_b(h_b), self.k_a(h_a), self.v_a(h_a),
                                  a_mask, bias_ba)
        h_b_new = self.out_b(h_b + ctx_b)

        # pair 特征
        B, Na, C = h_a_new.shape
        Nb = h_b_new.shape[1]
        pair_mask = a_mask.unsqueeze(2) & b_mask.unsqueeze(1)        # [B,Na,Nb]

        # 距离截断：仅保留 < cutoff 的节点对，避免 Na*Nb 全连接爆显存
        if pair_dist_cutoff is not None and dist_mat is not None:
            pair_mask = pair_mask & (dist_mat < pair_dist_cutoff)

        # 先用 mask 索引拿出有效对的节点坐标，再 gather 特征后 cat —
        # 避免先物化完整 [B,Na,Nb,2C] 张量。
        b_idx, a_idx, c_idx = pair_mask.nonzero(as_tuple=True)       # 各 [M]
        if b_idx.numel() == 0:
            pair_feat = h_a_new.new_empty((0, C))
            return h_a_new, h_b_new, pair_feat, pair_mask
        h_a_sel = h_a_new[b_idx, a_idx]                               # [M, C]
        h_b_sel = h_b_new[b_idx, c_idx]                               # [M, C]
        pair_feat = self.pair_mlp(
            torch.cat([h_a_sel, h_b_sel], dim=-1)
        )                                                             # [M, C]

        return h_a_new, h_b_new, pair_feat, pair_mask


# ─────────────────────────────────────────────────────────────
# SurfaceEncoder（共享权重）
# ─────────────────────────────────────────────────────────────

class SurfaceEncoder(nn.Module):
    """
    蛋白质表面网格编码器。
    受体和配体共享同一套权重（两者都是蛋白质）。

    节点特征（11维）：
        法向量(3) + 静电势(1) + 疏水性(1) + 氢键供体(1) + 氢键受体(1)
        + 曲率(1) + 形状指数(1) + 氨基酸极性(1) + rSASA(1)
    边特征（3维）：
        Cartesian 相对坐标（由 T.Cartesian 自动生成）
    """
    def __init__(
        self,
        in_channels=11,
        edge_features=3,
        hidden_dim=128,
        n_transformer_blocks=6,
        transformer_heads=4,
        dropout_rate=0.15,
        use_global_attn=True,
        global_attn_layers=2,
    ):
        super().__init__()
        self.node_enc = nn.Linear(in_channels, hidden_dim)
        self.edge_enc = nn.Linear(edge_features, hidden_dim)

        # 3 层基础 MetaLayer（捕捉局部邻域几何）
        self.local_convs = nn.ModuleList([
            MetaLayer(EdgeModel(hidden_dim), NodeModel(hidden_dim), None)
            for _ in range(3)
        ])

        # TransformerResBlock 堆叠（中远程注意力）
        self.transformer_blocks = nn.Sequential(*[
            TransformerResBlock(hidden_dim, heads=transformer_heads,
                                dropout_rate=dropout_rate)
            for _ in range(n_transformer_blocks)
        ])

        # 全局 Self-Attention（可选）
        self.use_global_attn = use_global_attn
        if use_global_attn:
            self.global_attn = GlobalSelfAttention(
                hidden_dim=hidden_dim,
                nhead=8,
                num_layers=global_attn_layers,
                dropout_rate=dropout_rate,
            )

    def forward(self, data):
        data = data.clone()  # 避免修改原始数据，防止多 epoch 时 x 维度累积
        data.edge_attr = None
        data = T.Cartesian(norm=False, max_value=None, cat=False)(data)

        data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        data.x = self.node_enc(data.x)
        data.edge_attr = self.edge_enc(data.edge_attr)

        for conv in self.local_convs:
            data.x, data.edge_attr, _ = conv(
                data.x, data.edge_index, data.edge_attr, None, data.batch
            )

        data = self.transformer_blocks(data)

        if self.use_global_attn:
            x_dense, mask = to_dense_batch(data.x, data.batch, fill_value=0)
            x_dense = self.global_attn(x_dense, mask)
            data.x = x_dense[mask]

        return data


# ─────────────────────────────────────────────────────────────
# DeepDock_PPI — 主模型
# ─────────────────────────────────────────────────────────────

class DeepDock_PPI(nn.Module):
    """
    蛋白质-蛋白质对接评分函数。

    forward(data_rec, data_lig) 返回：
        pi, sigma, mu : MDN 参数，各 [M, n_gaussians]
        dist          : 真实距离 [M, 1]（detach）
        C_batch       : batch 索引 [M]
    """
    def __init__(
        self,
        in_channels=11,
        edge_features=3,
        hidden_dim=128,
        n_gaussians=10,
        n_transformer_blocks=6,
        transformer_heads=4,
        dropout_rate=0.15,
        use_global_attn=True,
        global_attn_layers=2,
        cross_attn_heads=8,
        n_cross_attn_layers=2,
        dist_threshold=10.0,
    ):
        super().__init__()

        self.encoder = SurfaceEncoder(
            in_channels=in_channels,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            n_transformer_blocks=n_transformer_blocks,
            transformer_heads=transformer_heads,
            dropout_rate=dropout_rate,
            use_global_attn=use_global_attn,
            global_attn_layers=global_attn_layers,
        )

        self.cross_attn = nn.ModuleList([
            GeoBiasedCrossAttention(
                hidden_dim=hidden_dim,
                nhead=cross_attn_heads,
                dropout_rate=dropout_rate,
            )
            for _ in range(n_cross_attn_layers)
        ])

        self.z_pi    = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu    = nn.Linear(hidden_dim, n_gaussians)

        # Energy/score head（监督回归到 DockQ）
        self.energy_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.energy_alpha = nn.Parameter(torch.tensor(1.0))

        self.dist_threshold = dist_threshold

    def forward(self, data_rec, data_lig):
        device = data_rec.x.device

        # ── 编码（共享权重）──
        h_rec = self.encoder(data_rec)
        h_lig = self.encoder(data_lig)

        # ── dense batch ──
        h_r, r_mask = to_dense_batch(h_rec.x,   h_rec.batch, fill_value=0)
        h_l, l_mask = to_dense_batch(h_lig.x,   h_lig.batch, fill_value=0)
        p_r, _      = to_dense_batch(h_rec.pos,  h_rec.batch, fill_value=0)
        p_l, _      = to_dense_batch(h_lig.pos,  h_lig.batch, fill_value=0)

        assert h_r.size(0) == h_l.size(0), "batch size 不匹配"

        # ── 3D 距离矩阵 ──
        dist_mat = self._dist_matrix(p_r, p_l)   # [B, N_r, N_l]
        B, Nr, Nl = dist_mat.shape

        # ── 多层双向 Cross-Attention ──
        # 提前按距离截断 pair，避免 Na*Nb 全连接矩阵爆显存。
        # 阈值取 dist_threshold * 1.2，给 MDN 训练/打分的 10 Å 过滤留裕度。
        pair_cutoff = self.dist_threshold * 1.2
        for layer in self.cross_attn:
            h_r, h_l, pair_feat, pair_mask = layer(
                h_r, h_l, r_mask, l_mask, dist_mat,
                pair_dist_cutoff=pair_cutoff,
            )

        # ── MDN 输出 ──
        if pair_feat.numel() == 0:
            n_gaussians = self.z_pi.out_features
            pi = data_rec.x.new_empty((0, n_gaussians))
            sigma = data_rec.x.new_empty((0, n_gaussians))
            mu = data_rec.x.new_empty((0, n_gaussians))
            dist = data_rec.x.new_empty((0, 1))
            C_batch = torch.empty((0,), dtype=torch.long, device=device)
            pred_energy = data_rec.x.new_zeros(B)
            return pi, sigma, mu, dist, C_batch, pred_energy

        pi    = F.softmax(self.z_pi(pair_feat),    dim=-1)
        sigma = F.elu(self.z_sigma(pair_feat)) + 1.1
        mu    = F.elu(self.z_mu(pair_feat))    + 1.0
        pi = torch.nan_to_num(pi, nan=1.0 / pi.size(-1), posinf=1.0 / pi.size(-1), neginf=1.0 / pi.size(-1))
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sigma = torch.nan_to_num(sigma, nan=1.1, posinf=50.0, neginf=1e-3).clamp(min=1e-3, max=50.0)
        mu = torch.nan_to_num(mu, nan=1.0, posinf=50.0, neginf=0.0).clamp(min=0.0, max=50.0)
        dist  = dist_mat[pair_mask].unsqueeze(1).detach()

        C_batch = (
            torch.arange(B, device=device)
            .view(B, 1, 1).expand(B, Nr, Nl)[pair_mask]
        )

        # ── Energy/score 输出（每复合物一个标量，sigmoid 归到 [0,1]）──
        # 仅在距离 < dist_threshold 的有效界面对上聚合
        valid = dist.squeeze(1) < self.dist_threshold
        if valid.sum() > 0:
            energy_unit = self.energy_ffn(pair_feat[valid]).squeeze(-1)  # [Mv]
            valid_batch = C_batch[valid]
            n_per_b = torch.zeros(B, device=device).scatter_add_(
                0, valid_batch, torch.ones_like(energy_unit)
            )
            sum_per_b = torch.zeros(B, device=device).scatter_add_(
                0, valid_batch, energy_unit
            )
            mean_per_b = sum_per_b / n_per_b.clamp(min=1)
            penal = torch.log(n_per_b.clamp(min=1))
            pred_energy = torch.sigmoid(mean_per_b - self.energy_alpha * penal)
            # 没有界面对的样本（极差 decoy）赋零
            pred_energy = torch.where(n_per_b > 0, pred_energy,
                                       torch.zeros_like(pred_energy))
        else:
            pred_energy = torch.zeros(B, device=device)

        return pi, sigma, mu, dist, C_batch, pred_energy

    def forward_legacy(self, data_rec, data_lig):
        """5 元组版 forward，用于旧代码兼容（不返回 pred_energy）。"""
        pi, sigma, mu, dist, C_batch, _ = self.forward(data_rec, data_lig)
        return pi, sigma, mu, dist, C_batch

    @staticmethod
    def _dist_matrix(X, Y):
        """X [B,Na,3], Y [B,Nb,3] → [B,Na,Nb]"""
        X, Y = X.double(), Y.double()
        d = (
            -2 * torch.bmm(X, Y.permute(0, 2, 1))
            + (Y ** 2).sum(-1).unsqueeze(1)
            + (X ** 2).sum(-1).unsqueeze(-1)
        )
        return d.clamp(min=0).sqrt().float()


# ─────────────────────────────────────────────────────────────
# 损失函数 & 评分函数
# ─────────────────────────────────────────────────────────────

def mdn_loss_fn(pi, sigma, mu, y):
    """MDN 负对数似然损失。"""
    normal = Normal(mu, sigma)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    return -torch.logsumexp(torch.log(pi) + loglik, dim=1)


def ppi_train_loss(pi, sigma, mu, dist, dist_threshold=10.0):
    """只对界面内节点对（dist < threshold）计算 MDN loss。"""
    mask = dist.squeeze(1) <= dist_threshold
    if mask.sum() == 0:
        return torch.zeros(1, requires_grad=True, device=pi.device)
    return mdn_loss_fn(pi[mask], sigma[mask], mu[mask], dist[mask]).mean()


def ppi_score(pi, sigma, mu, dist, dist_threshold=10.0):
    """
    推理时计算对接评分（越大越好）。
    = Σ log P(d_ij) for d_ij < threshold
    """
    mask = dist.squeeze(1) < dist_threshold
    if mask.sum() == 0:
        return NO_INTERFACE_SCORE
    normal = Normal(mu[mask], sigma[mask])
    loglik = normal.log_prob(dist[mask].expand_as(normal.loc))
    return torch.logsumexp(torch.log(pi[mask]) + loglik, dim=1).mean().item()


def ppi_score_diff(pi, sigma, mu, dist, dist_threshold=10.0):
    """
    与 ppi_score 相同，但保留梯度（用于 contrastive 训练）。
    返回带 grad 的标量 tensor。
    """
    mask = dist.squeeze(1) < dist_threshold
    if mask.sum() == 0:
        return pi.sum() * 0.0 + NO_INTERFACE_SCORE
    normal = Normal(mu[mask], sigma[mask])
    loglik = normal.log_prob(dist[mask].expand_as(normal.loc))
    return torch.logsumexp(torch.log(pi[mask]) + loglik, dim=1).mean()
