import torch
import torch.nn as nn
import torch.fft
from typing import Optional
import torch.nn.functional as F
import numpy as np

# =============================================================================
# [新增] Dynamic EA 模块 (Test-Time Adaptation)
# =============================================================================
class DynamicEA(nn.Module):
    """
    Dynamic Euclidean Alignment (learnable, data-adaptive EMA momentum).

    对每个输入分布学习一个最优的 EMA 权重 alpha(x)，并做协方差白化对齐。
    - 原版: alpha 为固定超参 (momentum)，且训练模式不变换、仅评估模式白化。
    - 本版: alpha = sigma(MLP(协方差统计)); 训练与评估都做动态白化(可配 whiten_train)，
      使 alpha 端到端可学习，并对不同被试/会话自动得到不同的 alpha。
    """
    def __init__(self, num_channels=44, momentum=0.9, epsilon=1e-5,
                 learn_alpha=False, whiten_train=False, alpha_min=0.05, alpha_max=0.95,
                 ea_mode="whiten"):
        super(DynamicEA, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.learn_alpha = learn_alpha
        self.whiten_train = whiten_train
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ea_mode = ea_mode          # "whiten" (covariance whitening) | "interp" (alpha blend, no whitening)
        self.fixed_alpha = momentum
        # 参考协方差统计量 (EMA 状态，不参与反传)
        self.register_buffer('running_cov', torch.eye(num_channels))
        self.is_initialized = False
        if learn_alpha:
            # alpha = sigmoid(MLP(4 个协方差统计标量)) -> (alpha_min, alpha_max)
            self.alpha_head = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
            self._init_alpha(momentum)

    def _init_alpha(self, init_alpha):
        # 初始化使学习到的 alpha 接近给定初值 momentum
        s = (init_alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
        logit = float(np.log(max(s, 1e-6) / (1 - min(s, 1 - 1e-6))))
        with torch.no_grad():
            self.alpha_head[-1].weight.zero_()
            self.alpha_head[-1].bias.fill_(logit)

    def _compute_covariance(self, x):
        """当前 Batch 的平均协方差"""
        batch_size, channels, time_points = x.size()
        x_centered = x - x.mean(dim=2, keepdim=True)
        covs = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (time_points - 1)
        return covs.mean(dim=0)

    def _cov_stats(self, current_cov):
        """稳健的协方差统计特征，用于计算数据自适应的 alpha(x)"""
        running = self.running_cov.detach()
        eps = self.epsilon
        rel_dev = torch.norm(current_cov - running) / (torch.norm(running) + eps)
        energy = torch.trace(current_cov) / (torch.trace(running) + eps)
        diag_ratio = torch.norm(torch.diag(current_cov), 1) / (torch.norm(torch.diag(running), 1) + eps)
        fro = torch.norm(current_cov)
        return torch.stack([rel_dev, energy, diag_ratio, fro])

    def _alpha_from_cov(self, current_cov):
        psi = self._cov_stats(current_cov).unsqueeze(0)
        logit = self.alpha_head(psi).squeeze()
        s = torch.sigmoid(logit)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * s

    def _compute_whitening_matrix(self, R):
        """白化矩阵 P = R^(-1/2)"""
        R = R + torch.eye(self.num_channels, device=R.device) * self.epsilon
        e, v = torch.linalg.eigh(R)
        e = torch.clamp(e, min=1e-6)
        d_inv_sqrt = torch.diag(e.pow(-0.5))
        P = torch.matmul(torch.matmul(v, d_inv_sqrt), v.transpose(-1, -2))
        return P

    def forward(self, x):
        # x: (Batch, Channel, Time)
        current_cov = self._compute_covariance(x)
        if self.learn_alpha:
            alpha = self._alpha_from_cov(current_cov)
        else:
            alpha = torch.tensor(self.fixed_alpha, device=x.device)

        if self.ea_mode == "interp":
            # [gentle] alpha 只做插值、不白化：混合原始信号与按时间均值-中心化信号 (无 eigh/白化)
            x_mu = x - x.mean(dim=2, keepdim=True)
            x_aligned = (1.0 - alpha) * x + alpha * x_mu
        else:
            # 本 batch 的可微参考协方差 (单步 EMA 混合) -> 梯度可传到 alpha
            ref = alpha * self.running_cov + (1.0 - alpha) * current_cov
            do_whiten = (not self.training) or self.whiten_train
            if do_whiten:
                P = self._compute_whitening_matrix(ref)
                x_aligned = torch.einsum('ij,bjt->bit', P, x)
            else:
                x_aligned = x

        # 持久化 EMA 状态 (重新赋值而非 in-place，避免破坏 ref 的反传图)——running_cov 逐步追踪当前分布
        with torch.no_grad():
            if not self.is_initialized:
                self.running_cov = current_cov.detach().clone()
                self.is_initialized = True
            else:
                self.running_cov = (alpha.detach() * self.running_cov + (1 - alpha.detach()) * current_cov).detach()
        return x_aligned


# ---------------------------------------------------------
# 1. 预处理 / 特征提取模块
# ---------------------------------------------------------
class TorchPSD(nn.Module):
    """Spectral band-power features.
    spec_mode:
      "none" : fixed 5 rectangular bands (delta/theta/alpha/beta/gamma)   [original]
      "C"    : fixed bands scaled by learnable per-band weights
      "B"    : fixed bands re-weighted by a per-sample band attention (band squeeze-and-excite)
      "D"    : learnable Gaussian filterbank (learnable band centers/widths over Hz)
    """
    def __init__(self, fs=250, window_size=250, step=125, spec_mode="none", n_bands=5):
        super(TorchPSD, self).__init__()
        self.fs = fs
        self.window_size = window_size
        self.step = step
        self.spec_mode = spec_mode
        self.n_bands = int(n_bands)
        self.bands = [(0.5, 4), (4, 7), (8, 13), (14, 30), (31, 40)]
        self.register_buffer('window', torch.hamming_window(window_size))
        if spec_mode == "C":
            self.band_weights = nn.Parameter(torch.ones(self.n_bands))
        elif spec_mode == "B":
            hidden = max(2, self.n_bands // 2 + 1)
            self.band_attn = nn.Sequential(nn.Linear(self.n_bands, hidden), nn.ReLU(),
                                           nn.Linear(hidden, self.n_bands), nn.Sigmoid())
        elif spec_mode == "D":
            centers = [2.0, 5.5, 10.5, 22.0, 35.0]
            sig0 = [1.5, 1.5, 2.5, 8.0, 4.5]
            self.band_mu = nn.Parameter(torch.tensor(centers[:self.n_bands], dtype=torch.float32))
            self.band_log_sigma = nn.Parameter(torch.tensor(sig0[:self.n_bands], dtype=torch.float32))

    def forward(self, x):
        # center=False 是关键，避免边界填充带来的伪影
        stft_res = torch.stft(x.reshape(-1, x.shape[-1]), n_fft=self.window_size, hop_length=self.step,
                              window=self.window, return_complex=True, normalized=True, center=False)
        power = torch.abs(stft_res) ** 2                        # (B*C, n_freq, frames)
        freqs = torch.fft.rfftfreq(self.window_size, d=1 / self.fs).to(x.device)
        B, C, _ = x.shape

        if self.spec_mode == "D":
            mu = torch.clamp(self.band_mu, 0.5, 40.0)
            sig = F.softplus(self.band_log_sigma) + 1e-6        # (n_bands,)
            gaus = torch.exp(-0.5 * ((freqs.unsqueeze(0) - mu.unsqueeze(1)) / sig.unsqueeze(1)) ** 2)  # (n_bands,n_freq)
            gaus = gaus / (gaus.sum(1, keepdim=True) + 1e-6)
            band_power = torch.einsum('bfk,nf->bnk', power, gaus)   # (B*C, n_bands, frames)
            features = torch.log(band_power + 1e-6)
        else:
            band_power_list = []
            for low, high in self.bands:
                mask = (freqs >= low) & (freqs <= high)
                band_power_list.append(power[:, mask, :].sum(dim=1))
            band_power = torch.stack(band_power_list, dim=1)    # (B*C, n_bands, frames)
            features = torch.log(band_power + 1e-6)
            if self.spec_mode == "C":
                features = features * self.band_weights.unsqueeze(0).unsqueeze(-1)

        features = features.view(B, C, self.n_bands, -1)
        if self.spec_mode == "B":
            stats = features.mean(dim=(1, 3))                   # (B, n_bands)
            at = self.band_attn(stats).unsqueeze(1).unsqueeze(-1)
            features = features * at
        return features

# ---------------------------------------------------------
# 2. 频域分支
# ---------------------------------------------------------
class FreConv(nn.Module):
    def __init__(self):
        super(FreConv, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(44, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ELU()
        )
        self.spectro_temporal_conv = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU()
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.spatial_conv(x)
        x = x.squeeze(2)
        x = self.spectro_temporal_conv(x) 
        return x

# ---------------------------------------------------------
# 3. 时域分支
# ---------------------------------------------------------
class STConv(nn.Module):
    def __init__(self, input_channels=44, out_channels=64, dropout_rate=0.5):
        super(STConv, self).__init__()
        kernel_sizes = [31, 63, 125]
        self.D = 2
        nb_filters = out_channels // self.D
        self.F1 = nb_filters
        self.F2 = out_channels
        self.dropout_rate = dropout_rate
        self.pool_kernel = (1, 8)
        self.pool_stride = (1, 8)

        self.branch_short = self._make_branch(input_channels, kernel_sizes[0], self.F1, self.D, self.dropout_rate)
        self.branch_med = self._make_branch(input_channels, kernel_sizes[1], self.F1, self.D, self.dropout_rate)
        self.branch_long = self._make_branch(input_channels, kernel_sizes[2], self.F1, self.D, self.dropout_rate)

        total_concat_channels = 3 * self.F2
        self.fusion = nn.Sequential(
            nn.Conv2d(total_concat_channels, self.F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.F2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

    def _make_branch(self, in_ch, k_time, f1, d, drop_rate):
        return nn.Sequential(
            nn.Conv2d(1, f1, (1, k_time), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (in_ch, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out_s = self.branch_short(x)
        out_m = self.branch_med(x)
        out_l = self.branch_long(x)
        concat = torch.cat([out_s, out_m, out_l], dim=1)
        output = self.fusion(concat)
        return output

class GatedMultiHeadAttention(nn.Module):
    """
    gate_mode:
      "elementwise" : original —— query-based gate on attn OUTPUT (value)  (attn * sigmoid(gate))
      "headwise"    : query-based gate per head (one scalar/head)
      "none"        : no gating  (plain SDPA + o_proj)
      "content"     : improved —— content-aware (q,k) gate applied to attention logits BEFORE softmax,
                      so it actually selects WHICH positions are attended (time-selective)
    """
    def __init__(self, embed_dim, num_heads, gate_mode="elementwise", dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.gate_mode = gate_mode
        self.dropout = dropout
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if gate_mode in ("elementwise", "headwise"):
            extra = num_heads if gate_mode == "headwise" else embed_dim
            self.q_proj = nn.Linear(embed_dim, embed_dim + extra, bias=bias)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            if gate_mode == "content":
                # content-aware pair gate: concat(q,k) per (query,key) -> logit bias (before softmax)
                self.gate_proj = nn.Linear(self.head_dim * 2, 1, bias=True)

    def forward(self, hidden_states, attention_mask=None, is_causal=False):
        bs, seq, _ = hidden_states.size()
        H, hd = self.num_heads, self.head_dim
        k = self.k_proj(hidden_states).view(bs, seq, H, hd).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bs, seq, H, hd).transpose(1, 2)
        q = self.q_proj(hidden_states)

        if self.gate_mode in ("elementwise", "headwise"):
            if self.gate_mode == "headwise":
                q = q.view(bs, seq, H, hd + 1); qs, g = torch.split(q, [hd, 1], dim=-1)
            else:
                q = q.view(bs, seq, H, hd * 2); qs, g = torch.split(q, [hd, hd], dim=-1)
            qs = qs.transpose(1, 2); g = g.transpose(1, 2)
            s = F.scaled_dot_product_attention(qs, k, v, attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
            s = s * torch.sigmoid(g)
        else:
            qs = q.view(bs, seq, H, hd).transpose(1, 2)
            if self.gate_mode == "content":
                scores = (qs @ k.transpose(-1, -2)) / (hd ** 0.5)             # (bs,H,seq,seq)
                q_pair = qs.unsqueeze(3).expand(-1, -1, -1, seq, -1)          # (bs,H,seq_q,seq_k,hd)
                k_pair = k.unsqueeze(2).expand(-1, -1, seq, -1, -1)          # (bs,H,seq_q,seq_k,hd)
                pair_gate = self.gate_proj(torch.cat([q_pair, k_pair], dim=-1)).squeeze(-1)   # logit (bs,H,seq,seq)
                scores = scores + pair_gate
                s = torch.softmax(scores, dim=-1) @ v
            else:  # "none"
                s = F.scaled_dot_product_attention(qs, k, v, attn_mask=attention_mask,
                        dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
        s = s.transpose(1, 2).contiguous().view(bs, seq, self.embed_dim)
        return self.o_proj(s)

class Attention(nn.Module):
    def __init__(self, dim, seq_len=35, num_heads=8, bias=False, qkv_bias=False, attn_drop=0.7, proj_drop=0.5,
                 gate_mode="elementwise"):
        super(Attention, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 9), padding=(0, 4), groups=dim, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ELU()
        )
        self.gated_attn = GatedMultiHeadAttention(embed_dim=dim, num_heads=num_heads, gate_mode=gate_mode, dropout=attn_drop, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        local_feat = self.local_conv(x)
        x_reshaped = x.squeeze(2).permute(0, 2, 1)

        if x_reshaped.shape[1] == self.pos_embed.shape[1]:
            x_reshaped = x_reshaped + self.pos_embed
        else:
            pos_embed_reshaped = self.pos_embed.permute(0, 2, 1)
            pos_embed_resized = F.interpolate(pos_embed_reshaped, size=x_reshaped.shape[1], mode='linear', align_corners=False)
            x_reshaped = x_reshaped + pos_embed_resized.permute(0, 2, 1)

        global_feat = self.gated_attn(x_reshaped)
        global_feat = global_feat.permute(0, 2, 1).unsqueeze(2)
        out = global_feat + local_feat
        out = self.proj_drop(out)
        return out

class SemanticFusion(nn.Module):
    def __init__(self, time_input_dim, freq_input_dim, common_dim=64):
        super(SemanticFusion, self).__init__()
        self.time_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(time_input_dim, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        self.freq_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(freq_input_dim, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        self.fusion_out = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.ELU()
        )

    def forward(self, x_time, x_freq):
        h_time = self.time_projector(x_time)
        h_freq = self.freq_projector(x_freq)
        concat = torch.cat([h_time, h_freq], dim=1)
        out = self.fusion_out(concat)
        return out


# =============================================================================
# [Design A] FeatureAlign: 特征层数据自适应对齐 (learnable alpha)
# 作用在融合特征(64维)上，对每个 batch 做「零均值 + 协方差白化」，用可学习 alpha 控制强度。
# 静态 EA 已把输入协方差抹平，残余的跨被试漂移主要在特征层(均值/类条件)，故把 alpha 移到特征层，
# 既保留静态 EA，又让 alpha 真正学到逐被试差异。
# =============================================================================
class FeatureAlign(nn.Module):
    def __init__(self, dim=64, eps=1e-5, alpha_init=0.5, learn_alpha=True,
                 alpha_min=0.0, alpha_max=1.0):
        super(FeatureAlign, self).__init__()
        self.dim = dim
        self.eps = eps
        self.learn_alpha = learn_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_init = alpha_init
        if learn_alpha:
            # alpha = sigmoid(MLP(3 个特征统计标量)) -> [alpha_min, alpha_max]
            self.alpha_head = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
            self._init_alpha(alpha_init)

    def _init_alpha(self, a):
        s = (a - self.alpha_min) / (self.alpha_max - self.alpha_min)
        logit = float(np.log(max(s, 1e-6) / (1 - min(s, 1 - 1e-6))))
        with torch.no_grad():
            self.alpha_head[-1].weight.zero_()
            self.alpha_head[-1].bias.fill_(logit)

    def _stats(self, mu, cov):
        # mu: (D,), cov: (D, D)
        mean_norm = torch.norm(mu)
        off_iso = torch.norm(cov - torch.eye(self.dim, device=cov.device) * torch.trace(cov) / self.dim) / (torch.norm(cov) + self.eps)
        trace_ratio = torch.trace(cov) / (self.dim + self.eps)
        return torch.stack([mean_norm, off_iso, trace_ratio])

    def _alpha_from(self, mu, cov):
        psi = self._stats(mu, cov).unsqueeze(0)
        logit = self.alpha_head(psi).squeeze()
        s = torch.sigmoid(logit)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * s

    def forward(self, f):
        # f: (B, D)
        B, D = f.shape
        mu = f.mean(0)
        fc = f - mu
        # 稳健白化：用 SVD（对秩亏/重复特征值稳健，只作用于数据真实子空间），避免 eigh 收敛失败
        U, S, Vt = torch.linalg.svd(fc, full_matrices=False)      # fc (B,D)
        lamb = (S ** 2) / max(B - 1, 1)
        inv_sqrt = 1.0 / torch.sqrt(lamb + self.eps)
        P = (Vt.transpose(-1, -2) * inv_sqrt.unsqueeze(0)) @ Vt   # (D,D) = cov^{-1/2}
        w = fc @ P                                                # zero-mean + unit-cov whitened
        # alpha 输入统计（rof特诊断用，不参与分解）
        cov = (fc.transpose(0, 1) @ fc) / max(B - 1, 1) + torch.eye(D, device=f.device) * self.eps
        alpha = self._alpha_from(mu, cov) if self.learn_alpha else torch.tensor(self.alpha_init, device=f.device)
        # blend original-centered vs whitened; 始终零均值(去均值漂移), alpha 控制协方差白化强度
        out = (1.0 - alpha) * fc + alpha * w
        return out


# =============================================================================
# Net: 融合了 Dynamic EA 的最终网络
# =============================================================================
class Net(nn.Module):
    def __init__(self, num_classes=4, input_time_length=1125, fs=250, ea_momentum=0.9,
                 use_dynamic_ea=False, use_gasl=True, use_tgma=True, use_dbp_fusion=True,
                 learn_alpha=False, whiten_train=False,
                 use_feature_align=False, fa_learn_alpha=True,
                 attn_gate="none", spec_mode="none", ea_mode="whiten"):
        super(Net, self).__init__()
        self.attn_gate = attn_gate
        self.spec_mode = spec_mode
        self.ea_mode = ea_mode
        self.use_dynamic_ea = use_dynamic_ea
        self.use_gasl = use_gasl
        self.use_tgma = use_tgma
        self.use_dbp_fusion = use_dbp_fusion
        self.learn_alpha = learn_alpha
        self.whiten_train = whiten_train
        self.use_feature_align = use_feature_align

        # ============================================================
        # 1. 【核心集成】动态自适应 EA
        # ============================================================
        # 放在网络的第一层，对原始信号进行协方差修正
        self.dynamic_ea = DynamicEA(num_channels=44, momentum=ea_momentum,
                                    learn_alpha=learn_alpha, whiten_train=whiten_train,
                                    ea_mode=self.ea_mode)

        # 2. 频域分支组件
        self.torch_psd = TorchPSD(fs=fs, window_size=250, step=125, spec_mode=self.spec_mode)
        self.fre_conv = FreConv()
        
        # 重新计算维度: (B, 64, 3, 6) -> 64*3*6 = 1152
        self.freq_flat_dim = 64 * 3 * 6 

        # 3. 时域分支组件
        self.st_conv = STConv(input_channels=44, out_channels=64, dropout_rate=0.0)
        self.st_conv_out_len = input_time_length // 8 

        self.attention = Attention(
            dim=64,
            seq_len=self.st_conv_out_len,
            attn_drop=0.7,
            proj_drop=0.6,
            gate_mode=self.attn_gate
        )

        self.target_time_len = 35
        self.pre_flatten_pool = nn.AdaptiveAvgPool2d((1, self.target_time_len))
        self.time_flat_dim = 64 * self.target_time_len

        # [Ablation] naive-concat projection used when DBPF (projection fusion) is ablated
        self.concat_proj = nn.Linear(self.time_flat_dim + self.freq_flat_dim, 64)

        # 4. 融合与分类
        self.fusion = SemanticFusion(
            time_input_dim=self.time_flat_dim,
            freq_input_dim=self.freq_flat_dim,
            common_dim=64
        )
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.classifier_head = nn.Linear(64, num_classes)
        if use_feature_align:
            self.feature_align = FeatureAlign(dim=64, alpha_init=0.5, learn_alpha=fa_learn_alpha)

    def forward(self, x):
        # Input x: (B, 44, 1125)

        # ============================================================
        # Step 1: 动态对齐 (适应测试数据漂移)
        # ============================================================
        x = self.dynamic_ea(x) if self.use_dynamic_ea else x

        # --- Path 1: 频域 (GASL) ---
        if self.use_gasl:
            x_freq = self.torch_psd(x)      # (B, 22, 5, 8)
            x_freq_out = self.fre_conv(x_freq)  # (B, 64, 3, 6)
        else:
            x_freq_out = None

        # --- Path 2: 时域 ---
        x_time = self.st_conv(x)  # (B, 64, 1, 140)
        if self.use_tgma:
            x_time = self.attention(x_time)
        x_time = self.pre_flatten_pool(x_time)  # (B, 64, 1, 35)

        # --- 语义融合 (DBPF) ---
        if not self.use_gasl:
            # No spectral branch -> temporal-only projection into latent space
            x_fused = self.fusion.time_projector(x_time)       # (B, 64)
        elif not self.use_dbp_fusion:
            # DBPF ablated -> naive concat of the two branches, then a linear projection
            x_flat = torch.cat([x_time.flatten(1), x_freq_out.flatten(1)], dim=1)
            x_fused = self.concat_proj(x_flat)                 # (B, 64)
        else:
            x_fused = self.fusion(x_time, x_freq_out)          # (B, 64)

        # --- 分类 ---
        features = self.feature_extractor(x_fused)
        if self.use_feature_align:
            features = self.feature_align(features)   # [Design A] 特征层可学习 α 对齐
        logits = self.classifier_head(features)

        return logits, features

if __name__ == "__main__":
    model = Net(num_classes=4, input_time_length=1125)
    print("Model initialized successfully.")
    # 简单的 dummy 输入测试
    x = torch.randn(2, 44, 1125)
    logits, feat = model(x)
    print(f"Output shape: {logits.shape}")