"""
CTSF 기본 모델 정의
⚠️ 경고: 이 파일의 모델 구조는 절대 변경하지 마세요!
W1~W5 실험은 이 모델을 수정하지 않고, 실험별 variant만 추가합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== RevIN ====================
class RevIN(nn.Module):
    """Reversible Instance Normalization (PatchTST 등에서 사용)"""
    def __init__(self, dim=1, affine=True):
        super().__init__()
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, dim, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x, mode):
        # x : (B, C, L)  또는  (B, 1, H) when denorm
        if mode == 'norm':
            self.mu = x.mean(2, keepdim=True)
            self.sig = x.std(2, keepdim=True) + 1e-5
            y = (x - self.mu) / self.sig
            if self.affine:
                y = self.gamma * y + self.beta
            return y
        elif mode == 'denorm':
            y = x
            if self.affine:
                y = (y - self.beta) / (self.gamma + 1e-7)
            return y * self.sig + self.mu


# ==================== Conv Block ====================
class ResConvBlock(nn.Module):
    def __init__(self, d_embed, kernel, p_drop):
        super().__init__()
        self.input_residual = None
        self.conv = nn.Conv1d(d_embed, d_embed, kernel, padding=(kernel-1)//2)
        self.bn = nn.BatchNorm1d(d_embed)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.ln_post = nn.LayerNorm(d_embed)

    def forward(self, x):  # x: (Tp,B,d)
        if self.input_residual is not None:
            x = x + self.input_residual
            self.input_residual = None

        u = x.permute(1, 2, 0)  # (B,d,Tp)
        u = self.drop(self.act(self.bn(self.conv(u))))
        out = u + x.permute(1, 2, 0)  # local skip
        return self.ln_post(out.permute(2, 0, 1))  # (Tp,B,d)


# ==================== GRU Block ====================
class GRUBlock(nn.Module):
    def __init__(self, dim, p_drop):
        super().__init__()
        self.input_residual = None
        self.gru = nn.GRU(dim, dim, 1)
        self.drop = nn.Dropout(p_drop)
        self.ln_post = nn.LayerNorm(dim)

    def forward(self, x):  # x: (Tp,B,d)
        if self.input_residual is not None:
            x = x + self.input_residual
            self.input_residual = None
        z, _ = self.gru(x)
        return self.ln_post(self.drop(z))  # (Tp,B,d)


# ==================== Cross-Modal Co-Attention ====================
class CrossModalCoAttn(nn.Module):
    def __init__(self, dim, n_heads, p_drop):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=p_drop)
        self.ln = nn.LayerNorm(dim)
        self.last_attn = None  # 저장용 버퍼

    def forward(self, feat_c, feat_r, *, save=False):
        """
        feat_c : (Tp, B, d)  Conv-stream query
        feat_r : (Tp, B, d)  GRU-stream key/value
        save   : True 면 self.last_attn 에 attention weight(softmax) 보관
        """
        Q = self.q_proj(feat_c)
        KV = self.kv_proj(feat_r)
        K, V = KV.chunk(2, dim=-1)
        z, w = self.attn(Q, K, V, need_weights=True)  # w: (B, Tp, Tp)
        if save:
            self.last_attn = w.detach()
        return self.ln(z + Q)  # Residual + LayerNorm


# ==================== Fast HyperConv1D ====================
class HyperConv1D(nn.Module):
    """
    X:(T,B,d)  cond_vec:(B,d)  →  depth-wise conv 결과 (T,B,d)
    k must be odd (padding = (k-1)//2)
    """
    def __init__(self, d_embed, k=3):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.k = k
        self.d = d_embed
        self.gen = nn.Linear(d_embed, d_embed * k)

    def forward(self, X: torch.Tensor, cond_vec: torch.Tensor):
        # ---------- shapes ----------
        T, B, d = X.shape  # d == self.d
        pad = (self.k - 1) // 2

        # ---------- ① 커널 생성 ----------
        # (B,d*k) → (B*d,1,k)  depth-wise kernel
        W = self.gen(cond_vec).view(B * d, 1, self.k)

        # ---------- ② 입력을 group-conv 형태로 ----------
        # X:(T,B,d) → (1,B*d,T)
        X_grp = X.permute(1, 2, 0).reshape(1, B * d, T)
        X_grp = F.pad(X_grp, (pad, pad))

        # ---------- ③ single group-conv ----------
        Y = F.conv1d(X_grp, W, groups=B * d)  # (1,B*d,T)

        # ---------- ④ 원래 모양 복원 ----------
        Y = Y.view(B, d, T).permute(2, 0, 1)  # (T,B,d)
        return Y


# ==================== Cross-HyperConv + Cross-Stitch ====================
class CrossHyperConvBlock(nn.Module):
    """
    Dual-direction HyperConv + α Cross-Stitch
    ⚠️ 이 블록은 실험별로 동작을 제어할 수 있지만, 구조 자체는 변경하지 않습니다.
    """
    def __init__(self, d_embed, k=3, p_drop=0.1):
        super().__init__()
        self.hc_gc = HyperConv1D(d_embed, k)  # GRU → Conv
        self.hc_cg = HyperConv1D(d_embed, k)  # Conv → GRU

        # α 초기 diag=0.95, off-diag=0.02
        alpha = torch.eye(2) * 0.90 + 0.05
        self.alpha = nn.Parameter(alpha)

        self.ln_c = nn.LayerNorm(d_embed)
        self.ln_r = nn.LayerNorm(d_embed)
        self.drop = nn.Dropout(p_drop)

        # 실험 제어 플래그 (기본값: 모두 활성)
        self.use_gc = True  # GRU → Conv
        self.use_cg = True  # Conv → GRU

    def forward(self, C, R):  # (T,B,d)
        C_in, R_in = C, R

        # GRU → Conv 경로
        if self.use_gc:
            cond_gc = R_in[-1]  # (B,d)
            C_h = self.hc_gc(C_in, cond_gc)  # GRU → Conv
        else:
            C_h = C_in  # 우회(항등)

        # Conv → GRU 경로
        if self.use_cg:
            cond_cg = C_in[-1]
            R_h = self.hc_cg(R_in, cond_cg)  # Conv → GRU
        else:
            R_h = R_in  # 우회(항등)

        # Cross-Stitch 결합
        a = torch.relu(self.alpha)
        if not self.use_gc:
            a = a.clone()
            a[0, 1] = 0.0  # GRU→Conv 혼합 차단
        if not self.use_cg:
            a = a.clone()
            a[1, 0] = 0.0  # Conv→GRU 혼합 차단

        C_mix = a[0, 0] * C_h + a[0, 1] * R_h
        R_mix = a[1, 0] * C_h + a[1, 1] * R_h

        C = self.ln_c(self.drop(C_mix) + C_in)  # Residual + LN
        R = self.ln_r(self.drop(R_mix) + R_in)
        return C, R


# ==================== HybridTS (CTSF Main Model) ====================
class HybridTS(nn.Module):
    """
    CTSF 메인 모델
    ⚠️ 경고: 이 클래스의 구조는 절대 변경하지 마세요!
    실험별 변형은 experiment_variants.py에서 wrapper로 처리합니다.
    """
    def __init__(self, cfg, n_vars: int):
        super().__init__()
        d = cfg["d_embed"]
        dp = cfg["dropout_rate"]
        L = cfg["lookback"]
        Pl = cfg["patch_len"]
        self.Tp = L // Pl
        depth = cfg["cnn_depth"]

        # Pre-net
        self.revin = RevIN(dim=n_vars, affine=True)
        self.patch = nn.Conv1d(n_vars, d, kernel_size=Pl, stride=Pl)

        # Backbone
        self.conv_blks = nn.ModuleList([ResConvBlock(d, 3, dp) for _ in range(depth)])
        self.gru_blks = nn.ModuleList([GRUBlock(d, dp) for _ in range(depth)])
        self.xhconv_blks = nn.ModuleList([
            CrossHyperConvBlock(d, k=3, p_drop=dp) for _ in range(depth)
        ])

        # Fusion & Head
        self.fuse = CrossModalCoAttn(d, cfg["n_heads"], dp)
        self.ln_flat = nn.LayerNorm(d)
        self.fc = nn.Linear(self.Tp * d, cfg["horizon"] * n_vars)

        # 실험 제어용 속성
        self.cfg = cfg
        self.n_vars = n_vars
        self.depth = depth

    def forward(self, x: torch.Tensor, *, save_attn: bool = False):
        B, C, _ = x.shape
        z0 = self.patch(self.revin(x, mode='norm')).permute(2, 0, 1)  # (T,B,d)

        zc = zr = z0
        depth = len(self.conv_blks)
        use_hconv_depth = depth

        for i, (cb, gb) in enumerate(zip(self.conv_blks, self.gru_blks)):
            zc = cb(zc)
            zr = gb(zr)

            if i < use_hconv_depth:  # HyperConv+Stitch 적용 범위 제한
                zc, zr = self.xhconv_blks[i](zc, zr)

        mem = self.fuse(zc, zr, save=save_attn)
        flat = self.ln_flat((mem + z0).permute(1, 0, 2)).reshape(B, -1)
        out = self.fc(flat).view(B, self.cfg["horizon"], C).permute(0, 2, 1)
        return self.revin(out, mode='denorm')

    def set_cross_directions(self, use_gc=True, use_cg=True):
        """교차 연결 방향 제어 (실험용)"""
        for blk in self.xhconv_blks:
            blk.use_gc = bool(use_gc)
            blk.use_cg = bool(use_cg)

    def set_cross_layers(self, active_layers):
        """
        특정 층의 교차 연결만 활성화 (W4 실험용)
        active_layers: list of int, 활성화할 층 인덱스
        """
        for i, blk in enumerate(self.xhconv_blks):
            if i in active_layers:
                blk.use_gc = True
                blk.use_cg = True
            else:
                blk.use_gc = False
                blk.use_cg = False
