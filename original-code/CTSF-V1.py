# same as CTSF-V1.ipynb

import random, os, gc, json, math
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

torch.cuda.set_device(0)

# ------------------------------ 1. 하이퍼파라미터 ----------------------------- #
CFG = dict(
    # 랜덤 시드 고정 (재현성 보장)
    seed=11,
    
    device="cuda" if torch.cuda.is_available() else "cpu",

    # 최적 모델 저장 경로
    save_path="best-CTSF-V1.pt", # weather / ETTm2 / ETTm1 / ETTh1 / ETTh2
    # 결과 이미지 / 로그 폴더
    out_dir="results",
    # model tagging
    model_tag="HyperConv",

    # ——— 데이터 관련 ———
    # CSV 파일 경로
    csv_path="datasets/ETTh2.csv",
    # 예측하려는 타깃 컬럼 이름, None 이면 CSV의 모든 수치형 컬름 사용
    value_cols=None,
    # 학습/검증/테스트 분할 비율
    train_ratio=0.6,
    val_ratio=0.2,
    # 모델이 한 번에 볼 과거 시점 길이 (lookback window size)
    lookback=720,       # 예: 384시간(16일)
    # 예측할 미래시점 길이 (horizon)
    horizon=192,        # 96 / 192
    # 1D 신호를 패치 단위로 쪼갤 때의 패치 길이
    patch_len=36,
    # 한 배치(batch)당 샘플 개수
    batch_size=128,      # ↑를 늘리면 GPU 효율↑, 메모리 소요↑

    # ——— 모델 구조 관련 ———
    # 임베딩 차원 수 (모델 내부 feature 차원)
    d_embed=256,
    # CNN 레이어 + GRU Block depth (Residual Conv Block 개수)
    cnn_depth=7,
    # Multihead Attention 헤드 개수
    n_heads=4,
    
    # ——— 학습 관련 ———
    # 초기 학습률(Learning rate)
    lr=1e-4,
    # AdamW only
    weight_decay = 1e-3,
    # 총 학습 epoch 수
    epochs=100,
    # 학습률 워밍업(warm-up)에 사용할 epoch 수
    lr_warmup_epochs=5,  # epoch 동안 선형 증가
    # 그래디언트 클리핑 임계값 (gradient clipping)
    grad_clip=0.5,
    # 드롭아웃 비율 (Dropout) — 과적합 방지용
    dropout_rate=0.35,
    # Patch-Stat 가중치
    lam_patch=0.03,
    # α-L1 규제 가중치
    lamb_reg=1e-3,
)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

# --- RevIN -----------------------------------------------------------
class RevIN(nn.Module):
    """Reversible Instance Normalization (PatchTST 등에서 사용)"""
    def __init__(self, dim=1, affine=True):
        super().__init__()
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones (1, dim, 1))
            self.beta  = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x, mode):
        # x : (B, C, L)  또는  (B, 1, H) when denorm
        if mode == 'norm':
            self.mu  = x.mean(2, keepdim=True)
            self.sig = x.std (2, keepdim=True) + 1e-5
            y = (x - self.mu) / self.sig
            if self.affine: y = self.gamma * y + self.beta
            return y
        elif mode == 'denorm':
            y = x
            if self.affine: y = (y - self.beta) / (self.gamma + 1e-7)
            return y * self.sig + self.mu

# ------------------------------ 2. 데이터셋 ----------------------------- #
class TSDataset(Dataset):
    def __init__(self, series, L, H):
        self.s, self.L, self.H = series, L, H
        self.N = len(series) - L - H + 1

    def __len__(self): return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.s[idx      : idx+self.L  ]).unsqueeze(0)  # (1,L)
        y = torch.from_numpy(self.s[idx+self.L : idx+self.L+self.H])        # (H,)
        return x, y


def load_split_dataloaders(CFG):
    df = pd.read_csv(CFG["csv_path"], encoding='utf-8')

    # ─ 변수 선택 ─
    if CFG["value_cols"] is None:
        feature_cols = df.select_dtypes("number").columns.tolist()
    else:
        feature_cols = CFG["value_cols"]

    data = (df[feature_cols]
            .interpolate("linear", limit_direction="both")
            .to_numpy(np.float32))          # (T, C)
    mu, std = data.mean(0, keepdims=True), data.std(0, keepdims=True) + 1e-6
    data = (data - mu) / std                # 표준화
    C  = data.shape[1]                      # 변수 수
    L, H = CFG["lookback"], CFG["horizon"]

    total = len(data) - L - H + 1
    n_tr  = int(total * CFG["train_ratio"])
    n_val = int(total * CFG["val_ratio"])
    margin = L + H
    tr_end, val_st = n_tr, n_tr + margin
    val_end, te_st = val_st + n_val, val_st + n_val + margin

    # ── 다변수 Dataset ──
    class MVDataset(Dataset):
        def __init__(self, arr):
            self.arr = arr
        def __len__(self): return len(self.arr) - L - H + 1
        def __getitem__(self, idx):
            x = self.arr[idx      : idx+L    ].T  # (C, L)
            y = self.arr[idx+L   : idx+L+H ].T  # (C, H)
            return torch.from_numpy(x), torch.from_numpy(y)

    def mk(sub, sh):
        return DataLoader(MVDataset(sub), batch_size=CFG["batch_size"],
                          shuffle=sh, drop_last=True,
                          num_workers=4, pin_memory=True)

    return (mk(data[:tr_end+L+H-1], True),
            mk(data[val_st:val_end+L+H-1], False),
            mk(data[te_st:], False),
            (mu, std, C))

# ------------------------------ 3. 모델 ----------------------------- #
# ────────── Conv Block (unchanged) ──────────
class ResConvBlock(nn.Module):
    def __init__(self, d_embed, kernel, p_drop):
        super().__init__()
        self.input_residual = None
        self.conv = nn.Conv1d(d_embed, d_embed, kernel, padding=(kernel-1)//2)
        self.bn   = nn.BatchNorm1d(d_embed)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.ln_post = nn.LayerNorm(d_embed)

    def forward(self, x):                 # x: (Tp,B,d)
        if self.input_residual is not None:
            x = x + self.input_residual
            self.input_residual = None

        u = x.permute(1,2,0)              # (B,d,Tp)
        u = self.drop(self.act(self.bn(self.conv(u))))
        out = u + x.permute(1,2,0)        # local skip
        return self.ln_post(out.permute(2,0,1))   # (Tp,B,d)


# ────────── GRU Block (unchanged) ──────────
class GRUBlock(nn.Module):
    def __init__(self, dim, p_drop):
        super().__init__()
        self.input_residual = None
        self.gru  = nn.GRU(dim, dim, 1)
        self.drop = nn.Dropout(p_drop)
        self.ln_post = nn.LayerNorm(dim)

    def forward(self, x):                 # x: (Tp,B,d)
        if self.input_residual is not None:
            x = x + self.input_residual
            self.input_residual = None
        z, _ = self.gru(x)
        return self.ln_post(self.drop(z))    # (Tp,B,d)

# ────────── Cross‑Modal Co‑Attention (new!) ──────────
class CrossModalCoAttn(nn.Module):
    def __init__(self, dim, n_heads, p_drop):
        super().__init__()
        self.q_proj  = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.attn    = nn.MultiheadAttention(dim, n_heads, dropout=p_drop)
        self.ln      = nn.LayerNorm(dim)
        self.last_attn = None                 # 저장용 버퍼

    def forward(self, feat_c, feat_r, *, save=False):
        """
        feat_c : (Tp, B, d)  Conv-stream query
        feat_r : (Tp, B, d)  GRU-stream key/value
        save   : True 면 self.last_attn 에 attention weight(softmax) 보관
        """
        Q = self.q_proj(feat_c)
        KV = self.kv_proj(feat_r)
        K, V = KV.chunk(2, dim=-1)
        z, w = self.attn(Q, K, V, need_weights=True)   # w: (B, Tp, Tp)
        if save:
            self.last_attn = w.detach()
        return self.ln(z + Q)          # Residual + LayerNorm

# ─────────────────────────────────────────────────────────────
# Fast HyperConv1D  (single group-conv, no Python for-loop)
# ─────────────────────────────────────────────────────────────
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
        T, B, d = X.shape             # d == self.d
        pad = (self.k - 1) // 2

        # ---------- ① 커널 생성 ----------
        # (B,d*k) → (B*d,1,k)  depth-wise kernel
        W = self.gen(cond_vec).view(B * d, 1, self.k)

        # ---------- ② 입력을 group-conv 형태로 ----------
        # X:(T,B,d) → (1,B*d,T)
        X_grp = X.permute(1, 2, 0).reshape(1, B * d, T)
        X_grp = F.pad(X_grp, (pad, pad))

        # ---------- ③ single group-conv ----------
        Y = F.conv1d(X_grp, W, groups=B * d)     # (1,B*d,T)

        # ---------- ④ 원래 모양 복원 ----------
        Y = Y.view(B, d, T).permute(2, 0, 1)     # (T,B,d)
        return Y
        
# ─────────────────────────────────────────────────────────────
# Cross-HyperConv + Cross-Stitch
# ─────────────────────────────────────────────────────────────
class CrossHyperConvBlock(nn.Module):
    """
    Dual-direction HyperConv + α Cross-Stitch
    """
    def __init__(self, d_embed, k=3, p_drop=0.1):
        super().__init__()
        self.hc_gc = HyperConv1D(d_embed, k)   # GRU → Conv
        self.hc_cg = HyperConv1D(d_embed, k)   # Conv → GRU

        # α 초기 diag=0.95, off-diag=0.02
        alpha = torch.eye(2) * 0.90 + 0.05
        self.alpha = nn.Parameter(alpha)

        self.ln_c = nn.LayerNorm(d_embed)
        self.ln_r = nn.LayerNorm(d_embed)
        self.drop = nn.Dropout(p_drop)

    def forward(self, C, R):                   # (T,B,d)
        cond_gc = R[-1]                        # (B,d)
        cond_cg = C[-1]

        C_h = self.hc_gc(C, cond_gc)           # GRU → Conv
        R_h = self.hc_cg(R, cond_cg)           # Conv → GRU

        a = torch.relu(self.alpha)             # ensure α ≥ 0
        C_mix = a[0,0]*C_h + a[0,1]*R_h
        R_mix = a[1,0]*C_h + a[1,1]*R_h

        C = self.ln_c(self.drop(C_mix) + C)    # Residual + LN
        R = self.ln_r(self.drop(R_mix) + R)
        return C, R

# ─────────────────────────────────────────────────────────────
# HybridTS with Dual-HyperConv + Cross-Stitch
# ─────────────────────────────────────────────────────────────
class HybridTS(nn.Module):
    def __init__(self, cfg, n_vars:int):
        super().__init__()
        d, dp = CFG["d_embed"], CFG["dropout_rate"]
        L, Pl = CFG["lookback"], CFG["patch_len"]
        self.Tp = L // Pl
        depth   = CFG["cnn_depth"]

        # Pre-net
        self.revin = RevIN(dim=n_vars, affine=True)
        self.patch = nn.Conv1d(n_vars, d, kernel_size=Pl, stride=Pl)

        # Backbone
        self.conv_blks = nn.ModuleList([ResConvBlock(d, 3, dp) for _ in range(depth)])
        self.gru_blks  = nn.ModuleList([GRUBlock   (d,    dp) for _ in range(depth)])
        self.xhconv_blks = nn.ModuleList([
            CrossHyperConvBlock(d, k=3, p_drop=dp) for _ in range(depth)
        ])

        # Fusion & Head
        self.fuse    = CrossModalCoAttn(d, CFG["n_heads"], dp)
        self.ln_flat = nn.LayerNorm(d)
        self.fc      = nn.Linear(self.Tp*d, CFG["horizon"]*n_vars)

    def forward(self, x: torch.Tensor, *, save_attn: bool = False):
        B, C, _ = x.shape
        z0 = self.patch(self.revin(x, mode='norm')).permute(2, 0, 1)  # (T,B,d)

        zc = zr = z0
        depth = len(self.conv_blks)
        use_hconv_depth = depth

        for i, (cb, gb) in enumerate(zip(self.conv_blks, self.gru_blks)):
            zc = cb(zc)
            zr = gb(zr)

            if i < use_hconv_depth:          # HyperConv+Stitch 적용 범위 제한
                zc, zr = self.xhconv_blks[i](zc, zr)

        mem  = self.fuse(zc, zr, save=save_attn)
        flat = self.ln_flat((mem + z0).permute(1, 0, 2)).reshape(B, -1)
        out  = self.fc(flat).view(B, CFG["horizon"], C).permute(0, 2, 1)
        return self.revin(out, mode='denorm')

# ────────────────── hook 업데이트 ──────────────────
def register_hooks(model):
    grads = defaultdict(list)
    for i,(cb,gb) in enumerate(zip(model.conv_blks, model.gru_blks)):
        cb.register_full_backward_hook(
            lambda m,gi,go,idx=i: grads[f'conv_{idx}'].append(go[0].norm().item()))
        gb.register_full_backward_hook(
            lambda m,gi,go,idx=i: grads[f'gru_{idx}'].append(go[0].norm().item()))
    for i,xh in enumerate(model.xhconv_blks):
        xh.register_full_backward_hook(
            lambda m,gi,go,idx=i: grads[f'hconv_{idx}'].append(go[0].norm().item()))
    return grads

# ---------------------------------------------------------------------

def _to_np(x):            # CPU numpy 변환
    return x.detach().cpu().numpy()

def regression_metrics(pred, target):
    """
    pred, target : (N, H) torch tensors, real-scale
    반환 dict keys: rmse, mae, rrmse, r2, cc, nse, pbias
    """
    p = pred.flatten()
    t = target.flatten()
    n = t.numel()

    mse  = torch.mean((p-t)**2)
    rmse = torch.sqrt(mse)
    mae  = torch.mean(torch.abs(p-t))
    rrmse= rmse / (t.mean().clamp(min=1e-6)).abs()

    # R² (coefficient of determination)
    ss_tot = torch.sum((t - t.mean())**2)
    r2     = 1 - mse * n / ss_tot

    # Pearson CC
    cov = torch.mean((p-p.mean())*(t-t.mean()))
    cc  = cov / (p.std()*t.std()+1e-6)

    # NSE (Nash–Sutcliffe efficiency)
    nse = 1 - torch.sum((p-t)**2) / (ss_tot + 1e-6)

    # PBIAS (%)
    pbias = 100 * torch.sum(t - p) / (torch.sum(t)+1e-6)

    return dict(rmse=rmse.item() , mae=mae.item() , rrmse=rrmse.item(),
                r2=r2.item()     , cc=cc.item()  , nse=nse.item(),
                pbias=pbias.item())
    
# --- loss ------------------------------------------------------------
def ps_loss(pred, tgt):
    """
    Patch-Stat + MSE  (α 규제·skip 규제 제거 버전)
    pred, tgt : (B, C, H)  – RevIN 정규화 공간
    """
    lam_ps = CFG["lam_patch"]

    # ① MSE
    mse = F.mse_loss(pred, tgt)

    # ② Patch-Stat (평균·표준편차 패널티)
    B, C, H = pred.shape
    patch = 8 * max(1, H // 24)          # ≈24-step
    p = pred.reshape(B * C, H).unfold(1, patch, patch)
    t = tgt .reshape(B * C, H).unfold(1, patch, patch)

    mean_err = (p.mean(-1) - t.mean(-1)).abs().mean()
    std_err  = (p.std (-1) - t.std (-1)).abs().mean()

    return mse + lam_ps * (mean_err + std_err)

# ---------------------------------------------------------------------

# ─────────────────────────────────────────────
# 1) train_one_epoch 
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, opt, mu, std, epoch):
    model.train()
    tot = mse_sum = mse_real = cnt = 0
    for xb, yb in tqdm(loader, desc=f"Train {epoch:02d}", leave=False):
        xb, yb = xb.to(CFG["device"]), yb.to(CFG["device"])

        pred  = model(xb)
        loss  = ps_loss(pred, yb)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        opt.step()

        bs = xb.size(0)
        tot       += loss.item()*bs
        mse_sum   += F.mse_loss(pred, yb).item()*bs
        mse_real  += F.mse_loss(pred*std+mu, yb*std+mu).item()*bs
        cnt       += bs
    return tot/cnt, mse_sum/cnt, mse_real/cnt


# ─────────────────────────────────────────────
# 2) evaluate 
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, mu, std, tag="Val", epoch=None, metrics=False):
    model.eval()
    tot = mse_sum = mse_real = cnt = 0
    pr, tg = [], []

    for xb, yb in tqdm(loader, desc=f"{tag}", leave=False):
        xb, yb = xb.to(CFG["device"]), yb.to(CFG["device"])

        pred  = model(xb)
        loss  = ps_loss(pred, yb)

        bs = xb.size(0)
        tot       += loss.item()*bs
        mse_sum   += F.mse_loss(pred, yb).item()*bs
        mse_real  += F.mse_loss(pred*std+mu, yb*std+mu).item()*bs
        cnt       += bs

        if metrics:
            pr.append((pred*std+mu).cpu());  tg.append((yb*std+mu).cpu())

    extra = {}
    if metrics:
        P = torch.cat(pr);  T = torch.cat(tg)
        extra = regression_metrics(P, T)
        H = P.size(2)
        extra["per_step"] = [regression_metrics(P[:,:,h], T[:,:,h]) for h in range(H)]

    return tot/cnt, mse_sum/cnt, mse_real/cnt, extra
    
# ───────────────────────────────────────────────────────────
# EarlyStop
# ───────────────────────────────────────────────────────────
class EarlyStop:
    def __init__(self, patience=15, min_delta=1e-3):
        self.best = 1e18; self.wait=0; self.p=patience; self.md=min_delta
    def step(self, val):
        if val < self.best - self.md:
            self.best=val; self.wait=0; return False
        self.wait += 1
        return self.wait >= self.p

# ───────────────────────────────────────────────
# register_hooks  (Dual-HyperConv + Cross-Stitch 전용)
# ───────────────────────────────────────────────
def register_hooks(model):
    """
    저장되는 grads[key] 항목
      conv_ℓ   : Conv 블록 ℓ  출력 ∥∇y∥₂
      gru_ℓ    : GRU  블록 ℓ  출력 ∥∇y∥₂
      hyper_ℓ  : Cross-HyperConv 블록 ℓ, Conv-스트림 ∥∇y∥₂
      stitch_ℓ : Cross-HyperConv 블록 ℓ, GRU-스트림 ∥∇y∥₂
    """
    grads = defaultdict(list)

    # ── helper ───────────────────────────────
    def make_single_hook(key):
        def _hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is not None:                       # None 이면 건너뛰기
                grads[key].append(g.norm().item())
        return _hook

    def make_dual_hook(idx):
        """Cross-HyperConv 블록: (Conv_grad, GRU_grad) 두 개"""
        def _hook(module, grad_input, grad_output):
            gC, gR = grad_output
            if gC is not None:
                grads[f'hyper_{idx}'].append(gC.norm().item())
            if gR is not None:
                grads[f'stitch_{idx}'].append(gR.norm().item())
        return _hook

    # ── Conv / GRU 블록 ───────────────────────
    for i, (cb, gb) in enumerate(zip(model.conv_blks, model.gru_blks)):
        cb.register_full_backward_hook(make_single_hook(f'conv_{i}'))
        gb.register_full_backward_hook(make_single_hook(f'gru_{i}'))

    # ── Cross-HyperConv 블록 ──────────────────
    for i, xhb in enumerate(model.xhconv_blks):
        xhb.register_full_backward_hook(make_dual_hook(i))

    return grads
    
# --------------------------------------------------------------
#  유틸: 결과 + 하이퍼파라미터를 CSV에 병합 저장
# --------------------------------------------------------------

def save_results_csv(dataset_tag, model_tag, seed,
                     test_mse_std, test_mse_real,
                     extra, cfg, out_dir):

    fpath = out_dir / f"metrics_{dataset_tag}.csv"

    # --- ① extra 에서 스칼라만 추출 ---
    extra_clean = {k: v for k, v in extra.items()
                   if isinstance(v, (int, float, np.floating))}

    # --- ② 새 행 생성 ---
    row = {
        "dataset"   : dataset_tag,
        "model"     : model_tag,
        "seed"      : seed,
        "horizon"   : cfg["horizon"],
        "depth"   : cfg["cnn_depth"],
        "lam_patch" : cfg["lam_patch"],
        "lamb_reg"  : cfg["lamb_reg"],
        "mse_std"   : test_mse_std,
        "mse_real"  : test_mse_real,
    }
    row.update(extra_clean)          # per_step 등 비스칼라는 제외
    df_new = pd.DataFrame([row])

    # --- ③ CSV 병합 후 저장 ---
    if fpath.exists():
        df = pd.read_csv(fpath)
        dup = (
            (df.dataset == dataset_tag) &
            (df.model   == model_tag  ) &
            (df.seed    == seed       ) &
            (df.horizon   == cfg["horizon"]) & 
            (df.depth   == cfg["cnn_depth"]) & 
            (df.lam_patch == cfg["lam_patch"]) &
            (df.lamb_reg  == cfg["lamb_reg"])
        )
        df = df[~dup]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(fpath, index=False)
    print(f"saved metrics → {fpath}")

# ===========================================================
# [CTSF-V1 단일 셀 패치] 방향별 교차 on/off + '직접 근거' 지표(무파라미터)
#  - 원본 백본/훈련/스케줄러는 그대로 사용합니다.
#  - 변경/추가 라인 표기: # [PATCH] / # [NEW]
# ===========================================================
# -----------------------------------------------------------
# 0) CrossHyperConvBlock에 '방향 스위치' 패치
#    - GRU→Conv(use_gc), Conv→GRU(use_cg) on/off
# -----------------------------------------------------------
if not hasattr(CrossHyperConvBlock, "_orig_forward"):       # [PATCH] 중복패치 방지
    CrossHyperConvBlock._orig_forward = CrossHyperConvBlock.forward

def _forward_with_dirswitch(self, C, R):                     # (T,B,d) 튜플 반환
    use_gc = getattr(self, "use_gc", True)   # GRU → Conv
    use_cg = getattr(self, "use_cg", True)   # Conv → GRU

    C_in, R_in = C, R
    if use_gc:
        cond_gc = R_in[-1]                   # (B,d)
        C_h = self.hc_gc(C_in, cond_gc)      # GRU → Conv
    else:
        C_h = C_in                            # [PATCH] 우회(항등)

    if use_cg:
        cond_cg = C_in[-1]
        R_h = self.hc_cg(R_in, cond_cg)      # Conv → GRU
    else:
        R_h = R_in                            # [PATCH] 우회(항등)

    a = torch.relu(self.alpha)
    if not use_gc:
        a = a.clone(); a[0,1] = 0.0           # [PATCH] GRU→Conv 혼합 차단
    if not use_cg:
        a = a.clone(); a[1,0] = 0.0           # [PATCH] Conv→GRU 혼합 차단

    C_mix = a[0,0]*C_h + a[0,1]*R_h
    R_mix = a[1,0]*C_h + a[1,1]*R_h

    C = self.ln_c(self.drop(C_mix) + C_in)    # Residual + LN
    R = self.ln_r(self.drop(R_mix) + R_in)
    return C, R

CrossHyperConvBlock.forward = _forward_with_dirswitch        # [PATCH]

def set_dir_switches(model, use_gc=True, use_cg=True):       # [NEW]
    """모든 교차 블록에 방향 스위치 주입"""
    for blk in model.xhconv_blks:
        blk.use_gc = bool(use_gc)
        blk.use_cg = bool(use_cg)

# -----------------------------------------------------------
# 1) 마지막 블록 내부 신호 후크(직접 근거 수집용)
#    - Conv→GRU: Conv(교차前) vs GRU(교차後) 변화량 시퀀스 의존성
#    - GRU→Conv: hc_gc.gen(Linear) 생성 커널과 시간대(hour) 의존성
# -----------------------------------------------------------
def _get_last_module(model, candidates):
    for name in candidates:
        modlist = getattr(model, name, None)
        if modlist is not None and hasattr(modlist, "__len__") and len(modlist) > 0:
            return modlist[-1]
    raise AttributeError(f"None of {candidates} found in model (or empty).")

class _LastBlockHooks:
    """
    evaluate 시점에 내부 신호 수집:
      - Conv 마지막 블록(교차 前 Conv 출력)
      - CrossHyperConv 마지막 블록(교차 後 C/R 출력)
      - GRU→Conv 하이퍼컨브의 커널 생성기(Linear 출력)
    """
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.C_last_pre = None
        self.C_last_post = None
        self.R_last_post = None
        self.W_gc_list = []

        # 이름 폴백: conv_blks > resconv_blks > cnn_blks
        self.res_last = _get_last_module(model, ["conv_blks","resconv_blks","cnn_blks"])
        # 이름 폴백: xhconv_blks > xh_blks > cross_hc_blks
        self.xh_last  = _get_last_module(model, ["xhconv_blks","xh_blks","cross_hc_blks"])

        if not hasattr(self.xh_last, "hc_gc") or not hasattr(self.xh_last.hc_gc, "gen"):
            raise AttributeError("xh_last.hc_gc.gen not found – HyperConv1D(gen) 경로 확인 필요")
        self.gc_gen   = self.xh_last.hc_gc.gen

    def attach(self):
        def _h_res(mod, inp, out):
            self.C_last_pre = out.detach()
        def _h_xh(mod, inp, out):
            C_out, R_out = out
            self.C_last_post = C_out.detach()
            self.R_last_post = R_out.detach()
        def _h_gc(mod, inp, out):
            self.W_gc_list.append(out.detach().cpu())

        self.handles.append(self.res_last.register_forward_hook(_h_res))
        self.handles.append(self.xh_last.register_forward_hook(_h_xh))
        self.handles.append(self.gc_gen.register_forward_hook(_h_gc))

    def detach(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles = []
        
# -----------------------------------------------------------
# 2) HP2 적용 + 테스트 시간대(hour) 행렬 생성
# -----------------------------------------------------------
def apply_HP2(CFG, *, csv_path, seed, horizon, device=None, out_root="results_dirswitch", model_tag=None):  # [NEW]
    CFG.update(dict(
        seed=int(seed),
        csv_path=str(csv_path),
        horizon=int(horizon),
        lookback=720, patch_len=36,              # HP2
        out_dir=str(out_root),
        model_tag=(model_tag or "HyperConv"),
    ))
    if device is not None:
        CFG["device"] = device
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])

def _find_datetime_col(df):                                                           # [NEW]
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["date", "time", "datetime"]):
            return c
    return None

def build_test_hour_matrix(CFG):                                                      # [NEW]
    """테스트 세트 예측 시작시각의 '시(hour)' 행렬 (N_test,H)"""
    df = pd.read_csv(CFG["csv_path"])
    col = _find_datetime_col(df)
    if col is None: return None
    ts = pd.to_datetime(df[col]); hours = ts.dt.hour.to_numpy()

    L, H = CFG["lookback"], CFG["horizon"]
    total = len(df) - L - H + 1
    n_tr  = int(total * CFG["train_ratio"])
    n_val = int(total * CFG["val_ratio"])
    margin = L + H
    te_st = n_tr + margin + n_val + margin

    N_test = (len(df) - te_st) - L - H + 1
    if N_test <= 0: return None

    mat = np.zeros((N_test, H), dtype=np.int16)
    for i in range(N_test):
        base = te_st + i + L
        mat[i,:] = hours[base:base+H]
    return mat

# -----------------------------------------------------------
# 3) 무파라미터 보조 수학 유틸(상관/거리상관)
# -----------------------------------------------------------

# --- TOD 유틸: 하루 내 상대 위치 → 원형 인코딩 [sin, cos] ---
def _tod_sincos_from_ts(ts_series):
    ts = pd.to_datetime(ts_series)
    # 하루 내 초(second-of-day) 추출
    sod = (ts.dt.hour.astype(np.int64)*3600 +
           ts.dt.minute.astype(np.int64)*60 +
           ts.dt.second.astype(np.int64)).to_numpy(dtype=np.float32)
    ang = 2*np.pi*(sod/86400.0)  # 24h 주기
    return np.stack([np.sin(ang), np.cos(ang)], axis=1)  # (T,2)

def _find_datetime_col(df):
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["date","time","datetime"]):
            return c
    return None

def build_test_tod_vector(CFG):
    """
    테스트 세트 각 샘플(윈도)의 '예측 시작 시점'에 대한 TOD [sin,cos] 라벨 생성.
    반환: (N_test, 2)
    """
    df = pd.read_csv(CFG["csv_path"])
    col = _find_datetime_col(df)
    if col is None:
        return None

    tod_all = _tod_sincos_from_ts(df[col])  # (T_all, 2)

    L, H = CFG["lookback"], CFG["horizon"]
    total = len(df) - L - H + 1
    n_tr  = int(total * CFG["train_ratio"])
    n_val = int(total * CFG["val_ratio"])
    margin = L + H
    te_st = n_tr + margin + n_val + margin

    N_test = (len(df) - te_st) - L - H + 1
    if N_test <= 0:
        return None

    tod_vec = np.zeros((N_test, 2), dtype=np.float32)
    for i in range(N_test):
        base = te_st + i + L                # 예측 시작(첫 step) 시각
        tod_vec[i, :] = tod_all[base, :]    # [sin, cos]
    return tod_vec  # (N_test, 2)

def _zscore_t(a, dim=1, eps=1e-8):
    m = a.mean(dim=dim, keepdim=True); s = a.std(dim=dim, keepdim=True) + eps
    return (a - m) / s

def _time_deriv_norm(x):  # (T,B,d)->(B,T-1)
    dx = x[1:] - x[:-1]
    n  = torch.linalg.vector_norm(dx, dim=-1)
    return n.transpose(0,1).contiguous()

def _pearson_corr(a, b):  # (B,T)->(B,)
    a = _zscore_t(a, dim=1); b = _zscore_t(b, dim=1)
    return (a*b).mean(dim=1).clamp(-1,1)

def _rank_2d(x_np):
    return np.argsort(np.argsort(x_np, axis=1), axis=1).astype(np.float32)

def _spearman_corr(a, b):
    a_np, b_np = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    ra = torch.tensor(_rank_2d(a_np), device=a.device)
    rb = torch.tensor(_rank_2d(b_np), device=b.device)
    return _pearson_corr(ra, rb)

def _event_gain_and_hit(conv_d_b, gru_d_b, q_gain=0.80, q_hit=0.90):
    """
    conv_d_b, gru_d_b: (B,T)
    반환: gain_mean, hit_mean
      - gain: Conv 상위 q_gain 구간에서의 GRU 업데이트 평균 / 그 외 평균
      - hit : Conv 상위 q_hit 이벤트(피크)가 있을 때 ±1 step 이내에 GRU 상위 q_hit 이벤트가 있었던 비율
    """
    gains, hits = [], []
    a_np = conv_d_b.detach().cpu().numpy()
    b_np = gru_d_b.detach().cpu().numpy()
    B, T = a_np.shape
    for i in range(B):
        a = a_np[i]; b = b_np[i]
        # --- gain ---
        thr_g = np.quantile(a, q_gain)
        hi = a >= thr_g; lo = ~hi
        if hi.any() and lo.any():
            gain = float(b[hi].mean() / (b[lo].mean() + 1e-12))
            gains.append(gain)
        # --- hit ---
        thr_h_a = np.quantile(a, q_hit)
        thr_h_b = np.quantile(b, q_hit)
        a_top = a >= thr_h_a
        b_top = b >= thr_h_b
        # GRU 톱 이벤트를 ±1 확장
        b_dil = b_top.copy()
        if T > 1:
            b_dil[:-1] |= b_top[1:]
            b_dil[1:]  |= b_top[:-1]
        if a_top.sum() > 0:
            hit = float((a_top & b_dil).sum() / a_top.sum())
            hits.append(hit)
    gain_mean = float(np.mean(gains)) if gains else np.nan
    hit_mean  = float(np.mean(hits))  if hits  else np.nan
    return gain_mean, hit_mean

# ---------- 안전한 상관 계산(0-lag) ----------
def _safe_corr(aa, bb):
    aa = np.asarray(aa, dtype=np.float64)
    bb = np.asarray(bb, dtype=np.float64)
    if aa.size != bb.size or aa.size < 2:
        return np.nan
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    va = (aa**2).sum()
    vb = (bb**2).sum()
    if va <= 1e-12 or vb <= 1e-12:
        return np.nan
    r = float(np.dot(aa, bb) / np.sqrt(va * vb))
    if not np.isfinite(r):
        return np.nan
    # 수치 안정: 경계값 딱 1/−1로 포화되는 것을 미세하게 완화
    r = max(min(r, 1.0 - 1e-9), -1.0 + 1e-9)
    return r

# ---------- 라그별 최대 상관(겹침-가중 방식, 하이퍼파라미터 무사용) ----------
def _maxcorr_and_lag_mean(conv_d_b, gru_d_b):
    """
    conv_d_b, gru_d_b: (B, N)
    - |L| ≤ N - min_overlap 범위에서 탐색
    - min_overlap = max(3, floor(sqrt(N))) : 데이터 길이에 따른 '자동' 안전치
      (임의 하이퍼파라미터가 아니라 N으로부터 결정)
    - 라그별 상관 r(L)에 겹침비율 w(L)=M/N 가중 → 짧은 겹침의 과대평가 방지
    반환: (배치평균 최대상관 |r|, 배치평균 최적 라그)
    """
    a = conv_d_b.detach().cpu().numpy().astype(np.float64)
    b = gru_d_b.detach().cpu().numpy().astype(np.float64)
    Na, Nb = a.shape[1], b.shape[1]
    N = min(Na, Nb)
    if N < 5:
        return float("nan"), float("nan")

    min_overlap = max(3, int(np.sqrt(N)))   # 데이터 길이에 따른 자동 기준
    bests, lags = [], []
    for i in range(a.shape[0]):
        ai = a[i, :N]; bi = b[i, :N]
        # z-score
        ai = (ai - ai.mean()) / (ai.std() + 1e-8)
        bi = (bi - bi.mean()) / (bi.std() + 1e-8)

        best_eff, best_raw, bestL = -1.0, 0.0, 0
        # |L| ≤ N - min_overlap  ⇒ 겹침 M=N-|L| ≥ min_overlap 보장
        for L in range(-(N - min_overlap), (N - min_overlap) + 1):
            if L >= 0:
                aa, bb = ai[:N-L], bi[L:]
            else:
                aa, bb = ai[-L:],  bi[:N+L]
            M = aa.shape[0]
            if M < min_overlap:
                continue
            r = _safe_corr(aa, bb)
            if not np.isfinite(r):
                continue
            w = M / N
            eff = abs(r) * w
            if eff > best_eff:
                best_eff, best_raw, bestL = eff, abs(r), L
        bests.append(best_raw); lags.append(bestL)

    return float(np.mean(bests)), float(np.mean(lags))

# ---------- 거리상관(dCor) 안정화: U-centering ----------
def _pairwise_dmat(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    XX = (X**2).sum(axis=1, keepdims=True)
    D2 = XX + XX.T - 2.0 * X.dot(X.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, dtype=np.float64)

def _dcor_u(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if n < 4 or Y.shape[0] != n:  # 표본 너무 작거나 길이 불일치
        return np.nan
    A = _pairwise_dmat(X); B = _pairwise_dmat(Y)
    np.fill_diagonal(A, 0.0); np.fill_diagonal(B, 0.0)
    rowA = A.sum(axis=1)/(n-1); rowB = B.sum(axis=1)/(n-1)
    gA = A.sum()/(n*(n-1));    gB = B.sum()/(n*(n-1))
    A = A - rowA[:,None] - rowA[None,:] + gA
    B = B - rowB[:,None] - rowB[None,:] + gB
    np.fill_diagonal(A, 0.0); np.fill_diagonal(B, 0.0)
    denom = n*(n-3)
    if denom <= 0: return np.nan
    dcov2 = (A*B).sum()/denom
    dvarx = (A*A).sum()/denom
    dvary = (B*B).sum()/denom
    dcov2 = max(0.0, dcov2); dvarx = max(dvarx, 1e-16); dvary = max(dvary, 1e-16)
    val = np.sqrt(dcov2)/np.sqrt(dvarx*dvary)
    if not np.isfinite(val): return np.nan
    return float(min(max(val, 0.0), 1.0-1e-9))

def _dcor_bt_u(a_bt: torch.Tensor, b_bt: torch.Tensor) -> np.ndarray:
    a_np = a_bt.detach().cpu().numpy().astype(np.float64)
    b_np = b_bt.detach().cpu().numpy().astype(np.float64)
    out=[]
    for i in range(a_np.shape[0]):
        out.append(_dcor_u(a_np[i][:,None], b_np[i][:,None]))
    return np.array(out, dtype=np.float64)

# ---------- TOD 라벨(시간대) 생성(이미 build_test_tod_vector가 있으면 재사용) ----------
def _find_datetime_col(df):
    for c in df.columns:
        cl=str(c).lower()
        if any(k in cl for k in ["date","time","datetime"]): return c
    return None

def build_test_tod_vector(CFG):
    df = pd.read_csv(CFG["csv_path"]); col = _find_datetime_col(df)
    if col is None: return None
    ts = pd.to_datetime(df[col])
    sod = (ts.dt.hour*3600 + ts.dt.minute*60 + ts.dt.second).to_numpy(np.float64)
    ang = 2*np.pi*(sod/86400.0)
    tod_all = np.stack([np.sin(ang), np.cos(ang)], axis=1)
    L,H = CFG["lookback"], CFG["horizon"]
    total = len(df)-L-H+1
    n_tr  = int(total*CFG["train_ratio"])
    n_val = int(total*CFG["val_ratio"])
    margin=L+H; te_st = n_tr+margin + n_val+margin
    N_test = (len(df)-te_st)-L-H+1
    if N_test<=0: return None
    out=np.zeros((N_test,2),dtype=np.float64)
    for i in range(N_test):
        base = te_st+i+L
        out[i,:]=tod_all[base,:]
    return out  # (N_test,2)

# ----- multi-output R² (E ~ TOD[sin,cos]) -----
def _multioutput_r2(Y: np.ndarray, X: np.ndarray) -> float:
    # Y: (N,d), X: (N,2)
    N = Y.shape[0]
    if N<4 or X.shape[0]!=N: return np.nan
    X1 = np.concatenate([X, np.ones((N,1))], axis=1)  # bias
    # 최소제곱 해(다중출력)
    beta, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    Yhat = X1 @ beta
    ss_res = ((Y - Yhat)**2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0))**2).sum(axis=0) + 1e-12
    r2 = 1.0 - (ss_res/ss_tot)
    r2 = np.clip(r2, 0.0, 1.0-1e-9)
    return float(np.nanmean(r2))

# ----- NEW: 경로 활성 여부 -----
def _gc_active(model):
    return any(getattr(blk, "use_gc", True) for blk in model.xhconv_blks)
def _cg_active(model):
    return any(getattr(blk, "use_cg", True) for blk in model.xhconv_blks)

# -----------------------------------------------------------
# 4) 테스트 한 바퀴 + '직접 근거' 지표 수집(무파라미터)
# -----------------------------------------------------------
@torch.no_grad()
def evaluate_with_direct_evidence(model, loader, mu, std, tod_vec=None):
    model.eval()
    device = CFG["device"]

    # 경로 on/off (없으면 True로 간주)
    gc_on = _gc_active(model) if '_gc_active' in globals() else True
    cg_on = _cg_active(model) if '_cg_active' in globals() else True

    # 누적 버퍼
    Pz_all, Tz_all, P_all, T_all = [], [], [], []

    # Conv→GRU
    cg_p_list, cg_s_list, cg_dcor_list = [], [], []
    gain_list, hit_list, maxc_list, blag_list = [], [], [], []

    # GRU→Conv
    W_list, E_list, Y_list = [], [], []  # 생성커널, Conv변화, TOD[sin,cos]

    hooks = _LastBlockHooks(model); hooks.attach()
    sample_idx_base = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        # TOD 라벨(예측 시작 시점)
        y_tod = None
        if tod_vec is not None:
            B = xb.size(0)
            y_tod = tod_vec[sample_idx_base: sample_idx_base+B, :]
            sample_idx_base += B

        # 예측
        pred = model(xb)
        # 표준화/실측 스케일 동시 누적
        Pz_all.append(pred.detach().cpu().numpy())
        Tz_all.append(yb.detach().cpu().numpy())
        P_all.append((pred*std+mu).detach().cpu().numpy())
        T_all.append((yb  *std+mu).detach().cpu().numpy())

        # 내부 신호 (마지막 블록)
        C_pre  = hooks.C_last_pre
        C_post = hooks.C_last_post
        R_post = hooks.R_last_post

        # ---- Conv→GRU: 변화량 의존 지표들 ----
        if (C_pre is not None) and (R_post is not None):
            conv_d = _time_deriv_norm(C_pre)   # (B,T-1)
            gru_d  = _time_deriv_norm(R_post)  # (B,T-1)

            # (1) 0-lag Pearson: 정의 함수로 1회만 계산
            cg_p_list.append(_pearson_corr(conv_d, gru_d).cpu().numpy())

            # (2) Spearman: 정의된 _spearman_corr 직접 호출
            cg_s_list.append(_spearman_corr(conv_d, gru_d).cpu().numpy())

            # (3) 거리상관(U-centering) — 비선형 의존
            cg_dcor_list.append(_dcor_bt_u(conv_d, gru_d))

            # (4) 이벤트 지표(피크 반응 크기/적중률)
            if '_event_gain_and_hit' in globals():
                g_gain, g_hit = _event_gain_and_hit(conv_d, gru_d, q_gain=0.80, q_hit=0.90)
                gain_list.append(g_gain); hit_list.append(g_hit)

            # (5) 최대 동행도/최적 지연(가중 라그 상관)
            if '_maxcorr_and_lag_mean' in globals():
                maxcorr, bestlag = _maxcorr_and_lag_mean(conv_d, gru_d)
            else:
                maxcorr, bestlag = float('nan'), float('nan')
            maxc_list.append(maxcorr); blag_list.append(bestlag)

        # ---- GRU→Conv: 생성 커널/Conv 변화/TOD 수집 ----
        if len(hooks.W_gc_list) > 0:
            W_list.append(hooks.W_gc_list[-1].numpy())  # (B, d*k)
        if (C_pre is not None) and (C_post is not None):
            delta = C_post - C_pre                       # (T,B,d)
            E = torch.sqrt(torch.mean(delta**2, dim=0))  # (B,d)
            E_list.append(E.detach().cpu().numpy())
        if y_tod is not None:
            Y_list.append(y_tod)

    hooks.detach()

    # ---- 성능 (표준화/실측) ----
    Pz = np.concatenate(Pz_all, axis=0); Tz = np.concatenate(Tz_all, axis=0)
    P  = np.concatenate(P_all , axis=0); T  = np.concatenate(T_all , axis=0)
    mse_std  = float(((Pz - Tz)**2).mean())
    mse_real = float(((P  - T )**2).mean())
    rmse     = float(np.sqrt(mse_real))

    # ---- Conv→GRU 요약 ----
    def _m(L): return float(np.mean(np.concatenate(L))) if L else float('nan')
    cg_pearson_mean  = _m(cg_p_list)
    cg_spearman_mean = _m(cg_s_list)
    cg_dcor_mean     = _m(cg_dcor_list)
    cg_event_gain    = float(np.nanmean(gain_list)) if gain_list else np.nan
    cg_event_hit     = float(np.nanmean(hit_list )) if hit_list  else np.nan
    cg_maxcorr       = float(np.nanmean(maxc_list)) if maxc_list else np.nan
    cg_bestlag       = float(np.nanmean(blag_list )) if blag_list else np.nan

    # ---- GRU→Conv 요약 (경로 off이면 NaN) ----
    gc_kernel_tod_dcor   = np.nan
    gc_feat_tod_dcor     = np.nan
    gc_feat_tod_r2       = np.nan
    gc_kernel_feat_dcor  = np.nan
    gc_kernel_feat_align = np.nan

    has_W = len(W_list)>0
    has_E = len(E_list)>0
    has_Y = len(Y_list)>0

    if gc_on:
        if has_Y and has_W:
            X = np.concatenate(W_list, axis=0)  # (N, d*k)
            Y = np.concatenate(Y_list, axis=0)  # (N, 2) TOD
            gc_kernel_tod_dcor = _dcor_u(X, Y)
        if has_Y and has_E:
            E = np.concatenate(E_list, axis=0)  # (N, d)
            Y = np.concatenate(Y_list, axis=0)  # (N, 2)
            gc_feat_tod_dcor = _dcor_u(E, Y)    # 참고용
            if '_multioutput_r2' in globals():
                gc_feat_tod_r2 = _multioutput_r2(E, Y)  # NEW: 설명력(0~1)
        if has_W and has_E:
            X = np.concatenate(W_list, axis=0)  # (N, d*k)
            E = np.concatenate(E_list, axis=0)  # (N, d)
            N, DK = X.shape
            if E.shape[1]>0 and DK % E.shape[1] == 0:
                d = E.shape[1]; k = DK // d
                X_sum = X.reshape(N, d, k).mean(axis=2)  # (N,d) tap 요약
                gc_kernel_feat_dcor  = _dcor_u(X_sum, E)
                gc_kernel_feat_align = gc_kernel_feat_dcor  # alias(해석 명확화)

    return dict(
        mse_std=mse_std, mse_real=mse_real, rmse=rmse,
        # Conv → GRU
        cg_pearson_mean=cg_pearson_mean,
        cg_spearman_mean=cg_spearman_mean,
        cg_dcor_mean=cg_dcor_mean,
        cg_event_gain=cg_event_gain,
        cg_event_hit=cg_event_hit,
        cg_maxcorr=cg_maxcorr,
        cg_bestlag=cg_bestlag,
        # GRU → Conv
        gc_kernel_tod_dcor=gc_kernel_tod_dcor,
        gc_feat_tod_dcor=gc_feat_tod_dcor,
        gc_feat_tod_r2=gc_feat_tod_r2,
        gc_kernel_feat_dcor=gc_kernel_feat_dcor,
        gc_kernel_feat_align=gc_kernel_feat_align,
    )

# --------- 단일 통합 CSV 저장(업서트) ----------
def save_results_unified(row: dict, out_root: str = "results_dirswitch"):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    fpath = out_root / "results.csv"
    cols_order = [
        "dataset","horizon","seed","mode","model_tag",
        "mse_std","mse_real","rmse",
        # Conv→GRU
        "cg_pearson_mean","cg_spearman_mean","cg_dcor_mean",
        "cg_event_gain","cg_event_hit","cg_maxcorr","cg_bestlag",
        # GRU→Conv (old+new)
        "gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2",
        "gc_kernel_feat_dcor","gc_kernel_feat_align",
    ]
    for c in cols_order:
        if c not in row: row[c] = np.nan
    new_df = pd.DataFrame([row])[cols_order]
    if fpath.exists():
        df = pd.read_csv(fpath)
        mask = ((df["dataset"]==row["dataset"]) &
                (df["horizon"]==row["horizon"]) &
                (df["seed"]==row["seed"]) &
                (df["mode"]==row["mode"]))
        df = df[~mask]
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(fpath, index=False)
    else:
        new_df.to_csv(fpath, index=False)
    return str(fpath)

# -----------------------------------------------------------
# 5) 단일 run (훈련/검증/테스트 + CSV 저장, ckpt 1개, 성공시 삭제)
# -----------------------------------------------------------
def train_eval_one_run(mode, *, keep_ckpt_if_crashed=True):
    device = CFG["device"]
    dataset_tag = Path(CFG["csv_path"]).stem
    out_root = CFG.get("out_dir","results_dirswitch")
    out_dir  = Path(out_root) / dataset_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_dl, val_dl, te_dl, (mu_np, std_np, C) = load_split_dataloaders(CFG)
    mu  = torch.tensor(mu_np , device=device).unsqueeze(-1)
    std = torch.tensor(std_np, device=device).unsqueeze(-1)

    model = HybridTS(CFG, n_vars=C).to(device)
    set_dir_switches(model,
        use_gc=(mode in ["both","gc_only"]),
        use_cg=(mode in ["both","cg_only"])
    )

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=CFG["lr"], total_steps=len(tr_dl)*CFG["epochs"],
        pct_start=0.15, div_factor=25., final_div_factor=1e4
    )
    stopper = EarlyStop(patience=20)

    run_tag = f"{dataset_tag}-h{CFG['horizon']}-s{CFG['seed']}-{mode}"
    ckpt_path = out_dir / f"ckpt_{run_tag}.pt"
    start_ep, best_val, best_state_cpu = 1, float("inf"), None
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sched"])
        start_ep = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        best_state_cpu = ckpt.get("best_model", None)
        print(f"[resume] {ckpt_path.name} @ epoch {ckpt['epoch']} (best_val={best_val:.6f})")
    if best_state_cpu is None:
        best_state_cpu = {k: v.detach().cpu() for k,v in model.state_dict().items()}

    # Train
    for ep in range(start_ep, CFG["epochs"] + 1):
        tr_loss, _, _ = train_one_epoch(model, tr_dl, opt, mu, std, ep)
        scheduler.step()
        val_loss, _, _, _ = evaluate(model, val_dl, mu, std, tag="Val", epoch=ep, metrics=False)
        print(f"[{ep:02d}] train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state_cpu = {k: v.detach().cpu() for k,v in model.state_dict().items()}

        if stopper.step(val_loss):
            print(f"[early stop] stop @ {ep}")
            torch.save({
                "epoch": ep,
                "model": {k: v.detach().cpu() for k,v in model.state_dict().items()},
                "opt": opt.state_dict(),
                "sched": scheduler.state_dict(),
                "best_val": best_val,
                "best_model": best_state_cpu,
            }, ckpt_path)
            break

        torch.save({
            "epoch": ep,
            "model": {k: v.detach().cpu() for k,v in model.state_dict().items()},
            "opt": opt.state_dict(),
            "sched": scheduler.state_dict(),
            "best_val": best_val,
            "best_model": best_state_cpu,
        }, ckpt_path)

    # === Test on BEST ===
    model.load_state_dict({k: v.to(device) for k,v in best_state_cpu.items()})
    tod_vec = build_test_tod_vector(CFG)  # ← hour가 아니라 TOD 사용
    direct = evaluate_with_direct_evidence(model, te_dl, mu, std, tod_vec=tod_vec)

    row = {
        "dataset": dataset_tag,
        "horizon": int(CFG["horizon"]),
        "seed":    int(CFG["seed"]),
        "mode":    mode,
        "model_tag": CFG.get("model_tag","HyperConv"),
        **direct
    }
    fpath = save_results_unified(row, out_root=out_root)
    print(f"[saved] {fpath}")

    # 정리
    try:
        if ckpt_path.exists(): os.remove(ckpt_path)
    except Exception as e:
        print(f"(warn) ckpt delete fail: {e}")
    del model, opt, scheduler, tr_dl, val_dl, te_dl, mu, std
    torch.cuda.empty_cache(); gc.collect()

# -----------------------------------------------------------
# 6) 스위트 실행기 (seed × horizon × dataset × mode)
# -----------------------------------------------------------

def _norm_str(x): 
    return str(x).strip()

def _try_int(x):
    try: return int(x)
    except: return None

def _csv_path_or_mnt(name, root_csv="datasets"):
    p1 = Path(root_csv)/f"{name}.csv"
    p2 = Path("/mnt/data")/f"{name}.csv"
    return str(p1 if p1.exists() else p2)

def _read_results_rows(results_root="results_dirswitch"):
    f = Path(results_root) / "results.csv"
    if not f.exists():
        return [], set()
    df = pd.read_csv(f)
    # 안전 변환
    if "dataset" in df.columns: df["dataset"] = df["dataset"].astype(str).str.strip()
    if "mode"    in df.columns: df["mode"]    = df["mode"].astype(str).str.strip()
    if "horizon" in df.columns: df["horizon"] = df["horizon"].apply(_try_int)
    if "seed"    in df.columns: df["seed"]    = df["seed"].apply(_try_int)

    rows = []
    done = set()
    for _, r in df.iterrows():
        key = (_norm_str(r.get("dataset","")),
               _try_int(r.get("horizon",None)),
               _try_int(r.get("seed",None)),
               _norm_str(r.get("mode","")))
        rows.append(key)
        done.add(key)
    return rows, done

def _in_filter(key, datasets, horizons, seeds, mode_cycle):
    ds, H, s, md = key
    return (ds in set(map(_norm_str, datasets)) and
            H  in set(map(int, horizons)) and
            s  in set(map(int, seeds)) and
            md in set(map(_norm_str, mode_cycle)))

def _next_mode_in_cycle(curr_mode, mode_cycle):
    try:
        i = list(mode_cycle).index(curr_mode)
        return mode_cycle[(i+1) % len(mode_cycle)]
    except ValueError:
        return mode_cycle[0]

def run_dir_switch_suite_plan(
    seeds=(42,2,3,5,7,11,13,17,19,23),
    horizons=(96,192,336,720),
    datasets=("ETTm1","ETTm2","ETTh1","ETTh2","weather"),
    mode_cycle=("both","cg_only","gc_only","none"),
    root_csv="datasets",
    results_root="results_dirswitch",
    resume_mode="next",     # 'next' | 'fill_missing' | 'all'
    overwrite=False,        # True면 완료 조합도 재실행(업서트 저장)
    max_jobs=None,
    device="cuda",
    dry_run=False
):
    # 0) 결과 파일 읽기
    rows, done_set_all = _read_results_rows(results_root)

    # 1) 현재 필터 내의 완료 집합만 추리기
    datasets = tuple(map(_norm_str, datasets))
    horizons = tuple(map(int, horizons))
    seeds    = tuple(map(int, seeds))
    mode_cycle = tuple(map(_norm_str, mode_cycle))

    grid_keys = set(
        (_norm_str(ds), int(H), int(s), _norm_str(md))
        for ds in datasets for H in horizons for s in seeds for md in mode_cycle
    )
    done_set = {k for k in done_set_all if k in grid_keys}

    # 2) CSV '마지막 줄'에서 필터에 해당하는 마지막 완료 조합 찾기
    last_key = None
    for key in reversed(rows):
        if key in grid_keys:
            last_key = key
            break

    # 3) 실행 계획 만들기
    plan = []

    def _append_if_needed(ds, H, s, md):
        key = (_norm_str(ds), int(H), int(s), _norm_str(md))
        if (not overwrite) and (key in done_set):
            return
        plan.append(key)

    if resume_mode == "next":
        # (a) last_key가 필터 밖이면, 필터 첫 조합부터 시작
        if (last_key is None) or (not _in_filter(last_key, datasets, horizons, seeds, mode_cycle)):
            start_ds_idx = 0; start_H_idx = 0; start_s_idx = 0
            start_mode_idx = 0
        else:
            ds_last, H_last, s_last, md_last = last_key
            try:
                start_ds_idx = list(datasets).index(ds_last)
                start_H_idx  = list(horizons).index(H_last)
                start_s_idx  = list(seeds).index(s_last)
            except ValueError:
                # 필터와 어긋나면 처음부터
                start_ds_idx = 0; start_H_idx = 0; start_s_idx = 0
                md_last = None

            # (b) 같은 (ds,H,seed)에서 '다음 모드'부터 남은 모드 소진
            if md_last in mode_cycle:
                first_mode_idx = (list(mode_cycle).index(md_last) + 1) % len(mode_cycle)
            else:
                first_mode_idx = 0

            ds = datasets[start_ds_idx]; H = horizons[start_H_idx]; s = seeds[start_s_idx]
            for mi in range(first_mode_idx, len(mode_cycle)):
                _append_if_needed(ds, H, s, mode_cycle[mi])

            # (c) 그 seed 이후의 seed들, 그 horizon의 나머지, 그 dataset 내부를 순회
            for s_i in range(start_s_idx+1, len(seeds)):
                s2 = seeds[s_i]
                for mi in range(0, len(mode_cycle)):
                    _append_if_needed(ds, H, s2, mode_cycle[mi])

            # (d) 같은 dataset의 나머지 horizon들
            for h_i in range(start_H_idx+1, len(horizons)):
                H2 = horizons[h_i]
                for s2 in seeds:
                    for md in mode_cycle:
                        _append_if_needed(ds, H2, s2, md)

            # (e) 나머지 dataset들
            for d_i in range(start_ds_idx+1, len(datasets)):
                ds2 = datasets[d_i]
                for H2 in horizons:
                    for s2 in seeds:
                        for md in mode_cycle:
                            _append_if_needed(ds2, H2, s2, md)

    elif resume_mode == "fill_missing":
        for ds in datasets:
            for H in horizons:
                for s in seeds:
                    for md in mode_cycle:
                        _append_if_needed(ds, H, s, md)

    elif resume_mode == "all":
        for ds in datasets:
            for H in horizons:
                for s in seeds:
                    for md in mode_cycle:
                        plan.append((_norm_str(ds), int(H), int(s), _norm_str(md)))
    else:
        raise ValueError("resume_mode must be 'next' | 'fill_missing' | 'all'")

    if max_jobs is not None:
        plan = plan[:int(max_jobs)]

    # 4) 계획 요약 출력
    print(f"[plan] mode={resume_mode} | overwrite={overwrite} | grid={len(grid_keys)} | done={len(done_set)} | to_run={len(plan)} | last={last_key}")
    if plan:
        preview = ", ".join([str(plan[i]) for i in range(min(6, len(plan)))])
        print(f"[plan] first up to 6: {preview}")
    else:
        print("[plan] nothing to run.")
        return

    if dry_run:
        print("[dry_run] 계획만 출력하고 종료합니다.")
        return

    # 5) 실행
    for ds, H, s, md in plan:
        apply_HP2(
            CFG,
            csv_path=_csv_path_or_mnt(ds, root_csv=root_csv),
            seed=s, horizon=H,
            device=device if device is not None else CFG["device"],
            out_root=results_root,
            model_tag="HyperConv"
        )
        print(f"\n=== RUN :: ds={ds} | H={H} | seed={s} | mode={md} ===")
        train_eval_one_run(md)

# -----------------------------------------------------------

run_dir_switch_suite_plan()