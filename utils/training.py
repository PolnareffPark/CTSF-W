"""
학습 및 평가 함수 모듈
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc


def ps_loss(pred, tgt, lam_patch=0.03):
    """
    Patch-Stat + MSE 손실 (기존 CTSF-V1.py와 동일)
    
    Args:
        pred, tgt: (B, C, H) – RevIN 정규화 공간
        lam_patch: Patch-Stat 가중치
    """
    # ① MSE
    mse = F.mse_loss(pred, tgt)

    # ② Patch-Stat (평균·표준편차 패널티)
    B, C, H = pred.shape
    patch = 8 * max(1, H // 24)  # ≈24-step
    p = pred.reshape(B * C, H).unfold(1, patch, patch)
    t = tgt.reshape(B * C, H).unfold(1, patch, patch)

    mean_err = (p.mean(-1) - t.mean(-1)).abs().mean()
    std_err = (p.std(-1) - t.std(-1)).abs().mean()

    return mse + lam_patch * (mean_err + std_err)


def train_one_epoch(model, loader, opt, mu, std, epoch, cfg, device):
    """한 epoch 학습"""
    model.train()
    tot = mse_sum = mse_real = cnt = 0
    
    # verbose 모드가 아니면 tqdm 억제
    disable_tqdm = not cfg.get("verbose", True)
    
    for xb, yb in tqdm(loader, desc=f"Train {epoch:02d}", leave=False, disable=disable_tqdm):
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = ps_loss(pred, yb, lam_patch=cfg.get("lam_patch", 0.03))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 0.5))
        opt.step()

        bs = xb.size(0)
        tot += loss.item() * bs
        mse_sum += F.mse_loss(pred, yb).item() * bs
        mse_real += F.mse_loss(pred * std + mu, yb * std + mu).item() * bs
        cnt += bs

    return tot / cnt, mse_sum / cnt, mse_real / cnt


@torch.no_grad()
def evaluate(model, loader, mu, std, tag="Val", epoch=None, device=None, metrics=False, cfg=None):
    """평가"""
    model.eval()
    tot = mse_sum = mse_real = cnt = 0
    pr, tg = [], []
    
    # verbose 모드가 아니면 tqdm 억제
    disable_tqdm = not cfg.get("verbose", True) if cfg else False

    for xb, yb in tqdm(loader, desc=f"{tag}", leave=False, disable=disable_tqdm):
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = ps_loss(pred, yb)

        bs = xb.size(0)
        tot += loss.item() * bs
        mse_sum += F.mse_loss(pred, yb).item() * bs
        mse_real += F.mse_loss(pred * std + mu, yb * std + mu).item() * bs
        cnt += bs

        if metrics:
            pr.append((pred * std + mu).cpu())
            tg.append((yb * std + mu).cpu())

    extra = {}
    if metrics:
        from utils.metrics import regression_metrics
        P = torch.cat(pr)
        T = torch.cat(tg)
        extra = regression_metrics(P, T)
        H = P.size(2)
        extra["per_step"] = [regression_metrics(P[:, :, h], T[:, :, h]) for h in range(H)]

    return tot / cnt, mse_sum / cnt, mse_real / cnt, extra


class EarlyStop:
    """조기 종료"""
    def __init__(self, patience=15, min_delta=1e-3):
        self.best = 1e18
        self.wait = 0
        self.p = patience
        self.md = min_delta

    def step(self, val):
        if val < self.best - self.md:
            self.best = val
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.p


def cleanup_resources(model=None, opt=None, scheduler=None, loaders=None):
    """GPU 메모리 및 리소스 정리"""
    if model is not None:
        del model
    if opt is not None:
        del opt
    if scheduler is not None:
        del scheduler
    if loaders is not None:
        for loader in loaders:
            del loader
    
    torch.cuda.empty_cache()
    gc.collect()
