"""
기본 실험 클래스
모든 실험의 공통 기능 제공
"""

import os
import torch
import gc
from pathlib import Path
from utils.training import train_one_epoch, evaluate, EarlyStop, cleanup_resources
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.csv_logger import save_results_unified
from data.dataset import load_split_dataloaders, build_test_tod_vector


class BaseExperiment:
    """기본 실험 클래스"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]
        self.dataset_tag = Path(cfg["csv_path"]).stem
        self.out_root = Path(cfg.get("out_dir", "results"))
        self.out_dir = self.out_root / self.dataset_tag
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 로딩
        self.train_loader, self.val_loader, self.test_loader, (mu_np, std_np, C) = \
            load_split_dataloaders(cfg)
        self.mu = torch.tensor(mu_np, device=self.device).unsqueeze(-1)
        self.std = torch.tensor(std_np, device=self.device).unsqueeze(-1)
        self.n_vars = C
        
        # 모델 생성 (하위 클래스에서 구현)
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # 옵티마이저 및 스케줄러
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 1e-3)
        )
        
        total_steps = len(self.train_loader) * cfg["epochs"]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg["lr"],
            total_steps=total_steps,
            pct_start=0.15,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        self.stopper = EarlyStop(
            patience=cfg.get("early_stop_patience", 20),
            min_delta=cfg.get("early_stop_min_delta", 1e-3)
        )
        
        # 체크포인트 경로
        self.run_tag = self._get_run_tag()
        self.ckpt_path = self.out_dir / f"ckpt_{self.run_tag}.pt"
    
    def _create_model(self):
        """모델 생성 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def _get_run_tag(self):
        """실행 태그 생성"""
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-{self.cfg.get('mode', 'default')}"
    
    def train(self):
        """학습"""
        start_ep, best_val, best_state_cpu = 1, float("inf"), None
        
        # 체크포인트 로드
        if self.ckpt_path.exists():
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["opt"])
            self.scheduler.load_state_dict(ckpt["sched"])
            start_ep = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", float("inf"))
            best_state_cpu = ckpt.get("best_model", None)
            print(f"[resume] {self.ckpt_path.name} @ epoch {ckpt['epoch']} (best_val={best_val:.6f})")
        
        if best_state_cpu is None:
            best_state_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        
        # 학습 루프
        for ep in range(start_ep, self.cfg["epochs"] + 1):
            tr_loss, _, _ = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self.mu, self.std, ep, self.cfg, self.device
            )
            self.scheduler.step()
            
            val_loss, _, _, _ = evaluate(
                self.model, self.val_loader, self.mu, self.std,
                tag="Val", epoch=ep, device=self.device, metrics=False
            )
            print(f"[{ep:02d}] train {tr_loss:.4f} | val {val_loss:.4f}")
            
            if val_loss < best_val:
                best_val = val_loss
                best_state_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            
            # 체크포인트 저장
            torch.save({
                "epoch": ep,
                "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "opt": self.optimizer.state_dict(),
                "sched": self.scheduler.state_dict(),
                "best_val": best_val,
                "best_model": best_state_cpu,
            }, self.ckpt_path)
            
            if self.stopper.step(val_loss):
                print(f"[early stop] stop @ {ep}")
                break
        
        # Best 모델 로드
        self.model.load_state_dict({k: v.to(self.device) for k, v in best_state_cpu.items()})
        return best_state_cpu
    
    def evaluate_test(self):
        """테스트 평가 (직접 근거 지표 포함)"""
        tod_vec = build_test_tod_vector(self.cfg)
        direct = evaluate_with_direct_evidence(
            self.model, self.test_loader, self.mu, self.std,
            tod_vec=tod_vec, device=self.device
        )
        return direct
    
    def save_results(self, direct_metrics):
        """결과 저장"""
        row = {
            "dataset": self.dataset_tag,
            "horizon": int(self.cfg["horizon"]),
            "seed": int(self.cfg["seed"]),
            "mode": self.cfg.get("mode", "default"),
            "model_tag": self.cfg.get("model_tag", "HyperConv"),
            **direct_metrics
        }
        fpath = save_results_unified(row, out_root=str(self.out_root))
        print(f"[saved] {fpath}")
        return fpath
    
    def cleanup(self):
        """리소스 정리"""
        # 체크포인트 삭제
        try:
            if self.ckpt_path.exists():
                os.remove(self.ckpt_path)
        except Exception as e:
            print(f"(warn) ckpt delete fail: {e}")
        
        cleanup_resources(
            model=self.model,
            opt=self.optimizer,
            scheduler=self.scheduler,
            loaders=[self.train_loader, self.val_loader, self.test_loader]
        )
    
    def run(self):
        """전체 실험 실행"""
        try:
            self.train()
            direct_metrics = self.evaluate_test()
            self.save_results(direct_metrics)
        finally:
            self.cleanup()
