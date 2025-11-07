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
import time
from data.dataset import load_split_dataloaders, build_test_tod_vector


class BaseExperiment:
    """기본 실험 클래스"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.dataset_tag = Path(cfg["csv_path"]).stem
        self.out_root = Path(cfg.get("out_dir", "results"))
        self.out_dir = self.out_root / self.dataset_tag
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_type = cfg.get("experiment_type", "W1")
        self.start_time = None
        self.train_time = None
        self.active_layers = None  # W4 실험에서 사용
        
        # 데이터 로딩
        self.train_loader, self.val_loader, self.test_loader, (mu_np, std_np, C) = \
            load_split_dataloaders(cfg)
        self.mu = torch.tensor(mu_np, device=self.device).unsqueeze(-1)
        self.std = torch.tensor(std_np, device=self.device).unsqueeze(-1)
        self.n_vars = C
        
        # 모델 생성 (하위 클래스에서 구현)
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # 옵티마이저 및 스케줄러 (기존 CTSF-V1.py와 동일)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 1e-3)
        )
        
        # OneCycleLR 스케줄러 (기존과 동일)
        total_steps = len(self.train_loader) * cfg["epochs"]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg["lr"],
            total_steps=total_steps,
            pct_start=0.15,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        # EarlyStop (기존과 동일: patience=20)
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
        self.start_time = time.time()
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
            # 간단한 로그만 출력 (전체 실험 시 과도한 출력 방지)
            if self.cfg.get("verbose", True):
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
        self.train_time = time.time() - self.start_time
        return best_state_cpu
    
    def evaluate_test(self):
        """테스트 평가 (직접 근거 지표 포함)"""
        tod_vec = build_test_tod_vector(self.cfg)
        
        # W2 실험이면 게이트 출력 수집
        collect_gates = (self.experiment_type == "W2")
        
        # W4 실험이면 중간층 표현 수집을 위한 훅 등록
        hooks_data = {}
        hooks = []
        if self.experiment_type == "W4" and self.active_layers:
            # 층별로 표현을 저장할 딕셔너리 초기화
            cnn_repr_by_layer = {i: [] for i in self.active_layers}
            gru_repr_by_layer = {i: [] for i in self.active_layers}
            
            def make_hook(layer_idx, repr_dict):
                def hook_fn(module, input, output):
                    # output: zc or zr (T, B, d)
                    # 전체 시퀀스를 평균하거나 마지막 표현을 사용
                    if isinstance(output, tuple):
                        repr_tensor = output[0]  # (T, B, d)
                    else:
                        repr_tensor = output  # (T, B, d)
                    
                    # (T, B, d) -> (B, d) 평균
                    repr_mean = repr_tensor.mean(dim=0).detach().cpu().numpy()  # (B, d)
                    repr_dict[layer_idx].append(repr_mean)
                return hook_fn
            
            # 각 활성 층에 훅 등록
            for i in self.active_layers:
                if i < len(self.model.conv_blks):
                    # CNN 경로
                    h1 = self.model.conv_blks[i].register_forward_hook(
                        make_hook(i, cnn_repr_by_layer)
                    )
                    hooks.append(h1)
                if i < len(self.model.gru_blks):
                    # GRU 경로
                    h2 = self.model.gru_blks[i].register_forward_hook(
                        make_hook(i, gru_repr_by_layer)
                    )
                    hooks.append(h2)
            
            # 나중에 hooks_data에 저장 (평가 후)
        
        try:
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=collect_gates
            )
        finally:
            # 훅 제거
            for h in hooks:
                h.remove()
            
            # W4: 수집된 표현을 hooks_data에 저장
            if self.experiment_type == "W4" and self.active_layers:
                # 각 배치의 표현을 연결하여 층별 리스트 생성
                # cnn_repr_by_layer[i] = [batch0, batch1, ...] -> np.concatenate -> (total_samples, d)
                import numpy as np
                cnn_repr_list = []
                gru_repr_list = []
                
                # 활성 층 순서대로 정리
                for i in self.active_layers:
                    if cnn_repr_by_layer[i]:
                        cnn_repr_list.append(np.concatenate(cnn_repr_by_layer[i], axis=0))
                    else:
                        cnn_repr_list.append(None)
                    
                    if gru_repr_by_layer[i]:
                        gru_repr_list.append(np.concatenate(gru_repr_by_layer[i], axis=0))
                    else:
                        gru_repr_list.append(None)
                
                hooks_data["cnn_representations"] = cnn_repr_list
                hooks_data["gru_representations"] = gru_repr_list
        
        # hooks_data 병합 (W2 실험용 게이트 데이터)
        w2_hooks_data = direct.pop('hooks_data', None)
        if w2_hooks_data:
            hooks_data.update(w2_hooks_data)
        
        # direct_evidence는 direct 자체 (성능 및 직접 근거 지표)
        direct_evidence = direct.copy()
        
        # 실험별 특화 지표 계산
        try:
            from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics
            
            # W4의 경우 self.active_layers 사용 (중복 계산 제거)
            active_layers = self.active_layers if self.experiment_type == "W4" else []
            
            exp_specific = compute_all_experiment_metrics(
                experiment_type=self.experiment_type,
                model=self.model,
                hooks_data=hooks_data if hooks_data else None,
                tod_vec=tod_vec,
                direct_evidence=direct_evidence,
                perturbation_type=self.cfg.get("perturbation"),
                active_layers=active_layers,
            )
            
            # 지표 통합 (w1_, w2_ 등의 접두사 유지)
            for k, v in exp_specific.items():
                direct[k] = v
        except ImportError:
            # 실험별 특화 지표 모듈이 없으면 스킵
            pass
        
        # 보고용 그림 지표 계산
        # 주의: 그림 지표 계산은 선택적이며, 실패해도 기본 평가는 계속 진행
        # 이유: CKA, 그래디언트 정렬 등은 추가 계산이 필요하고 메모리/시간 부담이 있을 수 있음
        from utils.plotting_metrics import compute_all_plotting_metrics
        
        # 작은 배치만 사용하여 계산 부담 최소화
        small_loader = torch.utils.data.DataLoader(
            self.test_loader.dataset,
            batch_size=min(16, self.cfg.get("batch_size", 128)),
            shuffle=False,
            num_workers=0
        )
        
        plotting_metrics = compute_all_plotting_metrics(
            model=self.model,
            loader=small_loader,
            mu=self.mu,
            std=self.std,
            tod_vec=tod_vec,
            device=self.device,
            experiment_type=self.experiment_type,
            hooks_data=hooks_data,
            compute_grad_align=self.cfg.get("compute_grad_align", False)
        )
        
        # 지표 통합
        for k, v in plotting_metrics.items():
            direct[k] = v
        
        return direct
    
    def save_results(self, direct_metrics):
        """결과 저장"""
        row = {
            "dataset": self.dataset_tag,
            "horizon": int(self.cfg["horizon"]),
            "seed": int(self.cfg["seed"]),
            "mode": self.cfg.get("mode", "default"),
            "model_tag": self.cfg.get("model_tag", "HyperConv"),
            "experiment_type": self.experiment_type,
            "train_time_s": self.train_time if self.train_time else 0.0,
            **direct_metrics
        }
        fpath = save_results_unified(row, out_root=str(self.out_root), experiment_type=self.experiment_type)
        print(f"[saved] {fpath}")
        
        # 보고용 그림 데이터 저장
        # 주의: 그림 데이터만 저장하며, 실제 그림 그리기 코드는 포함되지 않음
        # 그림은 별도 스크립트로 CSV 데이터를 읽어서 그려야 함
        from utils.plot_results import save_plot_data
        
        # 그림 타입별로 데이터 저장
        plot_types = {
            "W1": ["forest_plot", "cka_heatmap", "grad_align_bar"],
            "W2": ["forest_plot", "gate_tod_heatmap", "gate_distribution"],
            "W3": ["effect_size_bar", "rank_preservation", "lag_distribution"],
            "W4": ["rmse_line", "gate_usage_bar", "cka_heatmap"],
            "W5": ["degradation_bar", "gate_distribution"],
        }
        
        for plot_type in plot_types.get(self.experiment_type, []):
            # 해당 그림 타입에 필요한 데이터만 추출
            plot_data = {k: v for k, v in direct_metrics.items() 
                        if k.startswith(plot_type.split('_')[0]) or 
                        k in ['rmse', 'mse_real', 'mae'] or
                        'cka' in k or 'grad_align' in k or 'gate' in k or 'bestlag' in k}
            
            if plot_data:
                save_plot_data(
                    experiment_type=self.experiment_type,
                    dataset=self.dataset_tag,
                    plot_type=plot_type,
                    plot_data=plot_data,
                    results_root=str(self.out_root)
                )
        
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
