"""
W3 실험: 데이터 구조 원인 확인 (교란 시험)
"""

from experiments.base_experiment import BaseExperiment
from models.ctsf_model import HybridTS
import numpy as np
import torch
from scipy.signal import savgol_filter


def get_dataset_freq_minutes(dataset_name):
    """
    데이터셋별 시간 간격(분) 반환
    - ETTh: 60분 (1시간)
    - ETTm: 15분
    - weather: 10분
    """
    dataset_name = dataset_name.lower()
    if 'etth' in dataset_name:
        return 60
    elif 'ettm' in dataset_name:
        return 15
    elif 'weather' in dataset_name:
        return 10
    else:
        # 기본값 (ETTm과 동일)
        return 15


class PerturbedDataLoader:
    """교란된 데이터를 제공하는 DataLoader 래퍼"""
    def __init__(self, original_loader, perturbation_type, dataset_name, **kwargs):
        self.original_loader = original_loader
        self.perturbation_type = perturbation_type
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        
        # 데이터셋별 시간 간격에 맞춰 시프트 포인트 계산
        self.freq_minutes = get_dataset_freq_minutes(dataset_name)
        # 기본 시프트: 1시간 분량
        self.default_shift_hours = 1
        self.default_shift_points = (self.default_shift_hours * 60) // self.freq_minutes
    
    def __iter__(self):
        for xb, yb in self.original_loader:
            # 배치별로 교란 적용
            if self.perturbation_type == 'tod_shift':
                # 시간대 시프트는 배치 단위로 랜덤 적용
                xb_np = xb.cpu().numpy()
                B, C, L = xb_np.shape
                # 데이터셋별 시간 간격에 맞춰 시프트 포인트 계산
                shift_points = self.kwargs.get('shift_points', self.default_shift_points)
                for i in range(B):
                    shift = np.random.randint(-shift_points, shift_points + 1)
                    xb_np[i] = np.roll(xb_np[i], shift, axis=1)
                xb = torch.from_numpy(xb_np).to(xb.device)
            elif self.perturbation_type == 'smooth':
                # 평활화: Savitzky-Golay 필터 적용
                xb_np = xb.cpu().numpy()
                window = self.kwargs.get('window_length', 5)
                poly = self.kwargs.get('polyorder', 2)
                for i in range(xb_np.shape[0]):
                    for c in range(xb_np.shape[1]):
                        xb_np[i, c] = savgol_filter(xb_np[i, c], window, poly, mode='nearest')
                xb = torch.from_numpy(xb_np).to(xb.device)
            yield xb, yb
    
    def __len__(self):
        return len(self.original_loader)


class W3Experiment(BaseExperiment):
    """W3 실험: 데이터 교란"""
    
    def __init__(self, cfg):
        # 교란 설정 저장
        self.perturbation = cfg.get("perturbation")
        self.perturbation_kwargs = cfg.get("perturbation_kwargs", {})
        super().__init__(cfg)
        
        # 학습 데이터로더에 교란 적용
        if self.perturbation and self.perturbation != "none":
            self.train_loader = PerturbedDataLoader(
                self.train_loader, self.perturbation, self.dataset_tag, **self.perturbation_kwargs
            )
    
    def _create_model(self):
        # 기본 모델 사용
        model = HybridTS(self.cfg, self.n_vars)
        model.set_cross_directions(use_gc=True, use_cg=True)
        return model
    
    def _get_run_tag(self):
        perturbation = self.cfg.get("perturbation", "none")
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W3-{perturbation}"
    
    def evaluate_test(self):
        """
        W3 실험용 테스트 평가: baseline(교란 없음)과 교란 실험 모두 수행하여 비교
        """
        from data.dataset import build_test_tod_vector
        from utils.direct_evidence import evaluate_with_direct_evidence
        from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics
        
        tod_vec = build_test_tod_vector(self.cfg)
        collect_gates = False  # W3는 게이트 수집 불필요
        
        # 교란이 "none"이면 baseline 자체이므로 일반 평가만 수행
        if self.perturbation == "none":
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=collect_gates
            )
            
            # W3 지표는 비교가 필요하므로 baseline 자체에서는 계산하지 않음
            # win_errors는 CSV 저장 시 제외
            direct.pop('hooks_data', None)
            direct.pop('win_errors', None)  # 벡터는 CSV에 저장 불가
            
            return direct
        
        # 교란이 있는 경우: baseline(교란 없음)과 교란 실험 모두 수행
        # 1. Baseline 평가 (교란 없이)
        # 원본 test_loader를 사용 (PerturbedDataLoader가 아닌 원본)
        from data.dataset import load_split_dataloaders
        
        # 원본 데이터로더를 다시 생성 (교란 없이)
        cfg_baseline = self.cfg.copy()
        cfg_baseline["perturbation"] = "none"
        _, _, test_loader_baseline, _ = load_split_dataloaders(cfg_baseline)
        
        # verbose 모드일 때만 출력
        if self.cfg.get("verbose", True):
            print(f"[W3] Evaluating baseline (no perturbation)...")
        baseline_direct = evaluate_with_direct_evidence(
            self.model, test_loader_baseline, self.mu, self.std,
            tod_vec=tod_vec, device=self.device,
            collect_gate_outputs=collect_gates
        )
        
        # hooks_data와 win_errors 제거 (metrics 계산용으로만 사용)
        baseline_direct.pop('hooks_data', None)
        
        # 2. 교란 평가
        if self.cfg.get("verbose", True):
            print(f"[W3] Evaluating with perturbation: {self.perturbation}...")
        
        # 교란된 test_loader 생성
        test_loader_perturbed = PerturbedDataLoader(
            test_loader_baseline, self.perturbation, self.dataset_tag, **self.perturbation_kwargs
        )
        
        current_direct = evaluate_with_direct_evidence(
            self.model, test_loader_perturbed, self.mu, self.std,
            tod_vec=tod_vec, device=self.device,
            collect_gate_outputs=collect_gates
        )
        
        # hooks_data 제거
        current_direct.pop('hooks_data', None)
        
        # 3. W3 특화 지표 계산 (baseline_metrics와 perturb_metrics 전달)
        # win_errors는 baseline_direct와 current_direct에 포함되어 있음
        exp_specific = compute_all_experiment_metrics(
            experiment_type=self.experiment_type,
            model=self.model,
            hooks_data=None,
            tod_vec=tod_vec,
            perturbation_type=self.perturbation,
            baseline_metrics=baseline_direct.copy(),  # 복사본 전달 (pop으로 변경되므로)
            perturb_metrics=current_direct.copy(),    # 복사본 전달
            w3_dz_ci=self.cfg.get("w3_dz_ci", False), # 선택적 부트스트랩 CI
            active_layers=[],
        )
        
        # 4. win_errors 제거 (벡터는 CSV에 저장 불가)
        current_direct.pop('win_errors', None)
        
        # 5. W3 지표 통합
        for k, v in exp_specific.items():
            current_direct[k] = v
        
        # 6. 출력 (verbose 모드일 때만)
        if self.cfg.get("verbose", True):
            print(f"[W3] Baseline RMSE: {baseline_direct['rmse']:.4f}, Perturbed RMSE: {current_direct['rmse']:.4f}")
            if 'w3_intervention_effect_rmse' in exp_specific:
                print(f"[W3] Intervention effect (ΔRMSE%): {exp_specific['w3_intervention_effect_rmse']*100:.2f}%")
            if 'w3_intervention_cohens_d' in exp_specific:
                print(f"[W3] Cohen's d_z: {exp_specific['w3_intervention_cohens_d']:.4f}")
            if 'w3_rank_preservation_rate' in exp_specific:
                print(f"[W3] Rank preservation (교란>기준): {exp_specific['w3_rank_preservation_rate']*100:.1f}%")
            if 'w3_lag_distribution_change' in exp_specific:
                print(f"[W3] Lag distribution change: {exp_specific['w3_lag_distribution_change']:.4f}")
        
        return current_direct