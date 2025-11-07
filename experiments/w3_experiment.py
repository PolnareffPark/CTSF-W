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