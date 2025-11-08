"""
W5 실험: 게이트 고정 시험
"""

from experiments.base_experiment import BaseExperiment
from models.ctsf_model import HybridTS
import torch
import numpy as np


class GateFixedModel:
    """게이트를 평균값으로 고정하는 래퍼"""
    def __init__(self, model):
        self.model = model
        self.gate_means = {}
        self._compute_gate_means()
        self._register_hooks()
    
    def _compute_gate_means(self):
        """
        각 교차 블록의 게이트 평균값 계산
        
        참고: W5 실험에서는 학습된 alpha 파라미터의 평균값을 사용합니다.
        이는 실제 inference 시 동적으로 생성되는 게이트 값의 평균을 근사한 것입니다.
        더 정확한 방법은 학습 중 게이트 출력을 수집하는 것이지만,
        본 실험에서는 학습된 alpha 기반으로 충분히 의미 있는 비교가 가능합니다.
        """
        for i, blk in enumerate(self.model.xhconv_blks):
            # alpha는 학습된 파라미터 (Cross-Stitch 가중치)
            # ReLU를 적용하여 음수 방지
            self.gate_means[i] = torch.relu(blk.alpha).detach().clone()
    
    def _register_hooks(self):
        """게이트 출력을 가로채서 평균으로 교체하는 훅"""
        self.hooks = []
        for i, blk in enumerate(self.model.xhconv_blks):
            def make_hook(idx, mean_val):
                def hook(module, input, output):
                    # output은 (C, R) 튜플
                    # alpha를 평균으로 고정
                    module.alpha.data = mean_val.clone()
                return hook
            hook = blk.register_forward_hook(make_hook(i, self.gate_means[i]))
            self.hooks.append(hook)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def eval(self):
        return self.model.eval()
    
    def train(self):
        return self.model.train()
    
    def to(self, device):
        return self.model.to(device)
    
    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)
    
    @property
    def xhconv_blks(self):
        """내부 모델의 xhconv_blks를 전달"""
        return self.model.xhconv_blks
    
    def __getattr__(self, name):
        """내부 모델의 속성에 접근 (conv_blks, resconv_blks 등)"""
        # 무한 재귀 방지: 기본 속성들은 여기서 처리하지 않음
        if name in ('model', 'gate_means', 'hooks'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # 내부 모델의 속성 반환
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class W5Experiment(BaseExperiment):
    """W5 실험: 게이트 고정"""
    
    def _create_model(self):
        model = HybridTS(self.cfg, self.n_vars)
        model.set_cross_directions(use_gc=True, use_cg=True)
        return model
    
    def evaluate_test(self):
        """동적 게이트와 고정 게이트를 모두 평가하여 비교"""
        from utils.direct_evidence import evaluate_with_direct_evidence
        from data.dataset import build_test_tod_vector
        from utils.experiment_metrics.w5_metrics import compute_w5_metrics
        
        # TOD 벡터 준비
        tod_vec = build_test_tod_vector(self.cfg)
        
        # 1. 동적 게이트 모드 평가 (원래 모델 그대로)
        self.model.eval()
        dynamic_results = evaluate_with_direct_evidence(
            self.model, self.test_loader, self.mu, self.std,
            tod_vec=tod_vec, device=self.device
        )
        
        # 2. 게이트 고정 모드 평가
        fixed_model = GateFixedModel(self.model)
        fixed_model.eval()
        fixed_results = evaluate_with_direct_evidence(
            fixed_model, self.test_loader, self.mu, self.std,
            tod_vec=tod_vec, device=self.device
        )
        
        # 3. W5 특화 비교 지표 계산
        w5_metrics = compute_w5_metrics(
            self.model,
            fixed_model_metrics=fixed_results,
            dynamic_model_metrics=dynamic_results
        )
        
        # 4. 결과 병합
        # 동적 모델 결과를 기본으로 하고, W5 비교 지표 추가
        # 고정 모델의 주요 지표도 별도로 기록
        final_results = {**dynamic_results}
        final_results.update(w5_metrics)
        
        # 고정 모델의 주요 성능 지표를 별도 키로 추가 (분석 용이성)
        final_results['rmse_fixed'] = fixed_results.get('rmse', np.nan)
        final_results['mae_fixed'] = fixed_results.get('mae', np.nan)
        final_results['gc_kernel_tod_dcor_fixed'] = fixed_results.get('gc_kernel_tod_dcor', np.nan)
        final_results['cg_event_gain_fixed'] = fixed_results.get('cg_event_gain', np.nan)
        
        return final_results
    
    def _get_run_tag(self):
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W5"