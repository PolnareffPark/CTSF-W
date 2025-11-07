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


class W5Experiment(BaseExperiment):
    """W5 실험: 게이트 고정"""
    
    def _create_model(self):
        model = HybridTS(self.cfg, self.n_vars)
        model.set_cross_directions(use_gc=True, use_cg=True)
        return model
    
    def evaluate_test(self):
        """게이트 고정 모드인 경우 래퍼 적용"""
        if self.cfg.get("gate_fixed", False):
            # 학습된 모델을 래핑하여 게이트 고정
            fixed_model = GateFixedModel(self.model)
            fixed_model.eval()
            
            # 고정 모델로 평가
            from utils.direct_evidence import evaluate_with_direct_evidence
            from data.dataset import build_test_tod_vector
            
            tod_vec = build_test_tod_vector(self.cfg)
            direct = evaluate_with_direct_evidence(
                fixed_model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device
            )
            return direct
        else:
            return super().evaluate_test()
    
    def _get_run_tag(self):
        gate_fixed = "fixed" if self.cfg.get("gate_fixed", False) else "dynamic"
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W5-{gate_fixed}"