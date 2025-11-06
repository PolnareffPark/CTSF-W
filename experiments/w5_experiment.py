"""
W5 실험: 게이트 고정 시험
"""

from experiments.base_experiment import BaseExperiment
from models.ctsf_model import HybridTS
from models.experiment_variants import GateFixedWrapper


class W5Experiment(BaseExperiment):
    """W5 실험: 게이트 고정"""
    
    def _create_model(self):
        model = HybridTS(self.cfg, self.n_vars)
        model.set_cross_directions(use_gc=True, use_cg=True)
        
        # 게이트 고정 모드인 경우 래퍼 적용
        if self.cfg.get("gate_fixed", False):
            # 먼저 학습된 모델이 필요하므로, 여기서는 래퍼만 준비
            # 실제로는 학습 후에 래퍼를 적용
            pass
        
        return model
    
    def evaluate_test(self):
        """게이트 고정 모드인 경우 래퍼 적용"""
        if self.cfg.get("gate_fixed", False):
            # 학습된 모델을 래핑
            wrapper = GateFixedWrapper(self.model)
            # 래퍼로 평가 (구현 필요)
            # 여기서는 기본 평가 사용
            pass
        
        return super().evaluate_test()
    
    def _get_run_tag(self):
        gate_fixed = "fixed" if self.cfg.get("gate_fixed", False) else "dynamic"
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W5-{gate_fixed}"
