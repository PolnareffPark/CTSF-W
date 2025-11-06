"""
W2 실험: 동적 교차 vs. 정적 교차
"""

from experiments.base_experiment import BaseExperiment
from models.ctsf_model import HybridTS
from models.experiment_variants import StaticCrossModel


class W2Experiment(BaseExperiment):
    """W2 실험: Dynamic Cross vs Static Cross"""
    
    def _create_model(self):
        mode = self.cfg.get("mode", "dynamic")
        
        if mode == "dynamic":
            # 원안: 동적 교차 연결
            model = HybridTS(self.cfg, self.n_vars)
            model.set_cross_directions(use_gc=True, use_cg=True)
        elif mode == "static":
            # 대조군: 정적 교차 연결
            model = StaticCrossModel(self.cfg, self.n_vars)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return model
    
    def _get_run_tag(self):
        mode = self.cfg.get("mode", "dynamic")
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W2-{mode}"
