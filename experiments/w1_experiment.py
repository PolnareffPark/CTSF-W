"""
W1 실험: 층별 교차 vs. 최종 결합 (Late Fusion)
"""

from experiments.base_experiment import BaseExperiment
from models.ctsf_model import HybridTS
from models.experiment_variants import LastLayerFusionModel


class W1Experiment(BaseExperiment):
    """W1 실험: Per-layer Cross vs Last-layer Fusion"""
    
    def _create_model(self):
        mode = self.cfg.get("mode", "per_layer")
        
        if mode == "per_layer":
            # 원안: 모든 층에서 교차 연결
            model = HybridTS(self.cfg, self.n_vars)
            model.set_cross_directions(use_gc=True, use_cg=True)
        elif mode == "last_layer":
            # 대조군: 마지막 층에서만 결합
            model = LastLayerFusionModel(self.cfg, self.n_vars)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return model
    
    def _get_run_tag(self):
        mode = self.cfg.get("mode", "per_layer")
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W1-{mode}"
