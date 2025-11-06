"""
W4 실험: 교차 층 기여도 분석 (Shallow/Mid/Deep)
"""

from experiments.base_experiment import BaseExperiment
from models.ctsf_model import HybridTS


class W4Experiment(BaseExperiment):
    """W4 실험: Shallow/Mid/Deep 교차"""
    
    def _create_model(self):
        model = HybridTS(self.cfg, self.n_vars)
        
        cross_layers = self.cfg.get("cross_layers", "all")
        depth = self.cfg["cnn_depth"]
        
        if cross_layers == "all":
            # 모든 층 활성화
            active_layers = list(range(depth))
        elif cross_layers == "shallow":
            # 얕은 층만 (예: 처음 1/3)
            n = max(1, depth // 3)
            active_layers = list(range(n))
        elif cross_layers == "mid":
            # 중간 층만 (예: 중간 1/3)
            n = max(1, depth // 3)
            start = depth // 3
            active_layers = list(range(start, start + n))
        elif cross_layers == "deep":
            # 깊은 층만 (예: 마지막 1/3)
            n = max(1, depth // 3)
            active_layers = list(range(depth - n, depth))
        else:
            raise ValueError(f"Unknown cross_layers: {cross_layers}")
        
        model.set_cross_layers(active_layers)
        return model
    
    def _get_run_tag(self):
        cross_layers = self.cfg.get("cross_layers", "all")
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W4-{cross_layers}"
