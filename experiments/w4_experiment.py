"""
W4 실험: 교차 층 기여도 분석 (Shallow/Mid/Deep)
"""

from experiments.base_experiment import BaseExperiment
from models.ctsf_model import HybridTS


class W4Experiment(BaseExperiment):
    """W4 실험: Shallow/Mid/Deep 교차"""
    
    def _create_model(self):
        model = HybridTS(self.cfg, self.n_vars)
        
        # 모델로부터 실제 depth를 가져옴 (하드코딩 방지)
        depth = len(model.xhconv_blks)
        cross_layers = self.cfg.get("cross_layers", "all")
        
        if cross_layers == "all":
            # 모든 층 활성화
            active_layers = list(range(depth))
        elif cross_layers == "shallow":
            # 얕은 층만 (처음 1/3)
            # depth=8 → [0,1], depth=9 → [0,1,2]
            n = max(1, depth // 3)
            active_layers = list(range(n))
        elif cross_layers == "mid":
            # 중간 층만 (전체 중간 영역)
            # depth=8 → [2,3,4,5], depth=9 → [3,4,5], depth=7 → [2,3,4]
            # shallow와 deep을 제외한 모든 중간 층을 포함
            shallow_end = max(1, depth // 3)
            deep_start = depth - max(1, depth // 3)
            active_layers = list(range(shallow_end, deep_start))
        elif cross_layers == "deep":
            # 깊은 층만 (마지막 1/3)
            # depth=8 → [6,7], depth=9 → [6,7,8]
            n = max(1, depth // 3)
            active_layers = list(range(depth - n, depth))
        else:
            raise ValueError(f"Unknown cross_layers: {cross_layers}")
        
        # active_layers를 인스턴스 속성으로 저장 (BaseExperiment에서 재사용)
        self.active_layers = active_layers
        model.set_cross_layers(active_layers)
        return model
    
    def _get_run_tag(self):
        cross_layers = self.cfg.get("cross_layers", "all")
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W4-{cross_layers}"
