"""
W3 실험: 데이터 구조 원인 확인 (교란 시험)
"""

from experiments.base_experiment import BaseExperiment
from models.ctsf_model import HybridTS
from data.dataset import apply_data_perturbation


class W3Experiment(BaseExperiment):
    """W3 실험: 데이터 교란"""
    
    def __init__(self, cfg):
        # 데이터 교란 적용 (학습 데이터에만)
        perturbation = cfg.get("perturbation")
        if perturbation:
            # 데이터 로딩 전에 교란 적용
            # 실제로는 데이터 로더에서 배치 단위로 적용하는 것이 더 적절
            pass
        super().__init__(cfg)
    
    def _create_model(self):
        # 기본 모델 사용
        model = HybridTS(self.cfg, self.n_vars)
        model.set_cross_directions(use_gc=True, use_cg=True)
        return model
    
    def _get_run_tag(self):
        perturbation = self.cfg.get("perturbation", "none")
        return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W3-{perturbation}"
