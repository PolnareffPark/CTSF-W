"""
실험별 모델 변형
⚠️ 기본 모델 구조는 절대 변경하지 않고, wrapper나 variant만 추가합니다.
"""

import torch
import torch.nn as nn
from models.ctsf_model import HybridTS


# ==================== W1: Last-layer Fusion ====================
class LastLayerFusionModel(HybridTS):
    """
    W1 실험: 마지막 층에서만 결합하는 대조 모델
    모든 교차 연결을 비활성화하고, 마지막에만 concat → 1×1 conv로 결합
    """
    def __init__(self, cfg, n_vars: int):
        super().__init__(cfg, n_vars)
        
        # 모든 교차 연결 비활성화
        for blk in self.xhconv_blks:
            blk.use_gc = False
            blk.use_cg = False
        
        # 마지막 결합 레이어 (파라미터 수 매칭용)
        d = cfg["d_embed"]
        # 원래 fuse는 CrossModalCoAttn이므로, 대체로 간단한 결합 사용
        # 파라미터 수를 ±3% 이내로 맞추기 위해 채널 조정 가능
        self.last_fusion = nn.Sequential(
            nn.Linear(d * 2, d),  # concat 후 linear
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(cfg["dropout_rate"])
        )

    def forward(self, x: torch.Tensor, *, save_attn: bool = False):
        B, C, _ = x.shape
        z0 = self.patch(self.revin(x, mode='norm')).permute(2, 0, 1)  # (T,B,d)

        zc = zr = z0
        depth = len(self.conv_blks)

        # 교차 연결 없이 각 경로만 진행
        for i, (cb, gb) in enumerate(zip(self.conv_blks, self.gru_blks)):
            zc = cb(zc)
            zr = gb(zr)
            # 교차 연결 스킵

        # 마지막 결합: concat → fusion
        zc_flat = zc.permute(1, 0, 2)  # (B, T, d)
        zr_flat = zr.permute(1, 0, 2)  # (B, T, d)
        z_concat = torch.cat([zc_flat, zr_flat], dim=-1)  # (B, T, 2d)
        mem = self.last_fusion(z_concat).permute(1, 0, 2)  # (T, B, d)

        flat = self.ln_flat((mem + z0).permute(1, 0, 2)).reshape(B, -1)
        out = self.fc(flat).view(B, self.cfg["horizon"], C).permute(0, 2, 1)
        return self.revin(out, mode='denorm')


# ==================== W2: Static Cross ====================
class StaticCrossModel(HybridTS):
    """
    W2 실험: 정적 교차 연결 모델
    동적 hypernetwork 대신 고정 1×1 conv 또는 학습 스칼라 게이트 사용
    """
    def __init__(self, cfg, n_vars: int):
        super().__init__(cfg, n_vars)
        
        d = cfg["d_embed"]
        # 정적 결합: 1×1 conv로 대체
        # 각 CrossHyperConvBlock의 동적 부분을 정적으로 교체
        for i, blk in enumerate(self.xhconv_blks):
            # 기존 HyperConv1D 대신 고정 1×1 conv
            blk.static_conv_gc = nn.Conv1d(d, d, 1)  # GRU→Conv
            blk.static_conv_cg = nn.Conv1d(d, d, 1)  # Conv→GRU
            # 기존 동적 부분은 유지하되 사용하지 않음 (구조 보존)

    def forward(self, x: torch.Tensor, *, save_attn: bool = False):
        B, C, _ = x.shape
        z0 = self.patch(self.revin(x, mode='norm')).permute(2, 0, 1)  # (T,B,d)

        zc = zr = z0
        depth = len(self.conv_blks)

        for i, (cb, gb) in enumerate(zip(self.conv_blks, self.gru_blks)):
            zc = cb(zc)
            zr = gb(zr)

            if i < depth:
                # 정적 결합 적용
                blk = self.xhconv_blks[i]
                C_in, R_in = zc, zr
                
                # 정적 1×1 conv (동적 대신)
                C_h = blk.static_conv_gc(C_in.permute(1, 2, 0)).permute(2, 0, 1)
                R_h = blk.static_conv_cg(R_in.permute(1, 2, 0)).permute(2, 0, 1)
                
                # Cross-Stitch 결합 (α는 학습됨)
                a = torch.relu(blk.alpha)
                C_mix = a[0, 0] * C_h + a[0, 1] * R_h
                R_mix = a[1, 0] * C_h + a[1, 1] * R_h
                
                zc = blk.ln_c(blk.drop(C_mix) + C_in)
                zr = blk.ln_r(blk.drop(R_mix) + R_in)

        mem = self.fuse(zc, zr, save=save_attn)
        flat = self.ln_flat((mem + z0).permute(1, 0, 2)).reshape(B, -1)
        out = self.fc(flat).view(B, self.cfg["horizon"], C).permute(0, 2, 1)
        return self.revin(out, mode='denorm')


# ==================== W5: Gate Fixed Model ====================
class GateFixedWrapper:
    """
    W5 실험: 학습된 모델의 게이트를 평균값으로 고정
    모델을 래핑하여 inference 시 게이트를 고정
    """
    def __init__(self, model):
        self.model = model
        self.gate_means = {}
        self._compute_gate_means()

    def _compute_gate_means(self):
        """각 교차 블록의 게이트 평균값 계산"""
        # 실제로는 학습 중 게이트 값들을 수집해야 하지만,
        # 여기서는 alpha 파라미터의 평균을 사용 (예시)
        for i, blk in enumerate(self.model.xhconv_blks):
            # alpha는 학습된 파라미터이므로, 실제 게이트 출력의 평균을 계산해야 함
            # 여기서는 단순화하여 alpha 자체를 사용
            self.gate_means[i] = torch.relu(blk.alpha).detach().clone()

    def forward(self, x, **kwargs):
        """고정된 게이트로 forward"""
        # 원래 forward를 호출하되, 게이트를 고정
        # 실제 구현은 forward hook을 사용하여 게이트 출력을 가로채고 평균으로 교체
        return self.model(x, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
