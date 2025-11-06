"""
Forward/Backward Hooks 모듈
내부 신호 추적 및 직접 근거 지표 수집용
"""

from collections import defaultdict
import torch


class LastBlockHooks:
    """
    마지막 블록 내부 신호 수집 (직접 근거 지표용)
    evaluate 시점에 내부 신호 수집:
      - Conv 마지막 블록(교차 前 Conv 출력)
      - CrossHyperConv 마지막 블록(교차 後 C/R 출력)
      - GRU→Conv 하이퍼컨브의 커널 생성기(Linear 출력)
    """
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.C_last_pre = None
        self.C_last_post = None
        self.R_last_post = None
        self.W_gc_list = []

        # 마지막 블록 찾기
        self.res_last = self._get_last_module(model, ["conv_blks", "resconv_blks", "cnn_blks"])
        self.xh_last = self._get_last_module(model, ["xhconv_blks", "xh_blks", "cross_hc_blks"])

        if not hasattr(self.xh_last, "hc_gc") or not hasattr(self.xh_last.hc_gc, "gen"):
            raise AttributeError("xh_last.hc_gc.gen not found – HyperConv1D(gen) 경로 확인 필요")
        self.gc_gen = self.xh_last.hc_gc.gen

    def _get_last_module(self, model, candidates):
        """후보 이름 중 하나로 마지막 모듈 찾기"""
        for name in candidates:
            modlist = getattr(model, name, None)
            if modlist is not None and hasattr(modlist, "__len__") and len(modlist) > 0:
                return modlist[-1]
        raise AttributeError(f"None of {candidates} found in model (or empty).")

    def attach(self):
        """훅 등록"""
        def _h_res(mod, inp, out):
            self.C_last_pre = out.detach()

        def _h_xh(mod, inp, out):
            C_out, R_out = out
            self.C_last_post = C_out.detach()
            self.R_last_post = R_out.detach()

        def _h_gc(mod, inp, out):
            self.W_gc_list.append(out.detach().cpu())

        self.handles.append(self.res_last.register_forward_hook(_h_res))
        self.handles.append(self.xh_last.register_forward_hook(_h_xh))
        self.handles.append(self.gc_gen.register_forward_hook(_h_gc))

    def detach(self):
        """훅 제거"""
        for h in self.handles:
            try:
                h.remove()
            except:
                pass
        self.handles = []


def register_gradient_hooks(model):
    """
    Gradient 추적용 훅 등록
    
    Returns:
        grads: defaultdict, 각 블록별 gradient norm 기록
    """
    grads = defaultdict(list)

    def make_single_hook(key):
        def _hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is not None:
                grads[key].append(g.norm().item())
        return _hook

    def make_dual_hook(idx):
        """Cross-HyperConv 블록: (Conv_grad, GRU_grad) 두 개"""
        def _hook(module, grad_input, grad_output):
            gC, gR = grad_output
            if gC is not None:
                grads[f'hyper_{idx}'].append(gC.norm().item())
            if gR is not None:
                grads[f'stitch_{idx}'].append(gR.norm().item())
        return _hook

    # Conv / GRU 블록
    for i, (cb, gb) in enumerate(zip(model.conv_blks, model.gru_blks)):
        cb.register_full_backward_hook(make_single_hook(f'conv_{i}'))
        gb.register_full_backward_hook(make_single_hook(f'gru_{i}'))

    # Cross-HyperConv 블록
    for i, xhb in enumerate(model.xhconv_blks):
        xhb.register_full_backward_hook(make_dual_hook(i))

    return grads
