#!/usr/bin/env python
# coding=utf-8
"""`kt_tile_k2_bf16_nopf` = `kt_tile_k2_bf16`에서 SW 프리페치만 뺀 커널 — **비트 동일성** 게이트.

no-SW-pf 형제(`GemmKernelTileK2BF16NoPF`, tile_k2.hpp)의 존재 이유는 "SW 프리페처 없음"
ablation을 같은 자산·같은 plan으로 돌리는 것이다. 그 실험이 의미를 가지려면 **프리페치
말고는 아무것도 다르지 않다**는 것이 참이어야 하고, 프리페치는 정의상 결과를 바꾸지
않으므로 그 주장은 tolerance가 아니라 `torch.equal`로 검사할 수 있다. 두 커널은
weight 레이아웃(BufferB 타입 자체)·plan 인코딩·누산 순서를 공유하므로 여기서 값이
갈리면 프리페치 제거 과정에서 다른 것까지 건드렸다는 뜻이다.

덮는 경로 셋:
  - decode dense (m == 1, plan 없음 → 상속받은 AVX 루프)
  - decode sparse (m == 1, k2wl2 페어 마스크 → `gemv_slab` + 프리페치 커서)
  - prefill (qlen > 1 → 상속받은 AMX `amx_kernel`, B-panel 프리페치가 여기 있다)

기하·헬퍼는 `test_tile_k1_bf16.py`와 같다 (노드 N shard 256 배수, K 밴드 32 배수).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
from kt_kernel import kt_kernel_ext
from kt_kernel.experts_partial import PartialMoEWrapper

E, TOPK, H, I = 8, 2, 512, 256
GU_OFF, GU_ROWS = 128, 384   # gate/up cold 밴드 = hidden rows [128, 512)
DN_OFF, DN_ROWS = 0, 256     # down cold 밴드 = inter rows [0, 256)
MAX_LEN = 8
THREADS = 4
NG, GRID, PMAX, RENORM_IT = 201, 0.005, 0.9, 3

PF, NOPF = "kt_tile_k2_bf16", "kt_tile_k2_bf16_nopf"


def _cpuinfer():
    cfg = kt_kernel_ext.WorkerPoolConfig()
    cfg.subpool_count = 1
    cfg.subpool_numa_map = [0]
    cfg.subpool_thread_count = [THREADS]
    return kt_kernel_ext.CPUInfer(cfg)


def _config(sparse: bool, p=0.5, lam=4.0):
    cfg = kt_kernel_ext.moe.MOEConfig(E, TOPK, H, I, 0)
    cfg.max_len = MAX_LEN
    cfg.layer_idx = 0
    cfg.partial.enabled = True
    cfg.partial.gate.offset, cfg.partial.gate.rows = GU_OFF, GU_ROWS
    cfg.partial.up.offset, cfg.partial.up.rows = GU_OFF, GU_ROWS
    cfg.partial.down.offset, cfg.partial.down.rows = DN_OFF, DN_ROWS
    cfg.partial.n_total = I
    cfg.partial.node_gateup_n_offset = [0]
    cfg.partial.node_gateup_n_rows = [I]
    cfg.partial.node_down_n_offset = [0]
    cfg.partial.node_down_n_rows = [H]
    if sparse:
        sp = cfg.partial.sparsity
        sp.pmax, sp.grid, sp.ng, sp.renorm_it = PMAX, GRID, NG, RENORM_IT
        sp.p_gate = sp.p_up = sp.p_down = p
        sp.lam_gate = sp.lam_up = sp.lam_down = lam
    return cfg


def make_weights(seed):
    g = torch.Generator().manual_seed(seed)
    mk = lambda shape: (torch.randn(shape, generator=g) * 0.1).to(torch.bfloat16).contiguous()
    return {"gate": mk((E, I, GU_ROWS)), "up": mk((E, I, GU_ROWS)), "down": mk((E, H, DN_ROWS))}


def k2_tables(w, thr=0.0):
    """k2wl2 페어 마스크가 요구하는 셋: 열 노름²(`*_wn_sq`), 페어 내적(`*_pair_dot`), thr 곡선."""
    out = {}
    for name in ("gate", "up", "down"):
        ww = w[name].float()                      # [E, n, rows]
        assert ww.shape[2] % 2 == 0
        out[f"{name}_wn_sq"] = (ww * ww).sum(dim=1).contiguous()                     # [E, rows]
        even, odd = ww[:, :, 0::2], ww[:, :, 1::2]
        out[f"{name}_pair_dot"] = (even * odd).sum(dim=1).contiguous()               # [E, rows/2]
        out[f"thr_{name}"] = torch.full((E, NG), float(thr), dtype=torch.float32)
    return out


def build(kernel_key, w, cpuinfer, tables=None):
    cfg = _config(tables is not None)
    cfg.pool = cpuinfer.backend_
    wrap = PartialMoEWrapper(cfg, cpuinfer, kernel_key=kernel_key)
    wrap.load_weights_from_tensors(w["gate"], w["up"], w["down"], sparsity_tables=tables)
    return wrap


def routing(qlen, g):
    return torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(qlen)]).to(torch.int64).contiguous()


def run_gateup(wrap, ids, x, weights=None):
    out = torch.full((x.shape[0], TOPK, 2 * I), 777.0, dtype=torch.bfloat16).contiguous()
    wrap.forward_gateup(ids, x, out, weights)
    return out


def run_down(wrap, ids, act, weights=None):
    out = torch.full((act.shape[0], TOPK, H), 777.0, dtype=torch.bfloat16).contiguous()
    wrap.forward_down(ids, act, out, weights)
    return out


def rand_bf16(shape, g, scale=0.1):
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16).contiguous()


def _pair(kernel_keys, w, tables=None):
    ci = _cpuinfer()
    return [build(k, w, ci, tables) for k in kernel_keys], ci


@pytest.mark.cpu
def test_both_keys_are_registered():
    from kt_kernel.experts_partial import _PARTIAL_MOE_CLASSES

    assert _PARTIAL_MOE_CLASSES[NOPF]() is kt_kernel_ext.moe.TileK2BF16NoPF_MOE
    ci = _cpuinfer()
    # 마스크 종류는 형제와 같다 (페어 = k2wl2). 여기가 True면 sparsity 셋이 어긋난다.
    assert PartialMoEWrapper(_config(False), ci, kernel_key=NOPF).mask_per_k is False


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", [1, 5])
def test_dense_is_bit_identical(qlen):
    """qlen == 1은 AVX 루프, qlen > 1은 상속받은 AMX prefill (B-panel 프리페치가 있는 쪽)."""
    w = make_weights(21)
    (a, b), _ci = _pair((PF, NOPF), w)
    g = torch.Generator().manual_seed(22)
    for _ in range(2):
        ids = routing(qlen, g)
        x = rand_bf16((qlen, H), g)
        assert torch.equal(run_gateup(a, ids, x), run_gateup(b, ids, x)), f"gateup qlen={qlen}"
        act = rand_bf16((qlen, TOPK, I), g)
        assert torch.equal(run_down(a, ids, act), run_down(b, ids, act)), f"down qlen={qlen}"


@pytest.mark.cpu
@pytest.mark.parametrize("thr", [0.0, 0.05])
def test_sparse_decode_is_bit_identical(thr):
    """`gemv_slab` + 프리페치 커서 경로. thr=0은 전량 keep(커서가 가장 긴 경우),
    thr>0은 실제로 페어가 빠지는 경우 — 커서와 소비 스트림이 어긋나면 후자에서 갈린다."""
    w = make_weights(23)
    tables = k2_tables(w, thr)
    (a, b), _ci = _pair((PF, NOPF), w, tables)
    g = torch.Generator().manual_seed(24)
    for _ in range(3):
        ids = routing(1, g)
        weights = torch.softmax(torch.randn(1, TOPK, generator=g), dim=-1).contiguous()
        x = rand_bf16((1, H), g)
        assert torch.equal(run_gateup(a, ids, x, weights), run_gateup(b, ids, x, weights))
        act = rand_bf16((1, TOPK, I), g)
        assert torch.equal(run_down(a, ids, act, weights), run_down(b, ids, act, weights))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
