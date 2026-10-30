#!/usr/bin/env python
# coding=utf-8
"""Prism partial — `kt_tile_k1_fp8pt` (fp8 e4m3 per-tensor 배율 × **per-k 마스크**) 정확성.

두 축의 교차이므로 픽스처는 `test_partial_fp8pt.py`(per-tensor 계약)를 그대로 재사용하고,
여기서 더 보는 것은 **k1 축**이다:

- 같은 입력(코드 + 배율 `[E]`)에서 dequant 레퍼런스와 일치 — k1 pack 순열(`ktf8k1_off`)이
  틀리면 여기서 잡힌다 (순열 오류는 예외가 아니라 그럴듯하게 나쁜 값이다).
- **per-k sparse == 마스킹된 dense (비트일치)**: |x| < thr 인 k를 0으로 둔 dense와 같아야 한다.
  마스킹된 k는 dense에서 곱이 0이고 k1은 그 라인을 안 읽으므로 같은 항을 같은 순서로 더한다.
- thr 곡선 셋만으로 sparse가 켜진다 (wn_sq/pair_dot 불필요).
- K 정렬이 **32**여도 된다 (b128 커널이 거절하는 기하 — 픽스처의 rows 416/224가 그것이다).
- prefill(qlen>1) 토큰 루프도 레퍼런스와 일치.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import torch
from kt_kernel import kt_kernel_ext
from kt_kernel.experts_partial import PartialMoEWrapper
from test_partial_fp8pt import (
    DN_OFF,
    DN_ROWS,
    E,
    GU_OFF,
    GU_ROWS,
    HIDDEN,
    INTER,
    SP_NG,
    TOP_K,
    Weights,
    assert_close_bf16,
    int_x,
    make_config,
    ref_down,
    ref_gateup,
    routing,
    run_down,
    run_gateup,
)

K1 = "kt_tile_k1_fp8pt"
K2 = "kt_tile_k2_fp8pt"


@pytest.fixture(scope="module")
def cpuinfer():
    return kt_kernel_ext.CPUInfer(60)


def thr_tables(g, u, d):
    full = lambda v: torch.full((E, SP_NG), float(v), dtype=torch.float32)
    return {"thr_gate": full(g), "thr_up": full(u), "thr_down": full(d)}


def masked(x_row, off, rows, thr):
    out = x_row.clone()
    seg = out[off : off + rows].float()
    keep = seg.abs() >= thr
    out[off : off + rows] = torch.where(keep, seg, torch.zeros_like(seg)).to(torch.bfloat16)
    return out, keep


def median_abs_thr(vals):
    flat = vals.abs().reshape(-1).float().sort().values
    n = flat.numel()
    return 0.5 * (float(flat[n // 2]) + float(flat[n // 2 + 1]))


@pytest.mark.cpu
def test_reports_per_k_mask(cpuinfer):
    cfg = make_config(cpuinfer)
    assert PartialMoEWrapper(cfg, cpuinfer, kernel_key=K1).mask_per_k is True
    assert PartialMoEWrapper(cfg, cpuinfer, kernel_key=K2).mask_per_k is False
    assert [n for n, _, _ in PartialMoEWrapper(cfg, cpuinfer, kernel_key=K1).sparsity_keys()] == [
        "thr_gate", "thr_up", "thr_down"]


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", [1, 5])
def test_dense_matches_dequant_reference(cpuinfer, qlen):
    W = Weights(seed=61)
    w = W.load(cpuinfer, kernel_key=K1)
    g = torch.Generator().manual_seed(62)
    ids, x = routing(qlen, g), int_x((qlen, HIDDEN), g).contiguous()
    out = run_gateup(w, ids, x)
    assert not torch.any(torch.isnan(out)), "일부 출력이 쓰이지 않았다 (NaN 센티널 잔존)"
    assert_close_bf16(out, ref_gateup(W, ids, x))
    act = int_x((qlen, TOP_K, INTER), g).contiguous()
    out = run_down(w, ids, act)
    assert not torch.any(torch.isnan(out))
    assert_close_bf16(out, ref_down(W, ids, act))


@pytest.mark.cpu
def test_agrees_with_the_k2_sibling(cpuinfer):
    """같은 코드·배율에서 k1과 k2 pt가 같은 값을 낸다 (누산 순서만 다르다)."""
    W = Weights(seed=63)
    k1, k2 = W.load(cpuinfer, kernel_key=K1), W.load(cpuinfer, kernel_key=K2)
    g = torch.Generator().manual_seed(64)
    ids, x = routing(1, g), int_x((1, HIDDEN), g).contiguous()
    assert_close_bf16(run_gateup(k1, ids, x), run_gateup(k2, ids, x).double())


@pytest.mark.cpu
def test_thr_only_tables_and_zero_thr_is_dense(cpuinfer):
    W = Weights(seed=71)
    dense = W.load(cpuinfer, kernel_key=K1)
    sparse = W.load(cpuinfer, tables=thr_tables(0.0, 0.0, 0.0), kernel_key=K1)
    g = torch.Generator().manual_seed(72)
    ids, x = routing(1, g), int_x((1, HIDDEN), g).contiguous()
    wts = torch.rand(1, TOP_K, generator=g).contiguous()
    assert torch.equal(run_gateup(sparse, ids, x, wts), run_gateup(dense, ids, x))
    act = int_x((1, TOP_K, INTER), g).contiguous()
    assert torch.equal(run_down(sparse, ids, act, wts), run_down(dense, ids, act))


@pytest.mark.cpu
def test_sparse_gateup_matches_masked_dense(cpuinfer):
    W = Weights(seed=81)
    dense = W.load(cpuinfer, kernel_key=K1)
    g = torch.Generator().manual_seed(82)
    ids, x = routing(1, g), int_x((1, HIDDEN), g).contiguous()
    band = x[0, GU_OFF : GU_OFF + GU_ROWS]
    thr_g = median_abs_thr(band)
    sparse = W.load(cpuinfer, tables=thr_tables(thr_g, 0.0, 0.0), kernel_key=K1)
    out = run_gateup(sparse, ids, x, torch.rand(1, TOP_K, generator=g).contiguous())
    xg, keep = masked(x[0], GU_OFF, GU_ROWS, thr_g)
    assert 0 < int(keep.sum()) < GU_ROWS
    ref = run_gateup(dense, ids, xg.unsqueeze(0).contiguous())
    assert torch.equal(out[0, :, :INTER], ref[0, :, :INTER]), "gate half"
    assert torch.equal(out[0, :, INTER:], run_gateup(dense, ids, x)[0, :, INTER:]), "up half (thr=0)"


@pytest.mark.cpu
def test_sparse_down_matches_masked_dense(cpuinfer):
    W = Weights(seed=91)
    dense = W.load(cpuinfer, kernel_key=K1)
    g = torch.Generator().manual_seed(92)
    ids = routing(1, g)
    act = int_x((1, TOP_K, INTER), g).contiguous()
    thr_d = median_abs_thr(act[0, :, DN_OFF : DN_OFF + DN_ROWS])
    sparse = W.load(cpuinfer, tables=thr_tables(0.0, 0.0, thr_d), kernel_key=K1)
    out = run_down(sparse, ids, act, torch.rand(1, TOP_K, generator=g).contiguous())
    am = act.clone()
    kept = 0
    for j in range(TOP_K):
        am[0, j], keep = masked(act[0, j], DN_OFF, DN_ROWS, thr_d)
        kept += int(keep.sum())
    assert 0 < kept < TOP_K * DN_ROWS
    assert torch.equal(out, run_down(dense, ids, am.contiguous()))


@pytest.mark.cpu
def test_huge_thr_zeroes_output(cpuinfer):
    W = Weights(seed=95)
    sparse = W.load(cpuinfer, tables=thr_tables(1e9, 1e9, 1e9), kernel_key=K1)
    g = torch.Generator().manual_seed(96)
    ids = routing(1, g)
    wts = torch.rand(1, TOP_K, generator=g).contiguous()
    assert torch.count_nonzero(run_gateup(sparse, ids, int_x((1, HIDDEN), g).contiguous(), wts)) == 0
    assert torch.count_nonzero(run_down(sparse, ids, int_x((1, TOP_K, INTER), g).contiguous(), wts)) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
