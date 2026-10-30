#!/usr/bin/env python
# coding=utf-8
"""bf16 k1 타일 커널(`kt_tile_k1_bf16`)의 Prism partial 경로 정확성 — PartialMoEWrapper 경유.

`test_tile_k1_fp8.py`의 bf16 판이고, 보는 불변식이 같다:

- **dense가 레퍼런스와 맞는가.** pack 순열(`ktb1_off`)이 틀리면 여기서 잡힌다 — 순열 오류는
  "같은 바이트를 다른 순서로 읽는" 형태라 예외가 아니라 **그럴듯하게 나쁜 값**으로 나온다.
- **per-k sparse == 마스킹된 dense (비트일치).** 마스킹된 k는 dense 경로에서 x=0이라 곱이
  0이고, k1 커널은 그 k의 라인을 아예 안 읽는다. 두 계산이 같은 항을 **같은 순서로** 더하므로
  비트일치가 성립한다 — tolerance 뒤에 이중계산·누락이 숨지 못한다.
- **thr 곡선 셋만으로 sparse가 켜진다** (wn_sq/pair_dot 불필요). thr=0이면 전량 통과 = dense.
- **prefill(qlen>1) 토큰 루프**도 레퍼런스와 맞는다. 이 커널에는 CPU AMX prefill이 없으므로
  (tile-bf16-moe.hpp) m>1 경로가 실제로 도는지 여기서 확인해야 한다.
- **k2 형제와의 관계**: 같은 weight·같은 x에서 dense 결과가 서로 tolerance 안에 있다
  (누산 순서만 다르다). 값이 크게 갈리면 pack이나 디코드가 틀렸다는 뜻이다.

기하: 노드 N shard는 256 배수(inter 256, hidden 512), K 밴드는 32 배수.
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

K1, K2 = "kt_tile_k1_bf16", "kt_tile_k2_bf16"


def _cpuinfer():
    cfg = kt_kernel_ext.WorkerPoolConfig()
    cfg.subpool_count = 1
    cfg.subpool_numa_map = [0]
    cfg.subpool_thread_count = [THREADS]
    return kt_kernel_ext.CPUInfer(cfg)


def _config(sparse: bool, p=0.5, lam=0.0):
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
    """정확히 표현 가능한 bf16 정수 weight — dense/masked 비트일치 비교의 전제다."""
    g = torch.Generator().manual_seed(seed)
    mk = lambda shape: torch.randint(-4, 5, shape, generator=g).to(torch.bfloat16).contiguous()
    return {"gate": mk((E, I, GU_ROWS)), "up": mk((E, I, GU_ROWS)), "down": mk((E, H, DN_ROWS))}


def thr_tables(thr_g, thr_u, thr_d):
    full = lambda v: torch.full((E, NG), float(v), dtype=torch.float32)
    return {"thr_gate": full(thr_g), "thr_up": full(thr_u), "thr_down": full(thr_d)}


def build(kernel_key, w, cpuinfer, tables=None, p=0.5, lam=0.0):
    cfg = _config(tables is not None, p, lam)
    cfg.pool = cpuinfer.backend_
    wrap = PartialMoEWrapper(cfg, cpuinfer, kernel_key=kernel_key)
    wrap.load_weights_from_tensors(w["gate"], w["up"], w["down"], sparsity_tables=tables)
    return wrap


def routing(qlen, g):
    return torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(qlen)]).to(torch.int64).contiguous()


def run_gateup(w, ids, x, weights=None):
    out = torch.full((x.shape[0], TOPK, 2 * I), 777.0, dtype=torch.bfloat16).contiguous()
    w.forward_gateup(ids, x, out, weights)
    return out


def run_down(w, ids, act, weights=None):
    out = torch.full((act.shape[0], TOPK, H), 777.0, dtype=torch.bfloat16).contiguous()
    w.forward_down(ids, act, out, weights)
    return out


def ref_gateup(x, ids, w):
    xs = x[:, GU_OFF : GU_OFF + GU_ROWS].float()
    out = torch.empty(x.shape[0], TOPK, 2 * I)
    for m in range(x.shape[0]):
        for j in range(TOPK):
            e = int(ids[m, j])
            out[m, j, :I] = xs[m] @ w["gate"][e].float().t()
            out[m, j, I:] = xs[m] @ w["up"][e].float().t()
    return out


def ref_down(act, ids, w):
    out = torch.empty(act.shape[0], TOPK, H)
    for m in range(act.shape[0]):
        for j in range(TOPK):
            e = int(ids[m, j])
            out[m, j] = act[m, j, DN_OFF : DN_OFF + DN_ROWS].float() @ w["down"][e].float().t()
    return out


def rel_diff(out, ref):
    return (torch.mean(torch.abs(out.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)).item()


def rand_bf16(shape, g, scale=0.1):
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16).contiguous()


def _masked(x_row, off, rows, thr):
    """|x| < thr 인 k를 0으로 (bf16 왕복 무손실). keep 마스크도 돌려준다."""
    out = x_row.clone()
    seg = out[off : off + rows].float()
    keep = seg.abs() >= thr
    out[off : off + rows] = torch.where(keep, seg, torch.zeros_like(seg)).to(torch.bfloat16)
    return out, keep


def _median_abs_thr(vals):
    """|x| 정렬의 중앙 두 값 사이 중점 — 판정이 경계에서 흔들리지 않는다."""
    flat = vals.abs().reshape(-1).float().sort().values
    n = flat.numel()
    return 0.5 * (float(flat[n // 2]) + float(flat[n // 2 + 1]))


# ── 계약: 마스크 종류와 테이블 집합 ────────────────────────────────────────


@pytest.mark.cpu
def test_wrapper_reports_per_k_mask():
    cpuinfer = _cpuinfer()
    assert PartialMoEWrapper(_config(False), cpuinfer, kernel_key=K1).mask_per_k is True
    assert PartialMoEWrapper(_config(False), cpuinfer, kernel_key=K2).mask_per_k is False
    assert [n for n, _, _ in PartialMoEWrapper(_config(False), cpuinfer, kernel_key=K1).sparsity_keys()] == [
        "thr_gate", "thr_up", "thr_down"]


# ── dense: pack 순열이 맞는가 ──────────────────────────────────────────────


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", [1, 2, 5])
def test_dense_matches_reference(qlen):
    """qlen>1은 토큰 루프 경로다 (이 커널에는 AMX prefill이 없다)."""
    w = make_weights(11)
    cpuinfer = _cpuinfer()
    wrap = build(K1, w, cpuinfer)
    g = torch.Generator().manual_seed(12)
    for _ in range(2):
        ids = routing(qlen, g)
        x = rand_bf16((qlen, H), g)
        d = rel_diff(run_gateup(wrap, ids, x), ref_gateup(x, ids, w))
        assert d < 0.01, f"gateup qlen={qlen}: rel diff {d}"
        act = rand_bf16((qlen, TOPK, I), g)
        d = rel_diff(run_down(wrap, ids, act), ref_down(act, ids, w))
        assert d < 0.01, f"down qlen={qlen}: rel diff {d}"


@pytest.mark.cpu
def test_dense_agrees_with_the_k2_sibling():
    """같은 weight·x에서 k1과 k2가 같은 값을 낸다 (누산 순서만 다르다)."""
    w = make_weights(13)
    cpuinfer = _cpuinfer()
    k1, k2 = build(K1, w, cpuinfer), build(K2, w, cpuinfer)
    g = torch.Generator().manual_seed(14)
    ids = routing(1, g)
    x = rand_bf16((1, H), g)
    a, b = run_gateup(k1, ids, x), run_gateup(k2, ids, x)
    assert rel_diff(a, b.float()) < 1e-3
    act = rand_bf16((1, TOPK, I), g)
    a, b = run_down(k1, ids, act), run_down(k2, ids, act)
    assert rel_diff(a, b.float()) < 1e-3


# ── per-k sparse ──────────────────────────────────────────────────────────


@pytest.mark.cpu
def test_thr_only_tables_accepted_and_zero_thr_is_dense():
    w = make_weights(21)
    cpuinfer = _cpuinfer()
    dense = build(K1, w, cpuinfer)
    sparse = build(K1, w, cpuinfer, thr_tables(0.0, 0.0, 0.0))
    g = torch.Generator().manual_seed(22)
    ids = routing(1, g)
    x = rand_bf16((1, H), g)
    wts = torch.rand(1, TOPK, generator=g).contiguous()
    assert torch.equal(run_gateup(sparse, ids, x, wts), run_gateup(dense, ids, x))
    act = rand_bf16((1, TOPK, I), g)
    assert torch.equal(run_down(sparse, ids, act, wts), run_down(dense, ids, act))


@pytest.mark.cpu
def test_sparse_gateup_matches_masked_dense():
    w = make_weights(31)
    cpuinfer = _cpuinfer()
    dense = build(K1, w, cpuinfer)
    g = torch.Generator().manual_seed(32)
    ids = routing(1, g)
    x = rand_bf16((1, H), g)
    band = x[0, GU_OFF : GU_OFF + GU_ROWS]
    thr_g = _median_abs_thr(band)
    thr_u = _median_abs_thr(band[: GU_ROWS // 2])   # gate와 다른 thr → 마스크가 proj별인지도 본다
    sparse = build(K1, w, cpuinfer, thr_tables(thr_g, thr_u, 0.0))
    out = run_gateup(sparse, ids, x, torch.rand(1, TOPK, generator=g).contiguous())

    xg, keep_g = _masked(x[0], GU_OFF, GU_ROWS, thr_g)
    xu, keep_u = _masked(x[0], GU_OFF, GU_ROWS, thr_u)
    assert 0 < int(keep_g.sum()) < GU_ROWS and 0 < int(keep_u.sum()) < GU_ROWS
    assert not torch.equal(keep_g, keep_u)
    ref_g = run_gateup(dense, ids, xg.unsqueeze(0).contiguous())
    ref_u = run_gateup(dense, ids, xu.unsqueeze(0).contiguous())
    assert torch.equal(out[0, :, :I], ref_g[0, :, :I]), "gate half"
    assert torch.equal(out[0, :, I:], ref_u[0, :, I:]), "up half"


@pytest.mark.cpu
def test_sparse_down_matches_masked_dense():
    w = make_weights(41)
    cpuinfer = _cpuinfer()
    dense = build(K1, w, cpuinfer)
    g = torch.Generator().manual_seed(42)
    ids = routing(1, g)
    act = rand_bf16((1, TOPK, I), g)
    thr_d = _median_abs_thr(act[0, :, DN_OFF : DN_OFF + DN_ROWS])
    sparse = build(K1, w, cpuinfer, thr_tables(0.0, 0.0, thr_d))
    out = run_down(sparse, ids, act, torch.rand(1, TOPK, generator=g).contiguous())
    am = act.clone()
    kept = 0
    for j in range(TOPK):
        am[0, j], keep = _masked(act[0, j], DN_OFF, DN_ROWS, thr_d)
        kept += int(keep.sum())
    assert 0 < kept < TOPK * DN_ROWS
    assert torch.equal(out, run_down(dense, ids, am.contiguous()))


# ── score k1w1 (per-k × 열 노름) ───────────────────────────────────────────
#
# 마스크 종류를 **테이블이** 정한다: per-k 커널에 `*_wn_sq`를 같이 주면
# `|x_k|·wn_k >= thr` (제곱 비교 `x²·a >= thr²`), 안 주면 순수 `|x_k| >= thr`.
# 같은 커널·같은 thr에서 두 마스크가 **다른 k를 죽인다**는 것이 요점이다.


def wn_tables(w, keys=("gate", "up", "down")):
    """열(=k) L2 노름의 제곱. weight는 [E, N, K]이므로 N축으로 줄인다."""
    out = {}
    for name, t in (("gate", w["gate"]), ("up", w["up"]), ("down", w["down"])):
        if name in keys:
            a = (t.float() ** 2).sum(dim=1)          # [E, K]
            out[f"{name}_wn_sq"] = a.reshape(-1).contiguous()
    return out


@pytest.mark.cpu
def test_k1w1_matches_weighted_masked_dense():
    """thr + wn 을 주면 |x|·wn 판정이고, 그 마스킹된 dense와 **비트일치**한다."""
    w = make_weights(61)
    cpuinfer = _cpuinfer()
    dense = build(K1, w, cpuinfer)
    g = torch.Generator().manual_seed(62)
    ids = routing(1, g)
    act = rand_bf16((1, TOPK, I), g)
    a = wn_tables(w)["down_wn_sq"].reshape(E, I)     # [E, K]
    # expert별 점수가 다르므로 thr은 라우팅된 expert 기준으로 고른다.
    e0 = int(ids[0, 0])
    score = act[0, 0, DN_OFF : DN_OFF + DN_ROWS].float().abs() * a[e0, DN_OFF : DN_OFF + DN_ROWS].sqrt()
    flat = score.sort().values
    thr = 0.5 * (float(flat[flat.numel() // 2]) + float(flat[flat.numel() // 2 + 1]))
    tabs = {**thr_tables(0.0, 0.0, thr), **wn_tables(w)}
    sparse = build(K1, w, cpuinfer, tabs)
    out = run_down(sparse, ids, act, torch.rand(1, TOPK, generator=g).contiguous())

    am = act.clone()
    kept = 0
    for j in range(TOPK):
        e = int(ids[0, j])
        seg = am[0, j, DN_OFF : DN_OFF + DN_ROWS].float()
        keep = seg.abs() * a[e, DN_OFF : DN_OFF + DN_ROWS].sqrt() >= thr
        am[0, j, DN_OFF : DN_OFF + DN_ROWS] = torch.where(keep, seg, torch.zeros_like(seg)).to(torch.bfloat16)
        kept += int(keep.sum())
    assert 0 < kept < TOPK * DN_ROWS
    assert torch.equal(out, run_down(dense, ids, am.contiguous()))


@pytest.mark.cpu
def test_k1w1_and_k1_kill_different_k():
    """같은 thr에서 두 점수의 마스크가 달라야 한다 — 아니면 wn이 안 곱해진 것이다."""
    w = make_weights(63)
    cpuinfer = _cpuinfer()
    g = torch.Generator().manual_seed(64)
    ids, act = routing(1, g), rand_bf16((1, TOPK, I), g)
    thr = median_abs = _median_abs_thr(act[0, :, DN_OFF : DN_OFF + DN_ROWS])
    wts = torch.rand(1, TOPK, generator=g).contiguous()
    plain = build(K1, w, cpuinfer, thr_tables(0.0, 0.0, thr))
    weighted = build(K1, w, cpuinfer, {**thr_tables(0.0, 0.0, thr), **wn_tables(w)})
    a = run_down(plain, ids, act, wts)
    b = run_down(weighted, ids, act, wts)
    assert not torch.equal(a, b)


@pytest.mark.cpu
def test_partial_wn_tables_are_rejected():
    """셋 중 일부만 주면 즉사 — proj마다 점수가 갈리면 어느 자산으로 쟀는지 알 수 없다."""
    w = make_weights(65)
    cpuinfer = _cpuinfer()
    tabs = {**thr_tables(0.0, 0.0, 0.1), "down_wn_sq": wn_tables(w)["down_wn_sq"]}
    with pytest.raises(ValueError, match="셋 다 주거나"):
        build(K1, w, cpuinfer, tabs)


@pytest.mark.cpu
def test_huge_thr_zeroes_output():
    w = make_weights(51)
    cpuinfer = _cpuinfer()
    sparse = build(K1, w, cpuinfer, thr_tables(1e9, 1e9, 1e9))
    g = torch.Generator().manual_seed(52)
    ids = routing(1, g)
    wts = torch.rand(1, TOPK, generator=g).contiguous()
    assert torch.count_nonzero(run_gateup(sparse, ids, rand_bf16((1, H), g), wts)) == 0
    assert torch.count_nonzero(run_down(sparse, ids, rand_bf16((1, TOPK, I), g), wts)) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
