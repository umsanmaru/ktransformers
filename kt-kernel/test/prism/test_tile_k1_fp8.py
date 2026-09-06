#!/usr/bin/env python
# coding=utf-8
"""fp8b128 타일 커널(k1 / k2)의 Prism partial 경로 정확성 — PartialMoEWrapper를 통해.

대상:
- `kt_tile_k1_fp8b128`(per-k 마스크, score k1 = |x| >= thr)와 `kt_tile_k2_fp8b128`(페어 마스크,
  k2wl2)이 같은 e4m3 + 128×128 fp32 scale_inv 입력에서 fp32 레퍼런스와 일치하는가 (dense).
- per-k sparse: `sparse(x, thr) == dense(|x|<thr 인 k를 0으로 둔 x)` **비트 단위** — 마스킹된 k는
  dense 경로에서 곱이 0이라 두 계산이 같은 항을 같은 순서로 더한다.
- per-k 커널은 sparsity 테이블이 **thr 곡선 셋만**이어야 한다 (wn_sq/pair_dot 불필요·무시).
- prefill(qlen>1, 토큰 루프)도 레퍼런스와 일치.

기하: 노드 N shard 256 배수(inter 256, hidden 512), K 밴드 128 배수.
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
BLK = 128
MAX_LEN = 8
THREADS = 4
NG, GRID, PMAX, RENORM_IT = 201, 0.005, 0.9, 3

K1, K2 = "kt_tile_k1_fp8b128", "kt_tile_k2_fp8b128"


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


def e4m3_codes(shape, g):
    """유효 e4m3 코드 (지수 1..14, NaN/denormal 없음)."""
    s = torch.randint(0, 2, shape, generator=g)
    e = torch.randint(1, 15, shape, generator=g)
    m = torch.randint(0, 8, shape, generator=g)
    return ((s << 7) | (e << 3) | m).to(torch.uint8).contiguous()


def pow2_scales(shape, g):
    return torch.ldexp(torch.ones(shape), torch.randint(-3, 4, shape, generator=g).float()).contiguous()


def dequant(codes, scales):
    """codes [E, N, k] u8, scales [E, N/128, k/128] fp32 → fp32 [E, N, k]."""
    w = codes.view(torch.float8_e4m3fn).float()
    s = scales.repeat_interleave(BLK, dim=1).repeat_interleave(BLK, dim=2)
    return w * s


def make_weights(seed):
    g = torch.Generator().manual_seed(seed)
    cw = {
        "gate": e4m3_codes((E, I, GU_ROWS), g), "up": e4m3_codes((E, I, GU_ROWS), g),
        "down": e4m3_codes((E, H, DN_ROWS), g),
    }
    sc = {
        "gate": pow2_scales((E, I // BLK, GU_ROWS // BLK), g), "up": pow2_scales((E, I // BLK, GU_ROWS // BLK), g),
        "down": pow2_scales((E, H // BLK, DN_ROWS // BLK), g),
    }
    return cw, sc


def thr_tables(thr_g, thr_u, thr_d):
    full = lambda v: torch.full((E, NG), float(v), dtype=torch.float32)
    return {"thr_gate": full(thr_g), "thr_up": full(thr_u), "thr_down": full(thr_d)}


def build(kernel_key, cw, sc, cpuinfer, tables=None, p=0.5, lam=0.0):
    cfg = _config(tables is not None, p, lam)
    cfg.pool = cpuinfer.backend_
    w = PartialMoEWrapper(cfg, cpuinfer, kernel_key=kernel_key)
    w.load_weights_from_tensors(
        cw["gate"], cw["up"], cw["down"], sparsity_tables=tables,
        gate_scale=sc["gate"].reshape(-1), up_scale=sc["up"].reshape(-1), down_scale=sc["down"].reshape(-1),
    )
    return w


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


def ref_gateup(x, ids, wg, wu):
    xs = x[:, GU_OFF : GU_OFF + GU_ROWS].float()
    out = torch.empty(x.shape[0], TOPK, 2 * I)
    for m in range(x.shape[0]):
        for j in range(TOPK):
            e = int(ids[m, j])
            out[m, j, :I] = xs[m] @ wg[e].t()
            out[m, j, I:] = xs[m] @ wu[e].t()
    return out


def ref_down(act, ids, wd):
    out = torch.empty(act.shape[0], TOPK, H)
    for m in range(act.shape[0]):
        for j in range(TOPK):
            e = int(ids[m, j])
            out[m, j] = act[m, j, DN_OFF : DN_OFF + DN_ROWS].float() @ wd[e].t()
    return out


def rel_diff(out, ref):
    return (torch.mean(torch.abs(out.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)).item()


def rand_bf16(shape, g, scale=0.1):
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16).contiguous()


@pytest.mark.cpu
def test_k1_wrapper_reports_per_k_mask():
    cpuinfer = _cpuinfer()
    assert PartialMoEWrapper(_config(False), cpuinfer, kernel_key=K1).mask_per_k is True
    assert PartialMoEWrapper(_config(False), cpuinfer, kernel_key=K2).mask_per_k is False
    assert [n for n, _, _ in PartialMoEWrapper(_config(False), cpuinfer, kernel_key=K1).sparsity_keys()] == [
        "thr_gate", "thr_up", "thr_down"]


@pytest.mark.cpu
@pytest.mark.parametrize("kernel_key", [K1, K2])
@pytest.mark.parametrize("qlen", [1, 5])
def test_dense_matches_reference(kernel_key, qlen):
    cw, sc = make_weights(11)
    wg, wu, wd = dequant(cw["gate"], sc["gate"]), dequant(cw["up"], sc["up"]), dequant(cw["down"], sc["down"])
    cpuinfer = _cpuinfer()
    w = build(kernel_key, cw, sc, cpuinfer)
    g = torch.Generator().manual_seed(12)
    for _ in range(2):
        ids = routing(qlen, g)
        x = rand_bf16((qlen, H), g)
        d = rel_diff(run_gateup(w, ids, x), ref_gateup(x, ids, wg, wu))
        assert d < 0.01, f"{kernel_key} gateup qlen={qlen}: rel diff {d}"
        act = rand_bf16((qlen, TOPK, I), g)
        d = rel_diff(run_down(w, ids, act), ref_down(act, ids, wd))
        assert d < 0.01, f"{kernel_key} down qlen={qlen}: rel diff {d}"


@pytest.mark.cpu
def test_k1_sparse_thr_only_tables_accepted_and_zero_thr_is_dense():
    """per-k 커널은 thr 셋만으로 sparse가 켜진다; thr=0이면 전량 통과 → dense와 비트일치."""
    cw, sc = make_weights(21)
    cpuinfer = _cpuinfer()
    dense = build(K1, cw, sc, cpuinfer)
    tables = thr_tables(0.0, 0.0, 0.0)
    sparse = build(K1, cw, sc, cpuinfer, tables)
    g = torch.Generator().manual_seed(22)
    ids = routing(1, g)
    x = rand_bf16((1, H), g)
    wts = torch.rand(1, TOPK, generator=g).contiguous()
    assert torch.equal(run_gateup(sparse, ids, x, wts), run_gateup(dense, ids, x))
    act = rand_bf16((1, TOPK, I), g)
    assert torch.equal(run_down(sparse, ids, act, wts), run_down(dense, ids, act))


@pytest.mark.cpu
def test_k2_rejects_thr_only_tables():
    """페어 커널은 wn_sq/pair_dot이 필수 — thr만 주면 wrapper가 즉사한다."""
    cw, sc = make_weights(23)
    cpuinfer = _cpuinfer()
    with pytest.raises(ValueError, match="missing keys"):
        build(K2, cw, sc, cpuinfer, thr_tables(0.0, 0.0, 0.0))


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


@pytest.mark.cpu
def test_k1_sparse_gateup_matches_masked_dense():
    cw, sc = make_weights(31)
    cpuinfer = _cpuinfer()
    dense = build(K1, cw, sc, cpuinfer)
    g = torch.Generator().manual_seed(32)
    ids = routing(1, g)
    x = rand_bf16((1, H), g)
    band = x[0, GU_OFF : GU_OFF + GU_ROWS]
    thr_g = _median_abs_thr(band)
    thr_u = _median_abs_thr(band[: GU_ROWS // 2])  # gate와 다른 thr → 마스크가 proj별인지도 본다
    tables = thr_tables(thr_g, thr_u, 0.0)
    sparse = build(K1, cw, sc, cpuinfer, tables)
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
def test_k1_sparse_down_matches_masked_dense():
    cw, sc = make_weights(41)
    cpuinfer = _cpuinfer()
    dense = build(K1, cw, sc, cpuinfer)
    g = torch.Generator().manual_seed(42)
    ids = routing(1, g)
    act = rand_bf16((1, TOPK, I), g)
    thr_d = _median_abs_thr(act[0, :, DN_OFF : DN_OFF + DN_ROWS])
    sparse = build(K1, cw, sc, cpuinfer, thr_tables(0.0, 0.0, thr_d))
    out = run_down(sparse, ids, act, torch.rand(1, TOPK, generator=g).contiguous())
    am = act.clone()
    kept = 0
    for j in range(TOPK):
        am[0, j], keep = _masked(act[0, j], DN_OFF, DN_ROWS, thr_d)
        kept += int(keep.sum())
    assert 0 < kept < TOPK * DN_ROWS
    ref = run_down(dense, ids, am.contiguous())
    assert torch.equal(out, ref)


@pytest.mark.cpu
def test_k1_huge_thr_zeroes_output():
    cw, sc = make_weights(51)
    cpuinfer = _cpuinfer()
    sparse = build(K1, cw, sc, cpuinfer, thr_tables(1e9, 1e9, 1e9))
    g = torch.Generator().manual_seed(52)
    ids = routing(1, g)
    wts = torch.rand(1, TOPK, generator=g).contiguous()
    assert torch.count_nonzero(run_gateup(sparse, ids, rand_bf16((1, H), g), wts)) == 0
    assert torch.count_nonzero(run_down(sparse, ids, rand_bf16((1, TOPK, I), g), wts)) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
