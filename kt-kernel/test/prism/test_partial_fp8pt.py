#!/usr/bin/env python
# coding=utf-8
"""Prism partial — `kt_tile_k2_fp8pt` (FP8 e4m3, **per-tensor** fp32 배율) 정확성 테스트.

대상 체크포인트: Mistral-Medium-3.5-128B 류 static-tensor FP8 (`weight_block_size: null`,
`weight_scale_inv` 스칼라). 검증하는 것:

- 배율 텐서 계약: proj당 fp32 `[E]` (expert = 슬롯당 스칼라 1개). 모양이 다르면 wrapper가 즉사.
- K 정렬 완화: gate/up 밴드 rows가 128의 배수가 **아닌** 32의 배수 — b128 커널은 거절하는 기하.
- decode(qlen=1)/prefill(qlen>1, 토큰 루프) 둘 다 fp64 dequant 레퍼런스와 일치.
  weight가 e4m3 정확값 × 2의 거듭제곱 배율이고 x가 작은 정수라 곱은 정확하고 fp32 누산만 남는다
  → bf16 export 라운딩 한도(rel 1e-2)보다 훨씬 작은 오차여야 한다.
- expert마다 다른 배율이 제 슬롯에 붙는가 (배율을 섞으면 즉시 틀린다).
- sparse(pair mask) 경로: thr=0이면 dense와 비트일치, 큰 thr이면 0.
- NUMA 노드 2개: 노드 N shard가 256 배수여야 하므로 inter/hidden을 512로 잡는다.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
from kt_kernel import kt_kernel_ext
from kt_kernel.experts_partial import PartialMoEWrapper

KERNEL = "kt_tile_k2_fp8pt"
E = 4
TOP_K = 2
HIDDEN = 512          # gate/up K 전체 = down N (노드 2 → shard 256)
INTER = 512           # gate/up N = down K 전체 (노드 2 → shard 256)
GU_OFF, GU_ROWS = 64, 416    # 416 = 13×32, 128의 배수가 아님 (per-tensor에서만 허용)
DN_OFF, DN_ROWS = 32, 224    # 224 = 7×32
MAX_LEN = 16
CPUINFER_PARAM = 60
QLENS = [1, 5]

SP_NG, SP_GRID, SP_PMAX, SP_RENORM = 201, 0.005, 0.9, 3


def fp8_codes(shape, gen):
    """denormal/NaN 코드 없는 e4m3 바이트 (지수 1..14)."""
    e = torch.randint(1, 15, shape, generator=gen)
    m = torch.randint(0, 8, shape, generator=gen)
    s = torch.randint(0, 2, shape, generator=gen)
    return ((s << 7) | (e << 3) | m).to(torch.uint8)


def fp8_dequant(codes: torch.Tensor) -> torch.Tensor:
    c = codes.to(torch.int32)
    s = (c >> 7) & 1
    e = (c >> 3) & 0xF
    m = c & 7
    val = (1.0 + m.double() / 8.0) * torch.pow(2.0, (e - 7).double())
    val = torch.where(e == 0, torch.zeros_like(val), val)
    return torch.where(s == 1, -val, val)


@pytest.fixture(scope="module")
def cpuinfer():
    return kt_kernel_ext.CPUInfer(CPUINFER_PARAM)


def make_config(cpuinfer, sparsity_tables=None, p=0.5):
    cfg = kt_kernel_ext.moe.MOEConfig(E, TOP_K, HIDDEN, INTER, 0)
    cfg.max_len = MAX_LEN
    cfg.partial.enabled = True
    cfg.partial.gate.offset = GU_OFF
    cfg.partial.gate.rows = GU_ROWS
    cfg.partial.up.offset = GU_OFF
    cfg.partial.up.rows = GU_ROWS
    cfg.partial.down.offset = DN_OFF
    cfg.partial.down.rows = DN_ROWS
    cfg.partial.n_total = INTER
    cfg.pool = cpuinfer.backend_
    if sparsity_tables is not None:
        sp = cfg.partial.sparsity
        sp.pmax, sp.grid, sp.ng, sp.renorm_it = SP_PMAX, SP_GRID, SP_NG, SP_RENORM
        sp.p_gate = sp.p_up = sp.p_down = p
        sp.lam_gate = sp.lam_up = sp.lam_down = 0.0
    return cfg


class Weights:
    def __init__(self, seed=0, pow2_scales=True):
        g = torch.Generator().manual_seed(seed)
        self.gate = fp8_codes((E, INTER, GU_ROWS), g)
        self.up = fp8_codes((E, INTER, GU_ROWS), g)
        self.down = fp8_codes((E, HIDDEN, DN_ROWS), g)
        if pow2_scales:
            exp = torch.randint(-3, 4, (3, E), generator=g).float()
            sc = torch.pow(2.0, exp)
        else:
            sc = torch.rand((3, E), generator=g) * 2.0 + 0.01
        self.gate_scale = sc[0].contiguous()
        self.up_scale = sc[1].contiguous()
        self.down_scale = sc[2].contiguous()

    def deq(self, name):  # fp64 [E, N, K]
        codes = getattr(self, name)
        sc = getattr(self, name + "_scale").double().view(E, 1, 1)
        return fp8_dequant(codes) * sc

    def load(self, cpuinfer, tables=None, p=0.5):
        cfg = make_config(cpuinfer, tables, p)
        w = PartialMoEWrapper(cfg, cpuinfer, kernel_key=KERNEL)
        w.load_weights_from_tensors(
            self.gate, self.up, self.down, sparsity_tables=tables,
            gate_scale=self.gate_scale, up_scale=self.up_scale, down_scale=self.down_scale,
        )
        return w


def int_x(shape, gen):
    return torch.randint(-3, 4, shape, generator=gen).to(torch.bfloat16)


def routing(qlen, gen):
    return torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(qlen)]).contiguous()


def run_gateup(w, ids, x, weights=None):
    out = torch.full((x.shape[0], TOP_K, 2 * INTER), float("nan"), dtype=torch.bfloat16).contiguous()
    w.forward_gateup(ids, x, out, weights)
    return out


def run_down(w, ids, act, weights=None):
    out = torch.full((act.shape[0], TOP_K, HIDDEN), float("nan"), dtype=torch.bfloat16).contiguous()
    w.forward_down(ids, act, out, weights)
    return out


def ref_gateup(W: Weights, ids, x):
    xs = x[:, GU_OFF:GU_OFF + GU_ROWS].double()
    g, u = W.deq("gate"), W.deq("up")
    out = torch.empty(x.shape[0], TOP_K, 2 * INTER, dtype=torch.float64)
    for m in range(x.shape[0]):
        for j in range(TOP_K):
            e = int(ids[m, j])
            out[m, j, :INTER] = xs[m] @ g[e].t()
            out[m, j, INTER:] = xs[m] @ u[e].t()
    return out


def ref_down(W: Weights, ids, act):
    d = W.deq("down")
    out = torch.empty(act.shape[0], TOP_K, HIDDEN, dtype=torch.float64)
    for m in range(act.shape[0]):
        for j in range(TOP_K):
            e = int(ids[m, j])
            out[m, j] = act[m, j, DN_OFF:DN_OFF + DN_ROWS].double() @ d[e].t()
    return out


def assert_close_bf16(out, ref, tol=1e-2):
    """export가 bf16이라 상대오차 한도는 bf16 ulp(≈4e-3) 근처. 항 크기 합 대비로 잰다."""
    o = out.double()
    err = (o - ref).abs().max().item()
    mag = ref.abs().max().item() + 1e-9
    assert err / mag < tol, f"max err {err} vs |ref|max {mag}"


# ── 정확성 ────────────────────────────────────────────────────────────
@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_gateup_matches_dequant_reference(cpuinfer, qlen):
    W = Weights(seed=1)
    w = W.load(cpuinfer)
    g = torch.Generator().manual_seed(11)
    ids, x = routing(qlen, g), int_x((qlen, HIDDEN), g).contiguous()
    out = run_gateup(w, ids, x)
    assert not torch.any(torch.isnan(out)), "일부 출력이 쓰이지 않았다 (NaN 센티널 잔존)"
    assert_close_bf16(out, ref_gateup(W, ids, x))


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_down_matches_dequant_reference(cpuinfer, qlen):
    W = Weights(seed=2)
    w = W.load(cpuinfer)
    g = torch.Generator().manual_seed(12)
    ids = routing(qlen, g)
    act = int_x((qlen, TOP_K, INTER), g).contiguous()
    out = run_down(w, ids, act)
    assert not torch.any(torch.isnan(out))
    assert_close_bf16(out, ref_down(W, ids, act))


@pytest.mark.cpu
def test_non_pow2_scales(cpuinfer):
    """임의 fp32 배율 — 곱이 부정확해도 bf16 export 한도 안이어야 한다."""
    W = Weights(seed=3, pow2_scales=False)
    w = W.load(cpuinfer)
    g = torch.Generator().manual_seed(13)
    ids, x = routing(1, g), int_x((1, HIDDEN), g).contiguous()
    assert_close_bf16(run_gateup(w, ids, x), ref_gateup(W, ids, x))


@pytest.mark.cpu
def test_per_expert_scale_lands_on_its_own_slot(cpuinfer):
    """expert마다 다른 배율. 어떤 expert의 배율을 2배 하면 그 expert 출력만 2배."""
    W = Weights(seed=4)
    W.gate_scale = torch.tensor([1.0, 2.0, 4.0, 8.0])
    W.up_scale = torch.tensor([8.0, 4.0, 2.0, 1.0])
    W.down_scale = torch.tensor([0.5, 1.0, 2.0, 4.0])
    w = W.load(cpuinfer)
    g = torch.Generator().manual_seed(14)
    x = int_x((1, HIDDEN), g).contiguous()
    ids = torch.tensor([[0, 3]], dtype=torch.int64)
    out = run_gateup(w, ids, x).double()
    ref = ref_gateup(W, ids, x)
    assert_close_bf16(out, ref)
    # 배율 부착 검증: gate e=3 / gate e=0 비율은 8 (같은 x, 코드만 다르니 직접 비교는 불가 →
    # 레퍼런스에서 배율을 바꾼 모형과 비교한다)
    W2 = Weights(seed=4)
    W2.gate_scale = torch.tensor([1.0, 1.0, 1.0, 1.0])
    W2.up_scale = W.up_scale
    W2.down_scale = W.down_scale
    wrong = ref_gateup(W2, ids, x)
    assert not torch.allclose(out[0, 1, :INTER], wrong[0, 1, :INTER], rtol=1e-2, atol=1e-3)


# ── 계약 위반은 즉사 ─────────────────────────────────────────────────
@pytest.mark.cpu
def test_rejects_blockwise_scale_shape(cpuinfer):
    W = Weights(seed=5)
    cfg = make_config(cpuinfer)
    w = PartialMoEWrapper(cfg, cpuinfer, kernel_key=KERNEL)
    bad = torch.ones((E, INTER // 128, GU_ROWS // 32), dtype=torch.float32)  # [E] 가 아님
    with pytest.raises(ValueError, match="gate_scale"):
        w.load_weights_from_tensors(W.gate, W.up, W.down, gate_scale=bad, up_scale=W.up_scale,
                                    down_scale=W.down_scale)


@pytest.mark.cpu
def test_rejects_missing_scales(cpuinfer):
    W = Weights(seed=6)
    cfg = make_config(cpuinfer)
    w = PartialMoEWrapper(cfg, cpuinfer, kernel_key=KERNEL)
    with pytest.raises(ValueError):
        w.load_weights_from_tensors(W.gate, W.up, W.down)


@pytest.mark.cpu
def test_b128_kernel_rejects_this_geometry(cpuinfer):
    """같은 기하(rows 416/224)는 b128 커널에서는 배율 표가 정의되지 않아 거절돼야 한다 —
    per-tensor 커널이 정말로 정렬 제약을 푼다는 대조."""
    W = Weights(seed=7)
    cfg = make_config(cpuinfer)
    w = PartialMoEWrapper(cfg, cpuinfer, kernel_key="kt_tile_k2_fp8b128")
    with pytest.raises((ValueError, RuntimeError)):
        blk = lambda n, k: torch.ones((E, n // 128, max(k // 128, 1)), dtype=torch.float32)
        w.load_weights_from_tensors(W.gate, W.up, W.down, gate_scale=blk(INTER, GU_ROWS),
                                    up_scale=blk(INTER, GU_ROWS), down_scale=blk(HIDDEN, DN_ROWS))


# ── sparse (pair mask → plan) ────────────────────────────────────────
def sparsity_tables(W: Weights, thr=0.0):
    def of(codes, name):
        f = W.deq(name).float()  # [E, N, K]
        return (f * f).sum(dim=1).contiguous(), (f[:, :, 0::2] * f[:, :, 1::2]).sum(dim=1).contiguous()

    g_a, g_c = of(W.gate, "gate")
    u_a, u_c = of(W.up, "up")
    d_a, d_c = of(W.down, "down")
    full = lambda v: torch.full((E, SP_NG), float(v), dtype=torch.float32)
    return {
        "gate_wn_sq": g_a, "gate_pair_dot": g_c, "up_wn_sq": u_a, "up_pair_dot": u_c,
        "down_wn_sq": d_a, "down_pair_dot": d_c,
        "thr_gate": full(thr), "thr_up": full(thr), "thr_down": full(thr),
    }


def router_weights(qlen):
    return torch.full((qlen, TOP_K), 1.0 / TOP_K, dtype=torch.float32).contiguous()


@pytest.mark.cpu
def test_sparse_zero_threshold_bitexact_with_dense(cpuinfer):
    W = Weights(seed=8)
    tables = sparsity_tables(W, thr=0.0)
    w_sparse = W.load(cpuinfer, tables)
    w_dense = W.load(cpuinfer)
    g = torch.Generator().manual_seed(18)
    ids, x = routing(1, g), int_x((1, HIDDEN), g).contiguous()
    out_s = run_gateup(w_sparse, ids, x, router_weights(1))
    out_d = run_gateup(w_dense, ids, x)
    assert torch.equal(out_s, out_d)
    act = int_x((1, TOP_K, INTER), g).contiguous()
    assert torch.equal(run_down(w_sparse, ids, act, router_weights(1)), run_down(w_dense, ids, act))


@pytest.mark.cpu
def test_sparse_huge_threshold_zeroes_output(cpuinfer):
    W = Weights(seed=9)
    tables = sparsity_tables(W, thr=1e9)
    w = W.load(cpuinfer, tables)
    g = torch.Generator().manual_seed(19)
    ids, x = routing(1, g), int_x((1, HIDDEN), g).contiguous()
    out = run_gateup(w, ids, x, router_weights(1))
    assert torch.all(out == 0)


@pytest.mark.cpu
def test_sparse_matches_masked_dense(cpuinfer):
    """중간 thr: 살아남은 페어를 레퍼런스에서 같은 규칙으로 골라 계산하면 일치해야 한다."""
    W = Weights(seed=10)
    thr = 3.0
    tables = sparsity_tables(W, thr=thr)
    w = W.load(cpuinfer, tables)
    g = torch.Generator().manual_seed(20)
    ids, x = routing(1, g), int_x((1, HIDDEN), g).contiguous()
    out = run_gateup(w, ids, x, router_weights(1)).double()

    xs = x[0, GU_OFF:GU_OFF + GU_ROWS].double()
    gd, ud = W.deq("gate"), W.deq("up")
    kept_any = False
    for j in range(TOP_K):
        e = int(ids[0, j])
        for name, wd, a, c in (("gate", gd, tables["gate_wn_sq"], tables["gate_pair_dot"]),
                               ("up", ud, tables["up_wn_sq"], tables["up_pair_dot"])):
            x0, x1 = xs[0::2], xs[1::2]
            imp = (a[e, 0::2].double() * x0 * x0 + a[e, 1::2].double() * x1 * x1
                   + 2 * c[e].double() * x0 * x1).clamp_min(0).sqrt()
            keep = (imp >= thr).repeat_interleave(2)
            kept_any |= bool(keep.any()) and not bool(keep.all())
            ref = (xs * keep) @ wd[e].t()
            sl = slice(0, INTER) if name == "gate" else slice(INTER, 2 * INTER)
            assert_close_bf16(out[0, j, sl], ref)
    assert kept_any, "thr가 마스크를 전혀 만들지 않았다 — 테스트가 dense와 같아진다"
