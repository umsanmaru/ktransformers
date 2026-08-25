#!/usr/bin/env python
# coding=utf-8
"""Prism partial 진입점 정확성 테스트 — AMX BF16, decode(qlen=1) + prefill(qlen>1).

대상 (계약: sglang prism CONTRACTS.md ②/⑤):
- forward_gateup_partial: full-width x에서 구워진 K 밴드만 읽어 pre-act
  partial [qlen, k, 2*inter] export ([0,inter)=gate, [inter,2inter)=up).
- forward_down_partial: full-width act의 slot별 행에서 down K 밴드를 읽어
  [qlen, k, hidden] partial export. N(hidden)축 노드 분할 — 합산 없음.
- prefill 경로: 토큰→expert 그룹핑(m_local_pos_) 후 GEMM, (m, slot) 좌표로
  scatter export. qlen=16 × k=2 / E=8 이라 **토큰 간 중복 expert가 반드시
  발생** — 좌표 뒤섞임을 정수 비트일치가 검출한다.
- NUMA shard: 균등 기본값 + Plan 주입 불균등 테이블.
- 수치: 내부 누산 fp32(AMX 타일), 재료화 bf16 → 일반 입력은 tolerance,
  bf16-무손실 정수 입력은 비트일치 (계약 ⑤-5).
- guard: fused forward 거부, qlen 범위 밖 거부 (호출 스레드 예외).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
from kt_kernel import kt_kernel_ext

expert_num = 8
hidden_size = 256
intermediate_size = 128
num_experts_per_tok = 2
max_len = 16
CPUINFER_PARAM = 60

GATEUP_K_OFFSET = 64
GATEUP_K_ROWS = 192  # gate/up cold 밴드 = hidden rows [64, 256)
DOWN_K_OFFSET = 32
DOWN_K_ROWS = 96  # down cold 밴드 = inter rows [32, 128)

QLENS = [1, 16]  # decode / prefill(그룹핑·scatter 경로)


def build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer, shard_tables=None):
    """K-슬라이스 텐서 주입으로 partial 인스턴스 생성 (kt SFT 주입 경로와 동형).

    shard_tables: (gu_off, gu_rows, dn_off, dn_rows) — 불균등 NUMA 분할 테이블.
    None이면 TP ctor의 균등 분할.
    """
    config = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0
    )
    config.max_len = max_len
    config.partial.enabled = True
    config.partial.gate.offset = GATEUP_K_OFFSET
    config.partial.gate.rows = GATEUP_K_ROWS
    config.partial.up.offset = GATEUP_K_OFFSET
    config.partial.up.rows = GATEUP_K_ROWS
    config.partial.down.offset = DOWN_K_OFFSET
    config.partial.down.rows = DOWN_K_ROWS
    config.partial.n_total = intermediate_size
    if shard_tables is not None:
        gu_off, gu_rows, dn_off, dn_rows = shard_tables
        config.partial.node_gateup_n_offset = gu_off
        config.partial.node_gateup_n_rows = gu_rows
        config.partial.node_down_n_offset = dn_off
        config.partial.node_down_n_rows = dn_rows
    config.gate_proj = gate_cold.data_ptr()
    config.up_proj = up_cold.data_ptr()
    config.down_proj = down_cold.data_ptr()
    config.gate_scale = 0
    config.up_scale = 0
    config.down_scale = 0
    config.pool = cpuinfer.backend_

    moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
    phys_map = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpuinfer.submit(moe.load_weights_task(phys_map.data_ptr()))
    cpuinfer.sync()
    return moe


def make_weights(generator):
    gate_full = generator((expert_num, intermediate_size, hidden_size))
    up_full = generator((expert_num, intermediate_size, hidden_size))
    down_full = generator((expert_num, hidden_size, intermediate_size))
    gate_cold = gate_full[:, :, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS].contiguous()
    up_cold = up_full[:, :, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS].contiguous()
    down_cold = down_full[:, :, DOWN_K_OFFSET : DOWN_K_OFFSET + DOWN_K_ROWS].contiguous()
    return gate_cold, up_cold, down_cold


def make_routing(qlen):
    """행마다 독립 randperm — qlen>k*?면 토큰 간 중복 expert가 필연."""
    return torch.stack(
        [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
    ).contiguous()


def run_gateup(moe, expert_ids, x, weights=None):
    qlen_t = torch.tensor([x.shape[0]], dtype=torch.int32)
    out = torch.full(
        (x.shape[0], num_experts_per_tok, 2 * intermediate_size), 777.0, dtype=torch.bfloat16
    ).contiguous()
    moe.forward_gateup_partial(
        qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), x.data_ptr(), out.data_ptr(),
        0 if weights is None else weights.data_ptr()
    )
    return out


def run_down(moe, expert_ids, act, weights=None):
    qlen_t = torch.tensor([act.shape[0]], dtype=torch.int32)
    out = torch.full(
        (act.shape[0], num_experts_per_tok, hidden_size), 777.0, dtype=torch.bfloat16
    ).contiguous()
    moe.forward_down_partial(
        qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), act.data_ptr(), out.data_ptr(),
        0 if weights is None else weights.data_ptr()
    )
    return out


def gateup_reference(x, expert_ids, gate_cold, up_cold):
    """fp32 레퍼런스 — 라우팅/그룹핑 방식을 모름 (토큰별 직접 계산)."""
    qlen = x.shape[0]
    xs = x[:, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS].float()
    out = torch.empty(qlen, num_experts_per_tok, 2 * intermediate_size)
    for m in range(qlen):
        for j in range(num_experts_per_tok):
            e = int(expert_ids[m, j])
            out[m, j, :intermediate_size] = xs[m] @ gate_cold[e].float().t()
            out[m, j, intermediate_size:] = xs[m] @ up_cold[e].float().t()
    return out


def down_reference(act, expert_ids, down_cold):
    qlen = act.shape[0]
    out = torch.empty(qlen, num_experts_per_tok, hidden_size)
    for m in range(qlen):
        for j in range(num_experts_per_tok):
            e = int(expert_ids[m, j])
            a = act[m, j, DOWN_K_OFFSET : DOWN_K_OFFSET + DOWN_K_ROWS].float()
            out[m, j] = a @ down_cold[e].float().t()
    return out


def rel_diff(out, ref):
    return (torch.mean(torch.abs(out.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)).item()


randn_gen = lambda s: (torch.randn(s, dtype=torch.float32) / 10.0).to(torch.bfloat16)
int_gen = lambda s: torch.randint(-1, 2, s, dtype=torch.float32).to(torch.bfloat16)


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_gateup_partial_matches_reference(qlen):
    torch.manual_seed(0)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    for i in range(3):
        expert_ids = make_routing(qlen)
        x = randn_gen((qlen, hidden_size)).contiguous()
        d = rel_diff(run_gateup(moe, expert_ids, x), gateup_reference(x, expert_ids, gate_cold, up_cold))
        print(f"gateup qlen={qlen} iter {i}: rel diff = {d:.6f}")
        assert d < 0.02


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_gateup_partial_bitexact_on_integer_inputs(qlen):
    """계약 ⑤-5: bf16 무손실 입력에서 비트일치. prefill에서는 (m, slot) 좌표
    뒤섞임의 exact 검출기 — 토큰마다 다른 정수 x라 permutation이면 즉시 불일치."""
    torch.manual_seed(1)
    gate_cold, up_cold, down_cold = make_weights(int_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = make_routing(qlen)
    x = int_gen((qlen, hidden_size)).contiguous()
    out = run_gateup(moe, expert_ids, x)
    ref = gateup_reference(x, expert_ids, gate_cold, up_cold).to(torch.bfloat16)
    assert torch.equal(out, ref), f"integer gateup partial must be bit-exact (qlen={qlen})"


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_down_partial_matches_reference(qlen):
    torch.manual_seed(4)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    for i in range(3):
        expert_ids = make_routing(qlen)
        act = randn_gen((qlen, num_experts_per_tok, intermediate_size)).contiguous()
        d = rel_diff(run_down(moe, expert_ids, act), down_reference(act, expert_ids, down_cold))
        print(f"down qlen={qlen} iter {i}: rel diff = {d:.6f}")
        assert d < 0.02


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_down_partial_bitexact_on_integer_inputs(qlen):
    torch.manual_seed(5)
    gate_cold, up_cold, down_cold = make_weights(int_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = make_routing(qlen)
    act = int_gen((qlen, num_experts_per_tok, intermediate_size)).contiguous()
    out = run_down(moe, expert_ids, act)
    ref = down_reference(act, expert_ids, down_cold).to(torch.bfloat16)
    assert torch.equal(out, ref), f"integer down partial must be bit-exact (qlen={qlen})"


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_gateup_act_down_chain(qlen):
    """두 진입점의 합성: gateup → (torch act) → down == torch 동일 체인."""
    torch.manual_seed(6)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = make_routing(qlen)
    x = randn_gen((qlen, hidden_size)).contiguous()

    gu = run_gateup(moe, expert_ids, x)
    gate, up = gu.float().split(intermediate_size, dim=2)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).contiguous()
    out = run_down(moe, expert_ids, act)

    rg, ru = gateup_reference(x, expert_ids, gate_cold, up_cold).split(intermediate_size, dim=2)
    ref_act = (torch.nn.functional.silu(rg) * ru).to(torch.bfloat16)
    ref = down_reference(ref_act, expert_ids, down_cold)

    d = rel_diff(out, ref)
    print(f"chain qlen={qlen}: rel diff = {d:.6f}")
    assert d < 0.03


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_unequal_numa_shard_tables(qlen):
    """Plan 주입 불균등 분할 (inter 96/32, hidden 192/64) × decode/prefill.
    레퍼런스는 분할 방식을 모르므로, 통과 = 스테이징·export offset 전부 정합."""
    torch.manual_seed(7)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    tables = ([0, 96], [96, 32], [0, 192], [192, 64])
    try:
        moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer, shard_tables=tables)
    except RuntimeError as err:
        if "tp_count" in str(err):
            pytest.skip("requires exactly 2 NUMA nodes")
        raise

    expert_ids = make_routing(qlen)
    x = randn_gen((qlen, hidden_size)).contiguous()
    assert rel_diff(run_gateup(moe, expert_ids, x),
                    gateup_reference(x, expert_ids, gate_cold, up_cold)) < 0.02

    act = randn_gen((qlen, num_experts_per_tok, intermediate_size)).contiguous()
    assert rel_diff(run_down(moe, expert_ids, act),
                    down_reference(act, expert_ids, down_cold)) < 0.02


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", QLENS)
def test_task_path_via_wrapper(qlen):
    """K3: task 경로 (submit → TaskQueue worker 실행 → sync) + PartialMoEWrapper.
    동기 경로와 같은 입력에서 같은 결과여야 한다."""
    from kt_kernel.experts_partial import PartialMoEWrapper

    torch.manual_seed(8)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

    config = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0
    )
    config.max_len = max_len
    config.partial.enabled = True
    config.partial.gate.offset = GATEUP_K_OFFSET
    config.partial.gate.rows = GATEUP_K_ROWS
    config.partial.up.offset = GATEUP_K_OFFSET
    config.partial.up.rows = GATEUP_K_ROWS
    config.partial.down.offset = DOWN_K_OFFSET
    config.partial.down.rows = DOWN_K_ROWS
    config.partial.n_total = intermediate_size
    config.pool = cpuinfer.backend_

    wrapper = PartialMoEWrapper(config, cpuinfer)
    wrapper.load_weights_from_tensors(gate_cold, up_cold, down_cold)

    expert_ids = make_routing(qlen)
    x = randn_gen((qlen, hidden_size)).contiguous()
    qlen_t = torch.tensor([qlen], dtype=torch.int32)
    out = torch.full((qlen, num_experts_per_tok, 2 * intermediate_size), 777.0, dtype=torch.bfloat16).contiguous()

    wrapper.submit_forward_gateup(
        qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), x.data_ptr(), out.data_ptr()
    )
    wrapper.sync()
    ref = gateup_reference(x, expert_ids, gate_cold, up_cold)
    assert rel_diff(out, ref) < 0.02

    act = randn_gen((qlen, num_experts_per_tok, intermediate_size)).contiguous()
    out_d = torch.full((qlen, num_experts_per_tok, hidden_size), 777.0, dtype=torch.bfloat16).contiguous()
    wrapper.submit_forward_down(
        qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), act.data_ptr(), out_d.data_ptr()
    )
    wrapper.sync()
    assert rel_diff(out_d, down_reference(act, expert_ids, down_cold)) < 0.02

    # 동기 편의 API도 같은 결과
    out2 = torch.empty_like(out)
    wrapper.forward_gateup(expert_ids, x, out2)
    assert torch.equal(out, out2)


@pytest.mark.cpu
def test_fused_forward_rejected_on_partial_instance():
    """partial 인스턴스에 fused forward를 부르면 즉사 (조용한 오답 방지)."""
    torch.manual_seed(2)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = make_routing(1)
    weights = torch.rand((1, num_experts_per_tok), dtype=torch.float32).contiguous()
    x = randn_gen((1, hidden_size)).contiguous()
    output = torch.empty((1, hidden_size), dtype=torch.bfloat16).contiguous()
    qlen_t = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="partial"):
        moe.forward(
            qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(),
            weights.data_ptr(), x.data_ptr(), output.data_ptr(), False,
        )


@pytest.mark.cpu
def test_qlen_out_of_range_rejected_cleanly():
    """qlen > max_len은 abort가 아니라 예외로 거부 (호출 스레드 검증 회귀 테스트)."""
    torch.manual_seed(3)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    bad_qlen = max_len + 16
    expert_ids = make_routing(bad_qlen)
    x = randn_gen((bad_qlen, hidden_size)).contiguous()
    out = torch.empty((bad_qlen, num_experts_per_tok, 2 * intermediate_size), dtype=torch.bfloat16).contiguous()
    qlen_t = torch.tensor([bad_qlen], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="qlen"):
        moe.forward_gateup_partial(
            qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), x.data_ptr(), out.data_ptr()
        )


if __name__ == "__main__":
    for q in QLENS:
        test_gateup_partial_matches_reference(q)
        test_gateup_partial_bitexact_on_integer_inputs(q)
        test_down_partial_matches_reference(q)
        test_down_partial_bitexact_on_integer_inputs(q)
        test_gateup_act_down_chain(q)
    test_fused_forward_rejected_on_partial_instance()
    test_qlen_out_of_range_rejected_cleanly()
    print("ALL PASSED")


# ===========================================================================
# 입력기반 sparsity (k2wl2) — 계약 ①
# ===========================================================================
#
# 예산(p, lam, pmax, grid, ng, renorm_it)과 threshold 곡선이 config/테이블에
# 상주하고, step마다 오는 것은 라우터 가중 하나다. 그래서 유효 threshold를
# 특정 값으로 고정하려면 **thr 곡선을 그 값으로 채운다** — 격자 인덱스가
# 무엇이든 조회 결과가 같아지므로 s를 우회할 수 있다.
#
# 검증 전략: 마스킹된 페어는 dense에서 a=0이 되어 c + 0*b = c (정확)이므로
#
#     sparse(x, mask)  ==  dense(mask 적용된 x)      (비트 단위)

SP_NG, SP_GRID, SP_PMAX, SP_RENORM = 201, 0.005, 0.9, 3


def sparsity_tables_from_weights(gate_cold, up_cold, down_cold,
                                 thr_g=0.0, thr_u=0.0, thr_d=0.0, ng=SP_NG):
    """cold weight에서 점수 재료를 유도하고 thr 곡선은 상수로 채운다.

    wn_sq[j] = ||W[:, j]||²  (= a) / pair_dot[p] = <W[:,2p], W[:,2p+1]>  (= c)
    합성 난수가 아니라 실제 weight에서 뽑으므로 부호·스케일이 실전과 같다.
    """

    def of(w):  # w: [E, N, K]
        f = w.float()
        wn_sq = (f * f).sum(dim=1).contiguous()                       # [E, K]
        c = (f[:, :, 0::2] * f[:, :, 1::2]).sum(dim=1).contiguous()   # [E, K/2]
        return wn_sq, c

    g_a, g_c = of(gate_cold)
    u_a, u_c = of(up_cold)
    d_a, d_c = of(down_cold)
    full = lambda v: torch.full((expert_num, ng), float(v), dtype=torch.float32)
    return {
        "gate_wn_sq": g_a, "gate_pair_dot": g_c,
        "up_wn_sq": u_a, "up_pair_dot": u_c,
        "down_wn_sq": d_a, "down_pair_dot": d_c,
        "thr_gate": full(thr_g), "thr_up": full(thr_u), "thr_down": full(thr_d),
    }


def ramp_thr_tables(tables, cutoff_idx, ng=SP_NG):
    """격자 인덱스 < cutoff면 0, 그 이상은 차단. s의 방향을 관찰 가능하게 만든다."""
    step = torch.full((expert_num, ng), 1e9, dtype=torch.float32)
    step[:, :cutoff_idx] = 0.0
    out = dict(tables)
    for key in ("thr_gate", "thr_up", "thr_down"):
        out[key] = step.clone()
    return out


def build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables,
                     p=0.5, lam=0.0):
    """build_partial_moe와 같지만 sparsity 예산 + 테이블을 설치한다.

    테이블은 C++가 step마다 원시 포인터로 읽으므로 호출자가 붙잡고 있어야
    한다 — 여기서는 tables dict가 테스트 스코프에 살아 있다.
    """
    config = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0
    )
    config.max_len = max_len
    config.partial.enabled = True
    config.partial.gate.offset = GATEUP_K_OFFSET
    config.partial.gate.rows = GATEUP_K_ROWS
    config.partial.up.offset = GATEUP_K_OFFSET
    config.partial.up.rows = GATEUP_K_ROWS
    config.partial.down.offset = DOWN_K_OFFSET
    config.partial.down.rows = DOWN_K_ROWS
    config.partial.n_total = intermediate_size
    config.gate_proj = gate_cold.data_ptr()
    config.up_proj = up_cold.data_ptr()
    config.down_proj = down_cold.data_ptr()
    config.gate_scale = 0
    config.up_scale = 0
    config.down_scale = 0
    config.pool = cpuinfer.backend_

    sp = config.partial.sparsity
    sp.pmax, sp.grid, sp.ng, sp.renorm_it = SP_PMAX, SP_GRID, SP_NG, SP_RENORM
    sp.p_gate = sp.p_up = sp.p_down = p
    sp.lam_gate = sp.lam_up = sp.lam_down = lam
    for name, t in tables.items():
        setattr(sp, name, t.data_ptr())
    sp.enabled = True

    moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
    phys_map = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpuinfer.submit(moe.load_weights_task(phys_map.data_ptr()))
    cpuinfer.sync()
    return moe


def router_weights(qlen, values=None):
    """[qlen, k] fp32. values를 주면 그 행을 모든 토큰에 쓴다."""
    if values is None:
        return torch.rand(qlen, num_experts_per_tok, dtype=torch.float32).contiguous()
    row = torch.tensor(values, dtype=torch.float32)
    return row.unsqueeze(0).repeat(qlen, 1).contiguous()


def pair_energy(vec_band, a_sq, c):
    """C++ build_pair_mask의 e (clamp 후 제곱 에너지). fp32로 계산."""
    x = vec_band.float()
    x0, x1 = x[0::2], x[1::2]
    e = a_sq[0::2] * x0 * x0 + a_sq[1::2] * x1 * x1 + 2.0 * c * x0 * x1
    return torch.clamp(e, min=0.0)


def mask_band_(full, offset, rows, keep_pairs):
    """full[offset:offset+rows]의 마스킹된 페어를 0으로. bf16 왕복은 무손실."""
    out = full.clone()
    m = keep_pairs.repeat_interleave(2).to(torch.float32)
    seg = out[..., offset : offset + rows].float() * m
    out[..., offset : offset + rows] = seg.to(torch.bfloat16)
    return out.contiguous()


def pick_threshold(energies):
    """정렬된 e의 중앙 두 값 사이 중점. 판정이 float 오차로 흔들리지 않는다.

    파이썬 쪽 마스크도 C++와 똑같이 `thr*thr`로 비교하므로 sqrt 왕복 오차가
    있어도 두 판정이 정확히 일치한다.
    """
    flat = torch.cat([e.reshape(-1) for e in energies]).sort().values
    n = flat.numel()
    mid = 0.5 * (float(flat[n // 2]) + float(flat[n // 2 + 1]))
    return float(mid ** 0.5)


@pytest.mark.cpu
def test_sparse_zero_threshold_bitexact_with_dense():
    """thr=0이면 전량 통과 — sparse 커널이 dense와 **비트 동일**해야 한다.

    누산 순서를 dense와 같게 유지했다는 설계의 회귀 테스트다.
    """
    torch.manual_seed(20)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    dense = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)

    ids = make_routing(1)
    x = randn_gen((1, hidden_size)).contiguous()
    act = randn_gen((1, num_experts_per_tok, intermediate_size)).contiguous()
    w = router_weights(1)

    assert torch.equal(run_gateup(sparse, ids, x, w), run_gateup(dense, ids, x))
    assert torch.equal(run_down(sparse, ids, act, w), run_down(dense, ids, act))


@pytest.mark.cpu
def test_sparse_huge_threshold_zeroes_output():
    """전량 차단 → 타일 마스크가 0이라 B 로드 없이 0이 export된다."""
    torch.manual_seed(21)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold,
                                          thr_g=1e9, thr_u=1e9, thr_d=1e9)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)

    ids = make_routing(1)
    x = randn_gen((1, hidden_size)).contiguous()
    act = randn_gen((1, num_experts_per_tok, intermediate_size)).contiguous()
    w = router_weights(1)

    assert torch.count_nonzero(run_gateup(sparse, ids, x, w)) == 0
    assert torch.count_nonzero(run_down(sparse, ids, act, w)) == 0


@pytest.mark.cpu
def test_sparse_gateup_matches_masked_dense():
    """중간 threshold: sparse == dense(마스킹된 x), 비트 단위.

    gate와 up은 노름·내적이 달라 마스크도 다르므로, dense 레퍼런스를 각
    마스크로 두 번 돌려 해당 절반씩 취한다 — 이 비대칭이 "A 한 벌을 공유
    하면서 마스크는 proj별"이라는 설계가 실제로 지켜지는지 검증한다.
    """
    torch.manual_seed(22)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    base = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    dense = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    ids = make_routing(1)
    x = randn_gen((1, hidden_size)).contiguous()
    band = x[0, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS]
    experts = [int(ids[0, j]) for j in range(num_experts_per_tok)]
    e_gate = [pair_energy(band, base["gate_wn_sq"][e], base["gate_pair_dot"][e])
              for e in experts]
    e_up = [pair_energy(band, base["up_wn_sq"][e], base["up_pair_dot"][e])
            for e in experts]
    thr_g, thr_u = pick_threshold(e_gate), pick_threshold(e_up)

    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold,
                                          thr_g=thr_g, thr_u=thr_u)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)
    out = run_gateup(sparse, ids, x, router_weights(1))

    kept_total = 0
    for j, e in enumerate(experts):
        for which, energies, thr in (("gate", e_gate, thr_g), ("up", e_up, thr_u)):
            keep = energies[j] >= (thr * thr)
            kept_total += int(keep.sum())
            xm = mask_band_(x, GATEUP_K_OFFSET, GATEUP_K_ROWS, keep)
            ref = run_gateup(dense, ids, xm)
            lo, hi = (0, intermediate_size) if which == "gate" else (
                intermediate_size, 2 * intermediate_size)
            assert torch.equal(out[0, j, lo:hi], ref[0, j, lo:hi]), f"{which} slot {j}"

    total = num_experts_per_tok * 2 * (GATEUP_K_ROWS // 2)
    assert 0 < kept_total < total, f"마스크가 무의미 ({kept_total}/{total})"


@pytest.mark.cpu
def test_sparse_down_matches_masked_dense():
    torch.manual_seed(23)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    base = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    dense = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    ids = make_routing(1)
    act = randn_gen((1, num_experts_per_tok, intermediate_size)).contiguous()
    experts = [int(ids[0, j]) for j in range(num_experts_per_tok)]
    energies = [
        pair_energy(act[0, j, DOWN_K_OFFSET : DOWN_K_OFFSET + DOWN_K_ROWS],
                    base["down_wn_sq"][e], base["down_pair_dot"][e])
        for j, e in enumerate(experts)
    ]
    thr_d = pick_threshold(energies)
    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold, thr_d=thr_d)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)
    out = run_down(sparse, ids, act, router_weights(1))

    kept_total = 0
    for j in range(num_experts_per_tok):
        keep = energies[j] >= (thr_d * thr_d)
        kept_total += int(keep.sum())
        am = act.clone()
        am[0, j] = mask_band_(act[0, j], DOWN_K_OFFSET, DOWN_K_ROWS, keep)
        ref = run_down(dense, ids, am.contiguous())
        assert torch.equal(out[0, j], ref[0, j]), f"down slot {j}"

    total = num_experts_per_tok * (DOWN_K_ROWS // 2)
    assert 0 < kept_total < total, f"마스크가 무의미 ({kept_total}/{total})"


@pytest.mark.cpu
def test_lambda_shifts_budget_by_router_weight():
    """gate-dynamic λ의 방향 검증 — 라우터 가중이 큰 slot을 덜 자른다.

    thr 곡선을 계단(격자 < cutoff면 0, 이상은 차단)으로 두면 s가 곧 관찰
    가능해진다. λ>0에서 큰 가중의 slot은 s가 작아져 dense가 되고, 작은
    가중의 slot은 s가 pmax로 밀려 전량 차단된다. λ=0이면 둘 다 s=p로
    같아져 함께 차단된다 — 이 대비가 s 수식이 배선됐는지의 증거다.
    """
    torch.manual_seed(24)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    base = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    # p=0.5 → 격자 100. cutoff 50이면 s<0.25만 통과한다.
    tables = ramp_thr_tables(base, cutoff_idx=50)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

    ids = make_routing(1)
    x = randn_gen((1, hidden_size)).contiguous()
    w = router_weights(1, values=[0.9, 0.1])  # slot 0이 지배적

    lam0 = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables,
                            p=0.5, lam=0.0)
    out0 = run_gateup(lam0, ids, x, w)
    assert torch.count_nonzero(out0) == 0, "λ=0이면 두 slot 모두 s=p=0.5 → 차단"

    lam20 = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables,
                             p=0.5, lam=20.0)
    out20 = run_gateup(lam20, ids, x, w)
    assert torch.count_nonzero(out20[0, 0]) > 0, "가중 큰 slot은 dense여야 한다"
    assert torch.count_nonzero(out20[0, 1]) == 0, "가중 작은 slot은 차단돼야 한다"


@pytest.mark.cpu
def test_sparse_rejects_prefill():
    """sparsity는 decode 전용 (계약 ①) — qlen>1에 가중을 주면 즉사."""
    torch.manual_seed(25)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)

    ids = make_routing(4)
    x = randn_gen((4, hidden_size)).contiguous()
    with pytest.raises(RuntimeError, match="decode-only"):
        run_gateup(sparse, ids, x, router_weights(4))


@pytest.mark.cpu
def test_sparse_rejects_weights_without_tables():
    """sparsity를 끈 인스턴스에 가중을 주면 조용히 dense로 흐르지 않고 즉사."""
    torch.manual_seed(26)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    dense = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    ids = make_routing(1)
    x = randn_gen((1, hidden_size)).contiguous()
    with pytest.raises(RuntimeError, match="sparsity.enabled is false"):
        run_gateup(dense, ids, x, router_weights(1))


# ── 인덱스 기하 (sglang CONTRACTS.md ① 2026-08-25) ──────────────────────────
#
# 티어 멤버십이 밴드에서 **가변 per-expert 인덱스**로 바뀐 경로. 밴드는 이
# 표현의 퇴화형이므로 위 테스트들이 그 등가성을 계속 지키고, 여기서는 밴드로는
# 표현 불가능한 것 — expert마다 다른 행 수, 무작위 순열 — 을 검증한다.

K_STEP = 32  # packed 저장의 타일 단위 (expert별 행 수는 이 배수여야 한다)


def make_index(rows_per_expert, axis_len, seed, shuffle=True):
    """(row_off, idx) — expert별 가변 길이 + (선택) 무작위 순열."""
    g = torch.Generator().manual_seed(seed)
    row_off, idx, acc = [0], [], 0
    for kr in rows_per_expert:
        cols = torch.randperm(axis_len, generator=g)[:kr] if shuffle else torch.arange(kr)
        idx.append(cols)
        acc += kr
        row_off.append(acc)
    return (torch.tensor(row_off, dtype=torch.int32),
            torch.cat(idx).to(torch.int32))


def build_indexed_moe(gu_rows, dn_rows, cpuinfer, gen=randn_gen, seed=0, shuffle=True,
                      up_rows=None):
    """인덱스형 partial 인스턴스 + 레퍼런스용 (full weight, index) 반환.

    주입 weight는 expert 블록 [N, k(e)]를 이어 붙인 flat이다 — 열 r이 원본
    K열 idx_e[r]에 대응한다 (로더가 그렇게 잘라 넣는 것과 같은 형태).
    """
    gate_full = gen((expert_num, intermediate_size, hidden_size))
    up_full = gen((expert_num, intermediate_size, hidden_size))
    down_full = gen((expert_num, hidden_size, intermediate_size))
    gu_off, gu_idx = make_index(gu_rows, hidden_size, seed, shuffle)
    # up_rows를 주면 up이 **자기 인덱스**를 갖는다 → kt가 dual-pack으로 간다.
    if up_rows is None:
        up_off, up_idx = gu_off, gu_idx
    else:
        up_off, up_idx = make_index(up_rows, hidden_size, seed + 2, shuffle)
    dn_off, dn_idx = make_index(dn_rows, intermediate_size, seed + 1, shuffle)

    def flat(full, off, idx):
        blocks = []
        for e in range(expert_num):
            cols = idx[off[e] : off[e + 1]].to(torch.int64)
            blocks.append(full[e][:, cols].contiguous().reshape(-1))
        return torch.cat(blocks).contiguous()

    gate_flat, up_flat = flat(gate_full, gu_off, gu_idx), flat(up_full, up_off, up_idx)
    down_flat = flat(down_full, dn_off, dn_idx)

    config = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0
    )
    config.max_len = max_len
    config.partial.enabled = True
    config.partial.gate.row_off = gu_off.tolist()
    config.partial.gate.idx = gu_idx.tolist()
    config.partial.up.row_off = up_off.tolist()
    config.partial.up.idx = up_idx.tolist()
    config.partial.down.row_off = dn_off.tolist()
    config.partial.down.idx = dn_idx.tolist()
    config.partial.n_total = intermediate_size
    config.gate_proj = gate_flat.data_ptr()
    config.up_proj = up_flat.data_ptr()
    config.down_proj = down_flat.data_ptr()
    config.gate_scale = config.up_scale = config.down_scale = 0
    config.pool = cpuinfer.backend_

    moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
    phys_map = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpuinfer.submit(moe.load_weights_task(phys_map.data_ptr()))
    cpuinfer.sync()
    # flat 텐서는 pack 완료까지 살아 있어야 한다 (수명 규칙) — 호출자가 잡는다.
    keep = (gate_flat, up_flat, down_flat)
    return (moe, gate_full, up_full, down_full, (gu_off, gu_idx), (up_off, up_idx),
            (dn_off, dn_idx), keep)


def gateup_reference_indexed(x, expert_ids, gate_full, up_full, gu, up=None):
    g_off, g_idx = gu
    u_off, u_idx = up if up is not None else gu
    out = torch.empty(x.shape[0], num_experts_per_tok, 2 * intermediate_size)
    for m in range(x.shape[0]):
        for j in range(num_experts_per_tok):
            e = int(expert_ids[m, j])
            gc = g_idx[g_off[e] : g_off[e + 1]].to(torch.int64)
            uc = u_idx[u_off[e] : u_off[e + 1]].to(torch.int64)
            out[m, j, :intermediate_size] = x[m, gc].float() @ gate_full[e][:, gc].float().t()
            out[m, j, intermediate_size:] = x[m, uc].float() @ up_full[e][:, uc].float().t()
    return out


def down_reference_indexed(act, expert_ids, down_full, dn):
    off, idx = dn
    out = torch.empty(act.shape[0], num_experts_per_tok, hidden_size)
    for m in range(act.shape[0]):
        for j in range(num_experts_per_tok):
            e = int(expert_ids[m, j])
            cols = idx[off[e] : off[e + 1]].to(torch.int64)
            out[m, j] = act[m, j, cols].float() @ down_full[e][:, cols].float().t()
    return out


VARIABLE_GU = [64, 192, 32, 128, 96, 224, 32, 160]   # expert별 행 수 (K_STEP 배수)
VARIABLE_DN = [32, 96, 64, 128, 32, 64, 96, 32]


@pytest.fixture(scope="module")
def cpuinfer():
    return kt_kernel_ext.CPUInfer(CPUINFER_PARAM)


@pytest.mark.parametrize("qlen", QLENS)
def test_indexed_gateup_matches_reference(cpuinfer, qlen):
    """가변 k[e] + 셔플 인덱스 — 밴드로는 표현 불가능한 구성."""
    torch.manual_seed(0)
    moe, g, u, d, gu, up, dn, _keep = build_indexed_moe(VARIABLE_GU, VARIABLE_DN, cpuinfer)
    x = randn_gen((qlen, hidden_size)).contiguous()
    ids = make_routing(qlen)
    out = run_gateup(moe, ids, x)
    ref = gateup_reference_indexed(x, ids, g, u, gu)
    assert rel_diff(out, ref) < 0.02


@pytest.mark.parametrize("qlen", QLENS)
def test_indexed_gateup_bitexact_on_integer_inputs(cpuinfer, qlen):
    """정수 입력에서 비트일치 — 좌표 뒤섞임의 exact 검출기 (계약 ⑤-5).

    밴드에서는 이 테스트가 순서를 검증하지 못한다(어느 순서로 더해도 같은 행).
    셔플 인덱스에서 비로소 검출기가 된다.
    """
    torch.manual_seed(1)
    moe, g, u, d, gu, up, dn, _keep = build_indexed_moe(
        VARIABLE_GU, VARIABLE_DN, cpuinfer, gen=int_gen, seed=7)
    x = int_gen((qlen, hidden_size)).contiguous()
    ids = make_routing(qlen)
    out = run_gateup(moe, ids, x)
    ref = gateup_reference_indexed(x, ids, g, u, gu)
    assert torch.equal(out.float(), ref)


@pytest.mark.parametrize("qlen", QLENS)
def test_indexed_down_bitexact_on_integer_inputs(cpuinfer, qlen):
    torch.manual_seed(2)
    moe, g, u, d, gu, up, dn, _keep = build_indexed_moe(
        VARIABLE_GU, VARIABLE_DN, cpuinfer, gen=int_gen, seed=11)
    act = int_gen((qlen, num_experts_per_tok, intermediate_size)).contiguous()
    ids = make_routing(qlen)
    out = run_down(moe, ids, act)
    ref = down_reference_indexed(act, ids, d, dn)
    assert torch.equal(out.float(), ref)


def test_indexed_contiguous_matches_band(cpuinfer):
    """연속 인덱스(shuffle=False, 균일 길이)는 밴드 경로와 **비트일치**.

    전환기의 합격 기준 — 밴드가 인덱스의 퇴화형이라는 주장의 실행형이다.
    """
    torch.manual_seed(3)
    rows_gu = [GATEUP_K_ROWS] * expert_num
    rows_dn = [DOWN_K_ROWS] * expert_num
    moe_i, g, u, d, gu, up, dn, _keep = build_indexed_moe(
        rows_gu, rows_dn, cpuinfer, gen=int_gen, seed=5, shuffle=False)
    # 밴드 인스턴스: 같은 full weight의 [0, rows) 구간 (shuffle=False가 만든 것과 동일)
    gate_cold = g[:, :, :GATEUP_K_ROWS].contiguous()
    up_cold = u[:, :, :GATEUP_K_ROWS].contiguous()
    down_cold = d[:, :, :DOWN_K_ROWS].contiguous()
    cfg = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
    cfg.max_len = max_len
    cfg.partial.enabled = True
    cfg.partial.gate.offset, cfg.partial.gate.rows = 0, GATEUP_K_ROWS
    cfg.partial.up.offset, cfg.partial.up.rows = 0, GATEUP_K_ROWS
    cfg.partial.down.offset, cfg.partial.down.rows = 0, DOWN_K_ROWS
    cfg.partial.n_total = intermediate_size
    cfg.gate_proj, cfg.up_proj, cfg.down_proj = (
        gate_cold.data_ptr(), up_cold.data_ptr(), down_cold.data_ptr())
    cfg.gate_scale = cfg.up_scale = cfg.down_scale = 0
    cfg.pool = cpuinfer.backend_
    moe_b = kt_kernel_ext.moe.AMXBF16_MOE(cfg)
    phys = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpuinfer.submit(moe_b.load_weights_task(phys.data_ptr()))
    cpuinfer.sync()
    x = int_gen((4, hidden_size)).contiguous()
    ids = make_routing(4)
    assert torch.equal(run_gateup(moe_i, ids, x), run_gateup(moe_b, ids, x))


def test_indexed_rejects_unaligned_rows(cpuinfer):
    """expert별 행 수는 packed 타일(K_STEP)의 배수여야 한다 — 아니면 즉사."""
    with pytest.raises(RuntimeError, match="multiple of K_STEP"):
        build_indexed_moe([64, 48] + [64] * (expert_num - 2), VARIABLE_DN, cpuinfer)


def test_indexed_rejects_out_of_range(cpuinfer):
    """인덱스가 축 밖을 가리키면 즉사 (조용히 남의 열을 읽지 않는다)."""
    off, idx = make_index([64] * expert_num, hidden_size, 0)
    idx[3] = hidden_size + 1
    config = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
    config.max_len = max_len
    config.partial.enabled = True
    config.partial.gate.row_off = off.tolist()
    config.partial.gate.idx = idx.tolist()
    config.partial.up.row_off = off.tolist()
    config.partial.up.idx = idx.tolist()
    config.partial.down.offset = DOWN_K_OFFSET
    config.partial.down.rows = DOWN_K_ROWS
    config.partial.n_total = intermediate_size
    config.pool = cpuinfer.backend_
    with pytest.raises(RuntimeError, match="index out of axis range"):
        kt_kernel_ext.moe.AMXBF16_MOE(config)


# ── dual-pack (gate ≠ up) ───────────────────────────────────────────────────
#
# gate와 up이 다른 인덱스를 가지면 packed activation이 달라져 A 풀이 둘로
# 갈린다. 기하가 같으면 pack을 공유하므로(dual_pack() == false) 위 테스트들은
# 전부 단일 pack 경로를 계속 탄다 — 침습성 상한이 지켜지는지의 확인이기도 하다.

VARIABLE_UP = [96, 64, 128, 32, 160, 32, 224, 64]  # gate와 다른 행 수·순열


@pytest.mark.parametrize("qlen", QLENS)
def test_dual_pack_gateup_matches_reference(cpuinfer, qlen):
    torch.manual_seed(4)
    moe, g, u, d, gu, up, dn, _keep = build_indexed_moe(
        VARIABLE_GU, VARIABLE_DN, cpuinfer, up_rows=VARIABLE_UP)
    x = randn_gen((qlen, hidden_size)).contiguous()
    ids = make_routing(qlen)
    out = run_gateup(moe, ids, x)
    ref = gateup_reference_indexed(x, ids, g, u, gu, up)
    assert rel_diff(out, ref) < 0.02


@pytest.mark.parametrize("qlen", QLENS)
def test_dual_pack_bitexact_on_integer_inputs(cpuinfer, qlen):
    """gate와 up이 서로 다른 열을 읽는다 — 둘이 뒤바뀌면 여기서 걸린다."""
    torch.manual_seed(5)
    moe, g, u, d, gu, up, dn, _keep = build_indexed_moe(
        VARIABLE_GU, VARIABLE_DN, cpuinfer, gen=int_gen, seed=13, up_rows=VARIABLE_UP)
    x = int_gen((qlen, hidden_size)).contiguous()
    ids = make_routing(qlen)
    out = run_gateup(moe, ids, x)
    ref = gateup_reference_indexed(x, ids, g, u, gu, up)
    assert torch.equal(out.float(), ref)


def test_shared_geometry_does_not_dual_pack(cpuinfer):
    """gate == up이면 dual-pack이 아니다 — A 풀 2배는 자유도를 실제로 쓸 때만
    낸다. (config 레벨 판정이므로 인스턴스 생성 없이 확인한다.)"""
    cfg = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
    cfg.partial.enabled = True
    off, idx = make_index(VARIABLE_GU, hidden_size, 0)
    cfg.partial.gate.row_off = off.tolist()
    cfg.partial.gate.idx = idx.tolist()
    cfg.partial.up.row_off = off.tolist()
    cfg.partial.up.idx = idx.tolist()
    assert not cfg.dual_pack()
    up_off, up_idx = make_index(VARIABLE_UP, hidden_size, 2)
    cfg.partial.up.row_off = up_off.tolist()
    cfg.partial.up.idx = up_idx.tolist()
    assert cfg.dual_pack()


# ── 타일 경계 패딩 ─────────────────────────────────────────────────────────
#
# plan/자산이 지키는 정렬은 페어(%2)뿐인데 packed 저장은 K_STEP(32)의 배수를
# 요구한다. 로더가 올림하고 0 행을 채우며, kt는 real_rows로 실제 행 수를 안다.

def test_padded_rows_are_numerically_harmless(cpuinfer):
    """k_real이 타일 배수가 아니어도 결과는 실제 행만 쓴 것과 같다.

    패딩 행의 weight가 0이라 dense 경로에서 곱이 0이다 — 인덱스가 아무 열을
    가리켜도 무해하다는 것이 이 설계의 전제이고, 여기서 그것을 고정한다.
    """
    torch.manual_seed(6)
    real = [96, 66, 34, 128, 98, 64, 30, 96]          # 전부 %2, 대부분 %32 아님
    pad = [(r + K_STEP - 1) // K_STEP * K_STEP for r in real]
    assert pad != real                                  # 실제로 패딩이 있는 구성

    gate_full = int_gen((expert_num, intermediate_size, hidden_size))
    up_full = int_gen((expert_num, intermediate_size, hidden_size))
    down_full = int_gen((expert_num, hidden_size, intermediate_size))
    g = torch.Generator().manual_seed(21)
    row_off, idx, real_cols, acc = [0], [], [], 0
    for e in range(expert_num):
        cols = torch.randperm(hidden_size, generator=g)[: real[e]]
        real_cols.append(cols)
        # 패딩 열은 아무 값이나 가리켜도 된다 (weight가 0) — 0으로 채운다.
        idx.append(torch.cat([cols, torch.zeros(pad[e] - real[e], dtype=torch.long)]))
        acc += pad[e]
        row_off.append(acc)
    row_off = torch.tensor(row_off, dtype=torch.int32)
    idx_flat = torch.cat(idx)

    def flat_padded(full):
        blocks = []
        for e in range(expert_num):
            blk = torch.zeros(intermediate_size if full is not down_full else hidden_size,
                              pad[e], dtype=torch.bfloat16)
            blk[:, : real[e]] = full[e][:, real_cols[e]]   # 나머지는 0 (패딩)
            blocks.append(blk.contiguous().reshape(-1))
        return torch.cat(blocks).contiguous()

    dn_off, dn_idx = make_index([DOWN_K_ROWS] * expert_num, intermediate_size, 31)

    def flat_down():
        blocks = []
        for e in range(expert_num):
            cols = dn_idx[dn_off[e] : dn_off[e + 1]].to(torch.int64)
            blocks.append(down_full[e][:, cols].contiguous().reshape(-1))
        return torch.cat(blocks).contiguous()

    gate_flat, up_flat, down_flat = flat_padded(gate_full), flat_padded(up_full), flat_down()

    cfg = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
    cfg.max_len = max_len
    cfg.partial.enabled = True
    for side in (cfg.partial.gate, cfg.partial.up):
        side.row_off = row_off.tolist()
        side.idx = idx_flat.to(torch.int32).tolist()
        side.real_rows = real
    cfg.partial.down.row_off = dn_off.tolist()
    cfg.partial.down.idx = dn_idx.tolist()
    cfg.partial.n_total = intermediate_size
    cfg.gate_proj, cfg.up_proj, cfg.down_proj = (
        gate_flat.data_ptr(), up_flat.data_ptr(), down_flat.data_ptr())
    cfg.gate_scale = cfg.up_scale = cfg.down_scale = 0
    cfg.pool = cpuinfer.backend_
    moe = kt_kernel_ext.moe.AMXBF16_MOE(cfg)
    phys = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpuinfer.submit(moe.load_weights_task(phys.data_ptr()))
    cpuinfer.sync()

    x = int_gen((4, hidden_size)).contiguous()
    ids = make_routing(4)
    out = run_gateup(moe, ids, x)
    ref = torch.empty(4, num_experts_per_tok, 2 * intermediate_size)
    for m in range(4):
        for j in range(num_experts_per_tok):
            e = int(ids[m, j])
            cols = real_cols[e]
            ref[m, j, :intermediate_size] = x[m, cols].float() @ gate_full[e][:, cols].float().t()
            ref[m, j, intermediate_size:] = x[m, cols].float() @ up_full[e][:, cols].float().t()
    assert torch.equal(out.float(), ref)


def test_real_rows_beyond_one_tile_rejected(cpuinfer):
    """real_rows는 패딩 전 값이므로 padded와 한 타일 이내로 붙어 있어야 한다.
    (더 벌어졌다면 로더가 올림을 잘못한 것이고, 조용히 행을 버리게 된다.)"""
    off, idx = make_index([64] * expert_num, hidden_size, 0)
    cfg = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
    cfg.max_len = max_len
    cfg.partial.enabled = True
    for side in (cfg.partial.gate, cfg.partial.up):
        side.row_off = off.tolist()
        side.idx = idx.tolist()
        side.real_rows = [16] * expert_num          # 64와 두 타일 차이
    cfg.partial.down.offset, cfg.partial.down.rows = DOWN_K_OFFSET, DOWN_K_ROWS
    cfg.partial.n_total = intermediate_size
    cfg.pool = cpuinfer.backend_
    with pytest.raises(RuntimeError, match="real_rows"):
        kt_kernel_ext.moe.AMXBF16_MOE(cfg)
