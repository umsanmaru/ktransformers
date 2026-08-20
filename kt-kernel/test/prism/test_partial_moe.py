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
    config.partial.gateup.offset = GATEUP_K_OFFSET
    config.partial.gateup.rows = GATEUP_K_ROWS
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


def run_gateup(moe, expert_ids, x):
    qlen_t = torch.tensor([x.shape[0]], dtype=torch.int32)
    out = torch.full(
        (x.shape[0], num_experts_per_tok, 2 * intermediate_size), 777.0, dtype=torch.bfloat16
    ).contiguous()
    moe.forward_gateup_partial(
        qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), x.data_ptr(), out.data_ptr()
    )
    return out


def run_down(moe, expert_ids, act):
    qlen_t = torch.tensor([act.shape[0]], dtype=torch.int32)
    out = torch.full(
        (act.shape[0], num_experts_per_tok, hidden_size), 777.0, dtype=torch.bfloat16
    ).contiguous()
    moe.forward_down_partial(
        qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), act.data_ptr(), out.data_ptr()
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
