#!/usr/bin/env python
# coding=utf-8
"""Prism partial 진입점 정확성 테스트 — AMX BF16, decode(qlen=1).

대상 (계약: sglang prism CONTRACTS.md ②/⑤):
- forward_gateup_partial (K1): full-width x에서 구워진 K 밴드만 읽어
  pre-act partial [1, k, 2*inter] export ([0,inter)=gate, [inter,2inter)=up).
- forward_down_partial (K2): full-width act의 slot별 행에서 down K 밴드를
  읽어 [1, k, hidden] partial export. N(hidden)축 노드 분할 — 합산 없음.
- 합성: gateup → (torch act) → down 체인.
- NUMA shard: 균등 기본값 + Plan 주입 불균등 테이블.
- 수치: 내부 누산 fp32(AMX 타일), 재료화 bf16 → 일반 입력은 tolerance,
  bf16-무손실 정수 입력은 비트일치 (계약 ⑤-5).
- guard: fused forward 거부, qlen>1 거부 (호출 스레드에서 깨끗한 예외).
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


def build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer, shard_tables=None):
    """K-슬라이스 텐서 주입으로 partial 인스턴스 생성 (kt SFT 주입 경로와 동형).

    shard_tables: (gu_off, gu_rows, dn_off, dn_rows) — 불균등 NUMA 분할 테이블.
    None이면 TP ctor의 균등 분할.
    """
    config = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0
    )
    config.max_len = max_len
    # enabled=True면 모든 range가 명시값 (센티널 없음)
    config.partial.enabled = True
    config.partial.gateup.offset = GATEUP_K_OFFSET
    config.partial.gateup.rows = GATEUP_K_ROWS
    config.partial.down.offset = DOWN_K_OFFSET
    config.partial.down.rows = DOWN_K_ROWS
    config.partial.n_total = intermediate_size  # gate/up 출력축 full 크기
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


def run_partial(moe, expert_ids, x):
    qlen = torch.tensor([1], dtype=torch.int32)
    out = torch.full(
        (1, num_experts_per_tok, 2 * intermediate_size), 777.0, dtype=torch.bfloat16
    ).contiguous()
    moe.forward_gateup_partial(
        qlen.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), x.data_ptr(), out.data_ptr()
    )
    return out


def reference(x, expert_ids, gate_cold, up_cold):
    """fp32 레퍼런스: 밴드 x 슬라이스 × 밴드 weight (pre-act partial)."""
    xs = x[:, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS].float()
    outs = []
    for j in range(num_experts_per_tok):
        e = int(expert_ids[0, j])
        gate = xs @ gate_cold[e].float().t()  # [1, inter]
        up = xs @ up_cold[e].float().t()
        outs.append(torch.cat([gate, up], dim=1))
    return torch.stack(outs, dim=1)  # [1, k, 2*inter]


def make_weights(generator):
    gate_full = generator((expert_num, intermediate_size, hidden_size))
    up_full = generator((expert_num, intermediate_size, hidden_size))
    down_full = generator((expert_num, hidden_size, intermediate_size))
    gate_cold = gate_full[:, :, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS].contiguous()
    up_cold = up_full[:, :, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS].contiguous()
    down_cold = down_full[:, :, DOWN_K_OFFSET : DOWN_K_OFFSET + DOWN_K_ROWS].contiguous()
    return gate_cold, up_cold, down_cold


def run_down_partial(moe, expert_ids, act):
    qlen = torch.tensor([1], dtype=torch.int32)
    out = torch.full(
        (1, num_experts_per_tok, hidden_size), 777.0, dtype=torch.bfloat16
    ).contiguous()
    moe.forward_down_partial(
        qlen.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), act.data_ptr(), out.data_ptr()
    )
    return out


def down_reference(act, expert_ids, down_cold):
    """fp32 레퍼런스: slot별 act 행의 down K 밴드 × down cold weight."""
    outs = []
    for j in range(num_experts_per_tok):
        e = int(expert_ids[0, j])
        a = act[0, j, DOWN_K_OFFSET : DOWN_K_OFFSET + DOWN_K_ROWS].float().unsqueeze(0)
        outs.append(a @ down_cold[e].float().t())  # [1, hidden]
    return torch.stack(outs, dim=1)  # [1, k, hidden]


@pytest.mark.cpu
def test_gateup_partial_matches_reference():
    torch.manual_seed(0)
    gen = lambda s: (torch.randn(s, dtype=torch.float32) / 10.0).to(torch.bfloat16)
    gate_cold, up_cold, down_cold = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    for i in range(3):
        expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
        x = (torch.randn((1, hidden_size), dtype=torch.float32) / 10.0).to(torch.bfloat16).contiguous()
        out = run_partial(moe, expert_ids, x)
        ref = reference(x, expert_ids, gate_cold, up_cold)
        diff = torch.mean(torch.abs(out.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)
        print(f"iter {i}: rel diff = {diff:.6f}")
        assert diff < 0.02, f"gateup partial diff {diff:.6f} >= 0.02"


@pytest.mark.cpu
def test_gateup_partial_bitexact_on_integer_inputs():
    """계약 ⑤-5: bf16 무손실 입력(작은 정수)에서 비트일치 — 이중계산/누락의
    exact 검출기. 합의 상한 |sum| <= 192 < 256 이라 bf16에서 정수 무손실."""
    torch.manual_seed(1)
    gen = lambda s: torch.randint(-1, 2, s, dtype=torch.float32).to(torch.bfloat16)
    gate_cold, up_cold, down_cold = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
    x = torch.randint(-1, 2, (1, hidden_size), dtype=torch.float32).to(torch.bfloat16).contiguous()
    out = run_partial(moe, expert_ids, x)
    ref = reference(x, expert_ids, gate_cold, up_cold).to(torch.bfloat16)
    assert torch.equal(out, ref), "integer-input partial must be bit-exact"


@pytest.mark.cpu
def test_fused_forward_rejected_on_partial_instance():
    """partial 인스턴스에 fused forward를 부르면 즉사해야 한다 (조용한 오답 방지)."""
    torch.manual_seed(2)
    gen = lambda s: (torch.randn(s, dtype=torch.float32) / 10.0).to(torch.bfloat16)
    gate_cold, up_cold, down_cold = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
    weights = torch.rand((1, num_experts_per_tok), dtype=torch.float32).contiguous()
    x = torch.randn((1, hidden_size), dtype=torch.float32).to(torch.bfloat16).contiguous()
    output = torch.empty((1, hidden_size), dtype=torch.bfloat16).contiguous()
    qlen = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="partial"):
        moe.forward(
            qlen.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(),
            weights.data_ptr(), x.data_ptr(), output.data_ptr(), False,
        )


@pytest.mark.cpu
def test_qlen_gt_one_rejected_cleanly():
    """qlen>1은 K1 미지원 — abort(terminate)가 아니라 예외로 거부돼야 한다
    (검증이 worker가 아닌 호출 스레드에 있는지의 회귀 테스트)."""
    torch.manual_seed(3)
    gen = lambda s: (torch.randn(s, dtype=torch.float32) / 10.0).to(torch.bfloat16)
    gate_cold, up_cold, down_cold = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
    x = torch.randn((2, hidden_size), dtype=torch.float32).to(torch.bfloat16).contiguous()
    out = torch.empty((2, num_experts_per_tok, 2 * intermediate_size), dtype=torch.bfloat16).contiguous()
    qlen = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="qlen==1"):
        moe.forward_gateup_partial(
            qlen.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), x.data_ptr(), out.data_ptr()
        )


@pytest.mark.cpu
def test_down_partial_matches_reference():
    """K2: down partial — act full-width에서 K 밴드 소비, hidden 전폭 partial 출력."""
    torch.manual_seed(4)
    gen = lambda s: (torch.randn(s, dtype=torch.float32) / 10.0).to(torch.bfloat16)
    gate_cold, up_cold, down_cold = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    for i in range(3):
        expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
        act = (torch.randn((1, num_experts_per_tok, intermediate_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        ).contiguous()
        out = run_down_partial(moe, expert_ids, act)
        ref = down_reference(act, expert_ids, down_cold)
        diff = torch.mean(torch.abs(out.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)
        print(f"down iter {i}: rel diff = {diff:.6f}")
        assert diff < 0.02, f"down partial diff {diff:.6f} >= 0.02"


@pytest.mark.cpu
def test_down_partial_bitexact_on_integer_inputs():
    torch.manual_seed(5)
    gen = lambda s: torch.randint(-1, 2, s, dtype=torch.float32).to(torch.bfloat16)
    gate_cold, up_cold, down_cold = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
    act = torch.randint(-1, 2, (1, num_experts_per_tok, intermediate_size), dtype=torch.float32).to(
        torch.bfloat16
    ).contiguous()
    out = run_down_partial(moe, expert_ids, act)
    ref = down_reference(act, expert_ids, down_cold).to(torch.bfloat16)
    assert torch.equal(out, ref), "integer-input down partial must be bit-exact"


@pytest.mark.cpu
def test_gateup_act_down_chain():
    """두 진입점의 합성 검증: gateup partial → (torch에서 act) → down partial
    == torch로 계산한 동일 체인. act 행이 slot 좌표로 올바르게 소비되는지 확인."""
    torch.manual_seed(6)
    gen = lambda s: (torch.randn(s, dtype=torch.float32) / 10.0).to(torch.bfloat16)
    gate_cold, up_cold, down_cold = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
    x = (torch.randn((1, hidden_size), dtype=torch.float32) / 10.0).to(torch.bfloat16).contiguous()

    gu = run_partial(moe, expert_ids, x)  # [1, k, 2*inter] bf16
    gate, up = gu.float().split(intermediate_size, dim=2)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).contiguous()  # rejoin#1 상당 (cold-only)
    out = run_down_partial(moe, expert_ids, act)

    ref_gu = reference(x, expert_ids, gate_cold, up_cold)
    rg, ru = ref_gu.split(intermediate_size, dim=2)
    ref_act = (torch.nn.functional.silu(rg) * ru).to(torch.bfloat16)
    ref = down_reference(ref_act, expert_ids, down_cold)

    diff = torch.mean(torch.abs(out.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)
    print(f"chain rel diff = {diff:.6f}")
    assert diff < 0.03, f"chain diff {diff:.6f} >= 0.03"


@pytest.mark.cpu
def test_unequal_numa_shard_tables():
    """K2: Plan 주입 불균등 분할 — inter 96/32, hidden 192/64. 레퍼런스는
    분할 방식을 모르므로, 통과하면 스테이징 offset·export offset이 전부 옳다."""
    torch.manual_seed(7)
    gen = lambda s: (torch.randn(s, dtype=torch.float32) / 10.0).to(torch.bfloat16)
    gate_cold, up_cold, down_cold = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    tables = ([0, 96], [96, 32], [0, 192], [192, 64])
    try:
        moe = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer, shard_tables=tables)
    except RuntimeError as err:
        if "tp_count" in str(err):
            pytest.skip("requires exactly 2 NUMA nodes")
        raise

    expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
    x = (torch.randn((1, hidden_size), dtype=torch.float32) / 10.0).to(torch.bfloat16).contiguous()
    gu = run_partial(moe, expert_ids, x)
    ref_gu = reference(x, expert_ids, gate_cold, up_cold)
    diff = torch.mean(torch.abs(gu.float() - ref_gu)) / (torch.mean(torch.abs(ref_gu)) + 1e-8)
    assert diff < 0.02, f"unequal gateup diff {diff:.6f}"

    act = (torch.randn((1, num_experts_per_tok, intermediate_size), dtype=torch.float32) / 10.0).to(
        torch.bfloat16
    ).contiguous()
    out = run_down_partial(moe, expert_ids, act)
    ref = down_reference(act, expert_ids, down_cold)
    diff = torch.mean(torch.abs(out.float() - ref)) / (torch.mean(torch.abs(ref)) + 1e-8)
    assert diff < 0.02, f"unequal down diff {diff:.6f}"


if __name__ == "__main__":
    test_gateup_partial_matches_reference()
    test_gateup_partial_bitexact_on_integer_inputs()
    test_fused_forward_rejected_on_partial_instance()
    print("ALL PASSED")
