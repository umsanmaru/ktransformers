#!/usr/bin/env python
# coding=utf-8
"""Prism forward_gateup_partial (K1) 정확성 테스트 — AMX BF16, decode(qlen=1).

계약 (sglang prism CONTRACTS.md ②/⑤):
- 입력 x는 full-width [1, hidden]; 인스턴스는 로드 때 구워진 밴드
  [k_off, k_off+k_rows) 만 읽는다.
- out은 bf16 [1, k, 2*inter], 열 [0,inter)=gate / [inter,2inter)=up.
  pre-activation 순수 GEMM 부분합 (act/가중/합산 없음, overwrite).
- 수치: 내부 누산 fp32(AMX 타일), 재료화 bf16 → 일반 입력은 tolerance,
  bf16-무손실 정수 입력은 비트일치 (계약 ⑤-5).
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
GATEUP_K_ROWS = 192  # cold 밴드 = hidden rows [64, 256)


def build_partial_moe(gate_cold, up_cold, down_full, cpuinfer):
    """K-슬라이스 텐서 주입으로 partial 인스턴스 생성 (kt SFT 주입 경로와 동형)."""
    config = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0
    )
    config.max_len = max_len
    # enabled=True면 모든 range가 명시값 (센티널 없음) — down full도 명시
    config.partial.enabled = True
    config.partial.gateup.offset = GATEUP_K_OFFSET
    config.partial.gateup.rows = GATEUP_K_ROWS
    config.partial.down.offset = 0
    config.partial.down.rows = intermediate_size  # down은 full (K2 전까지)
    config.partial.n_total = intermediate_size  # gate/up 출력축 full 크기
    config.gate_proj = gate_cold.data_ptr()
    config.up_proj = up_cold.data_ptr()
    config.down_proj = down_full.data_ptr()
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
    down_full = generator((expert_num, hidden_size, intermediate_size)).contiguous()
    gate_cold = gate_full[:, :, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS].contiguous()
    up_cold = up_full[:, :, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS].contiguous()
    return gate_cold, up_cold, down_full


@pytest.mark.cpu
def test_gateup_partial_matches_reference():
    torch.manual_seed(0)
    gen = lambda s: (torch.randn(s, dtype=torch.float32) / 10.0).to(torch.bfloat16)
    gate_cold, up_cold, down_full = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_full, cpuinfer)

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
    gate_cold, up_cold, down_full = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_full, cpuinfer)

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
    gate_cold, up_cold, down_full = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_full, cpuinfer)

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
    gate_cold, up_cold, down_full = make_weights(gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    moe = build_partial_moe(gate_cold, up_cold, down_full, cpuinfer)

    expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].view(1, -1).contiguous()
    x = torch.randn((2, hidden_size), dtype=torch.float32).to(torch.bfloat16).contiguous()
    out = torch.empty((2, num_experts_per_tok, 2 * intermediate_size), dtype=torch.bfloat16).contiguous()
    qlen = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="qlen==1"):
        moe.forward_gateup_partial(
            qlen.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), x.data_ptr(), out.data_ptr()
        )


if __name__ == "__main__":
    test_gateup_partial_matches_reference()
    test_gateup_partial_bitexact_on_integer_inputs()
    test_fused_forward_rejected_on_partial_instance()
    print("ALL PASSED")
