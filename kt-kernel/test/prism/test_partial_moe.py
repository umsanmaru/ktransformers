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


def run_gateup(moe, expert_ids, x, thr=None):
    qlen_t = torch.tensor([x.shape[0]], dtype=torch.int32)
    out = torch.full(
        (x.shape[0], num_experts_per_tok, 2 * intermediate_size), 777.0, dtype=torch.bfloat16
    ).contiguous()
    moe.forward_gateup_partial(
        qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), x.data_ptr(), out.data_ptr(),
        0 if thr is None else thr.data_ptr()
    )
    return out


def run_down(moe, expert_ids, act, thr=None):
    qlen_t = torch.tensor([act.shape[0]], dtype=torch.int32)
    out = torch.full(
        (act.shape[0], num_experts_per_tok, hidden_size), 777.0, dtype=torch.bfloat16
    ).contiguous()
    moe.forward_down_partial(
        qlen_t.data_ptr(), num_experts_per_tok, expert_ids.data_ptr(), act.data_ptr(), out.data_ptr(),
        0 if thr is None else thr.data_ptr()
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
    config.partial.gateup.offset = GATEUP_K_OFFSET
    config.partial.gateup.rows = GATEUP_K_ROWS
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
# 입력기반 sparsity (k2wl2) — 계약 ① / sglang prism sparsity.py의 CPU 짝
# ===========================================================================
#
# 검증 전략: sparse 커널은 dense 커널과 **누산 순서가 같도록** 만들었고,
# 마스킹된 페어는 dense에서 a=0이 되어 c + 0*b = c (정확)이다. 따라서
#
#     sparse(x, mask)  ==  dense(mask 적용된 x)      (비트 단위)
#
# 가 성립해야 한다. tolerance가 아니라 비트일치로 검증할 수 있는 이유다.


def sparsity_tables_from_weights(gate_cold, up_cold, down_cold):
    """cold weight에서 점수 재료를 유도한다 (calib과 같은 정의).

    wn_sq[j] = ||W[:, j]||²  (출력 행에 대한 노름의 제곱 = a)
    pair_dot[p] = <W[:, 2p], W[:, 2p+1]>                    (= c)
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
    return {
        "gate_wn_sq": g_a, "gate_pair_dot": g_c,
        "up_wn_sq": u_a, "up_pair_dot": u_c,
        "down_wn_sq": d_a, "down_pair_dot": d_c,
    }


def install_sparsity(config, tables):
    sp = config.partial.sparsity
    for name, t in tables.items():
        setattr(sp, name, t.data_ptr())
    sp.enabled = True


def build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables):
    """build_partial_moe와 같지만 sparsity 테이블을 설치한다.

    테이블은 C++가 step마다 원시 포인터로 읽으므로 호출자가 붙잡고 있어야
    한다 — 여기서는 tables dict가 테스트 스코프에 살아 있다.
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
    config.gate_proj = gate_cold.data_ptr()
    config.up_proj = up_cold.data_ptr()
    config.down_proj = down_cold.data_ptr()
    config.gate_scale = 0
    config.up_scale = 0
    config.down_scale = 0
    config.pool = cpuinfer.backend_
    install_sparsity(config, tables)

    moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
    phys_map = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpuinfer.submit(moe.load_weights_task(phys_map.data_ptr()))
    cpuinfer.sync()
    return moe


def make_thr(qlen, gate=0.0, up=0.0, down=0.0):
    """[qlen, k, 3] fp32 — 순서는 (gate, up, down)."""
    t = torch.zeros(qlen, num_experts_per_tok, 3, dtype=torch.float32)
    t[:, :, 0], t[:, :, 1], t[:, :, 2] = gate, up, down
    return t.contiguous()


def pair_energy(vec_band, a_sq, c):
    """C++ build_pair_mask의 e (clamp 전 제곱 에너지). fp32로 계산."""
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

    파이썬 쪽 마스크도 C++와 똑같이 `thr*thr`로 비교하므로, sqrt 왕복 오차가
    있어도 두 판정은 정확히 일치한다.
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
    thr = make_thr(1)

    assert torch.equal(run_gateup(sparse, ids, x, thr), run_gateup(dense, ids, x))
    assert torch.equal(run_down(sparse, ids, act, thr), run_down(dense, ids, act))


@pytest.mark.cpu
def test_sparse_huge_threshold_zeroes_output():
    """전량 차단 → 타일 마스크가 0이라 B 로드 없이 0이 export된다."""
    torch.manual_seed(21)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)

    ids = make_routing(1)
    x = randn_gen((1, hidden_size)).contiguous()
    act = randn_gen((1, num_experts_per_tok, intermediate_size)).contiguous()
    thr = make_thr(1, gate=1e9, up=1e9, down=1e9)

    assert torch.count_nonzero(run_gateup(sparse, ids, x, thr)) == 0
    assert torch.count_nonzero(run_down(sparse, ids, act, thr)) == 0


@pytest.mark.cpu
def test_sparse_gateup_matches_masked_dense():
    """중간 threshold: sparse == dense(마스킹된 x), 비트 단위.

    gate와 up은 노름·내적이 달라 마스크도 다르므로, dense 레퍼런스를 각
    마스크로 두 번 돌려 해당 절반씩 취한다 — 이 비대칭이 "A 한 벌을 공유
    하면서 마스크는 proj별"이라는 설계가 실제로 지켜지는지 검증한다.
    """
    torch.manual_seed(22)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    dense = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)

    ids = make_routing(1)
    x = randn_gen((1, hidden_size)).contiguous()
    band = x[0, GATEUP_K_OFFSET : GATEUP_K_OFFSET + GATEUP_K_ROWS]

    experts = [int(ids[0, j]) for j in range(num_experts_per_tok)]
    e_gate = [pair_energy(band, tables["gate_wn_sq"][e], tables["gate_pair_dot"][e])
              for e in experts]
    e_up = [pair_energy(band, tables["up_wn_sq"][e], tables["up_pair_dot"][e])
            for e in experts]
    thr_g = pick_threshold(e_gate)
    thr_u = pick_threshold(e_up)

    out = run_gateup(sparse, ids, x, make_thr(1, gate=thr_g, up=thr_u))

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
    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    dense = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)

    ids = make_routing(1)
    act = randn_gen((1, num_experts_per_tok, intermediate_size)).contiguous()
    experts = [int(ids[0, j]) for j in range(num_experts_per_tok)]
    energies = [
        pair_energy(act[0, j, DOWN_K_OFFSET : DOWN_K_OFFSET + DOWN_K_ROWS],
                    tables["down_wn_sq"][e], tables["down_pair_dot"][e])
        for j, e in enumerate(experts)
    ]
    thr_d = pick_threshold(energies)
    out = run_down(sparse, ids, act, make_thr(1, down=thr_d))

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
def test_sparse_rejects_prefill():
    """sparsity는 decode 전용 (계약 ①) — qlen>1에 thr을 주면 즉사."""
    torch.manual_seed(24)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    tables = sparsity_tables_from_weights(gate_cold, up_cold, down_cold)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    sparse = build_sparse_moe(gate_cold, up_cold, down_cold, cpuinfer, tables)

    ids = make_routing(4)
    x = randn_gen((4, hidden_size)).contiguous()
    with pytest.raises(RuntimeError, match="decode-only"):
        run_gateup(sparse, ids, x, make_thr(4, gate=1.0))


@pytest.mark.cpu
def test_sparse_rejects_thr_without_tables():
    """sparsity를 끈 인스턴스에 thr을 주면 조용히 dense로 흐르지 않고 즉사."""
    torch.manual_seed(25)
    gate_cold, up_cold, down_cold = make_weights(randn_gen)
    cpuinfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    dense = build_partial_moe(gate_cold, up_cold, down_cold, cpuinfer)

    ids = make_routing(1)
    x = randn_gen((1, hidden_size)).contiguous()
    with pytest.raises(RuntimeError, match="sparsity.enabled is false"):
        run_gateup(dense, ids, x, make_thr(1, gate=1.0))
