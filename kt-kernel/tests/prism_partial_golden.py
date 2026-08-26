#!/usr/bin/env python
"""Prism partial(gateup/down) 출력의 골든 덤프/비교 — 리팩터링의 비트 동일성 검사.

CUDA가 없는 머신에서도 돈다. 랜덤 가중치·활성·점수 테이블·threshold 곡선으로
`PartialMoEWrapper.forward_gateup / forward_down`을 여러 라우팅에 대해 호출하고
결과를 저장한다. 두 빌드의 덤프를 `compare`로 맞춰 보면 연산 순서를 바꾸지 않은
리팩터링(예: 준비 단계의 워커 병렬화)이 출력을 한 비트도 바꾸지 않았음을 확인할
수 있다.

    python tests/prism_partial_golden.py dump  /tmp/before.pt
    (rebuild)
    python tests/prism_partial_golden.py dump  /tmp/after.pt
    python tests/prism_partial_golden.py compare /tmp/before.pt /tmp/after.pt

덮는 축: 커널 {kt_tile_k2_bf16, kt_amx_bf16} × K 인덱스 {band, indexed, split(gate≠up →
dual_pack)} × {dense(weights=None), sparse(weights)} × {gateup, down}.
"""
from __future__ import annotations

import sys

import torch

from kt_kernel import kt_kernel_ext
from kt_kernel.experts_partial import PartialMoEWrapper

E, TOPK, H, I = 16, 4, 512, 256
K_STEP = 32
NG, GRID, PMAX, RENORM_IT = 201, 0.005, 0.9, 3
P, LAM = 0.5, 4.0
STEPS = 6


def _index(axis: int, rows: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(axis, generator=g)[:rows].sort().values.to(torch.int32).tolist()


def build(kernel_key: str, mode: str, seed: int, threads: int = 4):
    torch.manual_seed(seed)
    # subpool 1개(node 0) 고정 — 노드 shard 테이블 길이 = subpool 수여야 한다.
    cfg_pool = kt_kernel_ext.WorkerPoolConfig()
    cfg_pool.subpool_count = 1
    cfg_pool.subpool_numa_map = [0]
    cfg_pool.subpool_thread_count = [threads]
    cpuinfer = kt_kernel_ext.CPUInfer(cfg_pool)
    cfg = kt_kernel_ext.moe.MOEConfig(E, TOPK, H, I, 0)
    cfg.max_len = 1
    cfg.layer_idx = 0
    cfg.partial.enabled = True
    cfg.partial.n_total = I

    rows = {}
    for proj, ki, axis in (("gate", cfg.partial.gate, H), ("up", cfg.partial.up, H),
                           ("down", cfg.partial.down, I)):
        kc = (axis * 3 // 4) // K_STEP * K_STEP  # cold 3/4, 타일 정렬
        rows[proj] = kc
        if mode == "band":
            ki.offset = axis - kc
            ki.rows = kc
        else:
            ki.row_off = [e * kc for e in range(E + 1)]
            salt = 7919 if (mode == "split" and proj == "up") else 0
            idx = []
            for e in range(E):  # expert마다 다른 인덱스 (가변 per-expert 계약)
                idx += _index(axis, kc, seed * 1000 + e + salt)
            ki.idx = idx

    cfg.partial.node_gateup_n_offset = [0]
    cfg.partial.node_gateup_n_rows = [I]
    cfg.partial.node_down_n_offset = [0]
    cfg.partial.node_down_n_rows = [H]

    sp = cfg.partial.sparsity
    sp.pmax, sp.grid, sp.ng, sp.renorm_it = PMAX, GRID, NG, RENORM_IT
    sp.p_gate, sp.lam_gate = P, LAM
    sp.p_up, sp.lam_up = P, LAM
    sp.p_down, sp.lam_down = P, LAM
    cfg.pool = cpuinfer.backend_

    n_cols = {"gate": I, "up": I, "down": H}
    weights, tables = {}, {}
    for proj in ("gate", "up", "down"):
        weights[proj] = torch.randn(E * n_cols[proj] * rows[proj]).mul_(0.02).to(torch.bfloat16).contiguous()
        npairs = rows[proj] // 2
        tables[f"{proj}_wn_sq"] = torch.rand(E * rows[proj], dtype=torch.float32).contiguous()
        tables[f"{proj}_pair_dot"] = torch.randn(E * npairs, dtype=torch.float32).mul_(0.1).contiguous()
        # expert마다 다른 곡선, 격자 위 단조 증가 — 중간 sparsity가 나오게
        base = torch.rand(E, 1) * 0.4
        tables[f"thr_{proj}"] = (base + torch.linspace(0, 0.6, NG).unsqueeze(0)).contiguous()

    wrapper = PartialMoEWrapper(cfg, cpuinfer, kernel_key=kernel_key)
    wrapper.load_weights_from_tensors(weights["gate"], weights["up"], weights["down"],
                                      sparsity_tables=tables)
    return wrapper, cpuinfer


def run_case(kernel_key: str, mode: str, sparse: bool, seed: int) -> list[torch.Tensor]:
    wrapper, cpuinfer = build(kernel_key, mode, seed)
    g = torch.Generator().manual_seed(seed + 17)
    outs = []
    for _ in range(STEPS):
        ids = torch.randperm(E, generator=g)[:TOPK].to(torch.int64).reshape(1, TOPK).contiguous()
        w = torch.softmax(torch.randn(1, TOPK, generator=g), dim=-1).contiguous() if sparse else None
        x = torch.randn(1, H, generator=g).to(torch.bfloat16)
        out_gu = torch.zeros(1, TOPK, 2 * I, dtype=torch.bfloat16)
        wrapper.forward_gateup(ids, x, out_gu, w)
        act = torch.randn(1, TOPK, I, generator=g).to(torch.bfloat16)
        out_dn = torch.zeros(1, TOPK, H, dtype=torch.bfloat16)
        wrapper.forward_down(ids, act, out_dn, w)
        outs += [out_gu.clone(), out_dn.clone()]
    del wrapper
    del cpuinfer
    return outs


def cases():
    for kernel_key in ("kt_tile_k2_bf16", "kt_amx_bf16"):
        for mode in ("band", "indexed", "split"):
            for sparse in (False, True):
                yield kernel_key, mode, sparse


def dump(path: str) -> None:
    blob = {}
    for n, (kk, mode, sparse) in enumerate(cases()):
        outs = run_case(kk, mode, sparse, seed=100 + n)
        nz = sum(int(torch.count_nonzero(t)) for t in outs)
        print(f"{kk:16} {mode:8} sparse={sparse!s:5} tensors={len(outs)} nonzero={nz}")
        blob[(kk, mode, sparse)] = outs
    torch.save(blob, path)
    print("->", path)


def compare(a: str, b: str) -> None:
    A, B = torch.load(a), torch.load(b)
    assert A.keys() == B.keys(), "case sets differ"
    bad = 0
    for key in A:
        for i, (ta, tb) in enumerate(zip(A[key], B[key])):
            if not torch.equal(ta, tb):
                bad += 1
                d = (ta.float() - tb.float()).abs().max().item()
                print(f"MISMATCH {key} tensor#{i} ({'gateup' if i % 2 == 0 else 'down'}) max|diff|={d}")
    print("identical" if bad == 0 else f"{bad} tensors differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
