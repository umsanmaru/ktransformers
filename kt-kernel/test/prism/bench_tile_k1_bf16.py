#!/usr/bin/env python
# coding=utf-8
"""bf16 k1 타일 vs k2 타일 — dense 패리티와 **per-k sp50의 granularity 이득** 실측.

cpu-mm `docs/FINDINGS.md` §11이 예측하는 것 둘을 kt 슬랩(256-n)·실 shape에서 확인한다:

  dense        k2/k1 ≈ 1.00   (0.995x — 메모리 바운드라 명령 수가 거의 무료)
  per-k sp50   k2/k1 ≈ 1.50   (페어 union의 keep 0.75 vs per-k 0.5 = obligated bytes 비)

**왜 이 비교가 공정한가.** sparsity가 per-k로 정해지면 페어 커널은 페어의 한쪽만 살아도
그 페어의 라인을 읽어야 한다 — keep이 `1-s²/(1-s)`배로 늘어난다. 그래서 k2 쪽은 union
keep(0.75)으로, k1 쪽은 per-k keep(0.5)으로 재는 것이 같은 마스크를 두 커널에 준 것이다.

캐시 상주를 피하려고 **슬롯을 회전**한다 (E개 weight를 돌아가며 계산 — hwpf-sockets 메모의
"cold는 회전 필요"와 같은 이유). 스레드는 한 소켓 안에 둔다.

    python test/prism/bench_tile_k1_bf16.py [--n 5120 --k 17408 --experts 8 --threads 14]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from kt_kernel import kt_kernel_ext
from kt_kernel.experts_partial import PartialMoEWrapper

NG, GRID, PMAX, RENORM_IT = 201, 0.005, 0.9, 3


def cpuinfer(threads, numa=0):
    cfg = kt_kernel_ext.WorkerPoolConfig()
    cfg.subpool_count = 1
    cfg.subpool_numa_map = [numa]
    cfg.subpool_thread_count = [threads]
    return kt_kernel_ext.CPUInfer(cfg)


def config(E, H, I, sparse, p):
    """down 슬롯만 쓰는 dense 퇴화형 (dense 레인 cold의 모습). K = I, N = H."""
    cfg = kt_kernel_ext.moe.MOEConfig(E, 1, H, I, 0)
    cfg.max_len = 4
    cfg.layer_idx = 0
    cfg.partial.enabled = True
    cfg.partial.gate.offset, cfg.partial.gate.rows = 0, 32     # 더미 (최소 타일)
    cfg.partial.up.offset, cfg.partial.up.rows = 0, 32
    cfg.partial.down.offset, cfg.partial.down.rows = 0, I
    cfg.partial.n_total = I
    cfg.partial.node_gateup_n_offset, cfg.partial.node_gateup_n_rows = [0], [I]
    cfg.partial.node_down_n_offset, cfg.partial.node_down_n_rows = [0], [H]
    if sparse:
        sp = cfg.partial.sparsity
        sp.pmax, sp.grid, sp.ng, sp.renorm_it = PMAX, GRID, NG, RENORM_IT
        sp.p_gate = sp.p_up = sp.p_down = p
        sp.lam_gate = sp.lam_up = sp.lam_down = 0.0
    return cfg


def build(key, E, H, I, tables, ci, g):
    cfg = config(E, H, I, tables is not None, 0.5)
    cfg.pool = ci.backend_
    w = PartialMoEWrapper(cfg, ci, kernel_key=key)
    dn = (torch.randn(E, H, I, generator=g) * 0.05).to(torch.bfloat16).contiguous()
    du = (torch.randn(E, I, 32, generator=g) * 0.05).to(torch.bfloat16).contiguous()
    w.load_weights_from_tensors(du, du.clone(), dn, sparsity_tables=tables)
    return w


def per_k_thr(x, keep):
    """|x|의 (1-keep) 분위 — per-k 마스크가 정확히 keep 비율을 남기는 threshold."""
    flat = x.abs().reshape(-1).float().sort().values
    return float(flat[int((1.0 - keep) * flat.numel())])


def pair_thr(x, keep):
    """페어 에너지 sqrt(x0²+x1²)의 (1-keep) 분위 (wn=1, c=0일 때의 k2wl2 점수)."""
    v = x.reshape(-1).float()
    e = (v[0::2] ** 2 + v[1::2] ** 2).sqrt().sort().values
    return float(e[int((1.0 - keep) * e.numel())])


def timeit(w, E, H, I, x, reps, weights):
    ids = torch.zeros(1, 1, dtype=torch.int64).contiguous()
    out = torch.zeros(1, 1, H, dtype=torch.bfloat16).contiguous()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for r in range(reps):
            ids[0, 0] = r % E                      # 슬롯 회전 — 캐시 상주 방지
            w.forward_down(ids, x, out, weights)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5120, help="출력 축 (hidden)")
    ap.add_argument("--k", type=int, default=17408, help="contraction 축 (inter)")
    ap.add_argument("--experts", type=int, default=8, help="회전 슬롯 수")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--keep", type=float, default=0.5, help="per-k keep 비율")
    ap.add_argument("--granularity-probe", action="store_true",
                    help="같은 keep을 64 B 단위 / 128 B(페어 정렬) 단위로 스킵해 비교한다. "
                         "둘이 크게 다르면 스킵 이득을 먹는 것은 커널이 아니라 메모리 "
                         "쪽(인접 라인 프리페치·섹터 granularity)이다")
    ap.add_argument("--numa", type=int, default=0,
                    help="워커 소켓. HW 프리페처 축이기도 하다 (hwpf 메모) — 스킵한 라인을 "
                         "L2 streamer가 다시 끌어오면 per-k 이득이 사라진다")
    args = ap.parse_args()
    H, I, E = args.n, args.k, args.experts
    ci = cpuinfer(args.threads, args.numa)
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(1, 1, I, generator=g) * 0.05).to(torch.bfloat16).contiguous()
    wts = torch.ones(1, 1, dtype=torch.float32).contiguous()
    gib = E * H * I * 2 / 2**30
    print(f"shape K={I} N={H} · slots {E} ({gib:.2f} GiB) · {args.threads} thr · numa {args.numa} · keep {args.keep}")

    full = lambda v: torch.full((E, NG), float(v), dtype=torch.float32)
    thr1 = per_k_thr(x, args.keep)
    union = 1.0 - (1.0 - args.keep) ** 2                # 페어 union keep
    thr2 = pair_thr(x, union)
    k2_tables = {
        "thr_gate": full(0.0), "thr_up": full(0.0), "thr_down": full(thr2),
        "gate_wn_sq": torch.ones(E * 32), "gate_pair_dot": torch.zeros(E * 16),
        "up_wn_sq": torch.ones(E * 32), "up_pair_dot": torch.zeros(E * 16),
        "down_wn_sq": torch.ones(E * I), "down_pair_dot": torch.zeros(E * I // 2),
    }
    k1_tables = {"thr_gate": full(0.0), "thr_up": full(0.0), "thr_down": full(thr1)}

    if args.granularity_probe:
        # 페어 정렬 x: x[2i] == x[2i+1] → per-k 마스크가 페어를 통째로 남긴다. keep은
        # 같고 스킵 단위만 64 B → 128 B가 된다.
        xp = x.clone().reshape(-1)
        xp[1::2] = xp[0::2]
        xp = xp.reshape(1, 1, I).contiguous()
        thr_p = per_k_thr(xp, args.keep)
        w1 = build("kt_tile_k1_bf16", E, H, I, k1_tables, ci, torch.Generator().manual_seed(7))
        t_rand = timeit(w1, E, H, I, x, args.reps, wts)
        del w1
        w2 = build("kt_tile_k1_bf16", E, H, I,
                   {"thr_gate": full(0.0), "thr_up": full(0.0), "thr_down": full(thr_p)},
                   ci, torch.Generator().manual_seed(7))
        t_pair = timeit(w2, E, H, I, xp, args.reps, wts)
        del w2
        kept_r = float((x.abs().reshape(-1).float() >= thr1).float().mean())
        kept_p = float((xp.abs().reshape(-1).float() >= thr_p).float().mean())
        print(f"  k1 keep {kept_r:.3f}, 64 B 단위 스킵    {t_rand:7.3f} ms")
        print(f"  k1 keep {kept_p:.3f}, 128 B 단위 스킵   {t_pair:7.3f} ms")
        print(f"  → 128 B 단위가 {t_rand / t_pair:.3f}배 빠르다 "
              f"(1.0에 가까우면 64 B 스킵이 실제로 바이트를 아낀다)")
        return

    rows = []
    for label, key, tables, w_arg in (
        ("dense k2", "kt_tile_k2_bf16", None, None),
        ("dense k1", "kt_tile_k1_bf16", None, None),
        (f"sp{int(100 * (1 - union))} k2 (union keep {union:.2f})", "kt_tile_k2_bf16", k2_tables, wts),
        (f"sp{int(100 * (1 - args.keep))} k1 (per-k keep {args.keep:.2f})", "kt_tile_k1_bf16", k1_tables, wts),
    ):
        w = build(key, E, H, I, tables, ci, torch.Generator().manual_seed(7))
        ms = timeit(w, E, H, I, x, args.reps, w_arg)
        rows.append((label, ms))
        print(f"  {label:34s} {ms:7.3f} ms   {H * I * 2 / 2**30 / (ms / 1e3):6.1f} GB/s(dense 기준)")
        del w
    print(f"\n  dense    k2/k1 = {rows[0][1] / rows[1][1]:.3f}   (FINDINGS §11 예측 0.995)")
    print(f"  per-k sp k2/k1 = {rows[2][1] / rows[3][1]:.3f}   (예측 {union / args.keep:.3f})")


if __name__ == "__main__":
    main()
