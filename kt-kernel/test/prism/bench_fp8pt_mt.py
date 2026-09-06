"""kt_tile_k2_fp8b128 vs kt_tile_k2_fp8pt — 실제 kt 경로(PartialMoEWrapper, CPUInfer 워커풀) 다중 스레드 벤치.

Mistral-Medium-3.5 cold 형상(hidden 12288, ffn 28672, cold K 밴드 25%)을 dense 레인처럼
expert 축 = layer 슬롯으로 접고(E개 회전 → L3 밖), gateup / down 진입점을 dense와 pair-sp50으로 잰다.
"""
import argparse, statistics, sys, time
import torch
from kt_kernel import kt_kernel_ext
from kt_kernel.experts_partial import PartialMoEWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=28, help="CPUInfer 총 스레드 (NUMA 2개에 균등)")
ap.add_argument("--experts", type=int, default=6, help="회전 슬롯 수 (L3 밖으로 밀어내기)")
ap.add_argument("--iters", type=int, default=120)
ap.add_argument("--warmup-seconds", type=float, default=1.5)
ap.add_argument("--kernels", default="kt_tile_k2_fp8b128,kt_tile_k2_fp8pt")
ap.add_argument("--kfrac", type=float, default=0.25, help="cold K 밴드 비율")
args = ap.parse_args()

HIDDEN, FFN = 12288, 28672
E, TOPK = args.experts, 1
DUMMY = 128  # b128도 받는 더미 K (k%128)
SP_NG, SP_GRID, SP_PMAX, SP_RENORM = 201, 0.005, 0.9, 3

cpuinfer = kt_kernel_ext.CPUInfer(args.threads)
g = torch.Generator().manual_seed(0)


def fp8_codes(shape):
    e = torch.randint(1, 15, shape, generator=g); m = torch.randint(0, 8, shape, generator=g)
    s = torch.randint(0, 2, shape, generator=g)
    return ((s << 7) | (e << 3) | m).to(torch.uint8)


def fp8_dequant(codes):
    c = codes.to(torch.int32); s = (c >> 7) & 1; e = (c >> 3) & 0xF; m = c & 7
    v = (1.0 + m.float() / 8.0) * torch.pow(2.0, (e - 7).float())
    return torch.where(s == 1, -v, v)


def scales(kernel, n, k):
    if kernel.endswith("fp8pt"):
        return torch.full((E,), 0.02, dtype=torch.float32)
    return torch.full((E, n // 128, k // 128), 0.02, dtype=torch.float32)


def make_cfg(entry, kernel, k_rows, sparse_tables=None):
    cfg = kt_kernel_ext.moe.MOEConfig(E, TOPK, HIDDEN, FFN, 0)
    cfg.max_len = 4
    cfg.partial.enabled = True
    cfg.partial.n_total = FFN
    if entry == "gateup":
        cfg.partial.gate.offset = cfg.partial.up.offset = 0
        cfg.partial.gate.rows = cfg.partial.up.rows = k_rows
        cfg.partial.down.offset, cfg.partial.down.rows = 0, DUMMY
    else:
        cfg.partial.gate.offset = cfg.partial.up.offset = 0
        cfg.partial.gate.rows = cfg.partial.up.rows = DUMMY
        cfg.partial.down.offset, cfg.partial.down.rows = 0, k_rows
    cfg.pool = cpuinfer.backend_
    if sparse_tables is not None:
        sp = cfg.partial.sparsity
        sp.pmax, sp.grid, sp.ng, sp.renorm_it = SP_PMAX, SP_GRID, SP_NG, SP_RENORM
        sp.p_gate = sp.p_up = sp.p_down = 0.5
        sp.lam_gate = sp.lam_up = sp.lam_down = 0.0
    return cfg


def pair_tables(w_codes, x, keep=0.5):
    """w_codes [E, N, K] → wn_sq/pair_dot, thr = 유지 비율 keep이 되는 imp 분위수(expert별 상수 곡선)."""
    f = fp8_dequant(w_codes)
    a = (f * f).sum(dim=1).contiguous()
    c = (f[:, :, 0::2] * f[:, :, 1::2]).sum(dim=1).contiguous()
    x0, x1 = x[0::2].float(), x[1::2].float()
    imp = (a[:, 0::2] * x0 * x0 + a[:, 1::2] * x1 * x1 + 2 * c * x0 * x1).clamp_min(0).sqrt()  # [E, K/2]
    thr = torch.quantile(imp, 1.0 - keep, dim=1)  # [E]
    kept = (imp >= thr[:, None]).float().mean().item()
    return a, c, thr.view(E, 1).expand(E, SP_NG).contiguous(), kept


def bench(entry, kernel, sparse):
    k_rows = int((HIDDEN if entry == "gateup" else FFN) * args.kfrac)
    n_real = FFN if entry == "gateup" else HIDDEN
    x = torch.randn(k_rows, generator=g).to(torch.bfloat16)
    if entry == "gateup":
        gate, up = fp8_codes((E, FFN, k_rows)), fp8_codes((E, FFN, k_rows))
        down = fp8_codes((E, HIDDEN, DUMMY))
        gs, us, ds = scales(kernel, FFN, k_rows), scales(kernel, FFN, k_rows), scales(kernel, HIDDEN, DUMMY)
    else:
        down = fp8_codes((E, HIDDEN, k_rows))
        gate, up = fp8_codes((E, FFN, DUMMY)), fp8_codes((E, FFN, DUMMY))
        gs, us, ds = scales(kernel, FFN, DUMMY), scales(kernel, FFN, DUMMY), scales(kernel, HIDDEN, k_rows)
    tables, kept = None, 1.0
    if sparse:
        ga, gc, gt, kg = pair_tables(gate, x if entry == "gateup" else torch.ones(DUMMY, dtype=torch.bfloat16))
        ua, uc, ut, ku = pair_tables(up, x if entry == "gateup" else torch.ones(DUMMY, dtype=torch.bfloat16))
        da, dc, dt, kd = pair_tables(down, x if entry == "down" else torch.ones(DUMMY, dtype=torch.bfloat16))
        kept = (kg + ku) / 2 if entry == "gateup" else kd
        tables = {"gate_wn_sq": ga, "gate_pair_dot": gc, "up_wn_sq": ua, "up_pair_dot": uc,
                  "down_wn_sq": da, "down_pair_dot": dc, "thr_gate": gt, "thr_up": ut, "thr_down": dt}
    cfg = make_cfg(entry, kernel, k_rows, tables)
    w = PartialMoEWrapper(cfg, cpuinfer, kernel_key=kernel)
    w.load_weights_from_tensors(gate, up, down, sparsity_tables=tables, gate_scale=gs, up_scale=us, down_scale=ds)
    del gate, up, down

    qlen_t = torch.tensor([1], dtype=torch.int32)
    ids = [torch.tensor([[e]], dtype=torch.int64) for e in range(E)]
    rw = torch.tensor([[1.0]], dtype=torch.float32).contiguous()
    if entry == "gateup":
        xin = torch.zeros(1, HIDDEN, dtype=torch.bfloat16); xin[0, :k_rows] = x
        out = torch.zeros(1, TOPK, 2 * FFN, dtype=torch.bfloat16)
        def call(e):
            w.submit_forward_gateup(qlen_t.data_ptr(), TOPK, ids[e].data_ptr(), xin.data_ptr(), out.data_ptr(),
                                    weights_ptr=rw.data_ptr() if sparse else 0)
            w.sync()
    else:
        act = torch.zeros(1, TOPK, FFN, dtype=torch.bfloat16); act[0, 0, :k_rows] = x
        out = torch.zeros(1, TOPK, HIDDEN, dtype=torch.bfloat16)
        def call(e):
            w.submit_forward_down(qlen_t.data_ptr(), TOPK, ids[e].data_ptr(), act.data_ptr(), out.data_ptr(),
                                  weights_ptr=rw.data_ptr() if sparse else 0)
            w.sync()

    t_end = time.perf_counter() + args.warmup_seconds
    e = 0
    while time.perf_counter() < t_end:
        call(e % E); e += 1
    times = []
    for i in range(args.iters):
        t0 = time.perf_counter(); call(i % E); times.append((time.perf_counter() - t0) * 1e6)
    med = statistics.median(times)
    nproj = 2 if entry == "gateup" else 1
    weight_bytes = nproj * n_real * k_rows
    eff = weight_bytes * kept / med / 1e3
    print(f"  {entry:6s} {'sp50' if sparse else 'dense':5s} {kernel:20s} K={k_rows:5d} N={n_real:5d} "
          f"kept={kept:.2f}  median {med:8.1f} us  p10 {sorted(times)[len(times)//10]:8.1f}  "
          f"eff {eff:6.1f} GB/s  ({weight_bytes/1e6:.0f} MB/slot × {E} slots)")
    return med


print(f"threads={args.threads} (per NUMA {args.threads//2}), E={E} rotating, iters={args.iters}", flush=True)
res = {}
for entry in ("gateup", "down"):
    for sparse in (False, True):
        for kernel in args.kernels.split(","):
            res[(entry, sparse, kernel)] = bench(entry, kernel, sparse)
        ks = args.kernels.split(",")
        if len(ks) == 2:
            print(f"    -> {ks[1]}/{ks[0]} = {res[(entry, sparse, ks[1])]/res[(entry, sparse, ks[0])]:.3f}")
