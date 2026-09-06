/**
 * Prism SW prefetch master switch (ablation knob).
 *
 * HW prefetch is off on both nutella3 sockets, so the bf16 cold path issues its own
 * `_mm_prefetch` in three places: the tile_k2 decode gemv (weight stream, PF_DIST ahead),
 * the RowMajorNC prefill kernel (B panel one k-step ahead), and moe_base's prep-input /
 * export-copy helpers. `KT_SW_PREFETCH=0` turns all of them off at once so the "no SW
 * prefetcher" ablation can run on the very same binary and plan. Default (unset or any
 * other value) keeps them on — GLM/DSV4 lanes that share this build are unaffected unless
 * they set the variable themselves. The mxfp4/fp8 tile_k2 ports keep their own prefetch
 * regardless (the ablation is bf16-only by design).
 *
 * Read once per process; the callers hoist the check out of their inner loops, so the
 * enabled path costs nothing extra.
 */
#ifndef CPUINFER_OPERATOR_AMX_LA_SW_PREFETCH_HPP
#define CPUINFER_OPERATOR_AMX_LA_SW_PREFETCH_HPP

#include <cstdio>
#include <cstdlib>

namespace amx {

inline bool sw_prefetch_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("KT_SW_PREFETCH");
    const bool v = !(e && e[0] == '0' && e[1] == '\0');
    if (!v) std::fprintf(stderr, "[kt-kernel] SW prefetch OFF (KT_SW_PREFETCH=0): bf16 cold path issues no _mm_prefetch\n");
    return v;
  }();
  return on;
}

}  // namespace amx

#endif
