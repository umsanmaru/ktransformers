/* tile_k1_bf16_port.hpp — the k=1 member of the bf16 tile family (cpu-mm
 * `avx512_bcast_ymem_kt` / `ymem_k1_body.h` ported into kt's slab geometry).
 * Same surface as tile_k1_fp8b128_port.hpp (ktf8k1_*), prefixed ktb1_*; what
 * changes from fp8 is the element size (2 B), so one 64 B line is 32 n instead
 * of 64, and there are no scales:
 *
 *   ktb1_pack_codes[_range]  kt row-major bf16 [n][K] -> k1 tile layout
 *   ktb1_build_plan          the per-token plan (koff/pval/seg) from a PER-K mask
 *   ktb1_gemv                one n-range of one expert, m=1 decode
 *
 * Tile layout (element index bits, innermost first) — the k2 (VNNI) stack with
 * the k-pair bit lifted above the 32-n run, so one k row's 32-n chunk is ONE line:
 *   n:5/1 | k:5/32 | n:3/1024 | k-cols/8192 | n-supers/8192*(K/32)
 *   off(n,k) = (n&31) + (k&31)*32 + ((n>>5)&7)*1024 + (k>>5)*8192
 *            + (n>>8)*8192*(K/32)                          (= element offset)
 * Tile column = 32 k x 256 n = 8192 elements (16 KB) = 8 groups x 32 rows x 64 B.
 * Tile column and super size are the k2 values, so BufferB size is unchanged
 * (N*K bf16) — only the permutation inside a group differs. Skipping a k skips
 * a 64 B line per group (k2: a VNNI pair skips one 64 B line per 16-n group,
 * i.e. the SAME bytes for twice the k).
 *
 * Numerics: fp32 broadcast of the activation (bf16 << 16, exact) x bf16 weight
 * widened to fp32 (also << 16, exact), fp32 FMA into the y accumulators. The
 * k2 kernel instead uses `dpbf16_ps` on pairs — both accumulate in fp32 and
 * both are exact in the multiply; only the accumulation ORDER differs, so
 * results differ within fp32 reassociation (the same tolerance the tier tests
 * already carry).
 *
 * Prefetch: consumption-order cursor pf_dist LINES ahead (T1); one cursor step
 * is one line. `pf_dist == 0` walks the same order with no _mm_prefetch at all
 * (the k2 bf16 kernel spells that as its own no-pf instantiation instead).
 *
 * Why k=1 (cpu-mm FINDINGS §11, measured on bf16, N=4096): with a per-k mask
 * the kept fraction is 1-s instead of the pair's 1-s^2, so at sp50 the pair
 * kernel reads 1.479x the bytes and loses by exactly that. Dense the two are
 * within 0.5 % (0.995x), so nothing is given up when the mask is off.
 */
#ifndef KTB1_TILE_PORT_HPP
#define KTB1_TILE_PORT_HPP

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "llama.cpp/ggml-impl.h" /* ggml_bf16_t */

#define KTB1_TILE_N 256      /* n per super-block (tile column width) */
#define KTB1_TILE_K 32       /* k per tile column == mask word width */
#define KTB1_LINE_N 32       /* n per 64 B line (32 bf16) */
#define KTB1_NGROUP 8        /* 256 n / 32 n */
#define KTB1_GRP_ELEMS 1024  /* 32 n x 32 k */
#define KTB1_COL_ELEMS 8192  /* 256 n x 32 k */

#ifndef KTB1_PROLOGUE_PF
#define KTB1_PROLOGUE_PF 1
#endif

namespace amx {

/* element offset of (n, k) in the k1 tile layout for a K-column matrix */
static inline size_t ktb1_off(int n, int k, int K) {
  return (size_t)(n & 31) + (size_t)(k & 31) * 32 + (size_t)((n >> 5) & 7) * KTB1_GRP_ELEMS +
         (size_t)(k >> 5) * KTB1_COL_ELEMS + (size_t)(n >> 8) * KTB1_COL_ELEMS * (size_t)(K / KTB1_TILE_K);
}

/* ---- BufferB body ------------------------------------------------------
 * src: kt row-major bf16, src[n*K + k] (ckpt direction [N, K]).
 * dst: N*K bf16 in tile layout. The _range form packs rows [n0, n1). */
static inline void ktb1_pack_codes_range(ggml_bf16_t* dst, const ggml_bf16_t* src, int N, int K, int n0,
                                         int n1) {
  (void)N;
  for (int n = n0; n < n1; ++n)
    for (int k = 0; k < K; ++k) dst[ktb1_off(n, k, K)] = src[(size_t)n * K + k];
}

static inline void ktb1_pack_codes(ggml_bf16_t* dst, const ggml_bf16_t* src, int N, int K) {
  ktb1_pack_codes_range(dst, src, N, K, 0, N);
}

/* ---- the activation plan (kt Seam B shape, per-k) ----------------------
 *   koff [K]        ELEMENT offset of each kept k's line inside its 256-n
 *                   super, n-group 0: (k>>5)*8192 + (k&31)*32, k-ascending
 *   pval [K]        the kept k's activation as fp32 BITS (bf16 << 16)
 *   seg  [K/32+1]   tile-column prefix: column c owns kept ks [seg[c], seg[c+1])
 * `mask` is kt's absolute-k KMask (uint32 per 32-k column, bit j = k 32c + j);
 * NULL means dense. */
static inline uint32_t ktb1_k_koff(int k) {
  return (uint32_t)(k >> 5) * KTB1_COL_ELEMS + (uint32_t)(k & 31) * KTB1_LINE_N;
}

static inline int ktb1_build_plan(const uint32_t* mask, const uint16_t* x, int K, uint32_t* koff,
                                  uint32_t* pval, uint32_t* seg) {
  const int ncols = K / KTB1_TILE_K;
  int np = 0;
  seg[0] = 0;
  for (int c = 0; c < ncols; ++c) {
    uint32_t bits = mask ? mask[c] : 0xFFFFFFFFu;
    while (bits) {
      const int j = __builtin_ctz(bits);
      bits &= bits - 1u;
      const int k = c * KTB1_TILE_K + j;
      koff[np] = ktb1_k_koff(k);
      pval[np] = (uint32_t)x[k] << 16;  /* bf16 -> fp32 bits, exact */
      ++np;
    }
    seg[c + 1] = (uint32_t)np;
  }
  return np;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)

/* cursor over (column, group, kept k) — one step = one 64 B line */
typedef struct {
  int c, g, j, live;
} ktb1_cur;

static inline void ktb1_cur_step(ktb1_cur* t, const uint32_t* cs, int ncols) {
  if ((uint32_t)++t->j < cs[t->c + 1]) return;
  if (++t->g < KTB1_NGROUP) {
    t->j = (int)cs[t->c];
    return;
  }
  t->g = 0;
  do {
    ++t->c;
  } while (t->c < ncols && cs[t->c + 1] == cs[t->c]);
  if (t->c >= ncols) {
    t->live = 0;
    return;
  }
  t->j = (int)cs[t->c];
}

static inline ktb1_cur ktb1_cur_start(const uint32_t* cs, int ncols, int nkept, int ahead) {
  ktb1_cur t;
  t.c = 0;
  t.g = 0;
  t.j = 0;
  t.live = nkept > 0;
  if (t.live) {
    while (t.c < ncols && cs[t.c + 1] == cs[t.c]) ++t.c;
    if (t.c >= ncols)
      t.live = 0;
    else
      t.j = (int)cs[t.c];
  }
  for (int i = 0; i < ahead && t.live; ++i) ktb1_cur_step(&t, cs, ncols);
  return t;
}

/* 32 bf16 words -> fp32 of lanes [0,16) / [16,32) (memory order = n order) */
static inline __m512 ktb1_lo(__m512i v) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(v)), 16));
}
static inline __m512 ktb1_hi(__m512i v) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1)), 16));
}

#define KTB1_PF(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)

/* ---- the m=1 decode kernel: y[n0..n1) = x . W for one expert -----------
 * W: ktb1_pack_codes output (N*K bf16). koff/pval/seg: the plan. n0/n1 are
 * multiples of 256. y is fp32 and OVERWRITTEN. pf_dist in 64 B lines, 0 = off.
 *
 * y lives in L1 and is loaded/stored once per (column, group) visit — the
 * "ymem" policy that won the cpu-mm A/B against register-resident accumulators
 * (FINDINGS §11), and the same policy the k2 bf16 kernel uses. */
static inline void ktb1_gemv(const ggml_bf16_t* W, const uint32_t* koff, const uint32_t* pval,
                             const uint32_t* cs, int nkept, int N, int K, float* y, int n0, int n1,
                             int pf_dist) {
  const int ncols = K / KTB1_TILE_K;
  (void)N;
  memset(y + n0, 0, (size_t)(n1 - n0) * sizeof(float));
  for (int ns = n0; ns < n1; ns += KTB1_TILE_N) {
    const ggml_bf16_t* const sb = W + (size_t)(ns >> 8) * KTB1_COL_ELEMS * (size_t)ncols;
    float* const ys = y + ns;
    ktb1_cur cu = ktb1_cur_start(cs, ncols, nkept, pf_dist > 0 ? pf_dist : 0);
    if (KTB1_PROLOGUE_PF && pf_dist > 0) {
      ktb1_cur pro = ktb1_cur_start(cs, ncols, nkept, 0);
      for (int i = 0; i < pf_dist && pro.live; ++i) {
        KTB1_PF(sb + koff[pro.j] + (size_t)pro.g * KTB1_GRP_ELEMS);
        ktb1_cur_step(&pro, cs, ncols);
      }
    }
    for (int c = 0; c < ncols; ++c) {
      const int jlo = (int)cs[c], jhi = (int)cs[c + 1];
      if (jlo == jhi) continue;
#pragma GCC unroll 8
      for (int g = 0; g < KTB1_NGROUP; ++g) {
        const ggml_bf16_t* const gb = sb + (size_t)g * KTB1_GRP_ELEMS;
        float* const yg = ys + KTB1_LINE_N * g;
        __m512 a0 = _mm512_loadu_ps(yg);
        __m512 a1 = _mm512_loadu_ps(yg + 16);
        for (int j = jlo; j < jhi; ++j) {
          if (pf_dist > 0 && cu.live) {
            KTB1_PF(sb + koff[cu.j] + (size_t)cu.g * KTB1_GRP_ELEMS);
            ktb1_cur_step(&cu, cs, ncols);
          }
          const __m512 v = _mm512_castsi512_ps(_mm512_set1_epi32((int)pval[j]));
          const __m512i w = _mm512_loadu_si512((const void*)(gb + koff[j]));
          a0 = _mm512_fmadd_ps(v, ktb1_lo(w), a0);
          a1 = _mm512_fmadd_ps(v, ktb1_hi(w), a1);
        }
        _mm512_storeu_ps(yg, a0);
        _mm512_storeu_ps(yg + 16, a1);
      }
    }
  }
}

#undef KTB1_PF
#endif /* AVX512F && AVX512BW */

}  // namespace amx

#endif /* KTB1_TILE_PORT_HPP */
