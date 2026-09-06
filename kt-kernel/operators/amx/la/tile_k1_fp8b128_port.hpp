/* tile_k1_fp8b128_port.hpp — the k=1 member of the fp8b128 tile family
 * (fp8 e4m3 weights, one fp32 scale per 128x128 block — kt-kernel's
 * `fp8-moe.hpp` / DeepSeek blockwise contract). Same surface as
 * tile_k2_fp8b128_port.hpp (ktf8_*), prefixed ktf8k1_*; what changes is the
 * k granularity of the layout, the plan and the skip:
 *
 *   ktf8k1_pack_codes[_range]  kt's row-major e4m3 bytes [n][K] -> k1 tile layout
 *   (scales)                   ktf8_pack_scales_range from the k2 port, unchanged —
 *                              the transposed [K/128][N/128] table is identical
 *   ktf8k1_build_plan          the per-token plan (koff/pval/seg) from a PER-K mask
 *   ktf8k1_gemv                one n-range of one expert, m=1 decode
 *
 * Tile layout (element index bits, innermost first) — the k2 stack with the
 * k-pair bit lifted above the 64-n run, so one k row's 64-n chunk is ONE line:
 *   n:6/1 | k:5/64 | n:2/2048 | k-cols/8192 | n-supers/8192*(K/32)
 *   off(n,k) = (n&63) + (k&31)*64 + ((n>>6)&3)*2048 + (k>>5)*8192
 *            + (n>>8)*8192*(K/32)                               (= byte offset)
 * Tile column = 32 k x 256 n = 8 KB = 4 groups x 32 rows x 64 B. Tile column,
 * group, super, scale table position and BufferB size are all the k2 values —
 * only the byte permutation inside a group differs. Skipping a k skips a 64 B
 * line per group (k2: a pair skips 128 B).
 *
 * Numerics: fp32 activation x exact e4m3 value (integer bit decode straight to
 * fp32), fp32 FMA accumulate over each 32-k column, then y += acc * s(kblk, nblk)
 * with the fp32 block scale. Same contract as the k2 tile and kt's fp8 kernel;
 * only the accumulation order differs (per-k FMA instead of dpbf16 pairs).
 *
 * Prefetch: consumption-order cursor pf_dist LINES ahead (T1; one cursor step =
 * one line, no /2 as in k2) and, on by default, a burst for the first pf_dist
 * lines of every 256-n super-block.
 *
 * Why k=1 (cpu-mm FINDINGS §11/§15): with a per-k mask the kept fraction is
 * 1-s instead of the pair's 1-s^2, and at fp8 that byte saving beats dpbf16's
 * halved instruction count from ~sp25 up; dense it trails k2 by ~10-16 %.
 */
#ifndef KTF8K1_TILE_PORT_HPP
#define KTF8K1_TILE_PORT_HPP

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tile_k2_fp8b128_port.hpp" /* KTF8_TILE_N/K, KTF8_BLK, COL/GRP bytes, ktf8_pack_scales_range */

#ifdef __cplusplus
extern "C" {
#endif

#define KTF8K1_ROW_BYTES 64 /* one k row x 64 n x 1 B = one line */
#ifndef KTF8K1_PROLOGUE_PF
#define KTF8K1_PROLOGUE_PF 1
#endif
/* decode path: 1 = e4m3 -> bf16 (32 lanes, the k2 decoder's 6 ops) then widen lo/hi
 * halves to fp32; 0 = e4m3 -> fp32 directly (16 lanes, 6 ops). Same values. Measured
 * 2026-09-05 (nutella3 socket 1, 8 thr, N=4096 K=2048 E=16 rotation, min of 30 x2):
 * via-bf16 0.107 ms dense / 0.054 per-k sp50 vs direct 0.116 / 0.059 -> via-bf16 default. */
#ifndef KTF8K1_DECODE_VIA_BF16
#define KTF8K1_DECODE_VIA_BF16 1
#endif

/* element (== byte) offset of (n, k) in the k1 tile layout for a K-column matrix */
static inline size_t ktf8k1_off(int n, int k, int K) {
  return (size_t)(n & 63) + (size_t)(k & 31) * 64 + (size_t)((n >> 6) & 3) * 2048 +
         (size_t)(k >> 5) * 8192 + (size_t)(n >> 8) * 8192 * (size_t)(K / 32);
}

/* ---- BufferB body: codes ------------------------------------------------
 * src: kt row-major e4m3 bytes, src[n*K + k]. dst: N*K bytes in tile layout.
 * The _range form packs rows [n0, n1) (worker split); bases are whole-matrix. */
static inline void ktf8k1_pack_codes_range(uint8_t *dst, const uint8_t *src, int N, int K, int n0,
                                           int n1) {
  (void)N;
  for (int n = n0; n < n1; ++n)
    for (int k = 0; k < K; ++k) dst[ktf8k1_off(n, k, K)] = src[(size_t)n * K + k];
}
static inline void ktf8k1_pack_codes(uint8_t *dst, const uint8_t *src, int N, int K) {
  ktf8k1_pack_codes_range(dst, src, N, K, 0, N);
}

/* ---- the activation plan (kt Seam B shape, per-k) -------------------------
 *   koff [K]        BYTE offset of each kept k's line inside its 256-n super,
 *                   n-group 0: (k>>5)*8192 + (k&31)*64, k-ascending
 *   pval [K]        the kept k's activation as fp32 BITS (bf16 << 16)
 *   seg  [K/32+1]   tile-column prefix: column c owns kept ks [seg[c], seg[c+1])
 * ktf8k1_build_plan encodes kt's absolute-k KMask (uint32 per 32-k column,
 * bit j = k 32c + j; NULL = dense). */
static inline uint32_t ktf8k1_k_koff(int k) {
  return (uint32_t)(k >> 5) * KTF8_COL_BYTES + (uint32_t)(k & 31) * KTF8K1_ROW_BYTES;
}

static inline int ktf8k1_build_plan(const uint32_t *mask, const uint16_t *x, int K, uint32_t *koff,
                                    uint32_t *pval, uint32_t *seg) {
  const int ncols = K / KTF8_TILE_K;
  int np = 0;
  seg[0] = 0;
  for (int c = 0; c < ncols; ++c) {
    uint32_t bits = mask ? mask[c] : 0xFFFFFFFFu;
    while (bits) {
      const int j = __builtin_ctz(bits);
      bits &= bits - 1u;
      const int k = c * KTF8_TILE_K + j;
      koff[np] = ktf8k1_k_koff(k);
      pval[np] = (uint32_t)x[k] << 16;
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
} ktf8k1_cur;

static inline void ktf8k1_cur_step(ktf8k1_cur *t, const uint32_t *cs, int ncols) {
  if ((uint32_t)++t->j < cs[t->c + 1]) return;
  if (++t->g < 4) { t->j = (int)cs[t->c]; return; }
  t->g = 0;
  do { ++t->c; } while (t->c < ncols && cs[t->c + 1] == cs[t->c]);
  if (t->c >= ncols) { t->live = 0; return; }
  t->j = (int)cs[t->c];
}

static inline ktf8k1_cur ktf8k1_cur_start(const uint32_t *cs, int ncols, int nkept, int ahead) {
  ktf8k1_cur t;
  t.c = 0; t.g = 0; t.j = 0; t.live = nkept > 0;
  if (t.live) {
    while (t.c < ncols && cs[t.c + 1] == cs[t.c]) ++t.c;
    if (t.c >= ncols) t.live = 0; else t.j = (int)cs[t.c];
  }
  for (int i = 0; i < ahead && t.live; ++i) ktf8k1_cur_step(&t, cs, ncols);
  return t;
}

/* 16 e4m3 codes -> 16 exact fp32 values: mantissa 3 bits into fp32's top
 * mantissa bits, exponent rebias 7 -> 127 (+120 << 23 = 0x3C000000) on nonzero
 * codes only, sign to bit 31. Same encoder convention as ktf8_fp8_32 (no
 * denormal codes, no NaN code). */
static inline __m512 ktf8k1_fp8_16(const uint8_t *p) {
  const __m512i w = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)p));
  const __m512i mag = _mm512_slli_epi32(_mm512_and_si512(w, _mm512_set1_epi32(0x7f)), 20);
  const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(w, _mm512_set1_epi32(0x80)), 24);
  const __mmask16 nz = _mm512_test_epi32_mask(mag, mag);
  return _mm512_castsi512_ps(_mm512_or_si512(
      sign, _mm512_mask_add_epi32(_mm512_setzero_si512(), nz, mag, _mm512_set1_epi32(0x3C000000))));
}

#if KTF8K1_DECODE_VIA_BF16
/* 32 e4m3 codes -> 32 exact bf16 words (same as ktf8_fp8_32, which lives inside the k2
 * port's AVX512BF16 block; this kernel needs only F+BW so it carries its own copy). */
static inline __m512i ktf8k1_fp8_32bf(const uint8_t *p) {
  const __m512i w = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)p));
  const __m512i mag = _mm512_slli_epi16(_mm512_and_si512(w, _mm512_set1_epi16(0x7f)), 4);
  const __m512i sign = _mm512_slli_epi16(_mm512_and_si512(w, _mm512_set1_epi16((short)0x80)), 8);
  const __mmask32 nz = _mm512_test_epi16_mask(mag, mag);
  return _mm512_or_si512(
      sign, _mm512_mask_add_epi16(_mm512_setzero_si512(), nz, mag, _mm512_set1_epi16(0x3C00)));
}
/* bf16 word vector (32 lanes) -> fp32 of lanes [0,16) / [16,32) */
static inline __m512 ktf8k1_bf16_lo(__m512i v) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(v)), 16));
}
static inline __m512 ktf8k1_bf16_hi(__m512i v) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1)), 16));
}
#endif

/* one 64 B line (64 n of one k) into 4 fp32 accumulators */
static inline void ktf8k1_line_fma(const uint8_t *p, __m512 v, __m512 *a0, __m512 *a1, __m512 *a2,
                                   __m512 *a3) {
#if KTF8K1_DECODE_VIA_BF16
  const __m512i w0 = ktf8k1_fp8_32bf(p), w1 = ktf8k1_fp8_32bf(p + 32);
  *a0 = _mm512_fmadd_ps(v, ktf8k1_bf16_lo(w0), *a0);
  *a1 = _mm512_fmadd_ps(v, ktf8k1_bf16_hi(w0), *a1);
  *a2 = _mm512_fmadd_ps(v, ktf8k1_bf16_lo(w1), *a2);
  *a3 = _mm512_fmadd_ps(v, ktf8k1_bf16_hi(w1), *a3);
#else
  *a0 = _mm512_fmadd_ps(v, ktf8k1_fp8_16(p), *a0);
  *a1 = _mm512_fmadd_ps(v, ktf8k1_fp8_16(p + 16), *a1);
  *a2 = _mm512_fmadd_ps(v, ktf8k1_fp8_16(p + 32), *a2);
  *a3 = _mm512_fmadd_ps(v, ktf8k1_fp8_16(p + 48), *a3);
#endif
}

#define KTF8K1_PF(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T1)

/* ---- the m=1 decode kernel: y[n0..n1) = x . W for one expert -----------
 * W: ktf8k1_pack_codes output (N*K bytes). S: transposed fp32 scales
 * ((K/128)*(N/128) floats). koff/pval/seg: the plan. n0/n1 multiples of 256.
 * y is fp32 and OVERWRITTEN. pf_dist in 64 B lines, 0 = no prefetch. */
static inline void ktf8k1_gemv(const uint8_t *W, const float *S, const uint32_t *koff,
                               const uint32_t *pval, const uint32_t *cs, int nkept, int N, int K,
                               float *y, int n0, int n1, int pf_dist) {
  const int ncols = K / KTF8_TILE_K;
  const int NB = N / KTF8_BLK;
  memset(y + n0, 0, (size_t)(n1 - n0) * sizeof(float));
  for (int ns = n0; ns < n1; ns += KTF8_TILE_N) {
    const uint8_t *const sb = W + (size_t)(ns >> 8) * KTF8_COL_BYTES * (size_t)ncols;
    const int nb0 = ns / KTF8_BLK; /* 64-n group g lies in n-block nb0 + (g >> 1) */
    float *const ys = y + ns;
    ktf8k1_cur cu = ktf8k1_cur_start(cs, ncols, nkept, pf_dist > 0 ? pf_dist : 0);
    if (KTF8K1_PROLOGUE_PF && pf_dist > 0) {
      ktf8k1_cur pro = ktf8k1_cur_start(cs, ncols, nkept, 0);
      for (int i = 0; i < pf_dist && pro.live; ++i) {
        KTF8K1_PF(sb + koff[pro.j] + (size_t)pro.g * KTF8_GRP_BYTES);
        ktf8k1_cur_step(&pro, cs, ncols);
      }
    }
    for (int c = 0; c < ncols; ++c) {
      const int jlo = (int)cs[c], jhi = (int)cs[c + 1];
      if (jlo == jhi) continue;
      const float *const scol = S + (size_t)(c >> 2) * (size_t)NB + nb0;
      for (int g = 0; g < 4; ++g) {
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        const uint8_t *const gb = sb + (size_t)g * KTF8_GRP_BYTES;
        for (int j = jlo; j < jhi; ++j) {
          if (pf_dist > 0 && cu.live) {
            KTF8K1_PF(sb + koff[cu.j] + (size_t)cu.g * KTF8_GRP_BYTES);
            ktf8k1_cur_step(&cu, cs, ncols);
          }
          const __m512 v = _mm512_castsi512_ps(_mm512_set1_epi32((int)pval[j]));
          ktf8k1_line_fma(gb + koff[j], v, &a0, &a1, &a2, &a3);
        }
        const __m512 s = _mm512_set1_ps(scol[g >> 1]);
        float *const yg = ys + 64 * g;
        _mm512_storeu_ps(yg, _mm512_fmadd_ps(a0, s, _mm512_loadu_ps(yg)));
        _mm512_storeu_ps(yg + 16, _mm512_fmadd_ps(a1, s, _mm512_loadu_ps(yg + 16)));
        _mm512_storeu_ps(yg + 32, _mm512_fmadd_ps(a2, s, _mm512_loadu_ps(yg + 32)));
        _mm512_storeu_ps(yg + 48, _mm512_fmadd_ps(a3, s, _mm512_loadu_ps(yg + 48)));
      }
    }
  }
}

#undef KTF8K1_PF
#endif /* AVX512F && AVX512BW */

#ifdef __cplusplus
}
#endif
#endif /* KTF8K1_TILE_PORT_HPP */
