/* port/kt_tile_k2_fp8b128.hpp — self-contained port surface of
 * avx512_tile_k2_fp8b128 (fp8 e4m3 weights, one fp32 scale per 128x128
 * block — kt-kernel's `fp8-moe.hpp` / DeepSeek blockwise contract) for
 * kt-kernel. No mmp dependencies, C and C++20. Same surface as
 * port/kt_tile_k2_mxfp4.hpp (see port/PORTING_MXFP4.md for the Seam A/B/C
 * mapping); only the dtype-specific parts differ:
 *
 *   ktf8_pack_codes[_range]   kt's row-major e4m3 bytes [n][K] -> tile layout
 *                             (a byte permutation; 1 B per element)
 *   ktf8_pack_scales[_range]  kt's fp32 scale_inv, n-major [N/128][K/128]
 *                             -> transposed [K/128][N/128] (arbitrary fp32:
 *                             no power-of-two requirement)
 *   ktf8_collect / ktf8_build_plan   the per-token plan (koff/pval/seg),
 *                             identical to the mxfp4 port
 *   ktf8_gemv                 one n-range of one expert, m=1 decode
 *
 * Tile layout (element index bits, innermost first) — the fp4 tile's element
 * stack at 1 byte per element:
 *   k:1/1 | n:6/2 | k:4/128 | n:2/2048 | k-cols/8192 | n-supers/8192*(K/32)
 *   off(n,k) = (k&1) + (n&63)*2 + ((k>>1)&15)*128 + ((n>>6)&3)*2048
 *            + (k>>5)*8192 + (n>>8)*8192*(K/32)               (= byte offset)
 * One k-pair's 64-n chunk = 128 B = two adjacent lines. Tile column = 32 k x
 * 256 n = 8 KB = 4 groups x 16 pairs x 128 B.
 *
 * Numerics: bf16 activation x exact e4m3 value (integer bit decode to bf16),
 * fp32 dpbf16 accumulate over each 32-k column, then y += acc * s(kblk, nblk)
 * with the fp32 block scale — a 64-n group never straddles a 128-n scale
 * block, so the scale is one scalar broadcast per (column, group). Against
 * kt's own fp8 kernel only the accumulation order differs.
 *
 * Prefetch: consumption-order cursor pf_dist lines ahead (T1; the cursor steps
 * 128 B chunks = pf_dist/2) and, on by default (KTF8_PROLOGUE_PF), a burst for
 * the first pf_dist lines of every 256-n super-block. No scale stream to
 * prefetch: an expert's whole scale table is (K/128)*(N/128) floats.
 * Measured (dev box, socket 1 HW-prefetch-off, N=4096 K=2048, 8 threads):
 * 0.817 ms dense / 0.447 ms pair-sp50; in kt's regime (256-n worker slabs,
 * HW prefetchers on) 1.8x faster dense than the flat per-column fp8 kernel,
 * whose slab sweeps the L2 streamer over-fetches. FINDINGS §15.
 */
#ifndef KTF8_TILE_PORT_HPP
#define KTF8_TILE_PORT_HPP

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KTF8_TILE_N 256
#define KTF8_TILE_K 32
#define KTF8_BLK 128        /* scale block edge, both axes */
#define KTF8_COL_BYTES 8192 /* 32 k x 256 n x 1 B */
#define KTF8_GRP_BYTES 2048 /* 16 pairs x 64 n x 2 k x 1 B */
#ifndef KTF8_PROLOGUE_PF
#define KTF8_PROLOGUE_PF 1
#endif

/* element (== byte) offset of (n, k) in the tile layout for a K-column matrix */
static inline size_t ktf8_off(int n, int k, int K) {
  return (size_t)(k & 1) + (size_t)(n & 63) * 2 + (size_t)((k >> 1) & 15) * 128 +
         (size_t)((n >> 6) & 3) * 2048 + (size_t)(k >> 5) * 8192 +
         (size_t)(n >> 8) * 8192 * (size_t)(K / 32);
}

/* ---- BufferB body: codes ------------------------------------------------
 * src: kt row-major e4m3 bytes, src[n*K + k]. dst: N*K bytes in tile layout.
 * The _range form packs rows [n0, n1) (worker split); bases are whole-matrix. */
static inline void ktf8_pack_codes_range(uint8_t *dst, const uint8_t *src,
                                         int N, int K, int n0, int n1) {
  (void)N;
  for (int n = n0; n < n1; ++n)
    for (int k = 0; k < K; ++k) dst[ktf8_off(n, k, K)] = src[(size_t)n * K + k];
}
static inline void ktf8_pack_codes(uint8_t *dst, const uint8_t *src, int N, int K) {
  ktf8_pack_codes_range(dst, src, N, K, 0, N);
}

/* ---- BufferB body: scales -----------------------------------------------
 * src: kt fp32 scale_inv, n-major: src[(n/128)*(K/128) + k/128] (the
 * multiplier in w = q * scale_inv). dst: transposed dst[(k/128)*(N/128) +
 * n/128], N*K/16384 floats. The _range form transposes n-blocks that
 * intersect [n0, n1); n0/n1 need not be block-aligned. */
static inline void ktf8_pack_scales_range(float *dst, const float *src, int N,
                                          int K, int n0, int n1) {
  const int KB = K / KTF8_BLK, NB = N / KTF8_BLK;
  const int b0 = n0 / KTF8_BLK, b1 = (n1 + KTF8_BLK - 1) / KTF8_BLK;
  for (int nb = b0; nb < b1; ++nb)
    for (int kb = 0; kb < KB; ++kb)
      dst[(size_t)kb * NB + nb] = src[(size_t)nb * KB + kb];
}
static inline void ktf8_pack_scales(float *dst, const float *src, int N, int K) {
  ktf8_pack_scales_range(dst, src, N, K, 0, N);
}

/* ---- the activation plan (kt Seam B shape) -------------------------------
 *   koff [K/2]      BYTE offset of each kept pair's chunk inside its 256-n
 *                   super, n-group 0: (p>>4)*8192 + (p&15)*128, k-ascending
 *   pval [K/2]      the pair's two bf16 activations, lo = x[2p], hi = x[2p+1]
 *   seg  [K/32+1]   tile-column prefix: column c owns pairs [seg[c], seg[c+1])
 * ktf8_collect applies the pair-L2 threshold rule; ktf8_build_plan encodes
 * kt's absolute-k PairMask (uint16 per 32-k column, bit j = pair 32c/2 + j;
 * NULL = dense). Same arrays, same order. */
static inline uint32_t ktf8_pair_koff(int p) {
  return (uint32_t)(p >> 4) * KTF8_COL_BYTES + (uint32_t)(p & 15) * 128u;
}

static inline int ktf8_collect(const uint16_t *x, int K, float thr,
                               uint32_t *koff, uint32_t *pval, uint32_t *seg) {
  const float t2 = thr * thr;
  const int ncols = K / KTF8_TILE_K;
  int np = 0;
  seg[0] = 0;
  for (int c = 0; c < ncols; ++c) {
    for (int j = 0; j < 16; ++j) {
      const int p = c * 16 + j;
      uint32_t lo = x[2 * p], hi = x[2 * p + 1];
      float a, b;
      uint32_t ua = lo << 16, ub = hi << 16;
      memcpy(&a, &ua, 4);
      memcpy(&b, &ub, 4);
      if (a * a + b * b > t2) {
        koff[np] = ktf8_pair_koff(p);
        pval[np] = lo | (hi << 16);
        ++np;
      }
    }
    seg[c + 1] = (uint32_t)np;
  }
  return np;
}

static inline int ktf8_build_plan(const uint16_t *mask, const uint16_t *x, int K,
                                  uint32_t *koff, uint32_t *pval, uint32_t *seg) {
  const int ncols = K / KTF8_TILE_K;
  int np = 0;
  seg[0] = 0;
  for (int c = 0; c < ncols; ++c) {
    unsigned bits = mask ? mask[c] : 0xFFFFu;
    while (bits) {
      const int j = __builtin_ctz(bits);
      bits &= bits - 1u;
      const int p = c * 16 + j;
      koff[np] = ktf8_pair_koff(p);
      pval[np] = (uint32_t)x[2 * p] | ((uint32_t)x[2 * p + 1] << 16);
      ++np;
    }
    seg[c + 1] = (uint32_t)np;
  }
  return np;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)

typedef struct {
  int c, g, j, live;
} ktf8_cur;

static inline void ktf8_cur_step(ktf8_cur *t, const uint32_t *cs, int ncols) {
  if ((uint32_t)++t->j < cs[t->c + 1]) return;
  if (++t->g < 4) { t->j = (int)cs[t->c]; return; }
  t->g = 0;
  do { ++t->c; } while (t->c < ncols && cs[t->c + 1] == cs[t->c]);
  if (t->c >= ncols) { t->live = 0; return; }
  t->j = (int)cs[t->c];
}

static inline ktf8_cur ktf8_cur_start(const uint32_t *cs, int ncols, int npairs,
                                      int ahead) {
  ktf8_cur t;
  t.c = 0; t.g = 0; t.j = 0; t.live = npairs > 0;
  if (t.live) {
    while (t.c < ncols && cs[t.c + 1] == cs[t.c]) ++t.c;
    if (t.c >= ncols) t.live = 0; else t.j = (int)cs[t.c];
  }
  for (int i = 0; i < ahead && t.live; ++i) ktf8_cur_step(&t, cs, ncols);
  return t;
}

/* 32 e4m3 codes -> 32 exact bf16 values: mantissa 3 bits into bf16's top
 * mantissa bits, exponent rebias 7 -> 127 (+120 << 7 = 0x3C00) on nonzero
 * codes only, sign to bit 15. Requires the encoder convention kt and this
 * repo share: no denormal codes (they flush to zero), no NaN code. */
static inline __m512i ktf8_fp8_32(const uint8_t *p) {
  const __m512i w = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)p));
  const __m512i mag = _mm512_slli_epi16(_mm512_and_si512(w, _mm512_set1_epi16(0x7f)), 4);
  const __m512i sign = _mm512_slli_epi16(_mm512_and_si512(w, _mm512_set1_epi16((short)0x80)), 8);
  const __mmask32 nz = _mm512_test_epi16_mask(mag, mag);
  return _mm512_or_si512(
      sign, _mm512_mask_add_epi16(_mm512_setzero_si512(), nz, mag,
                                  _mm512_set1_epi16(0x3C00)));
}

#define KTF8_PF(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T1)

/* ---- the m=1 decode kernel: y[n0..n1) = x . W for one expert -----------
 * W: ktf8_pack_codes output (N*K bytes). S: transposed fp32 scales
 * ((K/128)*(N/128) floats). koff/pval/seg: the plan. n0/n1 multiples of 256.
 * y is fp32 and OVERWRITTEN. pf_dist in 64 B lines, 0 = no prefetch. */
static inline void ktf8_gemv(const uint8_t *W, const float *S,
                             const uint32_t *koff, const uint32_t *pval,
                             const uint32_t *cs, int npairs, int N, int K,
                             float *y, int n0, int n1, int pf_dist) {
  const int ncols = K / KTF8_TILE_K;
  const int NB = N / KTF8_BLK;
  memset(y + n0, 0, (size_t)(n1 - n0) * sizeof(float));
  for (int ns = n0; ns < n1; ns += KTF8_TILE_N) {
    const uint8_t *const sb = W + (size_t)(ns >> 8) * KTF8_COL_BYTES * (size_t)ncols;
    const int nb0 = ns / KTF8_BLK; /* 64-n group g lies in n-block nb0 + (g >> 1) */
    float *const ys = y + ns;
    ktf8_cur cu = ktf8_cur_start(cs, ncols, npairs, pf_dist > 0 ? pf_dist / 2 : 0);
    if (KTF8_PROLOGUE_PF && pf_dist > 0) {
      ktf8_cur pro = ktf8_cur_start(cs, ncols, npairs, 0);
      for (int i = 0; i < pf_dist / 2 && pro.live; ++i) {
        const uint8_t *const pp = sb + koff[pro.j] + (size_t)pro.g * KTF8_GRP_BYTES;
        KTF8_PF(pp); KTF8_PF(pp + 64);
        ktf8_cur_step(&pro, cs, ncols);
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
            const uint8_t *const pp = sb + koff[cu.j] + (size_t)cu.g * KTF8_GRP_BYTES;
            KTF8_PF(pp); KTF8_PF(pp + 64);
            ktf8_cur_step(&cu, cs, ncols);
          }
          const __m512bh vv = (__m512bh)_mm512_set1_epi32((int)pval[j]);
          const uint8_t *const p = gb + koff[j];
          a0 = _mm512_dpbf16_ps(a0, vv, (__m512bh)ktf8_fp8_32(p));
          a1 = _mm512_dpbf16_ps(a1, vv, (__m512bh)ktf8_fp8_32(p + 32));
          a2 = _mm512_dpbf16_ps(a2, vv, (__m512bh)ktf8_fp8_32(p + 64));
          a3 = _mm512_dpbf16_ps(a3, vv, (__m512bh)ktf8_fp8_32(p + 96));
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

#undef KTF8_PF
#endif /* AVX512F && AVX512BW */

#ifdef __cplusplus
}
#endif
#endif /* KTF8_TILE_PORT_HPP */
