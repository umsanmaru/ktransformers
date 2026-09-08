/* port/kt_tile_k2_mxfp4.hpp — self-contained port surface of
 * avx512_tile_k2_mxfp4 (the "union" mxfp4 kernel, FINDINGS §13) for
 * kt-kernel. No mmp dependencies: byte arithmetic instead of layout tables,
 * C and C++20 compatible. The kt side wraps these functions into its
 * Buffer/kernel structs (see port/PORTING_MXFP4.md); this file is the whole
 * compute payload.
 *
 *   ktm4_pack_codes()        BufferB body: kt's row-major nibble rows
 *                            ([n][K/2], low nibble = even k — kt's BufferB
 *                            convention) -> fp4 tile layout. A pure BYTE
 *                            permutation: the two nibbles of a byte stay
 *                            together because the pair bit is the layout's
 *                            innermost bit.
 *   ktm4_pack_scales_f32()   kt's fp32 group scales d[n][K/32] -> transposed
 *                            E8M0 bytes [K/32][N]. Scales must be powers of
 *                            two (MXFP4 contract); returns the count that
 *                            were not, so the caller can refuse the weight.
 *   ktm4_pack_scales_bf16()  same from bf16 bits (kt's native checkpoint
 *                            scale dtype before its convert_or_copy).
 *   ktm4_collect()           BufferA-side: bf16 activation -> kept pair list
 *                            (pair-L2 rule) + per-tile-column prefix index.
 *   ktm4_gemv()              one n-range of one expert, m=1 decode.
 *
 * Tile layout (element index bits, innermost first):
 *   k:1/1 | n:6/2 | k:4/128 | n:2/2048 | k-cols/8192 | n-supers/8192*(K/32)
 *   off(n,k) = (k&1) + (n&63)*2 + ((k>>1)&15)*128 + ((n>>6)&3)*2048
 *            + (k>>5)*8192 + (n>>8)*8192*(K/32)          (elements; byte = off>>1)
 * One 64 B line = one k-pair x 64 n (128 nibbles). Tile column = 32 k x 256 n
 * = 4 KB of codes (4 groups of 64 n x 16 pairs) + 256 scale bytes, one E8M0
 * per (n, 32-k block), stored TRANSPOSED: scale[(k/32)*N + n].
 *
 * Numerics: W4A16 exactly as kt's own fp4 path — bf16 activation x exact
 * e2m1 value, fp32 accumulate, power-of-two scale applied once per 32-k
 * block (distributive: sum(c_k * 2^s * x_k) = 2^s * sum(c_k * x_k)). Only the
 * accumulation order differs from a row-wise reference (fp32 rounding).
 *
 * Sparsity: pair-L2 rule (x_2p^2 + x_2p+1^2 > thr^2); a skipped pair skips a
 * whole 64 B line per n-group. thr = 0 keeps every pair with a nonzero half
 * = dense-safe. Verification contract identical to kernels/common/collect.h.
 *
 * Prefetch: consumption-order cursor pf_dist lines ahead (T1), the NEXT kept
 * column's four scale lines at every column head, and — on by default,
 * KTM4_PROLOGUE_PF — a burst for the first pf_dist lines of every 256-n
 * super-block plus the first kept column's scale lines. All three are
 * measured (FINDINGS §13: scale prefetch 1.4-1.6x; prologue -11% dense /
 * -15% pair-sp50 at N=4096 on the HW-prefetch-off socket). pf_dist 128 is
 * the measured winner; 0 disables software prefetch entirely (on a socket
 * with HW prefetchers off that is the whole prefetch story — do not ship 0).
 */
#ifndef KTM4_TILE_PORT_HPP
#define KTM4_TILE_PORT_HPP

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KTM4_TILE_N 256
#define KTM4_TILE_K 32
#define KTM4_COL_BYTES 4096 /* 32 k x 256 n nibbles */
#define KTM4_GRP_BYTES 1024 /* 16 pairs x 64 n nibbles */
#ifndef KTM4_PROLOGUE_PF
#define KTM4_PROLOGUE_PF 1
#endif

/* element offset of (n, k) in the fp4 tile layout for a K-column matrix */
static inline size_t ktm4_off(int n, int k, int K) {
  return (size_t)(k & 1) + (size_t)(n & 63) * 2 + (size_t)((k >> 1) & 15) * 128 +
         (size_t)((n >> 6) & 3) * 2048 + (size_t)(k >> 5) * 8192 +
         (size_t)(n >> 8) * 8192 * (size_t)(K / 32);
}

/* ---- BufferB body: codes ------------------------------------------------
 * src: kt row-major nibble rows, src[n*(K/2) + p] holds k=2p (low nibble)
 * and k=2p+1 (high nibble). dst: N*K/2 bytes in tile layout. */
/* The _range forms pack rows [n0, n1) only, so a kt worker split over n
 * (from_raw_mat(src, ith, nth)) can pack its own slice; dst/src are the
 * WHOLE-matrix bases in both forms. */
static inline void ktm4_pack_codes_range(uint8_t *dst, const uint8_t *src,
                                         int N, int K, int n0, int n1) {
  const size_t rowb = (size_t)K / 2;
  for (int n = n0; n < n1; ++n)
    for (int p = 0; p < K / 2; ++p)
      dst[ktm4_off(n, 2 * p, K) >> 1] = src[(size_t)n * rowb + (size_t)p];
  (void)N;
}
static inline void ktm4_pack_codes(uint8_t *dst, const uint8_t *src, int N,
                                   int K) {
  ktm4_pack_codes_range(dst, src, N, K, 0, N);
}

/* ---- BufferB body: scales ---------------------------------------------- */
/* fp32 power-of-two -> E8M0 byte (the fp32 exponent field). Non-power-of-two
 * inputs are truncated to their exponent and COUNTED in the return value. */
static inline int ktm4_pack_scales_f32_range(uint8_t *dst, const float *src,
                                             int N, int K, int n0, int n1) {
  const int G = K / 32;
  int bad = 0;
  for (int n = n0; n < n1; ++n)
    for (int g = 0; g < G; ++g) {
      uint32_t u;
      memcpy(&u, &src[(size_t)n * G + g], 4);
      if ((u & 0x807FFFFFu) != 0u) ++bad; /* sign or mantissa bits set */
      dst[(size_t)g * N + n] = (uint8_t)((u >> 23) & 0xFFu);
    }
  return bad;
}
static inline int ktm4_pack_scales_f32(uint8_t *dst, const float *src, int N,
                                       int K) {
  return ktm4_pack_scales_f32_range(dst, src, N, K, 0, N);
}
/* bf16 bits (kt checkpoint scale dtype): exponent field is bits 14..7 */
static inline int ktm4_pack_scales_bf16_range(uint8_t *dst, const uint16_t *src,
                                              int N, int K, int n0, int n1) {
  const int G = K / 32;
  int bad = 0;
  for (int n = n0; n < n1; ++n)
    for (int g = 0; g < G; ++g) {
      const uint16_t b = src[(size_t)n * G + g];
      if ((b & 0x807Fu) != 0u) ++bad;
      dst[(size_t)g * N + n] = (uint8_t)((b >> 7) & 0xFFu);
    }
  return bad;
}
static inline int ktm4_pack_scales_bf16(uint8_t *dst, const uint16_t *src,
                                        int N, int K) {
  return ktm4_pack_scales_bf16_range(dst, src, N, K, 0, N);
}
/* already-E8M0 bytes laid [n][K/32] -> transposed [K/32][N] */
static inline void ktm4_pack_scales_e8m0(uint8_t *dst, const uint8_t *src,
                                         int N, int K) {
  const int G = K / 32;
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < G; ++g) dst[(size_t)g * N + n] = src[(size_t)n * G + g];
}

/* ---- the activation plan (kt Seam B shape) -------------------------------
 * The kernel consumes a per-token plan of three arrays, caller-allocated:
 *   koff [K/2]      BYTE offset of each kept pair's line inside its 256-n
 *                   super, n-group 0: (p>>4)*4096 + (p&15)*64, k-ascending
 *   pval [K/2]      the pair's two bf16 activations, lo = x[2p], hi = x[2p+1]
 *   seg  [K/32+1]   tile-column prefix: column c owns pairs [seg[c], seg[c+1])
 * Two builders produce it: ktm4_collect (this repo's pair-L2 threshold rule,
 * from the activation alone) and ktm4_build_plan (from kt's absolute-k
 * PairMask, one uint16 per 32-k column, bit j = pair (32c+2j, 32c+2j+1);
 * mask == NULL means dense). Same arrays, same order — a mask that encodes
 * the pair-L2 rule yields a bit-identical plan (tile_port_mxfp4_test). */
static inline uint32_t ktm4_pair_koff(int p) {
  return (uint32_t)(p >> 4) * KTM4_COL_BYTES + (uint32_t)(p & 15) * 64u;
}

static inline int ktm4_collect(const uint16_t *x, int K, float thr,
                               uint32_t *koff, uint32_t *pval, uint32_t *seg) {
  const float t2 = thr * thr;
  const int ncols = K / KTM4_TILE_K;
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
        koff[np] = ktm4_pair_koff(p);
        pval[np] = lo | (hi << 16);
        ++np;
      }
    }
    seg[c + 1] = (uint32_t)np;
  }
  return np;
}

static inline int ktm4_build_plan(const uint16_t *mask, const uint16_t *x, int K,
                                  uint32_t *koff, uint32_t *pval, uint32_t *seg) {
  const int ncols = K / KTM4_TILE_K;
  int np = 0;
  seg[0] = 0;
  for (int c = 0; c < ncols; ++c) {
    unsigned bits = mask ? mask[c] : 0xFFFFu;
    while (bits) {
      const int j = __builtin_ctz(bits);
      bits &= bits - 1u; /* ascending bit order = ascending k */
      const int p = c * 16 + j;
      koff[np] = ktm4_pair_koff(p);
      pval[np] = (uint32_t)x[2 * p] | ((uint32_t)x[2 * p + 1] << 16);
      ++np;
    }
    seg[c + 1] = (uint32_t)np;
  }
  return np;
}

#if defined(__AVX512F__) && defined(__AVX512BW__)

/* Consumption-order cursor over (column, n-group of 4, kept pair) lines. */
typedef struct {
  int c, g, j, live;
} ktm4_cur;

static inline void ktm4_cur_step(ktm4_cur *t, const uint32_t *cs, int ncols) {
  if ((uint32_t)++t->j < cs[t->c + 1]) return;
  if (++t->g < 4) { t->j = (int)cs[t->c]; return; }
  t->g = 0;
  do { ++t->c; } while (t->c < ncols && cs[t->c + 1] == cs[t->c]);
  if (t->c >= ncols) { t->live = 0; return; }
  t->j = (int)cs[t->c];
}

static inline ktm4_cur ktm4_cur_start(const uint32_t *cs, int ncols, int npairs,
                                      int ahead) {
  ktm4_cur t;
  t.c = 0; t.g = 0; t.j = 0; t.live = npairs > 0;
  if (t.live) {
    while (t.c < ncols && cs[t.c + 1] == cs[t.c]) ++t.c;
    if (t.c >= ncols) t.live = 0; else t.j = (int)cs[t.c];
  }
  for (int i = 0; i < ahead && t.live; ++i) ktm4_cur_step(&t, cs, ncols);
  return t;
}

/* bf16 of {0,.5,1,1.5,2,3,4,6} then their negations, indexed by the raw
 * 4-bit code (bit 3 = sign); entries 16..31 unused. */
static inline __m512i ktm4_e2m1_lut(void) {
  return _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          (short)0xC0C0, (short)0xC080, (short)0xC040,
                          (short)0xC000, (short)0xBFC0, (short)0xBF80,
                          (short)0xBF00, (short)0x8000, 0x40C0, 0x4080,
                          0x4040, 0x4000, 0x3FC0, 0x3F80, 0x3F00, 0x0000);
}

/* 16 bytes = 32 nibble codes -> 32 bf16 values, element order preserved:
 * word 0x00HL -> 0x0H0L (bytes L,H = even,odd element), then one vpermw. */
static inline __m512i ktm4_fp4_32(const uint8_t *p, __m512i lut) {
  const __m256i w = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p));
  const __m256i spread = _mm256_or_si256(
      _mm256_and_si256(w, _mm256_set1_epi16(0x000F)),
      _mm256_slli_epi16(_mm256_and_si256(w, _mm256_set1_epi16(0x00F0)), 4));
  return _mm512_permutexvar_epi16(_mm512_cvtepu8_epi16(spread), lut);
}

/* E8M0 bytes -> fp32 scale vector: exponent byte into the fp32 exponent
 * field (value 2^(b-127)); 16 lanes per call. */
static inline __m512 ktm4_e8m0_16(const uint8_t *p) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)p)), 23));
}

#define KTM4_PF(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T1)

/* ---- the m=1 decode kernel: y[n0..n1) = x . W for one expert -----------
 * W: ktm4_pack_codes output (N*K/2 bytes). S: transposed E8M0 scales
 * (N*K/32 bytes). koff/pval/seg: the plan (ktm4_collect or ktm4_build_plan).
 * n0/n1 multiples of 256 (the (ith,nth) split hands each worker whole
 * supers). y is fp32 and is OVERWRITTEN. pf_dist in 64 B lines of the
 * consumption stream, 0 = off. */
static inline void ktm4_gemv(const uint8_t *W, const uint8_t *S,
                             const uint32_t *koff, const uint32_t *pval,
                             const uint32_t *cs, int npairs, int N, int K,
                             float *y, int n0, int n1, int pf_dist) {
  const int ncols = K / KTM4_TILE_K;
  const __m512i lut = ktm4_e2m1_lut();
  memset(y + n0, 0, (size_t)(n1 - n0) * sizeof(float));
  for (int ns = n0; ns < n1; ns += KTM4_TILE_N) {
    const uint8_t *const sb = W + (size_t)(ns >> 8) * KTM4_COL_BYTES * (size_t)ncols;
    const uint8_t *const sc_row = S + (size_t)ns;
    float *const ys = y + ns;
    ktm4_cur cu = ktm4_cur_start(cs, ncols, npairs, pf_dist > 0 ? pf_dist : 0);
    if (KTM4_PROLOGUE_PF && pf_dist > 0) {
      /* the cursor starts pf_dist lines in; cover what it skipped */
      ktm4_cur pro = ktm4_cur_start(cs, ncols, npairs, 0);
      if (pro.live) {
        const uint8_t *const s0 = sc_row + (size_t)pro.c * (size_t)N;
        KTM4_PF(s0); KTM4_PF(s0 + 64); KTM4_PF(s0 + 128); KTM4_PF(s0 + 192);
      }
      for (int i = 0; i < pf_dist && pro.live; ++i) {
        KTM4_PF(sb + koff[pro.j] + (size_t)pro.g * KTM4_GRP_BYTES);
        ktm4_cur_step(&pro, cs, ncols);
      }
    }
    for (int c = 0; c < ncols; ++c) {
      const int jlo = (int)cs[c], jhi = (int)cs[c + 1];
      if (jlo == jhi) continue;
      const uint8_t *const sc = sc_row + (size_t)c * (size_t)N;
      if (pf_dist > 0) { /* next kept column's four scale lines */
        int cn = c + 1;
        while (cn < ncols && cs[cn + 1] == cs[cn]) ++cn;
        const uint8_t *const scn = sc_row + (size_t)(cn < ncols ? cn : c) * (size_t)N;
        KTM4_PF(scn); KTM4_PF(scn + 64); KTM4_PF(scn + 128); KTM4_PF(scn + 192);
      }
      for (int g = 0; g < 4; ++g) {
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        const uint8_t *const gb = sb + (size_t)g * KTM4_GRP_BYTES;
        for (int j = jlo; j < jhi; ++j) {
          if (pf_dist > 0 && cu.live) {
            KTM4_PF(sb + koff[cu.j] + (size_t)cu.g * KTM4_GRP_BYTES);
            ktm4_cur_step(&cu, cs, ncols);
          }
          const __m512bh vv = (__m512bh)_mm512_set1_epi32((int)pval[j]);
          const uint8_t *const p = gb + koff[j];
          a0 = _mm512_dpbf16_ps(a0, vv, (__m512bh)ktm4_fp4_32(p, lut));
          a1 = _mm512_dpbf16_ps(a1, vv, (__m512bh)ktm4_fp4_32(p + 16, lut));
          a2 = _mm512_dpbf16_ps(a2, vv, (__m512bh)ktm4_fp4_32(p + 32, lut));
          a3 = _mm512_dpbf16_ps(a3, vv, (__m512bh)ktm4_fp4_32(p + 48, lut));
        }
        float *const yg = ys + 64 * g;
        const uint8_t *const sg = sc + 64 * g;
        _mm512_storeu_ps(yg, _mm512_fmadd_ps(a0, ktm4_e8m0_16(sg), _mm512_loadu_ps(yg)));
        _mm512_storeu_ps(yg + 16, _mm512_fmadd_ps(a1, ktm4_e8m0_16(sg + 16), _mm512_loadu_ps(yg + 16)));
        _mm512_storeu_ps(yg + 32, _mm512_fmadd_ps(a2, ktm4_e8m0_16(sg + 32), _mm512_loadu_ps(yg + 32)));
        _mm512_storeu_ps(yg + 48, _mm512_fmadd_ps(a3, ktm4_e8m0_16(sg + 48), _mm512_loadu_ps(yg + 48)));
      }
    }
  }
}

#undef KTM4_PF
#endif /* AVX512F && AVX512BW */

#ifdef __cplusplus
}
#endif
#endif /* KTM4_TILE_PORT_HPP */
