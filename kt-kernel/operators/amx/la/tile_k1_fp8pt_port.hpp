/* la/tile_k1_fp8pt_port.hpp — the k=1 x per-tensor cross: fp8 e4m3 weights with
 * ONE fp32 scale per matrix (Mistral-Medium-3.5 "static-tensor" FP8) on the k1
 * tile layout (64 B line = one k row x 64 n).
 *
 * Everything layout-shaped is shared with the k1 b128 port
 * (tile_k1_fp8b128_port.hpp): the byte permutation `ktf8k1_off`, the packer
 * `ktf8k1_pack_codes[_range]`, the per-k plan encoder `ktf8k1_build_plan`, the
 * prefetch cursor `ktf8k1_cur_*` and the decode+FMA of one line
 * `ktf8k1_line_fma`. What differs is exactly what differed between the two k2
 * members (b128 vs pt): **there is no scale block to flush on**, so the 16 fp32
 * accumulators of a 256-n super (4 groups x 64 n) stay in registers across ALL
 * tile columns and are scaled once when the super is done.
 *
 * So this file is the second of two independent axes crossing:
 *
 *            |  128x128 block scale        |  per-tensor scale
 *   k=2 pair |  ktf8_gemv                  |  ktf8pt_gemv
 *   k=1 row  |  ktf8k1_gemv                |  ktf8ptk1_gemv   <- here
 *
 * Compared with ktf8k1_gemv (b128) per 8 KB tile column: no 4 x (load y, fmadd,
 * store y) flush, no scale-table load, no memset of y, and K need only be a
 * multiple of 32 (tile column) instead of 128 (scale block).
 *
 * Numerics: fp32 broadcast activation x exact e4m3 value, fp32 FMA over the
 * whole K, then y = acc * s once. Same W8A16 contract as kt's other fp8
 * kernels; only the accumulation grouping differs (K-wide instead of per
 * 128-k block). The checkpoint's static `activation_scale` is NOT used — the
 * activation stays bf16 on the CPU.
 */
#ifndef KTF8PTK1_TILE_PORT_HPP
#define KTF8PTK1_TILE_PORT_HPP

#include "tile_k1_fp8b128_port.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)

#define KTF8PTK1_PF(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T1)

/* y[n0..n1) = s * (x . W) for one matrix, m = 1 decode.
 * W: ktf8k1_pack_codes output (N*K bytes). s: the per-tensor fp32 scale.
 * koff/pval/seg: the per-k plan (ktf8k1_build_plan). n0/n1 multiples of 256.
 * y is fp32 and OVERWRITTEN. pf_dist in 64 B lines, 0 = no prefetch. */
static inline void ktf8ptk1_gemv(const uint8_t *W, float s, const uint32_t *koff,
                                 const uint32_t *pval, const uint32_t *cs, int nkept, int N, int K,
                                 float *y, int n0, int n1, int pf_dist) {
  (void)N;
  const int ncols = K / KTF8_TILE_K;
  const __m512 vs = _mm512_set1_ps(s);
  for (int ns = n0; ns < n1; ns += KTF8_TILE_N) {
    const uint8_t *const sb = W + (size_t)(ns >> 8) * KTF8_COL_BYTES * (size_t)ncols;
    float *const ys = y + ns;
    ktf8k1_cur cu = ktf8k1_cur_start(cs, ncols, nkept, pf_dist > 0 ? pf_dist : 0);
    if (KTF8K1_PROLOGUE_PF && pf_dist > 0) {
      ktf8k1_cur pro = ktf8k1_cur_start(cs, ncols, nkept, 0);
      for (int i = 0; i < pf_dist && pro.live; ++i) {
        KTF8PTK1_PF(sb + koff[pro.j] + (size_t)pro.g * KTF8_GRP_BYTES);
        ktf8k1_cur_step(&pro, cs, ncols);
      }
    }
    /* 16 accumulators = the whole 256-n super, live across every column. */
    __m512 a00 = _mm512_setzero_ps(), a01 = _mm512_setzero_ps(), a02 = _mm512_setzero_ps(), a03 = _mm512_setzero_ps();
    __m512 a10 = _mm512_setzero_ps(), a11 = _mm512_setzero_ps(), a12 = _mm512_setzero_ps(), a13 = _mm512_setzero_ps();
    __m512 a20 = _mm512_setzero_ps(), a21 = _mm512_setzero_ps(), a22 = _mm512_setzero_ps(), a23 = _mm512_setzero_ps();
    __m512 a30 = _mm512_setzero_ps(), a31 = _mm512_setzero_ps(), a32 = _mm512_setzero_ps(), a33 = _mm512_setzero_ps();
    for (int c = 0; c < ncols; ++c) {
      const int jlo = (int)cs[c], jhi = (int)cs[c + 1];
      if (jlo == jhi) continue;
      /* group g: n [64g, 64g+64) of this super; its chunk is +g*2 KB inside the column.
       * Same (c, g, j) consumption order as ktf8k1_gemv so the shared cursor is exact. */
#define KTF8PTK1_GROUP(G, A0, A1, A2, A3)                                                  \
      {                                                                                   \
        const uint8_t *const gb = sb + (size_t)(G) * KTF8_GRP_BYTES;                      \
        for (int j = jlo; j < jhi; ++j) {                                                 \
          if (pf_dist > 0 && cu.live) {                                                   \
            KTF8PTK1_PF(sb + koff[cu.j] + (size_t)cu.g * KTF8_GRP_BYTES);                 \
            ktf8k1_cur_step(&cu, cs, ncols);                                              \
          }                                                                               \
          const __m512 v = _mm512_castsi512_ps(_mm512_set1_epi32((int)pval[j]));          \
          ktf8k1_line_fma(gb + koff[j], v, &A0, &A1, &A2, &A3);                           \
        }                                                                                 \
      }
      KTF8PTK1_GROUP(0, a00, a01, a02, a03)
      KTF8PTK1_GROUP(1, a10, a11, a12, a13)
      KTF8PTK1_GROUP(2, a20, a21, a22, a23)
      KTF8PTK1_GROUP(3, a30, a31, a32, a33)
#undef KTF8PTK1_GROUP
    }
    /* one scale, one store per 16 n — overwrites y (no memset needed). */
    _mm512_storeu_ps(ys + 0, _mm512_mul_ps(a00, vs));
    _mm512_storeu_ps(ys + 16, _mm512_mul_ps(a01, vs));
    _mm512_storeu_ps(ys + 32, _mm512_mul_ps(a02, vs));
    _mm512_storeu_ps(ys + 48, _mm512_mul_ps(a03, vs));
    _mm512_storeu_ps(ys + 64, _mm512_mul_ps(a10, vs));
    _mm512_storeu_ps(ys + 80, _mm512_mul_ps(a11, vs));
    _mm512_storeu_ps(ys + 96, _mm512_mul_ps(a12, vs));
    _mm512_storeu_ps(ys + 112, _mm512_mul_ps(a13, vs));
    _mm512_storeu_ps(ys + 128, _mm512_mul_ps(a20, vs));
    _mm512_storeu_ps(ys + 144, _mm512_mul_ps(a21, vs));
    _mm512_storeu_ps(ys + 160, _mm512_mul_ps(a22, vs));
    _mm512_storeu_ps(ys + 176, _mm512_mul_ps(a23, vs));
    _mm512_storeu_ps(ys + 192, _mm512_mul_ps(a30, vs));
    _mm512_storeu_ps(ys + 208, _mm512_mul_ps(a31, vs));
    _mm512_storeu_ps(ys + 224, _mm512_mul_ps(a32, vs));
    _mm512_storeu_ps(ys + 240, _mm512_mul_ps(a33, vs));
  }
}

#undef KTF8PTK1_PF
#endif /* AVX512F && AVX512BW */

#ifdef __cplusplus
}
#endif
#endif /* KTF8PTK1_TILE_PORT_HPP */
