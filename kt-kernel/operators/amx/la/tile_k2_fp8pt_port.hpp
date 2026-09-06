/* la/tile_k2_fp8pt_port.hpp — fp8 e4m3 weights with ONE fp32 scale per matrix
 * (Mistral-Medium-3.5 "static-tensor" FP8: `weight_block_size: null`,
 * `weight_scale_inv` is a scalar per linear) on the tile_k2 fp8 layout.
 *
 * Everything that is layout-only is shared with the b128 port
 * (tile_k2_fp8b128_port.hpp): the byte permutation `ktf8_off`, the code
 * packer `ktf8_pack_codes[_range]`, the plan encoder `ktf8_build_plan`, the
 * prefetch cursor `ktf8_cur_*` and the e4m3 -> bf16 decoder `ktf8_fp8_32`.
 * Only the consumer differs, and it differs in one place: there is no scale
 * block to flush on, so the 16 fp32 accumulators (4 groups x 64 n) of a 256-n
 * super stay in registers across ALL tile columns and are scaled exactly once
 * when the super is done.
 *
 * Compared with ktf8_gemv (b128) per 8 KB tile column:
 *   - no 12 x (load y, fmadd, store y) flush, no scale-table load
 *   - no memset of y (the final store overwrites)
 *   - K need only be a multiple of 32 (tile column), not 128 (scale block) —
 *     the prism tier cut for this format is pair-aligned like bf16.
 *
 * Numerics: bf16 activation x exact e4m3 value, fp32 dpbf16 accumulation over
 * the whole K, then y = acc * s once. Same W8A16 contract as kt's fp8 kernels;
 * only the accumulation grouping differs (K-wide instead of per 128-k block).
 * The checkpoint's static `activation_scale` is NOT used: this is a bf16
 * activation kernel, the activation is never quantised to e4m3 on the CPU.
 */
#ifndef KTF8PT_TILE_PORT_HPP
#define KTF8PT_TILE_PORT_HPP

#include "tile_k2_fp8b128_port.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512BF16__)

#define KTF8PT_PF(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T1)

/* y[n0..n1) = s * (x . W) for one matrix, m = 1 decode.
 * W: ktf8_pack_codes output (N*K bytes). s: the per-tensor fp32 scale.
 * koff/pval/seg: the plan (ktf8_build_plan). n0/n1 multiples of 256.
 * y is fp32 and OVERWRITTEN. pf_dist in 64 B lines, 0 = no prefetch. */
static inline void ktf8pt_gemv(const uint8_t *W, float s, const uint32_t *koff,
                               const uint32_t *pval, const uint32_t *cs, int npairs,
                               int N, int K, float *y, int n0, int n1, int pf_dist) {
  (void)N;
  const int ncols = K / KTF8_TILE_K;
  const __m512 vs = _mm512_set1_ps(s);
  for (int ns = n0; ns < n1; ns += KTF8_TILE_N) {
    const uint8_t *const sb = W + (size_t)(ns >> 8) * KTF8_COL_BYTES * (size_t)ncols;
    float *const ys = y + ns;
    ktf8_cur cu = ktf8_cur_start(cs, ncols, npairs, pf_dist > 0 ? pf_dist / 2 : 0);
    if (KTF8_PROLOGUE_PF && pf_dist > 0) {
      ktf8_cur pro = ktf8_cur_start(cs, ncols, npairs, 0);
      for (int i = 0; i < pf_dist / 2 && pro.live; ++i) {
        const uint8_t *const pp = sb + koff[pro.j] + (size_t)pro.g * KTF8_GRP_BYTES;
        KTF8PT_PF(pp); KTF8PT_PF(pp + 64);
        ktf8_cur_step(&pro, cs, ncols);
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
       * Same (c, g, j) consumption order as ktf8_gemv so the shared cursor is exact. */
#define KTF8PT_GROUP(G, A0, A1, A2, A3)                                                    \
      {                                                                                   \
        const uint8_t *const gb = sb + (size_t)(G) * KTF8_GRP_BYTES;                      \
        for (int j = jlo; j < jhi; ++j) {                                                 \
          if (pf_dist > 0 && cu.live) {                                                   \
            const uint8_t *const pp = sb + koff[cu.j] + (size_t)cu.g * KTF8_GRP_BYTES;    \
            KTF8PT_PF(pp); KTF8PT_PF(pp + 64);                                            \
            ktf8_cur_step(&cu, cs, ncols);                                                \
          }                                                                               \
          const __m512bh vv = (__m512bh)_mm512_set1_epi32((int)pval[j]);                  \
          const uint8_t *const p = gb + koff[j];                                          \
          A0 = _mm512_dpbf16_ps(A0, vv, (__m512bh)ktf8_fp8_32(p));                        \
          A1 = _mm512_dpbf16_ps(A1, vv, (__m512bh)ktf8_fp8_32(p + 32));                   \
          A2 = _mm512_dpbf16_ps(A2, vv, (__m512bh)ktf8_fp8_32(p + 64));                   \
          A3 = _mm512_dpbf16_ps(A3, vv, (__m512bh)ktf8_fp8_32(p + 96));                   \
        }                                                                                 \
      }
      KTF8PT_GROUP(0, a00, a01, a02, a03)
      KTF8PT_GROUP(1, a10, a11, a12, a13)
      KTF8PT_GROUP(2, a20, a21, a22, a23)
      KTF8PT_GROUP(3, a30, a31, a32, a33)
#undef KTF8PT_GROUP
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

#undef KTF8PT_PF
#endif /* AVX512F && AVX512BW && AVX512BF16 */

#ifdef __cplusplus
}
#endif
#endif /* KTF8PT_TILE_PORT_HPP */
