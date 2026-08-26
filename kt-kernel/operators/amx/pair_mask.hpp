#pragma once
/**
 * @file pair_mask.hpp
 * @brief k2wl2 페어 마스크(계약 ①)의 AVX-512 구현.
 *
 * 한 워드 = 16 페어 = zmm 하나. x는 32 bf16(64 B)을 통째로 들어 shift/and로
 * x0·x1을, a_sq는 두 zmm(32 fp32, (a0,a1) 인터리브)을 even/odd permute로 a0·a1을
 * 얻고, c는 16 fp32 연속이다. 비교 결과 __mmask16이 곧 출력 워드다.
 *
 * **연산 순서는 스칼라 원본이 -ffast-math 아래에서 실제로 컴파일된 결과를 그대로
 * 재현한다** (moe_base.hpp의 옛 build_pair_mask, kt_kernel_ext.so objdump
 * 2026-08-26):
 *
 *     twoc = c + c
 *     e    = fma(a0, x0*x0, fma(a1, x1, twoc*x0) * x1)
 *     e    = max(e, 0)          // vmaxss: NaN이면 0
 *     bit  = (e >= thr2)        // vcomiss + cmovb: ordered GE
 *
 * 마스크 비트는 경계(e ≈ thr2)에서 1 ulp에 갈리므로 순서가 곧 수치 계약이다. 이
 * 파일은 컴파일러가 그 순서를 다시 재결합하지 못하게 fast-math와 fp-contract를
 * 끈다 — intrinsic 하나가 IEEE 연산 하나로 남는다. 그래서 스칼라 판(폴백)과
 * 벡터 판의 출력이 비트 동일하다 (tests/prism_partial_golden.py).
 */

#include <immintrin.h>

#include <cstdint>

#include "llama.cpp/ggml-impl.h"  // ggml_bf16_t, GGML_BF16_TO_FP32

#pragma GCC push_options
#pragma GCC optimize("no-fast-math", "fp-contract=off")

namespace amx {

constexpr int kPairsPerMaskWord = 16;

/// out[t] 비트 j = 페어 (32t + 2j, 32t + 2j + 1) 유지. k_real < k_rows이면 그 차이는
/// 타일 경계까지의 패딩 페어 — 비트를 끈다 (thr == 0일 때 서면 쓸모없는 B 로드).
/// 테이블(a_sq [k_rows], c [k_rows/2])과 src [k_rows]는 패딩까지 읽을 수 있어야 한다
/// — 로더가 그렇게 굽는다.
inline void build_pair_mask(const ggml_bf16_t* src, int k_rows, int k_real, const float* a_sq, const float* c,
                            float thr, uint16_t* out) {
  const float thr2 = thr * thr;
  const int pairs = k_rows / 2;
  const int pairs_real = k_real / 2;
  const int words = pairs / kPairsPerMaskWord;
#if defined(__AVX512F__)
  const __m512 vthr2 = _mm512_set1_ps(thr2);
  const __m512 zero = _mm512_setzero_ps();
  const __m512i hi16 = _mm512_set1_epi32((int)0xFFFF0000);
  const __m512i even = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  const __m512i odd = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
  for (int t = 0; t < words; t++) {
    // x: 32 bf16 → 16 × (x0 | x1 << 16). bf16 → fp32는 상위 16비트에 놓는 것.
    const __m512i xv = _mm512_loadu_si512(src + 2 * kPairsPerMaskWord * t);
    const __m512 x0 = _mm512_castsi512_ps(_mm512_slli_epi32(xv, 16));
    const __m512 x1 = _mm512_castsi512_ps(_mm512_and_si512(xv, hi16));
    // a_sq: (a0, a1) 인터리브 32개 → even/odd 분리
    const __m512 a_lo = _mm512_loadu_ps(a_sq + 2 * kPairsPerMaskWord * t);
    const __m512 a_hi = _mm512_loadu_ps(a_sq + 2 * kPairsPerMaskWord * t + 16);
    const __m512 a0 = _mm512_permutex2var_ps(a_lo, even, a_hi);
    const __m512 a1 = _mm512_permutex2var_ps(a_lo, odd, a_hi);
    const __m512 cv = _mm512_loadu_ps(c + kPairsPerMaskWord * t);

    const __m512 twoc = _mm512_add_ps(cv, cv);
    const __m512 x0sq = _mm512_mul_ps(x0, x0);
    __m512 e = _mm512_mul_ps(twoc, x0);
    e = _mm512_fmadd_ps(a1, x1, e);
    e = _mm512_mul_ps(e, x1);
    e = _mm512_fmadd_ps(a0, x0sq, e);
    e = _mm512_max_ps(e, zero);  // 두 번째 피연산자가 NaN 대체 — vmaxss와 같다
    __mmask16 bits = _mm512_cmp_ps_mask(e, vthr2, _CMP_GE_OQ);

    const int valid = pairs_real - kPairsPerMaskWord * t;  // 이 워드의 실제 페어 수
    if (valid < kPairsPerMaskWord) {
      bits &= (valid <= 0) ? (__mmask16)0 : (__mmask16)((1u << valid) - 1u);
    }
    out[t] = (uint16_t)bits;
  }
#else
  // 스칼라 폴백 — 위와 같은 순서 (이 파일은 fast-math가 꺼져 있어 그대로 남는다).
  for (int t = 0; t < words; t++) {
    uint16_t bits = 0;
    const int base = t * kPairsPerMaskWord;
    for (int j = 0; j < kPairsPerMaskWord; j++) {
      if (base + j >= pairs_real) {
        break;
      }
      const int p = base + j;
      const float x0 = GGML_BF16_TO_FP32(src[2 * p]);
      const float x1 = GGML_BF16_TO_FP32(src[2 * p + 1]);
      const float twoc = c[p] + c[p];
      float e = __builtin_fmaf(a_sq[2 * p], x0 * x0, __builtin_fmaf(a_sq[2 * p + 1], x1, twoc * x0) * x1);
      e = (e > 0.0f) ? e : 0.0f;  // NaN → 0
      if (e >= thr2) {
        bits |= (uint16_t)1 << j;
      }
    }
    out[t] = bits;
  }
#endif
}

}  // namespace amx

#pragma GCC pop_options
