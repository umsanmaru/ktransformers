#pragma once
/**
 * @file k_mask.hpp
 * @brief per-k 마스크(score `k1`, 2026-09-05 사용자 결정)의 AVX-512 구현.
 *
 *     keep_k = |x_k| >= thr
 *
 * 가중치 통계(wn_sq/pair_dot)를 쓰지 않는다 — 입력은 활성 x(bf16)와 thr 하나다. 한 워드
 * = 32 k = zmm 하나(32 bf16). bf16 → fp32는 상위 16비트 배치(`<<16`)로 정확하고, 비교는
 * 부호 비트를 지운 뒤 `>= thr` 하나라 연산 순서 문제가 없다(스칼라/벡터 비트 동일).
 * `thr`이 NaN이면 비교가 전부 거짓 → 전량 skip. thr == 0이면 전량 통과(GPU 측과 같은 결론).
 *
 * `build_pair_mask`(pair_mask.hpp)와 같은 자리에서 moe_base가 부른다 — 커널이 `kMaskPerK`를
 * 선언하면 이쪽, 아니면 페어 쪽 (act_plan.hpp `mask_per_k`).
 *
 * ## score `k1w1` (2026-09-07): 가중치 노름을 곱한 per-k
 *
 *     keep_k = |x_k| · wn_k >= thr        (wn_k = 열 k의 L2 노름)
 *
 * `build_k_mask_w`가 그 판정이고, **제곱으로 비교한다**: `x_k² · a_k >= thr²` where
 * `a = wn²`. 둘 다 음이 아니므로 동치이고, sqrt 32개가 사라진다 (k2wl2가 thr²로
 * 비교하는 것과 같은 이유·같은 규약). 마스크 종류는 커널이 아니라 **주입된 테이블**이
 * 정한다 — `a_sq`가 있으면 k1w1, 없으면 k1이다. 커널 입장에서는 어느 쪽이든 "k 한
 * 행을 건너뛴다"라 같은 코드가 소비한다.
 */

#include <immintrin.h>

#include <cstdint>

#include "llama.cpp/ggml-impl.h"  // ggml_bf16_t, GGML_BF16_TO_FP32

namespace amx {

constexpr int kKPerMaskWord = 32;

/// out[t] 비트 j = k (32t + j) 유지. k_real < k_rows이면 그 차이는 타일 경계까지의 패딩 행 —
/// 비트를 끈다 (thr == 0일 때 서면 쓸모없는 B 로드). src [k_rows]는 패딩까지 읽을 수 있어야
/// 한다 — 로더가 그렇게 굽는다.
inline void build_k_mask(const ggml_bf16_t* src, int k_rows, int k_real, float thr, uint32_t* out) {
  const int words = k_rows / kKPerMaskWord;
#if defined(__AVX512F__)
  // 16 bf16(256 b)씩 k 순서 그대로 fp32로 올린다(cvtepu16 → <<16) — 비트 j가 k j가 되므로
  // 짝/홀 lane을 다시 섞을 필요가 없다 (pdep/BMI2 의존 없음: kt는 -march=native 없이
  // AVX512 플래그만으로도 빌드된다).
  const __m512 vthr = _mm512_set1_ps(thr);
  const __m256i abs16 = _mm256_set1_epi16(0x7fff);
  for (int t = 0; t < words; t++) {
    const __m256i lo16 = _mm256_and_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + kKPerMaskWord * t)), abs16);
    const __m256i hi16 = _mm256_and_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + kKPerMaskWord * t + 16)), abs16);
    const __m512 lo = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo16), 16));  // k 0..15
    const __m512 hi = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi16), 16));  // k 16..31
    const __mmask16 mlo = _mm512_cmp_ps_mask(lo, vthr, _CMP_GE_OQ);
    const __mmask16 mhi = _mm512_cmp_ps_mask(hi, vthr, _CMP_GE_OQ);
    uint32_t bits = (uint32_t)mlo | ((uint32_t)mhi << 16);
    const int valid = k_real - kKPerMaskWord * t;
    if (valid < kKPerMaskWord) {
      bits &= (valid <= 0) ? 0u : ((valid >= 32) ? 0xFFFFFFFFu : ((1u << valid) - 1u));
    }
    out[t] = bits;
  }
#else
  for (int t = 0; t < words; t++) {
    uint32_t bits = 0;
    for (int j = 0; j < kKPerMaskWord; j++) {
      const int k = kKPerMaskWord * t + j;
      if (k >= k_real) break;
      float v = GGML_BF16_TO_FP32(src[k]);
      v = v < 0.0f ? -v : v;
      if (v >= thr) bits |= 1u << j;
    }
    out[t] = bits;
  }
#endif
}

/// score k1w1: out[t] 비트 j = `x_k² · a_k >= thr²` (k = 32t + j). `a`는 열 노름의
/// 제곱 `wn²`이고 **weight와 같은 인덱스로 잘린 것**이어야 한다 (티어 gather 순서).
/// `thr2`는 호출자가 제곱해 넘긴다 — 곡선 조회는 thr이고 제곱은 한 번만 하면 된다.
inline void build_k_mask_w(const ggml_bf16_t* src, int k_rows, int k_real, const float* a_sq,
                           float thr2, uint32_t* out) {
  const int words = k_rows / kKPerMaskWord;
#if defined(__AVX512F__)
  const __m512 vthr = _mm512_set1_ps(thr2);
  const __m256i abs16 = _mm256_set1_epi16(0x7fff);
  for (int t = 0; t < words; t++) {
    const int k0 = kKPerMaskWord * t;
    const __m256i lo16 = _mm256_and_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k0)), abs16);
    const __m256i hi16 = _mm256_and_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + k0 + 16)), abs16);
    const __m512 lo = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo16), 16));
    const __m512 hi = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi16), 16));
    // x² · a — |x|·wn >= thr 와 동치이고 sqrt가 없다.
    const __m512 elo = _mm512_mul_ps(_mm512_mul_ps(lo, lo), _mm512_loadu_ps(a_sq + k0));
    const __m512 ehi = _mm512_mul_ps(_mm512_mul_ps(hi, hi), _mm512_loadu_ps(a_sq + k0 + 16));
    uint32_t bits = (uint32_t)_mm512_cmp_ps_mask(elo, vthr, _CMP_GE_OQ) |
                    ((uint32_t)_mm512_cmp_ps_mask(ehi, vthr, _CMP_GE_OQ) << 16);
    const int valid = k_real - k0;
    if (valid < kKPerMaskWord) {
      bits &= (valid <= 0) ? 0u : ((valid >= 32) ? 0xFFFFFFFFu : ((1u << valid) - 1u));
    }
    out[t] = bits;
  }
#else
  for (int t = 0; t < words; t++) {
    uint32_t bits = 0;
    for (int j = 0; j < kKPerMaskWord; j++) {
      const int k = kKPerMaskWord * t + j;
      if (k >= k_real) break;
      const float v = GGML_BF16_TO_FP32(src[k]);
      if (v * v * a_sq[k] >= thr2) bits |= 1u << j;
    }
    out[t] = bits;
  }
#endif
}

}  // namespace amx
