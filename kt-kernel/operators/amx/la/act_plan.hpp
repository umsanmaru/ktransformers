#pragma once
/**
 * @file act_plan.hpp
 * @brief 레이아웃 무관 per-token 사이드 채널 (KT_MERGE_CONTRACT.md Seam B).
 *
 * 살아남은 k를 고르는 판정은 두 조각이다 — **정책**(임계값이 어디서 오는가)과
 * **인코딩**(살아남은 페어를 어떤 자료구조로 커널에 넘기는가). 정책은 operator
 * 것이고, 인코딩은 자기 blocking을 아는 커널 것이다. 이 헤더가 그 선이다:
 * operator는 절대 k만 다루는 `PairMask`를 채우고, 커널이 `build_plan`으로
 * 자기 타일 기하에 맞는 `ActivationPlan`을 만든다.
 *
 * 부수 효과 하나: `pval`이 페어 값을 직접 나르므로 sparse 경로에서 **BufferA
 * 접근이 사라진다**.
 */

#include <cstdint>
#include <type_traits>

#include "llama.cpp/ggml-impl.h"  // ggml_bf16_t (아래 trait의 decltype이 쓴다)

namespace amx {

/// 절대 k 기준 keep 마스크. uint16 하나가 연속된 32 k를 덮고, 비트 j가
/// k-페어 (32t + 2j, 32t + 2j + 1)을 유지한다. 어떤 packing도 참조하지 않는다
/// — 마스크는 "어떤 k가 살아남는가"이고 그것은 활성값과 임계값의 성질이다.
using PairMask = uint16_t;  // mask[k / 32]

/// 절대 k 기준 **per-k** keep 마스크. uint32 하나가 연속된 32 k를 덮고, 비트 j가
/// k = 32t + j 하나를 유지한다. 페어 마스크와 같은 격자(32-k 워드)라 moe_base의
/// 워드 수 산정(k / K_STEP)이 두 마스크에 같이 맞는다. skip 단위가 k 한 행인 커널
/// (tile_k1 계열)이 요구한다 — 판정은 `|x_k| >= thr` 하나뿐이고 가중치 통계를 쓰지 않는다.
using KMask = uint32_t;  // mask[k / 32]

/// 커널이 자기 blocking으로 인코딩한 결과. nullptr 필드는 dense를 뜻한다.
struct ActivationPlan {
  const uint32_t* koff = nullptr;  ///< 유지된 페어의 weight 원소 오프셋, k 오름차순
  const uint32_t* pval = nullptr;  ///< 같은 순서의 packed bf16 (lo | hi << 16)
  const uint32_t* seg = nullptr;   ///< 타일 컬럼 접두: 컬럼 c는 [seg[c], seg[c+1])
  int npairs = 0;
  int nseg = 0;
  /// 마지막 **부분** n-블록용 오프셋. 블록 폭이 커널의 N_BLOCK보다 좁으면 타일
  /// 컬럼 stride가 달라지므로 오프셋 표가 한 벌 더 필요하다. `pval`/`seg`는
  /// 그대로 공유한다 — 마스크는 k축이고 n-블록은 n축이라 직교하기 때문이다.
  /// nullptr = 부분 블록 없음 (슬랩 n이 N_BLOCK의 배수).
  const uint32_t* koff_tail = nullptr;
  int ngroup_tail = 0;  ///< 꼬리 블록의 n-그룹 수 (= 꼬리 폭 / GROUP_N)

  bool empty() const { return npairs == 0; }
};

/// 이 커널이 per-k 마스크(`KMask`)를 요구하는가. 커널이 `static constexpr bool
/// kMaskPerK = true`를 선언하면 참이고, 선언이 없으면 페어 마스크(`PairMask`)다 —
/// 기존 커널은 전부 무선언이라 무변경으로 페어 경로에 남는다.
template <class K, class = void>
struct mask_per_k : std::false_type {};
template <class K>
struct mask_per_k<K, std::enable_if_t<K::kMaskPerK>> : std::true_type {};

/// 이 커널의 `build_plan`이 슬랩 n을 받는가 (= 부분 n-블록을 지원하는가).
/// 지원하는 커널만 꼬리 오프셋 표를 만들 수 있고, 나머지는 5인자 서명 그대로다
/// — 그래서 operator 쪽 호출부를 커널별로 갈라 쓰지 않아도 된다.
template <class K, class = void>
struct plan_takes_slab_n : std::false_type {};
template <class K>
struct plan_takes_slab_n<K, std::void_t<decltype(K::build_plan(
                                (const PairMask*)nullptr, (const ggml_bf16_t*)nullptr, 0,
                                (void*)nullptr, (ActivationPlan*)nullptr, 0))>> : std::true_type {};

/// `build_plan` 호출의 단일 창구. 슬랩 n을 받는 커널에는 넘기고, 아닌 커널에는
/// 넘기지 않는다 (부분 블록은 그 커널에서 여전히 호출부가 막아야 한다).
template <class K>
inline void build_plan_for(const PairMask* mask, const ggml_bf16_t* x, int k, void* arena,
                           ActivationPlan* out, int n_slab) {
  if constexpr (plan_takes_slab_n<K>::value) {
    K::build_plan(mask, x, k, arena, out, n_slab);
  } else {
    (void)n_slab;
    K::build_plan(mask, x, k, arena, out);
  }
}

/// per-k 마스크 커널용 창구. 꼬리(부분 n-블록) 표는 per-k 커널이 아직 지원하지
/// 않으므로 5인자 서명만 있다 — 슬랩 n은 받되 쓰지 않는다 (호출부 서명 통일).
template <class K>
inline void build_plan_for(const KMask* mask, const ggml_bf16_t* x, int k, void* arena, ActivationPlan* out,
                           int n_slab) {
  static_assert(mask_per_k<K>::value, "build_plan_for(KMask*) needs a per-k kernel (kMaskPerK)");
  (void)n_slab;
  K::build_plan(mask, x, k, arena, out);
}

}  // namespace amx
