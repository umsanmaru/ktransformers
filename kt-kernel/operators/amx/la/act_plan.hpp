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

namespace amx {

/// 절대 k 기준 keep 마스크. uint16 하나가 연속된 32 k를 덮고, 비트 j가
/// k-페어 (32t + 2j, 32t + 2j + 1)을 유지한다. 어떤 packing도 참조하지 않는다
/// — 마스크는 "어떤 k가 살아남는가"이고 그것은 활성값과 임계값의 성질이다.
using PairMask = uint16_t;  // mask[k / 32]

/// 커널이 자기 blocking으로 인코딩한 결과. nullptr 필드는 dense를 뜻한다.
struct ActivationPlan {
  const uint32_t* koff = nullptr;  ///< 유지된 페어의 weight 원소 오프셋, k 오름차순
  const uint32_t* pval = nullptr;  ///< 같은 순서의 packed bf16 (lo | hi << 16)
  const uint32_t* seg = nullptr;   ///< 타일 컬럼 접두: 컬럼 c는 [seg[c], seg[c+1])
  int npairs = 0;
  int nseg = 0;

  bool empty() const { return npairs == 0; }
};

}  // namespace amx
