#pragma once
/**
 * @file tile_k1_fp8b128.hpp
 * @brief FP8(e4m3, 128×128 블록 배율) m == 1 **per-k** sparse decode GEMV — k=1 타일 (Seam B/C 껍데기).
 *
 * `GemmKernelTileK2FP8B128`(tile_k2_fp8b128.hpp)의 k=1 형제다. 계산 페이로드는
 * `tile_k1_fp8b128_port.hpp`(pack, plan 빌더, `ktf8k1_gemv`, 프리페치 커서)이고, 이 파일은 그것을
 * kt의 Buffer/커널 struct 어휘로 감싼다. k2와 다른 것은 셋뿐이다:
 *
 *   - 레이아웃: 64 B 라인 = **k 한 행 × 64 n** (k2: k 페어 × 64 n = 128 B). 타일 컬럼 8 KB, 그룹 2 KB,
 *     super 256 n, 전치 fp32 배율 `[K/128][N/128]`, BufferB 크기는 전부 k2와 같다 — GPU KT_TILE8 로더의
 *     `blk_off`/배율 위치도 그대로고, 그룹 안 바이트 순열만 다르다 (sglang은 KT_TILE8_K1 분기).
 *   - 마스크: `KMask`(uint32 / 32-k, 비트 = k 하나). `kMaskPerK = true`로 moe_base가 `|x_k| >= thr`
 *     per-k 마스크 빌더를 고른다 — 가중치 통계(wn_sq/pair_dot)를 쓰지 않는다.
 *   - 소비: fp32 broadcast × e4m3→fp32 정확 디코드, fp32 FMA 누산 (k2: dpbf16 페어). 32-k 컬럼
 *     부분합에 fp32 블록 배율 1회 — 수치 계약은 k2·kt fp8 원본과 같고 누산 순서만 다르다.
 *
 * prefill(m ≫ 1)은 k2와 같이 GPU가 slab을 제자리 읽는다 — `mat_mat`은 없다. CPU는 m == 1(plan 경로)과
 * 소배치(토큰 루프)만 맡는다.
 *
 * 기하 제약: n(노드 N shard)은 **256의 배수**, k는 **128의 배수** (k2와 동일).
 */

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "act_plan.hpp"
#include "amx_raw_buffers.hpp"
#include "tile_k1_fp8b128_port.hpp"

namespace amx {

struct GemmKernelTileK1FP8B128 {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 1.0;

  /// Seam B 마스크 종류: per-k (act_plan.hpp `mask_per_k`).
  static constexpr bool kMaskPerK = true;
  using MaskWord = KMask;

  static const int M_STEP = 1;
  static const int N_STEP = 32;               // kt Buffer 어휘 (BufferC::to_mat 32열 단위)
  static const int K_STEP = KTF8_TILE_K;      // 32 — 타일 컬럼 = 마스크 워드
  static constexpr int SCALE_BLK = KTF8_BLK;  // 128 — 배율 블록 (양 축)
  static inline const int N_BLOCK = KTF8_TILE_N;
  static inline const int K_BLOCK = 1 << 30;  // K 분할 없음

  static std::string name() { return "TILE_K1_FP8B128"; }
  static void config() {}

  /// 프리페치 거리 (소비 스트림의 64 B 라인 수). k2와 같은 값에서 출발 — 벤치로 재조정.
  static constexpr int PF_DIST = 128;

  static int recommended_nth(int n) { return std::max(1, n / N_BLOCK); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    const int supers = n / N_BLOCK;
    const int per = (supers + nth - 1) / nth;
    return {std::min(n, ith * per * N_BLOCK), std::min(n, (ith + 1) * per * N_BLOCK)};
  }

  using BufferA = BufferABF16Impl<GemmKernelTileK1FP8B128>;  // raw bf16 [m][k] — plan 빌더가 읽는다

  /// codes(k1 타일 레이아웃, N·K B) 뒤에 전치 fp32 배율. 크기·정렬은 k2 BufferB와 동일.
  struct BufferB {
    uint8_t* b;
    float* s;
    int n, k, k_group_size;

    static constexpr int N_STEP = GemmKernelTileK1FP8B128::N_STEP;
    static constexpr int N_BLOCK = GemmKernelTileK1FP8B128::N_BLOCK;
    static constexpr int K_STEP = GemmKernelTileK1FP8B128::K_STEP;
    static constexpr bool SCALE = true;

    static size_t codes_bytes(int n, int k) { return (size_t)n * (size_t)k; }
    static size_t required_size(int n, int k, int k_group_size) {
      (void)k_group_size;
      return ((codes_bytes(n, k) + 63) & ~(size_t)63) +
             (size_t)(n / KTF8_BLK) * (size_t)(k / KTF8_BLK) * sizeof(float);
    }

    BufferB(int n, int k, int k_group_size, void* ptr) : n(n), k(k), k_group_size(k_group_size) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      if (n % N_BLOCK || k % KTF8_BLK || k_group_size != KTF8_BLK) {
        printf("TileK1FP8B128 BufferB: n=%d k=%d group=%d (need n%%256==0, k%%128==0, group==128)\n", n, k,
               k_group_size);
        throw std::runtime_error("TileK1FP8B128 BufferB: geometry not aligned to the fp8 tile layout");
      }
      b = reinterpret_cast<uint8_t*>(ptr);
      s = reinterpret_cast<float*>(b + ((codes_bytes(n, k) + 63) & ~(size_t)63));
    }

    void from_raw_mat(const uint8_t* proj, int ith, int nth) {
      auto [n0, n1] = GemmKernelTileK1FP8B128::split_range_n(n, ith, nth);
      if (n0 >= n1) return;
      ktf8k1_pack_codes_range(b, proj, n, k, n0, n1);
    }

    /// 배율 pack은 k2와 바이트 단위로 같다 (전치만).
    void from_scales(const float* scale_inv, int ith, int nth) {
      auto [n0, n1] = GemmKernelTileK1FP8B128::split_range_n(n, ith, nth);
      if (n0 >= n1) return;
      ktf8_pack_scales_range(s, scale_inv, n, k, n0, n1);
    }

    void from_mat(const uint8_t* b_src, const float* d_src, int ith, int nth) {
      from_raw_mat(b_src, ith, nth);
      if (d_src != nullptr) from_scales(d_src, ith, nth);
    }
  };

  using BufferC = BufferCFP32ReduceImpl<GemmKernelTileK1FP8B128>;

  // ── 인코더 (Seam B) ──────────────────────────────────────────────────
  /// koff [K] + pval [K] + seg [K/32 + 1] (uint32) + 64 B 정렬 여유 — k2의 약 2배.
  static size_t plan_arena_bytes(int k) {
    return ((size_t)k * 2 + (size_t)k / K_STEP + 1) * sizeof(uint32_t) + 64;
  }

  /// `mask == nullptr`이면 dense. mask는 KMask(uint32/32-k 컬럼, bit j = k 32c + j).
  static void build_plan(const KMask* mask, const ggml_bf16_t* x, int k, void* arena, ActivationPlan* out) {
    uint32_t* base = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(arena) + 63) & ~(uintptr_t)63);
    uint32_t* koff = base;
    uint32_t* pval = koff + (size_t)k;
    uint32_t* seg = pval + (size_t)k;
    const int np = ktf8k1_build_plan(mask, reinterpret_cast<const uint16_t*>(x), k, koff, pval, seg);
    out->koff = koff;
    out->pval = pval;  // fp32 비트 (k2는 bf16 페어) — 소비자는 이 커널의 gemv만이다
    out->seg = seg;
    out->npairs = np;  // = 유지된 k 수
    out->nseg = k / K_STEP;
  }

  // ── 소비 커널 (Seam C) ───────────────────────────────────────────────
  static void gemv_row(int m, int t, int n, int k, int n0, int n1, BufferB* bb, BufferC* bc,
                       const ActivationPlan* plan) {
    assert(n % N_BLOCK == 0 && n0 % N_BLOCK == 0 && n1 % N_BLOCK == 0);
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int nb = n0; nb < n1; nb += N_BLOCK) {
      float* y = bc->get_submat(m, n, t, nb) - nb;  // y + nb == 이 super의 출력
      ktf8k1_gemv(bb->b, bb->s, plan->koff, plan->pval, plan->seg, plan->npairs, n, k, y, nb, nb + N_BLOCK,
                  PF_DIST);
    }
#else
    (void)m;
    (void)t; (void)n; (void)k; (void)n0; (void)n1; (void)bb; (void)bc; (void)plan;
    throw std::runtime_error("TileK1FP8B128 needs AVX512F/BW");
#endif
  }

  static void gemv_slab(int m, int n, int k, int n0, int n1, BufferA* ba, BufferB* bb, BufferC* bc,
                        const ActivationPlan* plan) {
    (void)ba;  // pval이 활성값을 나른다 (Seam B)
    gemv_row(m, 0, n, k, n0, n1, bb, bc, plan);
  }

  /// dense plan을 스레드 로컬 arena에 만든다 (mask 없는 decode, 소배치 토큰 루프용).
  static const ActivationPlan* dense_plan(const ggml_bf16_t* x, int k, ActivationPlan* out) {
    static thread_local std::vector<uint8_t> arena;
    const size_t need = plan_arena_bytes(k);
    if (arena.size() < need) arena.resize(need);
    build_plan(nullptr, x, k, arena.data(), out);
    return out;
  }
};

}  // namespace amx
