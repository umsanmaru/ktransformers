#pragma once
/**
 * @file tile_k2_fp8pt.hpp
 * @brief FP8(e4m3, **per-tensor** fp32 배율) m == 1 sparse decode GEMV — 타일 레이아웃 (Seam B/C 껍데기).
 *
 * 대상 체크포인트: Mistral-Medium-3.5-128B 류 "static-tensor" FP8 — `quantization_config`가
 * `weight_block_size: null`, `activation_scheme: "static"`이고 `weight_scale_inv`가 선형층당
 * **스칼라 하나**(BF16 shape [])다. DeepSeek/GLM/Qwen의 128×128 블록 배율(`kt_tile_k2_fp8b128`)과
 * 코드 바이트는 같고 배율 기하만 다르다.
 *
 * 저장: codes는 `GemmKernelTileK2FP8B128`과 **같은 타일 순열**(32k×256n, 128 B 청크 = k-페어 × 64 n)
 * 이고, 그 뒤 64 B 슬롯에 fp32 배율 하나. 그래서 GPU 쪽 KT_TILE8 코드 로더가 그대로 맞고, 배율만
 * "표 조회" 대신 "스칼라 1개"로 읽으면 된다 (sglang 레이아웃 태그는 별도 — 아래 §sglang).
 *
 * 계산: `tile_k2_fp8pt_port.hpp`의 `ktf8pt_gemv` — 배율 블록 경계가 없으므로 super(256 n) 하나의
 * 누산기 16개를 전 K에 걸쳐 레지스터에 유지하고 마지막에 한 번 곱한다. 컬럼당 flush(12 mem op)와
 * 배율 표 로드, y memset이 사라진다.
 *
 * 기하 제약: n(노드 N shard)은 **256의 배수**, k는 **32의 배수**(타일 컬럼) — b128의 128 제약이
 * 풀린다. 티어 절단(sglang `k_align`)은 bf16처럼 페어(2)까지 내려갈 수 있고, 로더가 k를 32로 올린다.
 *
 * `quant_config.group_size == 0`이 이 커널의 "per-tensor" 신호다 (`SCALE_BLK = 0`). 배율 텐서는
 * proj당 `[E]` fp32 (expert = prism dense 레인의 슬롯).
 *
 * §sglang: 커널 키 `kt_tile_k2_fp8pt`, slab 레이아웃 태그 `kt_tile8pt`(코드 = kt_tile8, 배율 = 코드
 * 끝 64 B 슬롯의 fp32 1개), tile_rows 32, n_align 256. 이 헤더는 kt 쪽만 다룬다.
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

#include "amx_raw_buffers.hpp"  // ggml_bf16_t 먼저 (act_plan.hpp가 전제한다)
#include "act_plan.hpp"
#include "tile_k2_fp8pt_port.hpp"

namespace amx {

struct GemmKernelTileK2FP8PT {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 1.0;

  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = KTF8_TILE_K;  // 32 — 타일 컬럼
  /// 배율 블록이 없다. operator는 `group_size == SCALE_BLK`(= 0)을 per-tensor 신호로 검사한다.
  static constexpr int SCALE_BLK = 0;
  static constexpr bool kPerTensorScale = true;
  static inline const int N_BLOCK = KTF8_TILE_N;
  static inline const int K_BLOCK = 1 << 30;

  static std::string name() { return "TILE_K2_FP8PT"; }
  static void config() {}

  static constexpr int PF_DIST = 128;  // b128과 같은 소비 스트림 (cpu-mm FINDINGS §15 우승자)

  static int recommended_nth(int n) { return std::max(1, n / N_BLOCK); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    const int supers = n / N_BLOCK;
    const int per = (supers + nth - 1) / nth;
    return {std::min(n, ith * per * N_BLOCK), std::min(n, (ith + 1) * per * N_BLOCK)};
  }

  using BufferA = BufferABF16Impl<GemmKernelTileK2FP8PT>;

  /// codes(타일 레이아웃, N·K B, 64 B 올림) 뒤 64 B 슬롯에 fp32 배율 1개.
  struct BufferB {
    uint8_t* b;  // ktf8_pack_codes 출력
    float* s;    // 배율 슬롯 (s[0]만 의미, 나머지 60 B는 0)
    int n, k, k_group_size;

    static constexpr int N_STEP = GemmKernelTileK2FP8PT::N_STEP;
    static constexpr int N_BLOCK = GemmKernelTileK2FP8PT::N_BLOCK;
    static constexpr int K_STEP = GemmKernelTileK2FP8PT::K_STEP;
    static constexpr bool SCALE = true;
    static constexpr size_t SCALE_SLOT_BYTES = 64;

    static size_t codes_bytes(int n, int k) { return (size_t)n * (size_t)k; }
    static size_t scale_offset(int n, int k) { return (codes_bytes(n, k) + 63) & ~(size_t)63; }
    static size_t required_size(int n, int k, int k_group_size) {
      (void)k_group_size;
      return scale_offset(n, k) + SCALE_SLOT_BYTES;
    }

    BufferB(int n, int k, int k_group_size, void* ptr) : n(n), k(k), k_group_size(k_group_size) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      if (n % N_BLOCK || k % K_STEP || k_group_size != 0) {
        printf("TileK2FP8PT BufferB: n=%d k=%d group=%d (need n%%256==0, k%%32==0, group==0 per-tensor)\n", n, k,
               k_group_size);
        throw std::runtime_error("TileK2FP8PT BufferB: geometry not aligned to the fp8 tile layout");
      }
      b = reinterpret_cast<uint8_t*>(ptr);
      s = reinterpret_cast<float*>(b + scale_offset(n, k));
    }

    /// kt 규약의 행우선 e4m3 바이트 [n][k] → 타일 레이아웃 (b128과 같은 순열).
    void from_raw_mat(const uint8_t* proj, int ith, int nth) {
      auto [n0, n1] = GemmKernelTileK2FP8PT::split_range_n(n, ith, nth);
      if (n0 >= n1) return;
      ktf8_pack_codes_range(b, proj, n, k, n0, n1);
    }

    /// `scale`은 fp32 스칼라 하나를 가리킨다. 워커 0만 쓴다 (슬롯이 하나라 분할할 것이 없다).
    void from_scales(const float* scale, int ith, int nth) {
      (void)nth;
      if (ith != 0) return;
      std::memset(s, 0, SCALE_SLOT_BYTES);
      s[0] = scale[0];
    }

    void from_mat(const uint8_t* b_src, const float* d_src, int ith, int nth) {
      from_raw_mat(b_src, ith, nth);
      if (d_src != nullptr) from_scales(d_src, ith, nth);
    }
  };

  using BufferC = BufferCFP32ReduceImpl<GemmKernelTileK2FP8PT>;

  // ── 인코더 (Seam B) — b128과 동일 (레이아웃이 같으니 plan도 같다) ──────────
  static size_t plan_arena_bytes(int k) {
    return ((size_t)k / 2 * 2 + (size_t)k / K_STEP + 1) * sizeof(uint32_t) + 64;
  }

  static void build_plan(const uint16_t* mask, const ggml_bf16_t* x, int k, void* arena, ActivationPlan* out) {
    uint32_t* base = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(arena) + 63) & ~(uintptr_t)63);
    uint32_t* koff = base;
    uint32_t* pval = koff + (size_t)k / 2;
    uint32_t* seg = pval + (size_t)k / 2;
    const int np = ktf8_build_plan(mask, reinterpret_cast<const uint16_t*>(x), k, koff, pval, seg);
    out->koff = koff;
    out->pval = pval;
    out->seg = seg;
    out->npairs = np;
    out->nseg = k / K_STEP;
  }

  // ── 소비 커널 (Seam C) ───────────────────────────────────────────────
  static void gemv_row(int m, int t, int n, int k, int n0, int n1, BufferB* bb, BufferC* bc,
                       const ActivationPlan* plan) {
    assert(n % N_BLOCK == 0 && n0 % N_BLOCK == 0 && n1 % N_BLOCK == 0);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512BF16__)
    for (int nb = n0; nb < n1; nb += N_BLOCK) {
      float* y = bc->get_submat(m, n, t, nb) - nb;
      ktf8pt_gemv(bb->b, bb->s[0], plan->koff, plan->pval, plan->seg, plan->npairs, n, k, y, nb, nb + N_BLOCK,
                  PF_DIST);
    }
#else
    (void)m; (void)t; (void)n; (void)k; (void)n0; (void)n1; (void)bb; (void)bc; (void)plan;
    throw std::runtime_error("TileK2FP8PT needs AVX512F/BW/BF16");
#endif
  }

  static void gemv_slab(int m, int n, int k, int n0, int n1, BufferA* ba, BufferB* bb, BufferC* bc,
                        const ActivationPlan* plan) {
    (void)ba;
    gemv_row(m, 0, n, k, n0, n1, bb, bc, plan);
  }

  static const ActivationPlan* dense_plan(const ggml_bf16_t* x, int k, ActivationPlan* out) {
    static thread_local std::vector<uint8_t> arena;
    const size_t need = plan_arena_bytes(k);
    if (arena.size() < need) arena.resize(need);
    build_plan(nullptr, x, k, arena.data(), out);
    return out;
  }
};

}  // namespace amx
