#pragma once
/**
 * @file tile_k2_fp8b128.hpp
 * @brief FP8(e4m3, 128×128 블록 배율) m == 1 sparse decode GEMV — 타일 레이아웃 (Seam B/C 껍데기).
 *
 * 계산 페이로드는 `tile_k2_fp8b128_port.hpp`(cpu-mm-platform `port/kt_tile_k2_fp8b128.hpp`
 * 사본 — pack, plan 빌더, `ktf8_gemv`, 프리페치 정책)다. 이 파일은 그것을 kt의 Buffer/커널
 * struct 어휘로 감싼다 (port/PORTING_FP8B128.md + PORTING_MXFP4.md의 Seam A/B/C 표).
 *
 * 레이아웃: 32k×256n 타일(1 B/원소), 128 B 청크 = k-페어 1 × n 64, 배율은 **전치** fp32
 * `[K/128][N/128]`. kt 원본(`BufferBFP8Impl`: 행우선 e4m3 + n-major scale_inv)과 다른 저장이라
 * 원본 `mat_mul`/`vec_mul_kgroup`은 이 BufferB를 읽을 수 없다 — prefill(m ≫ 1)은 GPU(sglang
 * prism_grouped_fp8.cuh KT_TILE8 로더)가 slab을 제자리 읽고, CPU는 m == 1(plan 경로)과
 * 소배치(토큰 루프)만 맡는다 (사용자 결정 2026-08-28: fp8/mxfp4 prefill에 AMX 구간 없음).
 *
 * 수치: W8A16 — bf16 활성 × 정확 e4m3 값(정수 비트 디코드), fp32 dpbf16 누산 후 32-k 컬럼
 * 부분합에 fp32 블록 배율 1회. kt fp8 원본과 같은 계약이고 누산 순서만 다르다.
 *
 * 기하 제약: n(노드 N shard)은 **256의 배수**(타일 super), k는 **128의 배수**(배율 블록).
 * 로더가 k를 128로 올리고 패딩 행을 0(=+0.0 코드)으로 채운다.
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
#include "tile_k2_fp8b128_port.hpp"

namespace amx {

struct GemmKernelTileK2FP8B128 {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 1.0;

  static const int M_STEP = 1;
  // N_STEP은 kt Buffer 어휘(BufferC::to_mat가 32열 단위로 변환한다). 워커 분할 단위는
  // N_BLOCK(= 타일 super 256 n)이다 — mxfp4 타일과 같은 이유.
  static const int N_STEP = 32;
  static const int K_STEP = KTF8_TILE_K;  // 32 — 타일 컬럼
  static constexpr int SCALE_BLK = KTF8_BLK;  // 128 — 배율 블록 (양 축)
  static inline const int N_BLOCK = KTF8_TILE_N;
  static inline const int K_BLOCK = 1 << 30;  // K 분할 없음 (커서가 K 전체를 한 스트림으로 본다)

  static std::string name() { return "TILE_K2_FP8B128"; }
  static void config() {}

  /// 프리페치 거리 (소비 스트림의 64 B 라인 수). cpu-mm FINDINGS §15 측정 우승자, T1 힌트.
  static constexpr int PF_DIST = 128;

  static int recommended_nth(int n) { return std::max(1, n / N_BLOCK); }
  /// 워커 슬라이스는 항상 whole super — 커서와 프리페치가 super 안에서 한 스트림이다.
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    const int supers = n / N_BLOCK;
    const int per = (supers + nth - 1) / nth;
    return {std::min(n, ith * per * N_BLOCK), std::min(n, (ith + 1) * per * N_BLOCK)};
  }

  using BufferA = BufferABF16Impl<GemmKernelTileK2FP8B128>;  // raw bf16 [m][k] — plan 빌더가 읽는다

  /// codes(타일 레이아웃, N·K B) 뒤에 전치 fp32 배율((K/128)·(N/128) float). 두 영역 모두 64 B 정렬.
  struct BufferB {
    uint8_t* b;  // ktf8_pack_codes 출력
    float* s;    // ktf8_pack_scales 출력 (transposed [K/128][N/128])
    int n, k, k_group_size;

    static constexpr int N_STEP = GemmKernelTileK2FP8B128::N_STEP;
    static constexpr int N_BLOCK = GemmKernelTileK2FP8B128::N_BLOCK;
    static constexpr int K_STEP = GemmKernelTileK2FP8B128::K_STEP;
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
        printf("TileK2FP8B128 BufferB: n=%d k=%d group=%d (need n%%256==0, k%%128==0, group==128)\n", n, k,
               k_group_size);
        throw std::runtime_error("TileK2FP8B128 BufferB: geometry not aligned to the fp8 tile layout");
      }
      b = reinterpret_cast<uint8_t*>(ptr);
      s = reinterpret_cast<float*>(b + ((codes_bytes(n, k) + 63) & ~(size_t)63));
    }

    /// kt 규약의 행우선 e4m3 바이트 [n][k] → 타일 레이아웃. 워커 슬라이스는 whole super라
    /// 바이트 순열이 겹치지 않는다.
    void from_raw_mat(const uint8_t* proj, int ith, int nth) {
      auto [n0, n1] = GemmKernelTileK2FP8B128::split_range_n(n, ith, nth);
      if (n0 >= n1) return;
      ktf8_pack_codes_range(b, proj, n, k, n0, n1);
    }

    /// kt 규약의 fp32 `scale_inv` n-major [n/128][k/128] → 전치 [k/128][n/128]. 임의 fp32라
    /// 검증할 것이 없다 (mxfp4의 2의 거듭제곱 요구와 다른 점).
    void from_scales(const float* scale_inv, int ith, int nth) {
      auto [n0, n1] = GemmKernelTileK2FP8B128::split_range_n(n, ith, nth);
      if (n0 >= n1) return;
      ktf8_pack_scales_range(s, scale_inv, n, k, n0, n1);
    }

    /// kt `BufferBFP8Impl::from_mat`과 같은 시그니처 (코드 + 배율 한 번에).
    void from_mat(const uint8_t* b_src, const float* d_src, int ith, int nth) {
      from_raw_mat(b_src, ith, nth);
      if (d_src != nullptr) from_scales(d_src, ith, nth);
    }
  };

  using BufferC = BufferCFP32ReduceImpl<GemmKernelTileK2FP8B128>;  // kt fp8 원본과 같은 fp32 BufferC

  // ── 인코더 (Seam B) ──────────────────────────────────────────────────
  /// koff [K/2] + pval [K/2] + seg [K/32 + 1] (uint32) + 64 B 정렬 여유.
  static size_t plan_arena_bytes(int k) {
    return ((size_t)k / 2 * 2 + (size_t)k / K_STEP + 1) * sizeof(uint32_t) + 64;
  }

  /// `mask == nullptr`이면 dense. mask는 kt PairMask(uint16/32-k 컬럼, bit j = 페어 j).
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
  /// 토큰 t의 출력 [n0, n1)을 **덧쓴다** (ktf8_gemv가 memset). BufferC는 kt의 N_BLOCK 블록
  /// 레이아웃이라 super마다 `get_submat(m, n, t, nb)` 주소로 쓴다 — m == 1이면 행우선과 같다.
  static void gemv_row(int m, int t, int n, int k, int n0, int n1, BufferB* bb, BufferC* bc,
                       const ActivationPlan* plan) {
    assert(n % N_BLOCK == 0 && n0 % N_BLOCK == 0 && n1 % N_BLOCK == 0);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512BF16__)
    for (int nb = n0; nb < n1; nb += N_BLOCK) {
      float* y = bc->get_submat(m, n, t, nb) - nb;  // y + nb == 이 super의 출력
      ktf8_gemv(bb->b, bb->s, plan->koff, plan->pval, plan->seg, plan->npairs, n, k, y, nb, nb + N_BLOCK, PF_DIST);
    }
#else
    (void)m;
    (void)t; (void)n; (void)k; (void)n0; (void)n1; (void)bb; (void)bc; (void)plan;
    throw std::runtime_error("TileK2FP8B128 needs AVX512F/BW/BF16");
#endif
  }

  /// m == 1 전용 (float_mat_vec의 Seam A 형태와 같은 시그니처).
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
