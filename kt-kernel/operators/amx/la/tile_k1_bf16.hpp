#pragma once
/**
 * @file tile_k1_bf16.hpp
 * @brief bf16 m == 1 **per-k** sparse decode GEMV — k=1 타일 (Seam B/C 껍데기).
 *
 * `GemmKernelTileK2BF16`(tile_k2.hpp)의 k=1 형제다. 계산 페이로드는
 * `tile_k1_bf16_port.hpp`(pack, plan 빌더, `ktb1_gemv`, 프리페치 커서)이고 이 파일은 그것을
 * kt의 Buffer/커널 struct 어휘로 감싼다. k2와 다른 것은 셋:
 *
 *   - 레이아웃: 64 B 라인 = **k 한 행 × 32 n** (k2: VNNI 페어 × 16 n). 타일 컬럼(8192 원소)·
 *     super(256 n)·BufferB 크기(N·K bf16)는 **그대로**이고 그룹 안 원소 순열만 다르다.
 *   - 마스크: `KMask`(uint32 / 32-k, 비트 = k 하나). `kMaskPerK = true`로 moe_base가
 *     `|x_k| >= thr` per-k 빌더(k_mask.hpp)를 고른다 — wn_sq/pair_dot을 쓰지 않는다.
 *   - 소비: fp32 broadcast × bf16→fp32 widen, fp32 FMA (k2: `dpbf16_ps` 페어). 곱은 둘 다
 *     정확하고 누산 순서만 다르다.
 *
 * **CPU prefill(`mat_mul`)이 없다.** k2 형제는 NC pack을 그대로 써서 상속받은 AMX prefill을
 * 공유했지만, 이 커널은 pack이 다르므로 AMX 경로에 그대로 물리면 **같은 바이트를 다른
 * 순열로 읽어 조용히 쓰레기를 낸다**. 그래서 operator를 `TILE_BF16_MOE_TP`(tile-bf16-moe.hpp)
 * 로 따로 두고 m == 1은 plan 경로, m > 1은 **토큰 루프**로 돈다 — fp8/mxfp4 타일 형제와 같은
 * 구조다. 대가는 prefill 비용이 M배라는 것이고, 그것이 이 커널을 decode 전용으로 만든다.
 *
 * 기하 제약: n(노드 N shard)은 **256의 배수**(타일 super), k는 **32의 배수**(마스크 워드).
 */

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "act_plan.hpp"
#include "amx_raw_buffers.hpp"
#include "tile_k1_bf16_port.hpp"

namespace amx {

struct GemmKernelTileK1BF16 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 2.0;

  /// Seam B 마스크 종류: per-k (act_plan.hpp `mask_per_k`).
  static constexpr bool kMaskPerK = true;
  using MaskWord = KMask;

  static const int M_STEP = 1;
  static const int N_STEP = 32;               // kt Buffer 어휘 (BufferC::to_mat 32열 단위)
  static const int K_STEP = KTB1_TILE_K;      // 32 — 타일 컬럼 = 마스크 워드
  static constexpr int SCALE_BLK = 0;         // 배율 없음 (bf16)
  static inline const int N_BLOCK = KTB1_TILE_N;
  static inline const int K_BLOCK = 1 << 30;  // K 분할 없음

  static std::string name() { return "TILE_K1_BF16"; }
  static void config() {}

  /// 프리페치 거리 (소비 스트림의 64 B 라인 수). k2 bf16과 같은 값에서 출발 —
  /// cpu-mm의 pf_dist sweep도 128을 골랐다 (FINDINGS §11).
  static constexpr int PF_DIST = 128;

  static int recommended_nth(int n) { return std::max(1, n / N_BLOCK); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    const int supers = n / N_BLOCK;
    const int per = (supers + nth - 1) / nth;
    return {std::min(n, ith * per * N_BLOCK), std::min(n, (ith + 1) * per * N_BLOCK)};
  }

  using BufferA = BufferABF16Impl<GemmKernelTileK1BF16>;  // raw bf16 [m][k] — plan 빌더가 읽는다

  /// k1 타일 레이아웃의 bf16 코드만 (배율 없음). 크기는 k2 pack과 같은 N·K다.
  struct BufferB {
    ggml_bf16_t* b;
    int n, k;

    static constexpr int N_STEP = GemmKernelTileK1BF16::N_STEP;
    static constexpr int N_BLOCK = GemmKernelTileK1BF16::N_BLOCK;
    static constexpr int K_STEP = GemmKernelTileK1BF16::K_STEP;
    static constexpr bool SCALE = false;

    static size_t required_size(int n, int k) { return sizeof(ggml_bf16_t) * (size_t)n * (size_t)k; }

    BufferB(int n, int k, void* ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      if (n % N_BLOCK || k % K_STEP) {
        printf("TileK1BF16 BufferB: n=%d k=%d (need n%%256==0, k%%32==0)\n", n, k);
        throw std::runtime_error("TileK1BF16 BufferB: geometry not aligned to the k1 tile layout");
      }
      b = reinterpret_cast<ggml_bf16_t*>(ptr);
    }
    void set_data(void* new_ptr) { b = reinterpret_cast<ggml_bf16_t*>(new_ptr); }

    void from_mat(const ggml_bf16_t* src, int ith, int nth) {
      auto [n0, n1] = GemmKernelTileK1BF16::split_range_n(n, ith, nth);
      if (n0 >= n1) return;
      ktb1_pack_codes_range(b, src, n, k, n0, n1);
    }
  };

  using BufferC = BufferCFP32ReduceImpl<GemmKernelTileK1BF16>;

  // ── 인코더 (Seam B) ──────────────────────────────────────────────────
  /// koff [K] + pval [K] + seg [K/32 + 1] (uint32) + 64 B 정렬 여유 — k2의 약 2배
  /// (k2는 페어당 한 항목, 여기는 k당 한 항목).
  static size_t plan_arena_bytes(int k) {
    return ((size_t)k * 2 + (size_t)k / K_STEP + 1) * sizeof(uint32_t) + 64;
  }

  /// `mask == nullptr`이면 dense. mask는 KMask(uint32/32-k 컬럼, bit j = k 32c + j).
  static void build_plan(const KMask* mask, const ggml_bf16_t* x, int k, void* arena,
                         ActivationPlan* out) {
    uint32_t* base =
        reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(arena) + 63) & ~(uintptr_t)63);
    uint32_t* koff = base;
    uint32_t* pval = koff + (size_t)k;
    uint32_t* seg = pval + (size_t)k;
    const int np = ktb1_build_plan(mask, reinterpret_cast<const uint16_t*>(x), k, koff, pval, seg);
    out->koff = koff;
    out->pval = pval;  // fp32 비트 (k2는 bf16 페어) — 소비자는 이 커널의 gemv만이다
    out->seg = seg;
    out->npairs = np;  // = 유지된 k 수
    out->nseg = k / K_STEP;
    out->koff_tail = nullptr;
    out->ngroup_tail = 0;
  }

  // ── 소비 커널 (Seam C) ───────────────────────────────────────────────
  static void gemv_row(int m, int t, int n, int k, int n0, int n1, BufferB* bb, BufferC* bc,
                       const ActivationPlan* plan) {
    assert(n % N_BLOCK == 0 && n0 % N_BLOCK == 0 && n1 % N_BLOCK == 0);
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int nb = n0; nb < n1; nb += N_BLOCK) {
      float* y = bc->get_submat(m, n, t, nb) - nb;  // y + nb == 이 super의 출력
      ktb1_gemv(bb->b, plan->koff, plan->pval, plan->seg, plan->npairs, n, k, y, nb, nb + N_BLOCK,
                PF_DIST);
    }
#else
    (void)m; (void)t; (void)n; (void)k; (void)n0; (void)n1; (void)bb; (void)bc; (void)plan;
    throw std::runtime_error("TileK1BF16 needs AVX512F/BW");
#endif
  }

  static void gemv_slab(int m, int n, int k, int n0, int n1, BufferA* ba, BufferB* bb, BufferC* bc,
                        const ActivationPlan* plan) {
    (void)m;
    (void)ba;  // pval이 활성값을 나른다 (Seam B)
    gemv_row(1, 0, n, k, n0, n1, bb, bc, plan);
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
