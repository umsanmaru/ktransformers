#pragma once
/**
 * @file tile_k2.hpp
 * @brief m == 1 sparse decode GEMV — 소프트웨어 프리페치 커서 (Seam C).
 *
 * `GemmKernel224BF16RowMajorNC`를 **상속**한다. 그 커널이 이미 이 커널이
 * 필요로 하는 모든 것을 갖고 있기 때문이다: 같은 `N_STEP/K_STEP/N_BLOCK/
 * K_BLOCK`, `split_range_n`, `recommended_nth`, `config()`, prefill용
 * `amx_kernel`, 그리고 `m > 1` 폴백용 `avx_kernel_4`. 여기서 더하는 것은
 * `gemv_slab`(m == 1 전용)과 그 인코더뿐이고, **weight 사본은 하나**다.
 *
 * kt 원본 `BufferBBF16Impl`은 건드리지 않는다 — 원본 커널은 비트 단위로
 * 그대로이고, 롤백은 템플릿 인자 하나다.
 *
 * ## 이 커널이 이기는 이유
 *
 * 소프트웨어 프리페치 **커서**다. 커서는 전역 소비 순서 위의 lookahead이고,
 * 타일 하나만 보는 microkernel 경계로는 표현할 수 없다 — "다음에 무엇이
 * 소비될지"를 알 수 없기 때문이다. 그래서 이 커널은 `float_mat_vec`의 루프를
 * 대신 소유한다 (Seam A의 opt-in 분기).
 *
 * ## 레이아웃 대응 (해석적으로 확인, 2026-08-26)
 *
 * NC의 `pack_block`을 전개하면 (n_block, k_block) 영역 안에서
 *
 *     off = k_step*8192 + n_strip*1024 + n_half*512 + k_pair*32 + (n%16)*2 + (k%2)
 *
 * 이고 `n_half + 2*n_strip == (n%256)/16`이므로, 이는 곧
 * `k1/1, n4/2, k4/32, n4/512, k/8192` — 이 커널이 co-design된 그 스택이다.
 * 따라서 `koff = k_step*8192 + k_pair*32`, `noff[g] = g*512`.
 */

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#include "act_plan.hpp"
#include "bf16_rowmajor.hpp"

namespace amx {

struct GemmKernelTileK2BF16 : GemmKernel224BF16RowMajorNC {
  using Base = GemmKernel224BF16RowMajorNC;
  using BufferA = Base::BufferA;
  using BufferB = Base::BufferB;
  using BufferC = Base::BufferC;

  static std::string name() { return "TILE_K2_BF16"; }

  /// 프리페치 거리 (소비 스트림의 64 B 라인 수). 측정 우승자, T1 힌트.
  static constexpr int PF_DIST = 128;

  /// 한 타일 컬럼 = N_BLOCK(256) n × K_STEP(32) k.
  static constexpr int TILE_ELEMS = 8192;
  /// n-그룹: 16 n씩 16개. 각 그룹의 chunk가 정확히 64 B 라인 하나다.
  static constexpr int GROUP_N = 16;
  static constexpr int NGROUP = 16;
  static constexpr int GROUP_STRIDE = 512;

  // ── 인코더 (Seam B) ──────────────────────────────────────────────────
  static size_t plan_arena_bytes(int k) {
    const size_t pairs = (size_t)k / 2;
    const size_t cols = (size_t)k / K_STEP + 2;
    // pairs × 3 = koff + koff_tail + pval. 꼬리 표는 슬랩 n이 N_BLOCK의 배수일
    // 때 안 쓰이지만 자리는 늘 잡는다 — arena 크기가 k만의 함수라는 성질(moe_base가
    // expert별로 미리 재는 근거)을 유지하는 값이 그 낭비보다 크다.
    return (pairs * 3 + cols) * sizeof(uint32_t) + 64;
  }

  /// `mask == nullptr`이면 dense (전 페어 유지). 결과는 read-only이고 모든
  /// 워커가 공유한다.
  ///
  /// `n_slab > 0`이고 그것이 N_BLOCK의 배수가 아니면 마지막 부분 블록용 오프셋
  /// 표(`koff_tail`)를 같이 낸다. 꼬리에서 달라지는 것은 **타일 컬럼 stride
  /// 하나**다 (`c * n_w * K_STEP`): `GROUP_STRIDE`(512 = 16 n × 32 k)는 폭과
  /// 무관하고, 64 B chunk 성질(`_mm512_loadu_si512` 한 번 = 16 n × 2 k)도 그대로다.
  /// 그래서 꼬리 폭이 `GROUP_N`의 배수이기만 하면 된다.
  ///
  /// `n_slab == 0`(기본)이면 꼬리 표를 만들지 않는다 — 기존 호출부는 그대로
  /// 동작하고, `gemv_slab`이 부분 블록을 만나면 예전처럼 죽는다.
  static void build_plan(const PairMask* mask, const ggml_bf16_t* x, int k, void* arena,
                         ActivationPlan* out, int n_slab = 0) {
    const int cols = k / K_STEP;
    uint32_t* base = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(arena) + 63) & ~(uintptr_t)63);
    uint32_t* koff = base;
    uint32_t* koff_tail = koff + (size_t)k / 2;
    uint32_t* pval = koff_tail + (size_t)k / 2;
    uint32_t* seg = pval + (size_t)k / 2;

    const int n_tail = (n_slab > 0) ? (n_slab % N_BLOCK) : 0;
    assert(n_tail % GROUP_N == 0);  // 꼬리는 n-그룹 경계에 맞아야 한다
    const uint32_t tail_col = (uint32_t)n_tail * K_STEP;

    const uint16_t* xs = reinterpret_cast<const uint16_t*>(x);
    int np = 0;
    seg[0] = 0;
    for (int c = 0; c < cols; c++) {
      uint16_t bits = mask ? mask[c] : (uint16_t)0xFFFF;
      const uint32_t col_base = (uint32_t)c * TILE_ELEMS;
      const uint32_t col_base_tail = (uint32_t)c * tail_col;
      while (bits) {
        const int j = __builtin_ctz(bits);
        bits &= (uint16_t)(bits - 1);
        const int kk = c * K_STEP + 2 * j;
        const uint32_t in_col = (uint32_t)j * (2 * GROUP_N);
        koff[np] = col_base + in_col;
        koff_tail[np] = col_base_tail + in_col;
        pval[np] = (uint32_t)xs[kk] | ((uint32_t)xs[kk + 1] << 16);
        np++;
      }
      seg[c + 1] = (uint32_t)np;
    }
    out->koff = koff;
    out->pval = pval;
    out->seg = seg;
    out->npairs = np;
    out->nseg = cols;
    out->koff_tail = n_tail ? koff_tail : nullptr;
    out->ngroup_tail = n_tail / GROUP_N;
  }

  // ── 소비 커서 ────────────────────────────────────────────────────────
  // (컬럼, 그룹, 유지된 페어) 순서로 64 B chunk를 훑는 lookahead. 소비 순서
  // 그 자체이므로 프리페치가 스트림을 정확히 앞서간다.
  struct Cursor {
    int c, g, j, live;
  };

  static inline void cursor_step(Cursor& t, const uint32_t* seg, int ncols, int ngroup = NGROUP) {
    if ((uint32_t)++t.j < seg[t.c + 1]) return;
    if (++t.g < ngroup) {
      t.j = (int)seg[t.c];
      return;
    }
    t.g = 0;
    do {
      ++t.c;
    } while (t.c < ncols && seg[t.c + 1] == seg[t.c]);
    if (t.c >= ncols) {
      t.live = 0;
      return;
    }
    t.j = (int)seg[t.c];
  }

  static inline Cursor cursor_start(const uint32_t* seg, int ncols, int npairs, int ahead,
                                    int ngroup = NGROUP) {
    Cursor t{0, 0, 0, npairs > 0};
    if (t.live) {
      while (t.c < ncols && seg[t.c + 1] == seg[t.c]) ++t.c;
      if (t.c >= ncols)
        t.live = 0;
      else
        t.j = (int)seg[t.c];
    }
    for (int i = 0; i < ahead && t.live; ++i) cursor_step(t, seg, ncols, ngroup);
    return t;
  }

  /// 256 n 슈퍼블록 하나. y는 L1에 두고 (컬럼, 그룹) 방문마다 load/store 한 쌍
  /// (ymem 정책) — 누산기 16개가 레지스터에 상주하는 reg 정책의 A/B 상대이고,
  /// 이 변종이 측정 우승자다.
  static void super_block(const ggml_bf16_t* W, float* ys, const uint32_t* koff, const uint32_t* pval,
                          const uint32_t* seg, int ncols, int npairs) {
    Cursor cu = cursor_start(seg, ncols, npairs, PF_DIST);
    for (int c = 0; c < ncols; c++) {
      const int jlo = (int)seg[c], jhi = (int)seg[c + 1];
      if (jlo == jhi) continue;
#pragma GCC unroll 16
      for (int g = 0; g < NGROUP; g++) {
        const uint32_t go = (uint32_t)g * GROUP_STRIDE;
        float* yg = ys + GROUP_N * g;
        __m512 a0 = _mm512_loadu_ps(yg);
        for (int j = jlo; j < jhi; j++) {
          if (cu.live) {
            _mm_prefetch((const char*)(W + koff[cu.j] + (uint32_t)cu.g * GROUP_STRIDE), _MM_HINT_T1);
            cursor_step(cu, seg, ncols);
          }
          const __m512bh vv = (__m512bh)_mm512_set1_epi32((int)pval[j]);
          a0 = _mm512_dpbf16_ps(a0, vv, (__m512bh)_mm512_loadu_si512(W + koff[j] + go));
        }
        _mm512_storeu_ps(yg, a0);
      }
    }
  }

  /// 부분 블록용. 본체는 `super_block`과 같고 그룹 수만 런타임이다 — 그래서
  /// `#pragma GCC unroll 16`을 붙일 수 없다. 언롤을 잃는 것은 슬랩 전체에서
  /// **마지막 블록 하나**뿐이므로 주 경로의 코드젠은 손대지 않는다.
  static void super_block_partial(const ggml_bf16_t* W, float* ys, const uint32_t* koff,
                                  const uint32_t* pval, const uint32_t* seg, int ncols, int npairs,
                                  int ngroup) {
    Cursor cu = cursor_start(seg, ncols, npairs, PF_DIST, ngroup);
    for (int c = 0; c < ncols; c++) {
      const int jlo = (int)seg[c], jhi = (int)seg[c + 1];
      if (jlo == jhi) continue;
      for (int g = 0; g < ngroup; g++) {
        const uint32_t go = (uint32_t)g * GROUP_STRIDE;
        float* yg = ys + GROUP_N * g;
        __m512 a0 = _mm512_loadu_ps(yg);
        for (int j = jlo; j < jhi; j++) {
          if (cu.live) {
            _mm_prefetch((const char*)(W + koff[cu.j] + (uint32_t)cu.g * GROUP_STRIDE), _MM_HINT_T1);
            cursor_step(cu, seg, ncols, ngroup);
          }
          const __m512bh vv = (__m512bh)_mm512_set1_epi32((int)pval[j]);
          a0 = _mm512_dpbf16_ps(a0, vv, (__m512bh)_mm512_loadu_si512(W + koff[j] + go));
        }
        _mm512_storeu_ps(yg, a0);
      }
    }
  }

  /// m == 1 전용. [n0, n1)의 시작은 N_BLOCK 정렬이고, 마지막 블록은 슬랩 n이
  /// N_BLOCK의 배수가 아니면 좁을 수 있다 (그때는 plan에 꼬리 표가 있어야 한다).
  static void gemv_slab(int m, int n, int k, int n0, int n1, BufferA* ba, BufferB* bb, BufferC* bc,
                        const ActivationPlan* plan) {
    (void)m;
    (void)ba;  // pval이 활성값을 나르므로 A 접근이 없다 (Seam B)
    // `koff`는 타일 컬럼 stride를 `N_BLOCK * K_STEP`으로 접어 넣은 표다. 마지막
    // 블록이 좁으면 그 stride가 달라지므로 `koff_tail`이 필요하다 — 없으면 조용히
    // 남의 열을 읽느니 여기서 죽는다 (Release에는 assert가 없으니 호출부의 검증이
    // 여전히 1차 방어선이다).
    assert(n % N_BLOCK == 0 || plan->koff_tail != nullptr);
    float* c_row = bc->c;
    // ymem 정책은 누산기를 메모리에서 읽어 더한다 — 시작 전에 0으로 둔다.
    for (int j = n0; j < n1; j++) c_row[j] = 0.0f;
    for (int nb = n0; nb < n1; nb += N_BLOCK) {
      const ggml_bf16_t* W = bb->b + (size_t)nb * k;
      const int w = std::min(N_BLOCK, n - nb);
      if (w == N_BLOCK) {
        super_block(W, c_row + nb, plan->koff, plan->pval, plan->seg, plan->nseg, plan->npairs);
      } else {
        assert(plan->koff_tail != nullptr && plan->ngroup_tail == w / GROUP_N);
        super_block_partial(W, c_row + nb, plan->koff_tail, plan->pval, plan->seg, plan->nseg,
                            plan->npairs, plan->ngroup_tail);
      }
    }
  }
};

}  // namespace amx
