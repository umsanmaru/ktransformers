// tile_k1_fp8b128 게이트: kt 래퍼(BufferB 기하 + per-k plan 인코더 + gemv_row의 BufferC 주소)가
// 스칼라 dequant 레퍼런스와 일치하는가. dense / mask 전량 1 / 실제 per-k sparse 세 경우, 그리고
// 레이아웃 순열 자체(ktf8k1_off)가 (n, k) 전단사인지.
//
// 빌드 (헤더 전용, ggml 링크 없음; hwloc 헤더는 conda env에서):
//   CP=$HOME/miniconda3/envs/ktk-prism
//   g++ -O2 -std=c++20 -march=native -Ioperators/amx -I. -I../third_party \
//       -I../third_party/llama.cpp -I$CP/include -o /tmp/tile_k1_fp8-gate \
//       operators/amx/test/tile_k1_fp8-gate.cpp
//   (-DKTF8K1_DECODE_VIA_BF16=1 로 다른 디코드 경로도 같은 게이트를 통과해야 한다)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "la/tile_k1_fp8b128.hpp"

using namespace amx;
using K = GemmKernelTileK1FP8B128;

static ggml_bf16_t f2b(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  ggml_bf16_t b;
  b.bits = (uint16_t)(u >> 16);
  return b;
}
static float b2f(ggml_bf16_t b) {
  uint32_t u = (uint32_t)b.bits << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static double fp8_val(uint8_t c) {
  const int s = (c >> 7) & 1, e = (c >> 3) & 0xF, m = c & 7;
  if (e == 0) return 0.0;
  return (s ? -1.0 : 1.0) * std::ldexp(1.0 + m / 8.0, e - 7);
}

// 레이아웃 순열이 [0, n·k) 위의 전단사인지 — 겹치면 pack이 조용히 남의 원소를 덮는다.
static int check_bijection(int n, int k) {
  std::vector<uint8_t> hit((size_t)n * k, 0);
  for (int j = 0; j < n; ++j)
    for (int kk = 0; kk < k; ++kk) {
      const size_t off = ktf8k1_off(j, kk, k);
      if (off >= hit.size() || hit[off]) return 1;
      hit[off] = 1;
    }
  return 0;
}

static int run_case(int n, int k, int mode) {  // mode 0=dense, 1=mask1, 2=sparse
  std::mt19937 g(4321 + n + k + mode);
  std::uniform_int_distribution<int> dx(-4, 4);
  std::uniform_int_distribution<int> de(1, 14), dm(0, 7), ds(0, 1);
  std::uniform_int_distribution<int> dse(-3, 3);

  const int NB = n / K::SCALE_BLK, KB = k / K::SCALE_BLK;
  std::vector<ggml_bf16_t> x(k);
  std::vector<uint8_t> w((size_t)n * k);
  std::vector<float> sc((size_t)NB * KB);  // n-major scale_inv [n/128][k/128]
  for (auto& v : x) v = f2b((float)dx(g));
  for (auto& v : w) v = (uint8_t)((ds(g) << 7) | (de(g) << 3) | dm(g));
  for (auto& v : sc) v = std::ldexp(1.0f, dse(g));

  void* bp = std::aligned_alloc(64, K::BufferB::required_size(n, k, K::SCALE_BLK));
  K::BufferB bb(n, k, K::SCALE_BLK, bp);
  bb.from_mat(w.data(), sc.data(), 0, 1);
  void* ap = std::aligned_alloc(64, K::BufferA::required_size(K::M_STEP, k));
  K::BufferA ba(K::M_STEP, k, ap);
  ba.from_mat(1, x.data(), 0, 1);
  void* cp = std::aligned_alloc(64, K::BufferC::required_size(K::M_STEP, n));
  K::BufferC bc(K::M_STEP, n, cp);

  // per-k 마스크: mode 2는 k를 개별로 죽인다 (컬럼마다 다른 패턴, 홀짝 비대칭 포함).
  std::vector<uint32_t> mask(k / K::K_STEP, 0xFFFFFFFFu);
  if (mode == 2)
    for (size_t c = 0; c < mask.size(); ++c)
      mask[c] = (c % 3 == 0) ? 0u : (c % 3 == 1 ? 0xA53CC3A5u : 0x0000FFF1u);

  std::vector<uint8_t> arena(K::plan_arena_bytes(k));
  ActivationPlan plan;
  K::build_plan(mode == 0 ? nullptr : mask.data(), x.data(), k, arena.data(), &plan);
  K::gemv_slab(1, n, k, 0, n, &ba, &bb, &bc, &plan);

  int bad = 0;
  double worst = 0.0;
  for (int j = 0; j < n; ++j) {
    double acc = 0.0, mag = 0.0;
    for (int kb = 0; kb < KB; ++kb) {
      const double s = (double)sc[(size_t)(j / K::SCALE_BLK) * KB + kb];
      for (int t = 0; t < K::SCALE_BLK; ++t) {
        const int kk = kb * K::SCALE_BLK + t;
        if (mode != 0) {
          const int col = kk / K::K_STEP, bit = kk % K::K_STEP;
          if (!((mask[col] >> bit) & 1)) continue;
        }
        const double term = (double)b2f(x[kk]) * fp8_val(w[(size_t)j * k + kk]) * s;
        acc += term;
        mag += std::fabs(term);
      }
    }
    const double got = bc.get_submat(1, n, 0, (j / K::N_BLOCK) * K::N_BLOCK)[j % K::N_BLOCK];
    const double err = std::fabs(got - acc) / (mag + 1e-30);
    if (err > 1e-6) ++bad;
    if (err > worst) worst = err;
  }
  const char* tag = mode == 0 ? "dense" : (mode == 1 ? "mask1" : "sparse");
  std::printf("tile_k1_fp8 n=%4d k=%4d %-6s: nkept=%5d err/mag=%.3g %s\n", n, k, tag, plan.npairs, worst,
              bad == 0 ? "ok" : "MISMATCH");
  std::free(bp);
  std::free(ap);
  std::free(cp);
  return bad;
}

int main() {
  int bad = 0;
  for (int k : {128, 256, 1024}) {
    if (check_bijection(256, k)) {
      std::printf("tile_k1_fp8 layout n=256 k=%d NOT a bijection\n", k);
      ++bad;
    }
    for (int mode = 0; mode < 3; ++mode) bad += run_case(256, k, mode);
  }
  bad += check_bijection(768, 512);
  bad += run_case(512, 1024, 0);
  bad += run_case(768, 512, 2);
  std::printf(bad ? "tile_k1_fp8 gate FAILED\n" : "tile_k1_fp8 gate OK\n");
  return bad != 0;
}
