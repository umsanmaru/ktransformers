// tile_k2_fp8pt 게이트: per-tensor 배율 커널(BufferB 기하 + plan 인코더 + 16-누산기 gemv)이
// 스칼라 dequant 레퍼런스와 일치하는가. dense / mask 전량 1 / 실제 sparse, 그리고 k가 128의
// 배수가 아닌 경우(b128에서는 불가능했던 기하)와 2의 거듭제곱이 아닌 배율.
//
// 빌드 (헤더 전용, ggml 링크 없음):
//   g++ -O2 -std=c++20 -march=native -Ioperators/amx -I. -I../third_party \
//       -I../third_party/llama.cpp -o /tmp/tile_k2_fp8pt-gate \
//       operators/amx/test/tile_k2_fp8pt-gate.cpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "la/tile_k2_fp8pt.hpp"

using namespace amx;
using K = GemmKernelTileK2FP8PT;

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

// mode 0=dense, 1=mask1, 2=sparse. pow2_scale=false면 임의 fp32 배율 (곱이 부정확 → 허용치 완화).
static int run_case(int n, int k, int mode, bool pow2_scale) {
  std::mt19937 g(4321 + n + k + mode + (pow2_scale ? 0 : 7));
  std::uniform_int_distribution<int> dx(-4, 4);
  std::uniform_int_distribution<int> de(1, 14), dm(0, 7), ds(0, 1);
  std::uniform_int_distribution<int> dse(-3, 3);
  std::uniform_real_distribution<float> dsf(0.001f, 3.0f);

  std::vector<ggml_bf16_t> x(k);
  std::vector<uint8_t> w((size_t)n * k);
  for (auto& v : x) v = f2b((float)dx(g));
  for (auto& v : w) v = (uint8_t)((ds(g) << 7) | (de(g) << 3) | dm(g));
  const float sc = pow2_scale ? std::ldexp(1.0f, dse(g)) : dsf(g);

  void* bp = std::aligned_alloc(64, K::BufferB::required_size(n, k, 0));
  K::BufferB bb(n, k, 0, bp);
  bb.from_mat(w.data(), &sc, 0, 1);
  void* ap = std::aligned_alloc(64, K::BufferA::required_size(K::M_STEP, k));
  K::BufferA ba(K::M_STEP, k, ap);
  ba.from_mat(1, x.data(), 0, 1);
  void* cp = std::aligned_alloc(64, K::BufferC::required_size(K::M_STEP, n));
  K::BufferC bc(K::M_STEP, n, cp);
  // 덧쓰기 계약 검사: 쓰레기로 채워두고 커널이 전부 덮는지 본다.
  std::memset(cp, 0x7f, K::BufferC::required_size(K::M_STEP, n));

  std::vector<uint16_t> mask(k / K::K_STEP, 0xFFFF);
  if (mode == 2)
    for (size_t c = 0; c < mask.size(); ++c) mask[c] = (uint16_t)((c % 3 == 0) ? 0x0000 : 0xA53C);

  std::vector<uint8_t> arena(K::plan_arena_bytes(k));
  ActivationPlan plan;
  K::build_plan(mode == 0 ? nullptr : mask.data(), x.data(), k, arena.data(), &plan);
  K::gemv_slab(1, n, k, 0, n, &ba, &bb, &bc, &plan);

  int bad = 0;
  double worst = 0.0;
  const double tol = pow2_scale ? 1e-6 : 4e-6;
  for (int j = 0; j < n; ++j) {
    double acc = 0.0, mag = 0.0;
    for (int kk = 0; kk < k; ++kk) {
      if (mode != 0) {
        const int col = kk / K::K_STEP, pj = (kk % K::K_STEP) / 2;
        if (!((mask[col] >> pj) & 1)) continue;
      }
      const double term = (double)b2f(x[kk]) * fp8_val(w[(size_t)j * k + kk]) * (double)sc;
      acc += term;
      mag += std::fabs(term);
    }
    const double got = bc.get_submat(1, n, 0, (j / K::N_BLOCK) * K::N_BLOCK)[j % K::N_BLOCK];
    const double err = std::fabs(got - acc) / (mag + 1e-30);
    if (err > tol) ++bad;
    if (err > worst) worst = err;
  }
  const char* tag = mode == 0 ? "dense" : (mode == 1 ? "mask1" : "sparse");
  std::printf("tile_k2_fp8pt n=%4d k=%5d %-6s scale=%-8.4g: npairs=%5d err/mag=%.3g %s\n", n, k, tag, sc,
              plan.npairs, worst, bad == 0 ? "ok" : "MISMATCH");
  std::free(bp);
  std::free(ap);
  std::free(cp);
  return bad;
}

int main() {
  int bad = 0;
  for (int k : {32, 96, 160, 1024}) {  // 96/160: 128의 배수가 아님 (per-tensor에서만 허용)
    for (int mode = 0; mode < 3; ++mode) bad += run_case(256, k, mode, true);
  }
  bad += run_case(512, 1024, 0, true);
  bad += run_case(768, 544, 2, true);
  bad += run_case(512, 12288, 0, false);  // Mistral hidden K, 임의 배율
  bad += run_case(256, 2048, 2, false);
  std::printf(bad ? "tile_k2_fp8pt gate FAILED\n" : "tile_k2_fp8pt gate OK\n");
  return bad != 0;
}
