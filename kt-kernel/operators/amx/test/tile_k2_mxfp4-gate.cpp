// tile_k2_mxfp4 게이트: kt 래퍼(BufferB 기하 + plan 인코더 + gemv_row의 BufferC 주소)가
// 스칼라 dequant 레퍼런스와 일치하는가. dense / mask 전량 1 / 실제 sparse 세 경우.
//
// tile_k2_fp8-gate.cpp의 형제다. 다른 것은 셋: 코드가 nibble(e2m1)이고, 배율이 32-k
// 컬럼마다 하나인 E8M0(2의 거듭제곱)이며, BufferB가 codes/scales를 따로 받는다
// (`from_raw_mat` + `from_scales_bf16`).
//
// 빌드 (헤더 전용, ggml 링크 없음):
//   g++ -O2 -std=c++20 -march=native -Ioperators/amx -I. -I../third_party \
//       -I../third_party/llama.cpp -o /tmp/tile_k2_mxfp4-gate \
//       operators/amx/test/tile_k2_mxfp4-gate.cpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// tile_k2_mxfp4.hpp는 BufferC로 BufferCReduceImpl(amx_buffers.hpp)을 쓰는데 그 헤더를
// 스스로 include하지 않는다 — tile_k2-gate.cpp와 같이 amx_kernels.hpp를 먼저 들인다.
#include "la/amx_kernels.hpp"
#include "la/tile_k2_mxfp4.hpp"

using namespace amx;
using K = GemmKernelTileK2MXFP4;

// 작은 정수·2의 거듭제곱만 쓰므로 상위 16비트 절단이 무손실이다.
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

// e2m1 nibble → 값. bit3 = 부호, bit2..0 = {0,.5,1,1.5,2,3,4,6}
// (`ktm4_e2m1_lut`의 bf16 상수와 같은 표).
static double fp4_val(uint8_t c) {
  static const double v[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  return ((c >> 3) & 1) ? -v[c & 7] : v[c & 7];
}

// mode 0=dense, 1=mask1, 2=sparse. stress=true면 활성을 작은 정수 대신 임의 bf16으로
// 뽑는다 — 코드(e2m1)와 배율(2의 거듭제곱)은 정확값이라 정수 활성에서는 fp32 누산이
// 무손실이고, 그러면 누산 **순서**를 검증하지 못한다.
static int run_case(int n, int k, int mode, bool stress = false) {
  std::mt19937 g(4321 + n + k + mode + (stress ? 7777 : 0));
  std::uniform_int_distribution<int> dx(-4, 4);
  std::uniform_real_distribution<float> dxr(-3.0f, 3.0f);
  std::uniform_int_distribution<int> dcode(0, 15);
  std::uniform_int_distribution<int> dse(-3, 3);

  const int SC = k / K::K_STEP;  // 배율은 32-k 컬럼마다 하나
  std::vector<ggml_bf16_t> x(k);
  std::vector<uint8_t> w((size_t)n * k / 2);      // 행우선 nibble [n][k/2], 하위 nibble = 짝수 k
  std::vector<ggml_bf16_t> sc((size_t)n * SC);    // [n][k/32], 2의 거듭제곱
  for (auto& v : x) v = f2b(stress ? dxr(g) : (float)dx(g));
  for (auto& v : w) v = (uint8_t)(dcode(g) | (dcode(g) << 4));
  for (auto& v : sc) v = f2b(std::ldexp(1.0f, dse(g)));

  void* bp = std::aligned_alloc(64, K::BufferB::required_size(n, k, K::K_STEP));
  K::BufferB bb(n, k, K::K_STEP, bp);
  bb.from_raw_mat(w.data(), 0, 1);
  bb.from_scales_bf16(sc.data());
  void* ap = std::aligned_alloc(64, K::BufferA::required_size(K::M_STEP, k));
  K::BufferA ba(K::M_STEP, k, ap);
  ba.from_mat(1, x.data(), 0, 1);
  void* cp = std::aligned_alloc(64, K::BufferC::required_size(K::M_STEP, n));
  K::BufferC bc(K::M_STEP, n, cp);

  // 마스크: mode 2는 페어의 절반을 죽인다 (컬럼마다 다른 패턴).
  std::vector<uint16_t> mask(k / K::K_STEP, 0xFFFF);
  if (mode == 2)
    for (size_t c = 0; c < mask.size(); ++c) mask[c] = (uint16_t)((c % 3 == 0) ? 0x0000 : 0xA53C);

  std::vector<uint8_t> arena(K::plan_arena_bytes(k));
  ActivationPlan plan;
  K::build_plan(mode == 0 ? nullptr : mask.data(), x.data(), k, arena.data(), &plan);
  K::gemv_slab(1, n, k, 0, n, &ba, &bb, &bc, &plan);

  // 레퍼런스: 유지된 페어만, fp64 누산. 커널은 fp32 누산이므로 판정은 **항의 크기 합**
  // 대비 오차다 (상쇄가 큰 난수 입력에서 |acc| 대비 상대오차는 의미가 없다).
  int bad = 0;
  double worst = 0.0;
  for (int j = 0; j < n; ++j) {
    double acc = 0.0, mag = 0.0;
    for (int kk = 0; kk < k; ++kk) {
      const int col = kk / K::K_STEP;
      if (mode != 0) {
        const int pj = (kk % K::K_STEP) / 2;
        if (!((mask[col] >> pj) & 1)) continue;
      }
      const uint8_t byte = w[((size_t)j * k + kk) / 2];
      const uint8_t code = (kk & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
      const double s = (double)b2f(sc[(size_t)j * SC + col]);
      const double term = (double)b2f(x[kk]) * fp4_val(code) * s;
      acc += term;
      mag += std::fabs(term);
    }
    const double got = bc.get_submat(1, n, 0, (j / K::N_BLOCK) * K::N_BLOCK)[j % K::N_BLOCK];
    const double err = std::fabs(got - acc) / (mag + 1e-30);
    if (err > 1e-6) ++bad;
    if (err > worst) worst = err;
  }
  const char* tag = mode == 0 ? "dense" : (mode == 1 ? "mask1" : "sparse");
  if (stress) tag = mode == 2 ? "sp+rnd" : "dn+rnd";
  std::printf("tile_k2_mxfp4 n=%4d k=%5d %-6s: npairs=%5d err/mag=%.3g %s\n", n, k, tag, plan.npairs, worst,
              bad == 0 ? "ok" : "MISMATCH");
  std::free(bp);
  std::free(ap);
  std::free(cp);
  return bad;
}

int main() {
  int bad = 0;
  for (int k : {32, 128, 256, 1024}) {
    for (int mode = 0; mode < 3; ++mode) bad += run_case(256, k, mode);
  }
  bad += run_case(512, 1024, 0);
  bad += run_case(768, 512, 2);
  bad += run_case(1024, 2048, 1);
  bad += run_case(512, 12288, 0);  // 실제 plan 치수에 가까운 긴 K
  // fp32 누산 순서 검증 (임의 활성 → 곱이 더 이상 정확값이 아니다)
  bad += run_case(256, 1024, 0, /*stress=*/true);
  bad += run_case(256, 1024, 2, /*stress=*/true);
  bad += run_case(512, 12288, 0, /*stress=*/true);
  std::printf(bad ? "tile_k2_mxfp4 gate FAILED\n" : "tile_k2_mxfp4 gate OK\n");
  return bad != 0;
}
