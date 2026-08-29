// tile_k2_fp8b128 게이트: kt 래퍼(BufferB 기하 + plan 인코더 + gemv_row의 BufferC 주소)가
// 스칼라 dequant 레퍼런스와 일치하는가. dense / mask 전량 1 / 실제 sparse 세 경우.
//
// 빌드 (헤더 전용, ggml 링크 없음):
//   g++ -O2 -std=c++20 -march=native -Ioperators/amx -I. -I../third_party \
//       -I../third_party/llama.cpp -o /tmp/tile_k2_fp8-gate \
//       operators/amx/test/tile_k2_fp8-gate.cpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "la/tile_k2_fp8b128.hpp"

using namespace amx;
using K = GemmKernelTileK2FP8B128;

// 작은 정수만 쓰므로 상위 16비트 절단이 무손실이다.
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

// e4m3 바이트 → 값 (지수 0 = 0, denormal/NaN 코드는 생성하지 않는다).
static double fp8_val(uint8_t c) {
  const int s = (c >> 7) & 1, e = (c >> 3) & 0xF, m = c & 7;
  if (e == 0) return 0.0;
  return (s ? -1.0 : 1.0) * std::ldexp(1.0 + m / 8.0, e - 7);
}

static int run_case(int n, int k, int mode) {  // mode 0=dense, 1=mask1, 2=sparse
  std::mt19937 g(9876 + n + k + mode);
  std::uniform_int_distribution<int> dx(-4, 4);
  std::uniform_int_distribution<int> de(1, 14), dm(0, 7), ds(0, 1);
  std::uniform_int_distribution<int> dse(-3, 3);

  const int NB = n / K::SCALE_BLK, KB = k / K::SCALE_BLK;
  std::vector<ggml_bf16_t> x(k);
  std::vector<uint8_t> w((size_t)n * k);
  std::vector<float> sc((size_t)NB * KB);  // n-major scale_inv [n/128][k/128]
  for (auto& v : x) v = f2b((float)dx(g));
  for (auto& v : w) v = (uint8_t)((ds(g) << 7) | (de(g) << 3) | dm(g));
  for (auto& v : sc) v = std::ldexp(1.0f, dse(g));  // 2의 거듭제곱 → 곱이 정확

  void* bp = std::aligned_alloc(64, K::BufferB::required_size(n, k, K::SCALE_BLK));
  K::BufferB bb(n, k, K::SCALE_BLK, bp);
  bb.from_mat(w.data(), sc.data(), 0, 1);
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
    for (int kb = 0; kb < KB; ++kb) {
      const double s = (double)sc[(size_t)(j / K::SCALE_BLK) * KB + kb];
      for (int t = 0; t < K::SCALE_BLK; ++t) {
        const int kk = kb * K::SCALE_BLK + t;
        if (mode != 0) {
          const int col = kk / K::K_STEP, pj = (kk % K::K_STEP) / 2;
          if (!((mask[col] >> pj) & 1)) continue;
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
  std::printf("tile_k2_fp8 n=%4d k=%4d %-6s: npairs=%5d err/mag=%.3g %s\n", n, k, tag, plan.npairs, worst,
              bad == 0 ? "ok" : "MISMATCH");
  std::free(bp);
  std::free(ap);
  std::free(cp);
  return bad;
}

int main() {
  int bad = 0;
  for (int k : {128, 256, 1024}) {
    for (int mode = 0; mode < 3; ++mode) bad += run_case(256, k, mode);
  }
  bad += run_case(512, 1024, 0);
  bad += run_case(768, 512, 2);
  std::printf(bad ? "tile_k2_fp8 gate FAILED\n" : "tile_k2_fp8 gate OK\n");
  return bad != 0;
}
