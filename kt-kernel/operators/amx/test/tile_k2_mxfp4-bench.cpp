// tile_k2_mxfp4 gemv_slab 마이크로 벤치 — tile_k2-bench.cpp(bf16)의 형제.
//
// 같은 방법론이라 두 커널의 수치를 직접 비교할 수 있다: L3(22.5 MB)를 훨씬 넘는 풀에
// BufferB 사본을 여러 벌 잡고 호출마다 다른 사본을 읽는다. 대조군은 같은 버퍼를 순수
// 순차 읽기한 시간이고, 커널이 그 값을 넘을 수 없다.
//
// bf16과 다른 것은 바이트 회계뿐이다: weight가 코드 n*k/2 B + E8M0 배율 n*k/32 B라
// 원소당 0.53125 B다 (bf16은 2 B) — llama.cpp block_mxfp4(17 B/32)와 같다.
//
// 빌드:
//   g++ -O2 -std=c++20 -march=native -fopenmp -Ioperators/amx -I. -I../third_party \
//       -I../third_party/llama.cpp -o /tmp/tile_k2_mxfp4-bench \
//       operators/amx/test/tile_k2_mxfp4-bench.cpp
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "la/amx_kernels.hpp"
#include "la/tile_k2_mxfp4.hpp"

using namespace amx;
using K = GemmKernelTileK2MXFP4;
using Clock = std::chrono::steady_clock;

static ggml_bf16_t f2b(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  ggml_bf16_t b;
  b.bits = (uint16_t)(u >> 16);
  return b;
}

static const size_t POOL_BYTES = 1024ull << 20;  // tile_k2-bench와 같은 1 GB

struct Case {
  double us;
  double gbps;
  int npairs;
  double raw_us;
  double raw_gbps;
};

// 이 커널이 한 번에 훑는 weight 바이트 (코드 + 배율).
static inline double wbytes(int n, int k) { return (double)n * k / 2.0 + (double)n * k / 32.0; }

static Case run(int n, int k, double keep, int iters) {
  std::mt19937 g(99 + n + k);
  std::uniform_int_distribution<int> dx(-4, 4);
  std::uniform_int_distribution<int> dcode(0, 15);
  std::uniform_int_distribution<int> dse(-3, 3);

  const int SC = k / K::K_STEP;
  std::vector<ggml_bf16_t> x(k);
  std::vector<uint8_t> w((size_t)n * k / 2);
  std::vector<ggml_bf16_t> sc((size_t)n * SC);
  for (auto& v : x) v = f2b((float)dx(g));
  for (auto& v : w) v = (uint8_t)(dcode(g) | (dcode(g) << 4));
  for (auto& v : sc) v = f2b(std::ldexp(1.0f, dse(g)));

  // keep 비율만큼 페어를 살린다 (컬럼당 앞쪽 비트 — 값이 아니라 시간만 본다)
  const int cols = k / K::K_STEP;
  const int keep_bits = (int)(16 * keep + 0.5);
  std::vector<uint16_t> mask(cols, 0);
  for (auto& mb : mask) mb = (keep_bits >= 16) ? 0xFFFF : (uint16_t)((1u << keep_bits) - 1);

  const size_t bsz = K::BufferB::required_size(n, k, K::K_STEP);
  const int copies = (int)std::max<size_t>(2, POOL_BYTES / bsz);
  static bool told = false, told2 = false;
  if (!told) {
    std::printf("   [진단] weight=%.2f MB, BufferB::required_size=%.2f MB, copies=%d, 풀=%.2f MB\n",
                wbytes(n, k) / 1e6, bsz / 1e6, copies, copies * bsz / 1e6);
    told = true;
  }
  std::vector<void*> bps(copies);
  std::vector<K::BufferB*> bbs(copies);
  for (int i = 0; i < copies; i++) {
    bps[i] = std::aligned_alloc(64, bsz);
    bbs[i] = new K::BufferB(n, k, K::K_STEP, bps[i]);
    bbs[i]->from_raw_mat(w.data(), 0, 1);
    bbs[i]->from_scales_bf16(sc.data());
  }
  void* ap = std::aligned_alloc(64, K::BufferA::required_size(K::M_STEP, k));
  K::BufferA ba(K::M_STEP, k, ap);
  ba.from_mat(1, x.data(), 0, 1);
  void* cp = std::aligned_alloc(64, K::BufferC::required_size(K::M_STEP, n));
  K::BufferC bc(K::M_STEP, n, cp);

  std::vector<uint8_t> arena(K::plan_arena_bytes(k));
  ActivationPlan plan;
  K::build_plan(mask.data(), x.data(), k, arena.data(), &plan);

  // 대조군 3종 (누산기 1/4, 프리페치 유무) — 커널이 넘을 수 없는 값.
  auto raw_read = [&](int i, int variant) {
    const char* q = (const char*)bbs[i % copies]->b;
    const size_t nb = (size_t)wbytes(n, k);
    const size_t PF = 128 * 64;
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    if (variant == 0) {
      for (size_t j = 0; j + 64 <= nb; j += 64) {
        if (j + PF < nb) _mm_prefetch(q + j + PF, _MM_HINT_T1);
        a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void*)(q + j)));
      }
    } else if (variant == 1) {
      for (size_t j = 0; j + 256 <= nb; j += 256) {
        if (j + PF < nb) _mm_prefetch(q + j + PF, _MM_HINT_T1);
        a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void*)(q + j)));
        a1 = _mm512_add_epi64(a1, _mm512_loadu_si512((const void*)(q + j + 64)));
        a2 = _mm512_add_epi64(a2, _mm512_loadu_si512((const void*)(q + j + 128)));
        a3 = _mm512_add_epi64(a3, _mm512_loadu_si512((const void*)(q + j + 192)));
      }
    } else {
      for (size_t j = 0; j + 256 <= nb; j += 256) {
        a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void*)(q + j)));
        a1 = _mm512_add_epi64(a1, _mm512_loadu_si512((const void*)(q + j + 64)));
        a2 = _mm512_add_epi64(a2, _mm512_loadu_si512((const void*)(q + j + 128)));
        a3 = _mm512_add_epi64(a3, _mm512_loadu_si512((const void*)(q + j + 192)));
      }
    }
    a0 = _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
    return _mm512_reduce_add_epi64(a0);
  };
  if (!told2) {  // 풀이 정말 물리 메모리를 갖고 있는가
    FILE* f = fopen("/proc/self/status", "r");
    char ln[256];
    while (f && fgets(ln, sizeof ln, f))
      if (!strncmp(ln, "VmRSS:", 6)) { std::printf("   [RSS] %s", ln); break; }
    if (f) fclose(f);
    size_t touched = 0;
    for (int i = 0; i < copies; i++) {
      volatile const char* q = (const char*)bbs[i]->b;
      for (size_t j = 0; j < bsz; j += 4096) touched += q[j];
    }
    std::printf("   [터치 확인] %d 사본 x %.2f MB 읽기 완료 (합 %zu)\n", copies, bsz / 1e6, touched);
  }
  double raw_us = 1e18;
  for (int v = 0; v < 3; v++) {
    volatile long long sk = 0;
    for (int i = 0; i < 20; i++) sk += raw_read(i, v);
    auto r0 = Clock::now();
    for (int i = 0; i < iters; i++) sk += raw_read(i, v);
    auto r1 = Clock::now();
    const double u = std::chrono::duration<double, std::micro>(r1 - r0).count() / iters;
    if (!told2) std::printf("   [대조군 v%d] %8.3f us = %5.1f GB/s\n", v, u, wbytes(n, k) / (u * 1e-6) / 1e9);
    raw_us = std::min(raw_us, u);
    (void)sk;
  }
  told2 = true;

  for (int i = 0; i < 20; i++) K::gemv_slab(1, n, k, 0, n, &ba, bbs[i % copies], &bc, &plan);
  auto t0 = Clock::now();
  for (int i = 0; i < iters; i++) K::gemv_slab(1, n, k, 0, n, &ba, bbs[i % copies], &bc, &plan);
  auto t1 = Clock::now();
  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

  // 실제로 읽은 바이트: 코드는 유지된 페어 비율만, 배율은 살아있는 컬럼 전부.
  const double frac = (double)plan.npairs / (k / 2);
  const double bytes = (double)n * k / 2.0 * frac + (double)n * k / 32.0;

  for (int i = 0; i < copies; i++) { delete bbs[i]; std::free(bps[i]); }
  std::free(ap); std::free(cp);
  return {us, bytes / (us * 1e-6) / 1e9, plan.npairs, raw_us, wbytes(n, k) / (raw_us * 1e-6) / 1e9};
}

int main(int argc, char** argv) {
  const int k = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 3000;
  std::vector<int> ns = {256, 512, 768, 1024, 1280};
  if (argc > 3) {
    ns.clear();
    for (const char* q = argv[3]; *q;) {
      ns.push_back((int)std::strtol(q, (char**)&q, 10));
      if (*q == ',') ++q;
    }
  }
  std::printf("tile_k2_mxfp4  k=%d, iters=%d (weight 원소당 0.53125 B)\n\n", k, iters);
  for (double keep : {1.0, 0.5}) {
    std::printf("== keep=%.1f ==\n", keep);
    std::printf("%6s %10s %9s %10s %9s %8s\n", "n", "us", "GB/s", "raw_us", "raw_GB/s", "eff");
    for (int n : ns) {
      Case c = run(n, k, keep, iters);
      std::printf("%6d %10.3f %9.1f %10.3f %9.1f %7.0f%%\n", n, c.us, c.gbps, c.raw_us, c.raw_gbps,
                  c.gbps / c.raw_gbps * 100.0);
    }
    std::printf("\n");
  }
  return 0;
}
