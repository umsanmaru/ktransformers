// tile_k2_fp8pt vs tile_k2_fp8b128 마이크로벤치 — 같은 코드 바이트, 같은 plan, 배율 기하만 다름.
// 단일 스레드, Mistral-Medium-3.5 형상의 노드 N shard(256 배수) 하나. dense / 페어 sp50.
// 캐시 밖 측정을 위해 weight 사본을 회전시킨다 (L3 상주 방지 — [[prism-tier-cache-residency]]).
//
// 빌드:
//   g++ -O2 -std=c++20 -march=native -Ioperators/amx -I. -I../third_party \
//       -I../third_party/llama.cpp -I$CONDA_PREFIX/include -o /tmp/tile_k2_fp8pt-bench \
//       operators/amx/test/tile_k2_fp8pt-bench.cpp
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "la/tile_k2_fp8b128.hpp"
#include "la/tile_k2_fp8pt.hpp"

using namespace amx;
using KB = GemmKernelTileK2FP8B128;
using KP = GemmKernelTileK2FP8PT;

static ggml_bf16_t f2b(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  ggml_bf16_t b;
  b.bits = (uint16_t)(u >> 16);
  return b;
}

struct Shape { const char* name; int n, k; };

template <class K>
static double bench(const char* tag, const Shape& sh, const std::vector<uint8_t>& w, const std::vector<float>& sc,
                    const std::vector<ggml_bf16_t>& x, const uint16_t* mask, int copies, int iters) {
  const int n = sh.n, k = sh.k;
  const int grp = K::SCALE_BLK;  // 128 또는 0
  std::vector<void*> bufs(copies);
  std::vector<typename K::BufferB*> bbs(copies);
  for (int c = 0; c < copies; ++c) {
    bufs[c] = std::aligned_alloc(64, K::BufferB::required_size(n, k, grp));
    bbs[c] = new typename K::BufferB(n, k, grp, bufs[c]);
    bbs[c]->from_mat(w.data(), sc.data(), 0, 1);
  }
  void* ap = std::aligned_alloc(64, K::BufferA::required_size(1, k));
  typename K::BufferA ba(1, k, ap);
  ba.from_mat(1, const_cast<ggml_bf16_t*>(x.data()), 0, 1);
  void* cp = std::aligned_alloc(64, K::BufferC::required_size(1, n));
  typename K::BufferC bc(1, n, cp);
  std::vector<uint8_t> arena(K::plan_arena_bytes(k));
  ActivationPlan plan;
  K::build_plan(mask, x.data(), k, arena.data(), &plan);

  for (int c = 0; c < copies; ++c) K::gemv_slab(1, n, k, 0, n, &ba, bbs[c], &bc, &plan);  // warm code path
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) K::gemv_slab(1, n, k, 0, n, &ba, bbs[i % copies], &bc, &plan);
  auto t1 = std::chrono::steady_clock::now();
  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  const double bytes = (double)plan.npairs * 128.0 * (n / 64);  // 유지 페어 × 64-n 청크
  std::printf("  %-8s %-6s n=%5d k=%5d npairs=%5d  %8.1f us  %6.1f GB/s\n", tag, K::name().c_str(), n, k,
              plan.npairs, us, bytes / us / 1e3);
  for (int c = 0; c < copies; ++c) { delete bbs[c]; std::free(bufs[c]); }
  std::free(ap);
  std::free(cp);
  return us;
}

int main(int argc, char** argv) {
  const int copies = argc > 1 ? std::atoi(argv[1]) : 24;  // 회전 사본 수 (L3 밖으로 밀어내기)
  const int iters = argc > 2 ? std::atoi(argv[2]) : 96;
  // Mistral-Medium-3.5: hidden 12288, ffn 28672. 노드 shard 예: 1/8 N × 1/4 K (cold 밴드) 정도.
  const Shape shapes[] = {
      {"gateup", 3584, 3072},   // N shard 28672/8, K 12288/4
      {"down", 1536, 7168},     // N shard 12288/8, K 28672/4
      {"o_proj", 1536, 3072},   // N shard 12288/8, K 12288/4
  };
  std::mt19937 g(77);
  std::uniform_int_distribution<int> de(1, 14), dm(0, 7), ds(0, 1);
  std::normal_distribution<float> dx(0.f, 1.f);
  for (const auto& sh : shapes) {
    std::vector<uint8_t> w((size_t)sh.n * sh.k);
    for (auto& v : w) v = (uint8_t)((ds(g) << 7) | (de(g) << 3) | dm(g));
    std::vector<float> sc((size_t)(sh.n / 128) * (sh.k / 128), 0.75f);  // b128 표; pt는 sc[0]만 읽는다
    std::vector<ggml_bf16_t> x(sh.k);
    for (auto& v : x) v = f2b(dx(g));
    std::vector<uint16_t> mask_half(sh.k / 32);
    for (auto& m : mask_half) m = (uint16_t)(g() & 0xFFFF);  // 페어 ~50% 유지
    std::printf("%s (weights %.1f MB/copy, %d copies rotating)\n", sh.name, sh.n * (double)sh.k / 1e6, copies);
    double b_d = bench<KB>("dense", sh, w, sc, x, nullptr, copies, iters);
    double p_d = bench<KP>("dense", sh, w, sc, x, nullptr, copies, iters);
    double b_s = bench<KB>("sp50", sh, w, sc, x, mask_half.data(), copies, iters);
    double p_s = bench<KP>("sp50", sh, w, sc, x, mask_half.data(), copies, iters);
    std::printf("  -> pt/b128: dense %.3f  sp50 %.3f\n", p_d / b_d, p_s / b_s);
  }
  return 0;
}
