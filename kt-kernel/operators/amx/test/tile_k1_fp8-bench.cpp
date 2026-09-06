// tile_k1 vs tile_k2 fp8b128 마이크로벤치 — dense / per-k sp / pair sp, E개 expert 회전(캐시 회피).
// 빌드: tile_k1_fp8-gate.cpp의 명령에 -fopenmp 추가. 실행: OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
//   /tmp/tile_k1_fp8-bench [N=4096] [K=2048] [E=16] [iters=20]
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "la/tile_k1_fp8b128.hpp"
#include "la/tile_k2_fp8b128.hpp"

using namespace amx;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class K, class Mask>
struct Set {
  int N, Kd, E;
  std::vector<void*> bufs;
  std::vector<std::unique_ptr<typename K::BufferB>> bb;
  std::vector<uint8_t> arena;
  std::vector<float> y;
  Set(int N, int Kd, int E, std::mt19937& g) : N(N), Kd(Kd), E(E), y((size_t)N) {
    std::vector<uint8_t> w((size_t)N * Kd);
    std::vector<float> sc((size_t)(N / 128) * (Kd / 128), 1.0f);
    std::uniform_int_distribution<int> dw(0, 255);
    for (int e = 0; e < E; ++e) {
      for (auto& v : w) v = (uint8_t)(dw(g) & 0x7f);  // 유효 e4m3(NaN 없음)
      void* p = std::aligned_alloc(64, K::BufferB::required_size(N, Kd, 128));
      bufs.push_back(p);
      bb.emplace_back(new typename K::BufferB(N, Kd, 128, p));
      bb.back()->from_mat(w.data(), sc.data(), 0, 1);
    }
    arena.resize(K::plan_arena_bytes(Kd));
  }
  ~Set() { for (void* p : bufs) std::free(p); }
  // 한 expert 전체 N을 supers 단위로 스레드 분할.
  double run(const Mask* mask, const ggml_bf16_t* x, int iters) {
    ActivationPlan plan;
    K::build_plan(mask, x, Kd, arena.data(), &plan);
    double best = 1e9;
    for (int it = 0; it < iters; ++it) {
      const int e = it % E;
      const double t0 = now();
#pragma omp parallel
      {
        const int nth = omp_get_num_threads(), ith = omp_get_thread_num();
        auto [n0, n1] = K::split_range_n(N, ith, nth);
        if (n0 < n1) {
          if constexpr (std::is_same_v<K, GemmKernelTileK1FP8B128>)
            ktf8k1_gemv(bb[e]->b, bb[e]->s, plan.koff, plan.pval, plan.seg, plan.npairs, N, Kd, y.data(), n0, n1, K::PF_DIST);
          else
            ktf8_gemv(bb[e]->b, bb[e]->s, plan.koff, plan.pval, plan.seg, plan.npairs, N, Kd, y.data(), n0, n1, K::PF_DIST);
        }
      }
      best = std::min(best, now() - t0);
    }
    return best * 1e3;
  }
};

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 4096, Kd = argc > 2 ? atoi(argv[2]) : 2048;
  const int E = argc > 3 ? atoi(argv[3]) : 16, iters = argc > 4 ? atoi(argv[4]) : 20;
  std::mt19937 g(7);
  std::vector<ggml_bf16_t> x(Kd);
  std::uniform_real_distribution<float> dx(-1.f, 1.f);
  for (auto& v : x) { float f = dx(g); uint32_t u; memcpy(&u, &f, 4); v.bits = (uint16_t)(u >> 16); }
  Set<GemmKernelTileK1FP8B128, uint32_t> k1(N, Kd, E, g);
  Set<GemmKernelTileK2FP8B128, uint16_t> k2(N, Kd, E, g);
  const int cols = Kd / 32;
  std::printf("N=%d K=%d E=%d threads=%d  (ms, min of %d; k1 decode=%s)\n", N, Kd, E, omp_get_max_threads(), iters,
              KTF8K1_DECODE_VIA_BF16 ? "bf16" : "fp32");
  std::printf("%-14s %8s %8s\n", "case", "k1", "k2");
  std::printf("%-14s %8.3f %8.3f\n", "dense", k1.run(nullptr, x.data(), iters), k2.run(nullptr, x.data(), iters));
  for (double s : {0.25, 0.5, 0.75}) {
    // per-k: 각 k 독립 생존 확률 1-s. pair(k2): 페어 생존 = 두 k 중 하나라도 산다 → 1-s².
    std::vector<uint32_t> m1(cols, 0);
    std::vector<uint16_t> m2(cols, 0);
    std::bernoulli_distribution keep(1.0 - s);
    for (int c = 0; c < cols; ++c)
      for (int j = 0; j < 32; ++j)
        if (keep(g)) { m1[c] |= 1u << j; m2[c] |= (uint16_t)(1u << (j >> 1)); }
    // survival-matched pair mask (같은 바이트): 페어 생존 1-s
    std::vector<uint16_t> m2m(cols, 0);
    for (int c = 0; c < cols; ++c)
      for (int j = 0; j < 16; ++j)
        if (keep(g)) m2m[c] |= (uint16_t)(1u << j);
    char tag[32];
    snprintf(tag, sizeof tag, "perk sp%02d", (int)(s * 100));
    std::printf("%-14s %8.3f %8.3f   (k2 pair-derived, kept pairs %.2f)\n", tag, k1.run(m1.data(), x.data(), iters),
                k2.run(m2.data(), x.data(), iters), 1.0 - s * s);
    snprintf(tag, sizeof tag, "pair sp%02d", (int)(s * 100));
    std::printf("%-14s %8s %8.3f   (k2 own pair mask, kept %.2f)\n", tag, "-", k2.run(m2m.data(), x.data(), iters), 1.0 - s);
  }
  return 0;
}
