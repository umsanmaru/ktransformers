// tile_k2 gemv_slab 마이크로 벤치 — 부분 n-블록의 값을 잰다.
//
// 세 질문: (1) 꼬리 블록이 온전한 블록 대비 원소당 얼마나 비싼가(언롤 손실),
// (2) 그래서 n을 256 배수로 **패딩**하는 우회보다 나은가, (3) 주 경로가
// 부분 블록 지원 때문에 느려지지 않았는가.
//
// 캐시: n·k가 작으면 weight가 L2에 들어앉아 작은 n이 부당하게 빨라진다. 그래서
// L3(소켓당 22.5 MB)보다 큰 풀을 잡고 호출마다 다른 사본을 읽는다.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include "la/amx_kernels.hpp"
#include "la/amx_raw_kernels.hpp"
#include "la/bf16_rowmajor.hpp"
#include "la/tile_k2.hpp"

using namespace amx;
using K = GemmKernelTileK2BF16;
using Clock = std::chrono::steady_clock;

static ggml_bf16_t f2b(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  ggml_bf16_t b;
  b.bits = (uint16_t)(u >> 16);
  return b;
}

static const size_t POOL_BYTES = 1024ull << 20;  // L3(22.5MB)의 45배 — 48MB로는 절반이 상주해
                                                // 단일 스레드가 53 GB/s로 보였다 (DRAM 상한 17)

struct Case {
  double us;        // 호출당
  double gbps;      // 실제로 읽은 weight 바이트 기준
  int npairs;
  double raw_us;    // 같은 버퍼 순수 순차 읽기 (대조군)
  double raw_gbps;
};

static Case run(int n, int k, double keep, int iters) {
  std::mt19937 g(99 + n + k);
  std::uniform_int_distribution<int> d(-4, 4);
  std::vector<ggml_bf16_t> x(k), w((size_t)n * k);
  for (auto& v : x) v = f2b((float)d(g));
  for (auto& v : w) v = f2b((float)d(g));

  // keep 비율만큼 페어를 살린다 (컬럼당 앞쪽 비트 — 값이 아니라 시간만 본다)
  const int cols = k / 32;
  const int keep_bits = (int)(16 * keep + 0.5);
  std::vector<uint16_t> mask(cols, 0);
  for (auto& mb : mask) mb = (keep_bits >= 16) ? 0xFFFF : (uint16_t)((1u << keep_bits) - 1);

  const size_t bsz = K::BufferB::required_size(n, k);
  const int copies = (int)std::max<size_t>(2, POOL_BYTES / bsz);
  static bool told = false;
  static bool told2 = false;
  if (!told) {
    std::printf("   [진단] n*k*2=%.2f MB, BufferB::required_size=%.2f MB, copies=%d, 풀=%.2f MB\n",
                2.0 * n * k / 1e6, bsz / 1e6, copies, copies * bsz / 1e6);
    told = true;
  }
  std::vector<void*> bps(copies);
  std::vector<K::BufferB*> bbs(copies);
  for (int i = 0; i < copies; i++) {
    bps[i] = std::aligned_alloc(64, bsz);
    bbs[i] = new K::BufferB(n, k, bps[i]);
    for (int t = 0; t * K::N_BLOCK < n; t++) bbs[i]->from_mat(w.data(), t, 1);
  }
  void* ap = std::aligned_alloc(64, K::BufferA::required_size(K::M_STEP, k));
  K::BufferA ba(K::M_STEP, k, ap);
  ba.from_mat(1, x.data(), 0, 1);
  void* cp = std::aligned_alloc(64, K::BufferC::required_size(K::M_STEP, n));
  K::BufferC bc(K::M_STEP, n, cp);

  std::vector<uint8_t> arena(K::plan_arena_bytes(k));
  ActivationPlan plan;
  K::build_plan(mask.data(), x.data(), k, arena.data(), &plan, n);

  // 대조군: 같은 버퍼·같은 회전을 순수 순차 읽기로. 커널이 이 값을 넘을 수 없다.
  // 대조군 3종: 누산기 1개 vs 4개, 프리페치 유무. 커널이 넘을 수 없는 값을
  // 찾는 게 목적이므로 가장 빠른 것을 기준으로 쓴다.
  auto raw_read = [&](int i, int variant) {
    const char* q = (const char*)bbs[i % copies]->b;
    const size_t nb = (size_t)n * k * sizeof(ggml_bf16_t);
    const size_t PF = 128 * 64;
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    if (variant == 0) {          // 누산기 1 + 프리페치
      for (size_t j = 0; j + 64 <= nb; j += 64) {
        if (j + PF < nb) _mm_prefetch(q + j + PF, _MM_HINT_T1);
        a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void*)(q + j)));
      }
    } else if (variant == 1) {   // 누산기 4 + 프리페치
      size_t j = 0;
      for (; j + 256 <= nb; j += 256) {
        if (j + PF < nb) _mm_prefetch(q + j + PF, _MM_HINT_T1);
        a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void*)(q + j)));
        a1 = _mm512_add_epi64(a1, _mm512_loadu_si512((const void*)(q + j + 64)));
        a2 = _mm512_add_epi64(a2, _mm512_loadu_si512((const void*)(q + j + 128)));
        a3 = _mm512_add_epi64(a3, _mm512_loadu_si512((const void*)(q + j + 192)));
      }
    } else {                     // 누산기 4, 프리페치 없음
      size_t j = 0;
      for (; j + 256 <= nb; j += 256) {
        a0 = _mm512_add_epi64(a0, _mm512_loadu_si512((const void*)(q + j)));
        a1 = _mm512_add_epi64(a1, _mm512_loadu_si512((const void*)(q + j + 64)));
        a2 = _mm512_add_epi64(a2, _mm512_loadu_si512((const void*)(q + j + 128)));
        a3 = _mm512_add_epi64(a3, _mm512_loadu_si512((const void*)(q + j + 192)));
      }
    }
    a0 = _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
    return _mm512_reduce_add_epi64(a0);
  };
  {  // 풀이 정말 물리 메모리를 갖고 있는가 (미터치 페이지는 공유 zero page로 매핑된다)
    static bool rss_told = false;
    if (!rss_told) {
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
      std::printf("   [터치 확인] %d 사본 x %.2f MB 읽기 완료 (합 %zu)\n", copies, bsz / 1e6,
                  touched);
      rss_told = true;
    }
  }
  double raw_us = 1e18;
  for (int v = 0; v < 3; v++) {
    volatile long long sk = 0;
    for (int i = 0; i < 20; i++) sk += raw_read(i, v);
    auto r0 = Clock::now();
    for (int i = 0; i < iters; i++) sk += raw_read(i, v);
    auto r1 = Clock::now();
    const double u = std::chrono::duration<double, std::micro>(r1 - r0).count() / iters;
    if (!told2) std::printf("   [대조군 v%d] %8.3f us = %5.1f GB/s\n", v, u,
                            2.0 * k * n / (u * 1e-6) / 1e9);
    raw_us = std::min(raw_us, u);
    (void)sk;
  }
  told2 = true;
  for (int i = 0; i < 20; i++) K::gemv_slab(1, n, k, 0, n, &ba, bbs[i % copies], &bc, &plan);
  auto t0 = Clock::now();
  for (int i = 0; i < iters; i++) K::gemv_slab(1, n, k, 0, n, &ba, bbs[i % copies], &bc, &plan);
  auto t1 = Clock::now();

  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  const double bytes = 2.0 * k * n * ((double)plan.npairs / (k / 2));
  for (int i = 0; i < copies; i++) { delete bbs[i]; std::free(bps[i]); }
  std::free(ap); std::free(cp);
  const double raw_bytes = 2.0 * k * n;
  return {us, bytes / (us * 1e-6) / 1e9, plan.npairs, raw_us,
          raw_bytes / (raw_us * 1e-6) / 1e9};
}

int main(int argc, char** argv) {
  const int k = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 3000;
  // argv[3]: 쉼표로 구분한 n 목록 (기본은 아래 고정 목록). 베이스라인 비교처럼
  // 온전한 블록만 돌려야 할 때 쓴다.
  std::vector<int> ns = {256, 512, 640, 768, 1024, 1056, 1280};
  if (argc > 3) {
    ns.clear();
    for (const char* q = argv[3]; *q;) {
      ns.push_back((int)std::strtol(q, (char**)&q, 10));
      if (*q == ',') ++q;
    }
  }
  std::printf("k=%d, iters=%d, 풀 48 MB 회전 (L3 22.5 MB 초과)\n\n", k, iters);

  for (double keep : {1.0, 0.5, 0.1}) {
    std::printf("== keep=%.1f ==\n", keep);
    std::printf("%6s %6s %10s %9s %10s %9s %8s %s\n", "n", "blocks", "us", "GB/s",
                "raw_us", "raw_GB/s", "eff", "구성");
    for (int n : ns) {
      Case c = run(n, k, keep, iters);
      const int full = n / 256, tail = n % 256;
      char cfg[64];
      if (tail) std::snprintf(cfg, sizeof cfg, "%d full + %d tail", full, tail);
      else std::snprintf(cfg, sizeof cfg, "%d full", full);
      std::printf("%6d %6.2f %10.3f %9.1f %10.3f %9.1f %7.0f%% %s\n", n, n / 256.0, c.us,
                  c.gbps, c.raw_us, c.raw_gbps, c.gbps / c.raw_gbps * 100.0, cfg);
    }
    // 꼬리 vs 패딩: 640을 (a) 2full+128tail로 (b) 768로 패딩해 3full로
    if (argc > 3) { std::printf("\n"); continue; }
    Case a = run(640, k, keep, iters), b = run(768, k, keep, iters);
    std::printf("  640 꼬리 %.3f us vs 768 패딩 %.3f us  →  %s (%.1f%%)\n\n", a.us, b.us,
                a.us < b.us ? "꼬리 승" : "패딩 승", (b.us / a.us - 1.0) * 100.0);
  }
  return 0;
}
