/**
 * @brief A/B benchmark: GemmKernel224BF16RowMajorNC vs GemmKernel224BF16, over
 *        the MoE protocol both kernels actually run under in kt.
 *
 * Self-contained -- it needs only la/bf16_rowmajor.hpp and the kt headers that
 * pulls in -- so it can be copied to a target machine together with the kernel
 * to re-fix the constants there. The measured recipe is machine specific: the
 * row padding, the non-temporal stores and the B prefetch were each chosen by
 * measurement, and a different microarchitecture or prefetcher setting can move
 * the answer.
 *
 * Both kernels see the identical per-trial routing and their run order flips
 * every trial so slow drift cancels. mode=null runs two independent instances
 * of GemmKernel224BF16 instead: that is the in-run noise floor any reported
 * delta has to beat, and it should be checked before believing a small result.
 * The timed region per kernel is the full forward path -- BufferA::from_mat,
 * mat_mul over activated_expert x recommended_nth(n) tasks, BufferC::to_mat.
 *
 * Outputs are compared element by element before timing starts; the two kernels
 * hand AMX the same tile pairs in the same order and must agree bitwise.
 *
 * usage: bf16_rowmajor-bench m n k [trials=7] [seed=0] [mode=ab|null]
 *   ab   the row-major kernel against kt's
 *   null two instances of kt's kernel -- the in-run noise floor
 *   m is the token count; the per-expert m each kernel sees is m * topk / experts.
 *
 * Build (from kt-kernel/operators/amx/test):
 *   g++ -O3 -std=c++20 -mavx512f -mavx512bw -mavx512vl -mavx512bf16 -mavx512vnni \
 *       -mamx-tile -mamx-bf16 -mamx-int8 -pthread -DHAVE_AMX \
 *       -I.. -I../../.. -I../../../.. -I../../../../third_party \
 *       -I../../../../third_party/llama.cpp \
 *       bf16_rowmajor-bench.cpp -o bf16_rowmajor-bench
 *
 * Run pinned to one socket, e.g.  numactl -N 0 -m 0 ./bf16_rowmajor-bench 512 2048 2048
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../la/bf16_rowmajor.hpp"

namespace {

constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kThreads = 8;

uint64_t now_ns() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

uint16_t f32_to_bf16(float f) {
  uint32_t x;
  __builtin_memcpy(&x, &f, 4);
  return (uint16_t)((x + 0x7fff + ((x >> 16) & 1)) >> 16);
}

// Minimal fork-join pool: the kernels are called exactly as kt calls them,
// (expert, ith) tasks over a fixed thread count.
class Pool {
 public:
  explicit Pool(int nth) : nth_(nth < 1 ? 1 : nth) {
    for (int i = 1; i < nth_; i++) workers_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    {
      std::unique_lock<std::mutex> lk(m_);
      stop_ = true;
      gen_++;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }

  void parallel(int n, const std::function<void(int)>& fn) {
    fn_ = &fn;
    ntask_ = n;
    remaining_ = n;
    next_.store(0);
    {
      std::unique_lock<std::mutex> lk(m_);
      gen_++;
    }
    cv_.notify_all();
    int idx;
    while ((idx = next_.fetch_add(1)) < n) {
      fn(idx);
      done_one();
    }
    std::unique_lock<std::mutex> lk(dm_);
    dcv_.wait(lk, [this] { return remaining_.load() == 0; });
  }

 private:
  void done_one() {
    if (remaining_.fetch_sub(1) == 1) {
      std::unique_lock<std::mutex> lk(dm_);
      dcv_.notify_all();
    }
  }
  void loop() {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [this, &seen] { return stop_ || gen_ != seen; });
      if (stop_) return;
      seen = gen_;
      lk.unlock();
      int idx;
      while ((idx = next_.fetch_add(1)) < ntask_) {
        (*fn_)(idx);
        done_one();
      }
    }
  }

  int nth_;
  std::vector<std::thread> workers_;
  std::mutex m_, dm_;
  std::condition_variable cv_, dcv_;
  const std::function<void(int)>* fn_ = nullptr;
  std::atomic<int> next_{0}, remaining_{0};
  int ntask_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

// One kernel's full MoE state: per-expert weights plus the activation and
// accumulator arenas the per-trial buffers are carved out of.
template <typename K>
struct Runner {
  int n, k, nth;
  Pool& pool;
  std::vector<std::vector<char>> b_mem;
  std::vector<std::shared_ptr<typename K::BufferB>> bb;
  std::vector<char> a_mem, c_mem;
  char *a_base, *c_base;
  std::vector<uint16_t> out;
  std::vector<std::shared_ptr<typename K::BufferA>> ba;
  std::vector<std::shared_ptr<typename K::BufferC>> bc;

  Runner(int n, int k, size_t in_rows, Pool& pool, const std::vector<std::vector<uint16_t>>& w_src)
      : n(n),
        k(k),
        nth(K::recommended_nth(n)),
        pool(pool),
        b_mem(kExperts),
        bb(kExperts),
        a_mem(K::BufferA::required_size((int)in_rows, k) + 64),
        c_mem(K::BufferC::required_size((int)in_rows, n) + 64),
        out(in_rows * (size_t)n),
        ba(kExperts),
        bc(kExperts) {
    a_base = (char*)(((uintptr_t)a_mem.data() + 63) & ~(uintptr_t)63);
    c_base = (char*)(((uintptr_t)c_mem.data() + 63) & ~(uintptr_t)63);
    for (int e = 0; e < kExperts; e++) {
      b_mem[e].resize(K::BufferB::required_size(n, k) + 64);
      void* p = (void*)(((uintptr_t)b_mem[e].data() + 63) & ~(uintptr_t)63);
      bb[e] = std::make_shared<typename K::BufferB>(n, k, p);
    }
    pool.parallel(kExperts * nth, [&](int t) { bb[t / nth]->from_mat((ggml_bf16_t*)w_src[t / nth].data(), t % nth, nth); });
  }

  // rows[e] is padded to 32-row granules; off[e] indexes the shared activation
  // and output arenas. Arena slices step by required_size, not by dense rows,
  // because the buffers may pad their row strides.
  double run(const std::vector<int>& rows, const std::vector<int>& off, const std::vector<uint16_t>& in) {
    size_t ao = 0, co = 0;
    for (int e = 0; e < kExperts; e++) {
      if (!rows[e]) {
        ba[e] = nullptr;
        bc[e] = nullptr;
        continue;
      }
      ba[e] = std::make_shared<typename K::BufferA>(rows[e], k, a_base + ao);
      bc[e] = std::make_shared<typename K::BufferC>(rows[e], n, c_base + co);
      ao += K::BufferA::required_size(rows[e], k);
      co += K::BufferC::required_size(rows[e], n);
    }
    uint64_t t0 = now_ns();
    pool.parallel(kExperts, [&](int e) {
      if (!rows[e]) return;
      ba[e]->from_mat(rows[e], (ggml_bf16_t*)(in.data() + (size_t)off[e] * k), 0, 1);
    });
    pool.parallel(kExperts * nth, [&](int t) {
      int e = t / nth, ith = t % nth;
      if (!rows[e]) return;
      K::config();
      amx::mat_mul(rows[e], n, k, ba[e], bb[e], bc[e], ith, nth);
    });
    pool.parallel(kExperts * nth, [&](int t) {
      int e = t / nth, ith = t % nth;
      if (!rows[e]) return;
      bc[e]->to_mat(rows[e], (ggml_bf16_t*)(out.data() + (size_t)off[e] * n), ith, nth);
    });
    return (now_ns() - t0) / 1e6;
  }
};

double report(const char* label, std::vector<double> v, int m, int n, int k) {
  std::vector<double> s = v;
  std::sort(s.begin(), s.end());
  double gflops = 2.0 * kTopK * m * (double)n * k / (s.front() / 1e3) / 1e9;
  std::printf("%-7s min=%.3f med=%.3f max=%.3f ms  %.1f GFLOP/s(min)\n", label, s.front(), s[s.size() / 2], s.back(),
              gflops);
  return s.front();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s m n k [trials=7] [seed=0] [mode=ab|null]\n", argv[0]);
    return 2;
  }
  const int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
  const int trials = argc > 4 ? atoi(argv[4]) : 7;
  const unsigned seed = argc > 5 ? (unsigned)atoi(argv[5]) : 0u;
  const std::string mode = argc > 6 ? argv[6] : "ab";
  if (n % 32 || k % 32) {
    std::fprintf(stderr, "n and k must be multiples of 32\n");
    return 2;
  }

  std::mt19937 g(seed);
  auto rb = [&] { return f32_to_bf16((float)((int)(g() % 9) - 4)); };

  std::vector<std::vector<uint16_t>> w_src(kExperts);  // row-major [n][k] per expert
  for (auto& w : w_src) {
    w.resize((size_t)n * k);
    for (auto& x : w) x = rb();
  }
  const size_t in_rows = (size_t)kTopK * m + (size_t)kExperts * 32;  // headroom for 32-row padding
  std::vector<uint16_t> in(in_rows * k);
  for (auto& x : in) x = rb();

  Pool pool(kThreads);
  std::printf("bf16_rowmajor-bench m=%d n=%d k=%d experts=%d topk=%d threads=%d nth_n=%d mode=%s\n", m, n, k, kExperts,
              kTopK, kThreads, amx::GemmKernel224BF16::recommended_nth(n), mode.c_str());

  Runner<amx::GemmKernel224BF16> base(n, k, in_rows, pool, w_src);
  std::unique_ptr<Runner<amx::GemmKernel224BF16RowMajorNC>> rowmaj;
  std::unique_ptr<Runner<amx::GemmKernel224BF16>> twin;
  if (mode == "null") {
    twin = std::make_unique<Runner<amx::GemmKernel224BF16>>(n, k, in_rows, pool, w_src);
  } else {
    rowmaj = std::make_unique<Runner<amx::GemmKernel224BF16RowMajorNC>>(n, k, in_rows, pool, w_src);
  }
  const char* label_b = mode == "null" ? "kt224b" : "rm+nc";
  auto run_b = [&](const std::vector<int>& rows, const std::vector<int>& off) {
    return mode == "null" ? twin->run(rows, off, in) : rowmaj->run(rows, off, in);
  };

  std::vector<int> rows(kExperts), off(kExperts);
  auto layout = [&] {
    rows.assign(kExperts, 0);
    std::vector<int> idx(kExperts);
    for (int e = 0; e < kExperts; e++) idx[e] = e;
    for (int t = 0; t < m; t++) {
      for (int j = 0; j < kTopK; j++) {
        int r = j + (int)(g() % (unsigned)(kExperts - j));
        std::swap(idx[j], idx[r]);
        rows[idx[j]]++;
      }
    }
    for (auto& c : rows) c = (c + 31) / 32 * 32;
    int o = 0;
    for (int e = 0; e < kExperts; e++) {
      off[e] = o;
      o += rows[e];
    }
  };

  // Warm up once and cross-check: same accumulation order, so bitwise equal.
  layout();
  base.run(rows, off, in);
  run_b(rows, off);
  if (mode != "null") {
    const std::vector<uint16_t>& contender = rowmaj->out;
    size_t live = 0, diff = 0;
    for (int e = 0; e < kExperts; e++) {
      for (int i = 0; i < rows[e]; i++, live++) {
        for (int j = 0; j < n; j++) {
          size_t idx = (size_t)(off[e] + i) * n + j;
          diff += base.out[idx] != contender[idx];
        }
      }
    }
    std::printf("cross-check: %zu rows, %zu mismatched elements%s\n", live, diff, diff ? "  ** MISMATCH **" : "");
    if (diff) return 1;
  }

  std::vector<double> ms_a, ms_b;
  for (int t = 0; t < trials; t++) {
    layout();  // fresh routing per trial, shared by both kernels
    if (t % 2 == 0) {
      ms_a.push_back(base.run(rows, off, in));
      ms_b.push_back(run_b(rows, off));
    } else {
      ms_b.push_back(run_b(rows, off));
      ms_a.push_back(base.run(rows, off, in));
    }
  }

  double min_a = report("kt224", ms_a, m, n, k);
  double min_b = report(label_b, ms_b, m, n, k);
  std::printf("ratio %s/kt224 = %.4f (min-based; <1 means %s faster)\n", label_b, min_b / min_a, label_b);
  return 0;
}
