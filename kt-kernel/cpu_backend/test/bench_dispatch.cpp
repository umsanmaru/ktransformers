// cpu_backend 디스패치 고정비 마이크로벤치 — 빈 태스크로 홉과 job 경계 비용만 잰다.
//
// 빌드 (CMake 미연결; conda의 hwloc/numa 기준):
//   g++ -O2 -std=c++20 -march=native -I. -I$CONDA_PREFIX/include \
//       cpu_backend/test/bench_dispatch.cpp cpu_backend/worker_pool.cpp cpu_backend/task_queue.cpp \
//       -o bench_dispatch -L$CONDA_PREFIX/lib -lhwloc -lnuma -lpthread -Wl,-rpath,$CONDA_PREFIX/lib
//   taskset -c <워커와 겹치지 않는 cpu> ./bench_dispatch
//
// 2026-08-26, Xeon 6534, 워커 8 = node 0 코어, 호출 스레드는 node 1 (µs/호출):
//   do_numa_job(empty)                        1.18   ← 호출 스레드 → m_0 → 복귀
//   do_numa_job{ ws(8) }                      2.40   ← work-stealing job 경계 1회 ≈ 1.2
//   do_numa_job{ ws(32) }                     3.12
//   do_numa_job{ ws(8); ws(32) }              4.20   ← prep job + GEMV job (현재 구조)
//   do_numa_job{ ws(40) }                     3.31   ← 융합 job이 아끼는 것은 ~0.9
//   TaskQueue enqueue + sync (no-op)          6.25   ← submit 경로가 더 내는 홉 (futex 왕복 2회)
//   TaskQueue{ do_numa_job{ ws(8); ws(32) } } 10.78  ← 실모델 submit 경로의 순수 디스패치
// 호출 스레드가 워커의 HT 형제(16-23)에 있으면 numa 홉은 0.2로 줄지만(같은 L2)
// 워커를 느리게 하므로 득이 아니다.
#include <chrono>
#include <cstdio>
#include <functional>
#include "cpu_backend/task_queue.h"
#include "cpu_backend/worker_pool.h"

template <class F>
static void bench(const char* name, F fn, int n = 20000) {
  for (int i = 0; i < 500; i++) fn();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; i++) fn();
  double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
  std::printf("%-58s %7.2f us\n", name, us);
}

int main() {
  WorkerPoolConfig cfg;
  cfg.subpool_count = 1; cfg.subpool_numa_map = {0}; cfg.subpool_thread_count = {8};
  WorkerPool pool(cfg);
  auto* dist = pool.dispense_backend();
  auto* sub = pool.get_subpool(0);
  auto nop = [](int) {};
  bench("do_numa_job(empty)                      [caller -> m_0 -> back]", [&] { dist->do_numa_job(nop); });
  for (int N : {8, 16, 32, 64})
    bench((std::string("do_numa_job{ ws(") + std::to_string(N) + " empty) }").c_str(),
          [&] { dist->do_numa_job([&](int) { sub->do_work_stealing_job(N, nop); }); });
  bench("do_numa_job{ ws(8) ; ws(32) }         [prep job + GEMV job 모사]",
        [&] { dist->do_numa_job([&](int) { sub->do_work_stealing_job(8, nop); sub->do_work_stealing_job(32, nop); }); });
  bench("do_numa_job{ ws(40) }                 [융합 job 모사]",
        [&] { dist->do_numa_job([&](int) { sub->do_work_stealing_job(40, nop); }); });
  bench("subpool ws(32) from caller directly    [numa 홉 없이]", [&] { sub->do_work_stealing_job(32, nop); });
  TaskQueue tq;
  bench("TaskQueue enqueue + sync (no-op)       [submit 경로의 홉]", [&] { tq.enqueue([] {}); tq.sync(0); });
  bench("TaskQueue{ do_numa_job{ ws(8); ws(32) } } + sync  [실모델 submit 경로 모사]",
        [&] { tq.enqueue([&] { dist->do_numa_job([&](int) { sub->do_work_stealing_job(8, nop); sub->do_work_stealing_job(32, nop); }); }); tq.sync(0); });
  return 0;
}
