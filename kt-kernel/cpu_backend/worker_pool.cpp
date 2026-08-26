/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "worker_pool.h"

#include <hwloc/bitmap.h>
#include <numa.h>
#include <numaif.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "hwloc.h"

// do_numa_job의 호출 스레드는 NUMA 풀이 끝나길 기다리며 spin한다. 그 스레드는
// 바인딩이 없어 워커의 HT 형제 위에 앉기 쉽고, pause 없는 spin은 형제 코어의
// 파이프라인·L2 fill 자원을 나눠 쓴다 (2026-08-26: gateup decode 398 → 363 µs를
// 스핀 스레드를 다른 소켓으로 옮기는 것만으로 얻었다). pause는 그 간섭을 줄인다.
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define KT_SPIN_PAUSE() _mm_pause()
#else
#define KT_SPIN_PAUSE() ((void)0)
#endif

thread_local int WorkerPool::thread_local_id = -1;

InNumaPool::InNumaPool(int max_thread_num) {
  printf("In Numa Worker Pool at NUMA %d, %d threads\n", numa_node_of_cpu(sched_getcpu()), max_thread_num);
  total_worker_count = max_thread_num;
  set_restricted_worker_count(total_worker_count);
  thread_state_ = std::unique_ptr<ThreadState[]>(new ThreadState[max_thread_num]);
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WAITING, std::memory_order_release);
  }
  workers_.resize(total_worker_count);
  for (int i = 1; i < total_worker_count; i++) {
    workers_[i] = std::thread(&InNumaPool::worker_thread, this, i, -1);
  }
}

InNumaPool::InNumaPool(int max_thread_num, int numa_id, int threads_id_start) {
  printf("===========In NumaPool============\n");
  hwloc_topology_t topology;
  hwloc_obj_t numa_obj, core_obj;
  hwloc_bitmap_t cpuset;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);
  printf("In Numa Worker Pool at NUMA %d, %d threads\n", numa_node_of_cpu(sched_getcpu()), max_thread_num);
  total_worker_count = max_thread_num;
  set_restricted_worker_count(total_worker_count);
  thread_state_ = std::unique_ptr<ThreadState[]>(new ThreadState[max_thread_num]);
  for (int i = 0; i < total_worker_count; i++) {
    thread_state_[i].status.store(ThreadStatus::WAITING, std::memory_order_release);
  }
  workers_.resize(total_worker_count);
  for (int i = 1; i < total_worker_count; i++) {
    workers_[i] = std::thread(&InNumaPool::worker_thread, this, i, numa_id);
    // set the thread name as: "numa_(numa_id)_t_(i+threads_id_start)"
    std::string thread_name = "numa_" + std::to_string(numa_id) + "_t_" + std::to_string(i + threads_id_start);
    pthread_t native_handle = workers_[i].native_handle();
    auto res_set_name = pthread_setname_np(native_handle, thread_name.c_str());
    if (res_set_name != 0) {
      fprintf(stderr, "Failed to set thread name: %s\n", strerror(res_set_name));
    }
    // 检查线程是否成功命名
    char name[16];
    pthread_getname_np(native_handle, name, sizeof(name));
    if (strcmp(name, thread_name.c_str()) == 0) {
      // printf("Thread name set successfully: %s\n", name);
    } else {
      // printf("Failed to set thread name: %s\n", name);
    }
    // Set the thread affinity to the specified NUMA node's CPU
    numa_obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, numa_id);
    if (!numa_obj) {
      fprintf(stderr, "NUMA node %d not found\n", numa_id);
      // throw std::runtime_error("NUMA node not found");
      continue;
    }
    core_obj = hwloc_get_obj_inside_cpuset_by_type(topology, numa_obj->cpuset, HWLOC_OBJ_CORE, i + threads_id_start);
    if (!core_obj) {
      fprintf(stderr,
              "Core %d inside NUMA node %d not found — thread stays unbound and "
              "may run on a remote socket (silent slowdown)\n",
              i + threads_id_start, numa_id);
      // throw std::runtime_error("Core not found inside NUMA node");
      continue;
    }
    cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_copy(cpuset, core_obj->cpuset);
    hwloc_bitmap_singlify(cpuset);
    auto res = hwloc_set_thread_cpubind(topology, native_handle, cpuset, HWLOC_CPUBIND_STRICT);
    if (res != 0) {
      fprintf(stderr, "Failed to set thread CPU binding: %s\n", strerror(errno));
    }
  }
}

InNumaPool::~InNumaPool() {
  for (int i = 0; i < total_worker_count; i++) {
    {
      std::lock_guard<std::mutex> lock(thread_state_[i].mutex);
      thread_state_[i].status.store(ThreadStatus::EXIT, std::memory_order_release);
    }
    thread_state_[i].cv.notify_one();
  }
  for (int i = 0; i < total_worker_count; i++) {
    if (workers_[i].joinable()) {
      workers_[i].join();
    }
  }
}

int InNumaPool::get_thread_num() {
  throw std::runtime_error("Deprecated");
  return total_worker_count;
}

void InNumaPool::set_restricted_worker_count(int count) { restricted_worker_count = count; }

void InNumaPool::wait() {
  for (int i = 0; i < worker_count; i++) {
    while (thread_state_[i].status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
    }
  }

#ifdef PROFILE_BALANCE
  size_t max_time = 0;
  size_t min_time = thread_state_[0].finish_ns;
  size_t sum = 0;
  for (int i = 0; i < worker_count; i++) {
    sum += thread_state_[i].finish_ns;
    max_time = std::max(max_time, thread_state_[i].finish_ns);
    min_time = std::min(min_time, thread_state_[i].finish_ns);
  }
  double balance = 1.0 * sum / (max_time * worker_count);
  printf("max_time: %ld, min_time: %ld, sum_time: %ld, balance: %f\n", max_time, min_time, sum, balance);

#endif
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr);
}

void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  do_work_stealing_job_async(task_num, init_func, compute_func, finalize_func);
  wait();
}

void InNumaPool::do_work_stealing_job_async(int task_num, std::function<void(int)> init_func,
                                            std::function<void(int)> compute_func,
                                            std::function<void(int)> finalize_func) {
  init_func_ = init_func;
  compute_func_ = compute_func;
  finalize_func_ = finalize_func;
  worker_count = std::min(restricted_worker_count, task_num);
  curr_.store(0, std::memory_order_release);
  end_ = task_num;
  for (int i = 0; i < worker_count; i++) {
    {
      std::lock_guard<std::mutex> lock(thread_state_[i].mutex);
      thread_state_[i].status.store(ThreadStatus::WORKING, std::memory_order_release);
    }
    thread_state_[i].cv.notify_one();
  }
  WorkerPool::thread_local_id = 0;
  process_tasks(0);
}

void InNumaPool::process_tasks(int thread_id) {
#ifdef PROFILE_BALANCE
  auto start = std::chrono::high_resolution_clock::now();
#endif
  auto& s = thread_state_[thread_id];
  if (init_func_ != nullptr) {
    init_func_(thread_id);
  }

  // omp-guided-style work scheduling
  while (true) {
    int old = curr_.load(std::memory_order_relaxed);
    int rem = end_ - old;
    if (rem <= 0) {
      break;
    }

    int block = (rem + worker_count - 1) / worker_count;
    block = 1;
    int task_id = curr_.fetch_add(block, std::memory_order_acq_rel);
    if (task_id >= end_) {
      break;
    }

    for (int i = 0; i < block; i++) {
      if (task_id + i >= end_) {
        break;
      }
      compute_func_(task_id + i);
    }
  }

  if (finalize_func_ != nullptr) {
    finalize_func_(thread_id);
  }

  s.status.store(ThreadStatus::WAITING, std::memory_order_release);
#ifdef PROFILE_BALANCE
  s.finish_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
#endif
}

void InNumaPool::worker_thread(int thread_id, int numa_id) {
  if (numa_id >= 0) {
    set_memory_to_numa(numa_id);
  }
  auto start = std::chrono::high_resolution_clock::now();
  WorkerPool::thread_local_id = thread_id;  // 设置线程本地变量
  while (true) {
    ThreadStatus status = thread_state_[thread_id].status.load(std::memory_order_acquire);
    if (status == ThreadStatus::WORKING) {
      process_tasks(thread_id);
      start = std::chrono::high_resolution_clock::now();
    } else if (status == ThreadStatus::WAITING) {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (duration > 50) {
        std::unique_lock<std::mutex> lock(thread_state_[thread_id].mutex);
        thread_state_[thread_id].cv.wait(lock, [&] {
          return thread_state_[thread_id].status.load(std::memory_order_acquire) != ThreadStatus::WAITING;
        });
      }
    } else if (status == ThreadStatus::EXIT) {
      return;
    }
  }
}

namespace {
// 노드 id로 인덱싱하는 보조 테이블의 크기를 정하기 위한 최대 노드 id.
// subpool_numa_map은 임의의 노드를 가리킬 수 있으므로 (단일 소켓 실행은
// map 전체가 GPU가 붙은 노드 하나다) 크기를 map 길이로 잡으면 안 된다.
int max_node_id(const std::vector<int>& numa_ids) {
  int m = 0;
  for (int id : numa_ids) {
    if (id > m) {
      m = id;
    }
  }
  return m;
}
}  // namespace

NumaJobDistributor::NumaJobDistributor(int numa_count) {
  std::vector<int> numa_ids;
  for (int i = 0; i < numa_count; i++) {
    numa_ids.push_back(i);
  }
  init(numa_ids);
}

NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids) { init(numa_ids); }
NumaJobDistributor::NumaJobDistributor(std::vector<int> numa_ids, std::vector<int> thread_count) {
  init(numa_ids, thread_count);
}

void NumaJobDistributor::init(std::vector<int> numa_ids) {
  this->numa_count = numa_ids.size();
  this->ready_bar = std::unique_ptr<std::barrier<>>(new std::barrier<>(numa_count + 1));
  this->numa_ids = numa_ids;
  for (size_t i = 0; i < numa_count; i++) {
    status.push_back(nullptr);
    mutexes.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
  }

  workers.resize(numa_count);
  for (int i = 0; i < numa_count; i++) {
    std::thread([this, i]() { workers[i] = std::thread(&NumaJobDistributor::worker_thread, this, i); }).join();
  }
  ready_bar->arrive_and_wait();
}

void NumaJobDistributor::init(std::vector<int> numa_ids, std::vector<int> thread_count) {
  hwloc_topology_t topology;
  hwloc_obj_t numa_obj, core_obj;
  hwloc_bitmap_t cpuset;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  this->numa_count = numa_ids.size();
  this->ready_bar = std::unique_ptr<std::barrier<>>(new std::barrier<>(numa_count + 1));
  this->numa_ids = numa_ids;
  for (size_t i = 0; i < numa_count; i++) {
    status.push_back(nullptr);
    mutexes.push_back(std::make_unique<std::mutex>());
    cvs.push_back(std::make_unique<std::condition_variable>());
  }

  workers.resize(numa_count);
  // **노드 id로 인덱싱하므로 크기가 노드 수여야 한다** (subpool 수가 아니다).
  // numa_count는 subpool 수이고 numa_ids[i]는 물리 노드 id다 — 단일 소켓
  // 실행에서 subpool 1개가 node 1에 붙으면 numa_threads_count[1]이 크기 1
  // 벡터의 범위 밖 읽기가 되어, 쓰레기 값이 코어 인덱스로 들어가 바인딩이
  // 조용히 생략된다. 바인딩 실패는 결과를 바꾸지 않고 느리게만 만들므로
  // (스레드가 원격 소켓에 앉아 데이터를 링크 건너 읽는다) 어떤 테스트도
  // 잡아주지 않고, 값이 매번 달라져 성능이 실행마다 추첨이 된다.
  std::vector<int> numa_threads_count(max_node_id(numa_ids) + 1, 0);
  for (int i = 0; i < numa_count; i++) {
    workers[i] = std::thread(&NumaJobDistributor::worker_thread, this, i);
    auto this_numa = numa_ids[i];
    auto start_id = numa_threads_count[this_numa];
    // set the thread name as: "worker_numa_(numa_id)_main_start_id(0)"
    // printf("nuam_id %d, start_id %d\n", this_numa, start_id);
    std::string thread_name = "numa_" + std::to_string(numa_ids[i]) + "_m_" + std::to_string(start_id);
    pthread_t native_handle = workers[i].native_handle();
    pthread_setname_np(native_handle, thread_name.c_str());
    // Set the thread affinity to the specified NUMA node's CPU (0)
    numa_obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, this_numa);
    if (!numa_obj) {
      fprintf(stderr, "NUMA node %d not found\n", this_numa);
      // throw std::runtime_error("NUMA node not found");
      continue;
    }
    core_obj = hwloc_get_obj_inside_cpuset_by_type(topology, numa_obj->cpuset, HWLOC_OBJ_CORE, start_id);
    if (!core_obj) {
      fprintf(stderr,
              "Core %d inside NUMA node %d not found — thread stays unbound and "
              "may run on a remote socket (silent slowdown)\n",
              start_id, this_numa);
      // throw std::runtime_error("Core not found inside NUMA node");
      continue;
    }
    // 精简 cpuset
    auto cpuset_simple = hwloc_bitmap_alloc();
    hwloc_bitmap_copy(cpuset_simple, core_obj->cpuset);
    hwloc_bitmap_singlify(cpuset_simple);
    // 打印绑定的具体的 CPU 物理索引
    unsigned long i_in;
    // hwloc_bitmap_foreach_begin(i_in, cpuset_simple) { printf("Thread %d bound to CPU %ld\n", start_id, i_in); }
    // hwloc_bitmap_foreach_end();
    auto res = hwloc_set_thread_cpubind(topology, native_handle, cpuset_simple, HWLOC_CPUBIND_STRICT);
    if (res != 0) {
      fprintf(stderr, "Failed to set thread CPU binding: %s\n", strerror(errno));
    }
    // 检查线程是否绑定到指定的 核上了
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_get_thread_cpubind(topology, native_handle, cpuset, HWLOC_CPUBIND_THREAD);
    // hwloc_bitmap_foreach_begin(i_in, cpuset) { printf("Thread %d is bound to CPU %ld\n", start_id, i_in); }
    // hwloc_bitmap_foreach_end();

    numa_threads_count[this_numa] += thread_count[i];
  }
  ready_bar->arrive_and_wait();
}

NumaJobDistributor::~NumaJobDistributor() {
  for (int i = 0; i < numa_count; i++) {
    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::EXIT, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  for (int i = 0; i < numa_count; i++) {
    if (workers[i].joinable()) {
      workers[i].join();
    }
  }
}

#ifdef USE_NUMA_JOB_DIRECT_WORK

void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {
  this->compute_func = compute_func;
  auto me_numa = numa_node_of_cpu(sched_getcpu());
  for (int i = 0; i < numa_count; i++) {
    if (i == me_numa) continue;

    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  compute_func(me_numa);
  for (int i = 0; i < numa_count; i++) {
    if (i == me_numa) continue;

    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
      KT_SPIN_PAUSE();
    }
  }
}
#else
void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {
  this->compute_func = compute_func;
  for (int i = 0; i < numa_count; i++) {
    {
      std::lock_guard<std::mutex> lock(*mutexes[i]);
      status[i]->store(ThreadStatus::WORKING, std::memory_order_release);
    }
    cvs[i]->notify_one();
  }
  for (int i = 0; i < numa_count; i++) {
    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {
      KT_SPIN_PAUSE();
    }
  }
}
#endif

void NumaJobDistributor::worker_thread(int numa_id) {
  auto start = std::chrono::high_resolution_clock::now();
  set_memory_to_numa(numa_id);
  status[numa_id] =
      std::move(std::unique_ptr<std::atomic<ThreadStatus>>(new std::atomic<ThreadStatus>(ThreadStatus::WAITING)));
  ready_bar->arrive_and_wait();
  while (true) {
    auto stat = status[numa_id]->load(std::memory_order_acquire);
    if (stat == ThreadStatus::WORKING) {
      auto me_numa = numa_node_of_cpu(sched_getcpu());
      // printf("numa work on %d, me %d\n", numa_id, me_numa);
      compute_func(numa_id);
      status[numa_id]->store(ThreadStatus::WAITING, std::memory_order_release);
      start = std::chrono::high_resolution_clock::now();
    } else if (stat == ThreadStatus::WAITING) {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (duration > 50) {
        std::unique_lock<std::mutex> lock(*mutexes[numa_id]);
        cvs[numa_id]->wait(lock, [&] {
          return status[numa_id]->load(std::memory_order_acquire) != ThreadStatus::WAITING;
        });
      }
    } else if (stat == ThreadStatus::EXIT) {
      return;
    }
  }
}

void WorkerPool::init(WorkerPoolConfig config) {
  printf("WorkerPool[0x%lx] %d subpools, [numa:threads]", (intptr_t)this, config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    printf("[%d:%d] ", config.subpool_numa_map[i], config.subpool_thread_count[i]);
  }
  printf("\n");

  for (int i = 0; i < config.subpool_count; i++) {
    numa_worker_pools.push_back(nullptr);
  }
  // 노드 id로 인덱싱한다 — 크기는 subpool 수가 아니라 노드 수다 (위 주석 참조).
  std::vector<int> numa_threads_count(max_node_id(config.subpool_numa_map) + 1, 0);
  for (int i = 0; i < config.subpool_count; i++) {
    auto this_numa = config.subpool_numa_map[i];
    auto this_thread_count = config.subpool_thread_count[i];
    auto this_thread_id_start = numa_threads_count[this_numa];
    std::thread([this, i, this_numa, this_thread_count, this_thread_id_start]() {
      set_to_numa(this_numa);
      numa_worker_pools[i] =
          std::move(std::unique_ptr<InNumaPool>(new InNumaPool(this_thread_count, this_numa, this_thread_id_start)));
      // numa_worker_pools[i] = std::move(std::unique_ptr<InNumaPool>(new InNumaPool(this_thread_count)));
    }).join();
    numa_threads_count[this_numa] += this_thread_count;
  }

  distributor = std::move(std::unique_ptr<NumaJobDistributor>(
      new NumaJobDistributor(config.subpool_numa_map, config.subpool_thread_count)));
  // distributor = std::move(std::unique_ptr<NumaJobDistributor>(new NumaJobDistributor(config.subpool_numa_map)));
}

WorkerPool::WorkerPool(WorkerPoolConfig config) : config(config) { init(config); }

WorkerPool::WorkerPool(int total_threads) {
  config.subpool_count = numa_num_configured_nodes();
  config.subpool_numa_map.resize(config.subpool_count);
  config.subpool_thread_count.resize(config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    config.subpool_numa_map[i] = i;
    config.subpool_thread_count[i] = total_threads / config.subpool_count;
  }
  init(config);
}

WorkerPool::WorkerPool(int total_threads, int single_numa_id) {
  set_to_numa(single_numa_id);
  config.subpool_count = numa_num_configured_nodes();
  config.subpool_numa_map.resize(config.subpool_count);
  config.subpool_thread_count.resize(config.subpool_count);
  for (int i = 0; i < config.subpool_count; i++) {
    config.subpool_numa_map[i] = single_numa_id;
    config.subpool_thread_count[i] = total_threads / config.subpool_count;
  }
  init(config);
}

WorkerPool::~WorkerPool() {}

int WorkerPool::get_thread_num() { return total_thread_count; }

void WorkerPool::set_restricted_worker_count(int count) {
  for (int i = 0; i < numa_count; i++) {
    numa_worker_pools[i]->set_restricted_worker_count(threads_per_numa);
  }
}

InNumaPool* WorkerPool::get_subpool(int numa_id) { return numa_worker_pools[numa_id].get(); }

NumaJobDistributor* WorkerPool::dispense_backend() { return distributor.get(); }

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  numa_worker_pools[0]->do_work_stealing_job(task_num, init_func, compute_func, finalize_func);
}

void WorkerPool::do_work_stealing_job(int task_num, std::function<void(int)> compute_func) {
  do_work_stealing_job(task_num, nullptr, compute_func, nullptr);
}
