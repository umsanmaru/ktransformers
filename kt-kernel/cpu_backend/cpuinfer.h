/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// NVTX (진단용). CUDA 빌드에서만, 헤더 전용 nvtx3. 비용은 프로파일러가 붙지 않으면 ~0.
#if defined(KTRANSFORMERS_USE_CUDA) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define KT_NVTX_PUSH(name) nvtxRangePushA(name)
#define KT_NVTX_POP() nvtxRangePop()
#define KT_NVTX_MARK(name) nvtxMarkA(name)
#else
#define KT_NVTX_PUSH(name) ((void)0)
#define KT_NVTX_POP() ((void)0)
#define KT_NVTX_MARK(name) ((void)0)
#endif
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#elif KTRANSFORMERS_USE_MACA
#include "vendors/maca.h"
#endif

#include "./vendors/vendor.h"
#include "llama.cpp/ggml-impl.h"
#include "task_queue.h"
#include "worker_pool.h"

class CPUInfer {
 public:
  CPUInfer(int thread_num) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    delete backend_;
    delete task_queue_;
  }

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  // stream host node 경유 submit. NVTX 마커 "kt.submit"을 콜백 안에서 찍는다 — nsys에서
  // "CPU cold 작업이 언제 시작됐나"를 보기 위해 (Prism SGLANG_PRISM_COLD_STREAM=1 경로는
  // host 스레드가 앞서 달려 executor의 host range가 실제 시점을 담지 못한다).
  struct HostSubmitArgs {
    void (*func)(void*);
    void* args;
  };
  static void host_submit_(void* p) {
    HostSubmitArgs* a = (HostSubmitArgs*)p;
    KT_NVTX_MARK("kt.submit");
    a->func(a->args);
    delete a;
  }
  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA)
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    HostSubmitArgs* wrapped = new HostSubmitArgs{func, args};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&host_submit_, (void*)wrapped);
#endif
  }
#endif

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
  };

  // 완료 신호 태스크 (Prism cold_async): 큐의 앞선 태스크(예: forward_*_partial)가 끝난 뒤
  // 실행되어 pinned int32 플래그에 값을 쓴다. GPU가 그 플래그를 stream에서 기다리므로
  // sync host node(블로킹 콜백)가 필요 없다 — 콜백 스레드가 막히지 않는다.
  struct SignalArgs {
    CPUInfer* cpuinfer;
    intptr_t flag_ptr;
    int32_t value;
  };
  static void signal_(void* p) {
    SignalArgs* a = (SignalArgs*)p;
    intptr_t ptr = a->flag_ptr;
    int32_t v = a->value;
    a->cpuinfer->task_queue_->enqueue([ptr, v]() {
      std::atomic_thread_fence(std::memory_order_seq_cst);
      *reinterpret_cast<volatile int32_t*>(ptr) = v;
    });
  }
  std::pair<intptr_t, intptr_t> signal_task(intptr_t flag_ptr, int32_t value) {
    SignalArgs* a = new SignalArgs{this, flag_ptr, value};  // submit이 cpuinfer 필드를 다시 채운다
    return std::make_pair((intptr_t)&signal_, (intptr_t)a);
  }

  static void sync_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    // "kt.sync(wait)" range = 이 host node가 실행된 시점부터 큐가 빌 때까지 = stream이
    // 이 지점에 도달한 뒤 **노출된** CPU 대기. (submit 마커 → 이 range 끝 = CPU 작업 창)
    KT_NVTX_PUSH("kt.sync(wait)");
    args->cpuinfer->task_queue_->sync(args->allow_n_pending);
    KT_NVTX_POP();
  }

  void sync(size_t allow_n_pending = 0) {
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    sync_(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA)
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
#endif
  }
#endif
 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;
};

#endif
