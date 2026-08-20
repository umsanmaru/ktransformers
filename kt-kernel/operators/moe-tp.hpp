#ifndef CPUINFER_OPERATOR_MOE_HPP
#define CPUINFER_OPERATOR_MOE_HPP

// #define CHECK

#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../cpu_backend/shared_mem_buffer.h"
#include "common.hpp"

// Forward declaration for Llamafile backend type checking
class LLAMA_MOE_TP;

template <typename T>
concept MOE_TP_PART = requires(T t, int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                               void* output, GeneralMOEConfig config, int tp_idx) {
  typename T::output_t;
  { new T(config, tp_idx) } -> std::same_as<T*>;
  { t.forward(qlen, k, expert_ids, weights, input, output) } -> std::same_as<void>;
  // { t.load_weights() } -> std::same_as<void>;
};

template <MOE_TP_PART T, typename Concrete = T>
class TP_MOE_Common : public MoE_Interface {
  static_assert(std::is_base_of_v<T, Concrete>);
  static_assert(std::is_constructible_v<Concrete, GeneralMOEConfig, int>);

 protected:
  std::vector<GeneralMOEConfig> tp_configs;
  int tp_count;
  int me_numa_id;
  std::vector<std::unique_ptr<T>> tps;

  std::vector<typename T::output_t*> local_output_numa;
  typename T::output_t* local_output = nullptr;

  bool weights_loaded = false;

#ifdef FORWARD_TIME_REPORT
  size_t forward_time_sum_ns = 0;
  size_t forward_count = 0;
#endif
 public:
  GeneralMOEConfig config;
  using input_t = typename T::input_t;
  using PartFactory = std::function<std::unique_ptr<T>(const GeneralMOEConfig&, int)>;

  TP_MOE_Common(const GeneralMOEConfig& config) : TP_MOE_Common(config, PartFactory{}) {}

  TP_MOE_Common(const GeneralMOEConfig& config, PartFactory part_factory) : config(config) {
    printf("TP MOE layer %d, pool: 0x%lx, expert num: %d, num_experts_per_tok: %d\n", config.layer_idx,
           (intptr_t)config.pool, config.expert_num, config.num_experts_per_tok);
    if (config.pool == nullptr) {
      printf("TP MOE layer %d, no worker pool\n", config.layer_idx);
      throw std::runtime_error("no worker pool");
    }

    this->config = config;
    tp_count = config.pool->config.subpool_count;
    if (config.intermediate_size % tp_count != 0) {
      printf("intermediate_size %d, tp count %d\n", config.intermediate_size, tp_count);
      throw std::runtime_error(
          "For TP, intermediate_size must be a "
          "multiple of NUMA node count");
    }

    // Check if this is Llamafile backend using compile-time type checking
    constexpr bool is_llamafile = std::is_same<T, LLAMA_MOE_TP>::value;
#ifndef QK_K
#define QK_K 256
#endif

    if (is_llamafile) {
      // For Llamafile backend: use QK_K-aligned TP splitting
      if (config.intermediate_size % QK_K != 0) {
        printf("intermediate_size %d must be divisible by QK_K %d for Llamafile backend\n", config.intermediate_size,
               QK_K);
        throw std::runtime_error("intermediate_size must be divisible by QK_K (256) for Llamafile backend");
      }

      int num_blocks = config.intermediate_size / QK_K;
      int base_blocks = num_blocks / tp_count;
      int extra_blocks = num_blocks % tp_count;

      if (base_blocks == 0) {
        printf("intermediate_size %d is too small for tp_count %d (num_blocks=%d)\n", config.intermediate_size,
               tp_count, num_blocks);
        throw std::runtime_error("intermediate_size too small: cannot distribute blocks to all TP instances");
      }

      printf("Llamafile TP splitting: intermediate_size=%d, tp_count=%d, QK_K=%d\n", config.intermediate_size, tp_count,
             QK_K);
      printf("  num_blocks=%d, base_blocks=%d, extra_blocks=%d\n", num_blocks, base_blocks, extra_blocks);

      int current_offset = 0;
      for (auto i = 0; i < tp_count; i++) {
        tps.push_back(nullptr);
        GeneralMOEConfig tp_config = config;

        // First extra_blocks TPs get one more block
        int num_blocks_for_this_tp = base_blocks + (i < extra_blocks ? 1 : 0);
        tp_config.intermediate_size = num_blocks_for_this_tp * QK_K;

        printf("  TP %d: intermediate_size=%d, offset=%d, blocks=%d\n", i, tp_config.intermediate_size, current_offset,
               num_blocks_for_this_tp);

        tp_configs.push_back(tp_config);
        current_offset += tp_config.intermediate_size;
      }
    } else {
      // For non-Llamafile backends: use simple equal division
      if (config.intermediate_size % tp_count != 0) {
        printf("intermediate_size %d, tp count %d\n", config.intermediate_size, tp_count);
        throw std::runtime_error(
            "For TP, intermediate_size must be a "
            "multiple of NUMA node count");
      }

      // Prism partial: 노드 shard를 Plan 주입 테이블에서 읽는다 (없으면 균등).
      // 검증은 여기(호출 스레드, ctor)에서 — 테이블-노드 불일치는 즉사.
      const auto& pg = config.partial;
      const bool has_shard_tables = config.is_partial() && !pg.node_gateup_n_rows.empty();
      if (has_shard_tables) {
        auto check_table = [&](const std::vector<int>& off, const std::vector<int>& rows, int full,
                               const char* name) {
          if ((int)off.size() != tp_count || (int)rows.size() != tp_count) {
            throw std::runtime_error(std::string("partial shard table size != tp_count: ") + name);
          }
          int sum = 0;
          for (int i = 0; i < tp_count; i++) {
            if (rows[i] <= 0 || off[i] < 0 || off[i] + rows[i] > full) {
              throw std::runtime_error(std::string("partial shard table out of range: ") + name);
            }
            sum += rows[i];
          }
          if (sum != full) {
            throw std::runtime_error(std::string("partial shard rows do not sum to full dim: ") + name);
          }
        };
        check_table(pg.node_gateup_n_offset, pg.node_gateup_n_rows, config.intermediate_size, "gateup_n");
        check_table(pg.node_down_n_offset, pg.node_down_n_rows, config.hidden_size, "down_n");
      } else if (config.is_partial() && config.hidden_size % tp_count != 0) {
        throw std::runtime_error("partial without shard tables: hidden_size must divide tp_count");
      }

      for (auto i = 0; i < tp_count; i++) {
        tps.push_back(nullptr);
        GeneralMOEConfig tp_config = config;
        if (config.is_partial()) {
          if (has_shard_tables) {
            tp_config.intermediate_size = pg.node_gateup_n_rows[i];
            tp_config.partial.gateup_n = {pg.node_gateup_n_offset[i], pg.node_gateup_n_rows[i]};
            tp_config.partial.down_n = {pg.node_down_n_offset[i], pg.node_down_n_rows[i]};
          } else {
            const int i_node = config.intermediate_size / tp_count;
            const int h_node = config.hidden_size / tp_count;
            tp_config.intermediate_size = i_node;
            tp_config.partial.gateup_n = {i * i_node, i_node};
            tp_config.partial.down_n = {i * h_node, h_node};
          }
        } else {
          tp_config.intermediate_size /= tp_count;
        }
        tp_configs.push_back(tp_config);
      }
    }

    std::vector<std::exception_ptr> construction_errors(tp_count);
    config.pool->dispense_backend()->do_numa_job([this, &construction_errors, &part_factory](int i) {
      try {
        tps[i] = part_factory ? part_factory(tp_configs[i], i) : std::unique_ptr<T>(new Concrete(tp_configs[i], i));
      } catch (...) {
        construction_errors[i] = std::current_exception();
      }
    });
    for (int i = 0; i < tp_count; ++i) {
      if (!construction_errors[i]) continue;
      try {
        std::rethrow_exception(construction_errors[i]);
      } catch (const std::exception& error) {
        throw std::runtime_error("TP/NUMA " + std::to_string(i) +
                                 " construction failed: " + error.what());
      } catch (...) {
        throw std::runtime_error("TP/NUMA " + std::to_string(i) +
                                 " construction failed with an unknown error");
      }
    }

    local_output_numa.resize(tp_count, nullptr);
    MemoryRequest mem_requests;
    for (auto i = 0; i < tp_count; i++) {
      mem_requests.append_pointer(
          &local_output_numa[i],
          (size_t)sizeof(typename T::output_t) * tp_configs[i].max_possible_qlen() * tp_configs[i].hidden_size);
    }
    mem_requests.append_pointer(
        (void**)&local_output,
        sizeof(typename T::output_t) * tp_configs[0].max_possible_qlen() * tp_configs[0].hidden_size);
    // printf("local output tp, %d,\n", tp_configs[0].max_possible_qlen());
    shared_mem_buffer.alloc(this, mem_requests);
  }

  ~TP_MOE_Common() { shared_mem_buffer.dealloc(this); }

  void warm_up() {
    int qlen = config.max_possible_qlen();
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config.num_experts_per_tok);
    std::vector<float> weights(qlen * config.num_experts_per_tok);
    for (int i = 0; i < qlen * config.num_experts_per_tok; i++) {
      expert_ids[i] = i % config.expert_num;
      weights[i] = 0.01;
    }
    forward(&qlen, config.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data(), false);
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental = false) {
    int qlen_local = qlen;
    forward(&qlen_local, k, expert_ids, weights, input, output, incremental);
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    forward(qlen_ptr, k, expert_ids, weights, input, output, false);
  }

  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
            incremental);
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental) {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
    if (config.is_partial()) [[unlikely]] {
      // 반드시 호출 스레드에서 거부해야 한다 — worker 스레드(do_numa_job)
      // 안에서 던지면 uncaught → std::terminate (base::forward의 guard는
      // 비-TP 직접 호출용 이중 방어로 유지).
      throw std::runtime_error("TP_MOE::forward called on a Prism partial instance");
    }
#ifdef FORWARD_TIME_REPORT
    auto start = std::chrono::high_resolution_clock::now();
#endif
    int qlen = *qlen_ptr;

    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });

    merge_results(qlen, output, incremental);
#ifdef FORWARD_TIME_REPORT
    auto end = std::chrono::high_resolution_clock::now();
    auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    int unique_experts = 0;
    {
      std::unordered_set<int64_t> expert_set;
      for (int i = 0; i < qlen * config.num_experts_per_tok; i++) {
        expert_set.insert(expert_ids[i]);
      }
      unique_experts = expert_set.size();
    }
    auto band_width =
        (1.0 * unique_experts * config.hidden_size * config.intermediate_size * 3 / 1e9) / (1.0 * forward_time / 1e6);
    auto GFLOPS =
        (1.0 * config.hidden_size * config.intermediate_size * qlen * 3 * config.num_experts_per_tok * 2 / 1e9) /
        (1.0 * forward_time / 1e6);
    if (qlen <= 10) {
      forward_time_sum_ns += forward_time;
      forward_count++;
    }
    auto average_bandwidth =
        (1.0 * forward_count * unique_experts * config.hidden_size * config.intermediate_size * 3 / 1e9) /
        (1.0 * forward_time_sum_ns / 1e6);
    printf(
        "forward time %ld, time stamp:%ld, band width %f GElement/s, ave bandwidth %f GElement/s (only "
        "decode), %f GFLOPS, me numa: %d\n",
        forward_time, end.time_since_epoch().count() / 1000 % 100000000, band_width, average_bandwidth, GFLOPS,
        numa_node_of_cpu(sched_getcpu()));
#endif
  }

  // ── Prism partial: gateup pre-act partial (계약 ②) ────────────────────
  // forward 가족과 같은 2단 구성: 타입 있는 본체 + intptr 캐스팅 binding.
  // gate/up은 inter 축 열 분할이므로 각 노드가 full-width out의 자기 shard
  // 열 구간에 직접 쓴다 — merge 단계가 없다 (down partial은 K2에서 hidden
  // 축 N-분할로 동형 구현 예정). routing weight 인자도 없다 (가중은 GPU
  // rejoin). T가 forward_gateup_partial을 제공할 때만 인스턴스화되는
  // 템플릿-지연 메서드다 (llamafile 등 미지원 백엔드는 바인딩하지 않음).
  void forward_gateup_partial(int* qlen_ptr, int k, const int64_t* expert_ids, const void* input, void* out) {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
    int qlen = *qlen_ptr;
    // 검증은 반드시 호출 스레드에서 — worker(do_numa_job) 안의 throw는
    // uncaught → std::terminate. (node 쪽 guard들은 비-TP 직접 호출용.)
    if (qlen != 1) [[unlikely]] {
      throw std::runtime_error("forward_gateup_partial: only qlen==1 (decode) is supported (K1)");
    }
    if (config.n_total() != config.intermediate_size) [[unlikely]] {
      // 상위 config의 n_total은 full inter와 같아야 노드 shard 열 쓰기가 안전
      throw std::runtime_error("forward_gateup_partial: partial.n_total != full intermediate_size");
    }
    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, qlen, k, expert_ids, input, out](int numa_id) {
      tps[numa_id]->forward_gateup_partial(qlen, k, expert_ids, input, out);
    });
  }

  void forward_gateup_partial_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t input, intptr_t out) {
    forward_gateup_partial((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const void*)input, (void*)out);
  }

  // ── Prism partial: down partial (계약 ②) — gateup과 동형 2단 구성 ─────
  // down도 N(hidden) 축 열 분할이므로 각 노드가 full-width out
  // [qlen×k×hidden]의 자기 shard 열 구간에 직접 쓴다 — 노드 합산 없음.
  // 입력 act는 full-width [qlen×k×inter] (GPU rejoin#1 결과가 D2H로 내려온 것).
  void forward_down_partial(int* qlen_ptr, int k, const int64_t* expert_ids, const void* act, void* out) {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
    int qlen = *qlen_ptr;
    if (qlen != 1) [[unlikely]] {
      throw std::runtime_error("forward_down_partial: only qlen==1 (decode) is supported (K2)");
    }
    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, qlen, k, expert_ids, act, out](int numa_id) {
      tps[numa_id]->forward_down_partial(qlen, k, expert_ids, act, out);
    });
  }

  void forward_down_partial_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t act, intptr_t out) {
    forward_down_partial((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const void*)act, (void*)out);
  }

  virtual void load_weights() = 0;

  virtual void merge_results(int qlen, void* output) = 0;

  virtual void merge_results(int qlen, void* output, bool incremental) {
    if (incremental == false) {
      merge_results(qlen, output);
    } else {
      throw std::runtime_error("Not Implemented");
    }
  };
};

template <MOE_TP_PART T>
class TP_MOE : public TP_MOE_Common<T> {
 public:
  using TP_MOE_Common<T>::TP_MOE_Common;
  void load_weights(const uint64_t* physical_to_logical_map) { throw std::runtime_error("Not Implemented"); }
  // void merge_results(int qlen, void *output, bool incremental) { throw std::runtime_error("Not Implemented"); }
};

#endif
