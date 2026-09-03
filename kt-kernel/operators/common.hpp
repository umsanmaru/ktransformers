#ifndef CPUINFER_OPERATOR_COMMON_HPP
#define CPUINFER_OPERATOR_COMMON_HPP

#include <map>

#include "../cpu_backend/worker_pool.h"
#include "ggml.h"

#if defined(__aarch64__) && defined(CPU_USE_KML)
#include <arm_sve.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#define ASSERT_RELEASE(x, text)                                                            \
  do {                                                                                     \
    if (!(x)) {                                                                            \
      fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", #x, __FILE__, __LINE__); \
      fprintf(stderr, "Error message: %s\n", (text));                                      \
      throw std::runtime_error((text));                                                    \
    }                                                                                      \
  } while (0)

#define PUSH_MEM_REQ(ptr, size) mem_requests.append_pointer(&(ptr), (size))

#define PROFILE_RECORD_TIME_STAMP(name)                                                             \
  do {                                                                                              \
    auto end_time = std::chrono::high_resolution_clock::now();                                      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - last).count(); \
    time_map[(name)] = duration;                                                                    \
    last = end_time;                                                                                \
  } while (0)

#define DO_TPS_LOAD_WEIGHTS(pool)                                                         \
  (pool)->dispense_backend()->do_numa_job([this, pool, config](int numa_id) {             \
    this->tps[numa_id]->config_.physical_to_logical_map = config.physical_to_logical_map; \
    this->tps[numa_id]->load_weights();                                                   \
  })

#define expert_map(m, x) (m != nullptr ? m[(x)] : (x))

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
T* offset_pointer(T* ptr, size_t byte_offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + byte_offset);
}

template <typename T>
size_t pointer_offset(T* ptr, T* b) {
  return reinterpret_cast<size_t>(b) - reinterpret_cast<size_t>(ptr);
}

template <typename T>
const T* offset_pointer(const T* ptr, size_t byte_offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) + byte_offset);
}

template <typename T>
T* offset_pointer_row_major(T* t, int row, int col, size_t ld) {
  return offset_pointer(t, row * ld) + col;
}

template <typename T>
T* offset_pointer_col_major(T* t, int row, int col, size_t ld) {
  return offset_pointer(t, col * ld) + row;
}

class TimePerf {
 protected:
  std::string time_perf_name;
  std::map<std::string, long> time_map;
  std::chrono::time_point<std::chrono::high_resolution_clock> last;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

  void forward_perf_start() {
    start_time = std::chrono::high_resolution_clock::now();
    last = start_time;
  }

  void perf_report() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::string output = time_perf_name + ", forward time: " + std::to_string(duration.count()) + " us";
    // for (auto [name, t] : time_map) {
    //   double p = 100.0 * t / duration.count();
    //   // if (p < 1.0) {
    //   //   continue; // Skip if the percentage is less than 1%
    //   // }
    //   output += ", " + name + ": " + std::to_string(t) + " us(" + std::to_string(size_t(round(p))) + "%)";
    // }
    // 反向遍历
    for (auto it = time_map.rbegin(); it != time_map.rend(); ++it) {
      const std::string& name = it->first;
      long t = it->second;
      double p = 100.0 * t / duration.count();
      // if (p < 1.0) {
      //   continue; // Skip if the percentage is less than 1%
      // }
      output += ", " + name + ": " + std::to_string(t) + " us(" + std::to_string(size_t(round(p))) + "%)";
    }
    printf("%s\n", output.c_str());
  }
};

struct TaskCounter {
  std::vector<size_t> fold = {}, card = {};

  TaskCounter(std::initializer_list<size_t> i) {
    card.push_back(1);
    for (auto j : i) {
      push_back(j);
    }
  }

  void push_back(size_t i) {
    fold.push_back(i);
    for (auto& c : card) {
      c *= i;
    }
    card.push_back(1);
  }
  void push_back(std::vector<size_t> i) {
    for (auto j : i) {
      push_back(j);
    }
  }
  size_t count() { return card[0]; }
  size_t at(size_t id, size_t which) { return id % card.at(which) / card.at(which + 1); }
};

struct GeneralConfig {
  size_t vocab_size;
  size_t hidden_size;

  size_t num_experts_per_tok;
  size_t n_routed_experts;
  size_t n_shared_experts;
  size_t max_qlen = 4096;

  void* lm_heads_ptr;
  ggml_type lm_heads_type;
  void* norm_weights_ptr;
  ggml_type norm_weights_type;
  void* token_embd_ptr;
  ggml_type token_embd_type;
  WorkerPool* pool = nullptr;
  GeneralConfig() {}
};

struct GeneralMLAConfig {
  size_t hidden_size;
  size_t q_lora_rank;
  size_t num_heads;
  size_t nope_size;
  size_t rope_size;
  size_t kv_lora_rank;

  int layer_idx = 0;
  WorkerPool* pool = nullptr;
  size_t token_count_in_page = 256;  // token count in a page
  size_t max_qlen = 1024;
  size_t max_kvlen = 4096;

  // rope
  size_t max_position_embeddings;
  double rope_scaling_factor = 1.0;
  double rope_theta = 10000.0;
  double rope_scaling_beta_fast;
  double rope_scaling_beta_slow;
  double rope_scaling_mscale;
  double rope_scaling_mscale_all_dim;
  double rope_scaling_original_max_position_embeddings;

  void* q_a_proj;
  void* q_a_norm = nullptr;
  void* q_b_proj;
  void* kv_a_proj_with_mqa;
  void* kv_a_norm = nullptr;
  void* kv_b_proj;
  void* o_proj;

  // for llamafile
  ggml_type q_a_proj_type;
  ggml_type q_a_norm_type;
  ggml_type q_b_proj_type;
  ggml_type kv_a_proj_with_mqa_type;
  ggml_type kv_a_norm_type;
  ggml_type kv_b_proj_type;
  ggml_type w_o_type;

  ggml_type input_type = GGML_TYPE_F32;
  ggml_type output_type = GGML_TYPE_F32;

  size_t m_block = 4;
  size_t n_block = 4;
  // for kvcache
  size_t page_count = 200;  // page count for kv cache

  GeneralMLAConfig() {}
  GeneralMLAConfig(size_t hidden_size, size_t q_lora_rank, size_t kv_lora_rank, size_t num_heads, size_t nope_size,
                   size_t rope_size)
      : hidden_size(hidden_size),
        q_lora_rank(q_lora_rank),
        kv_lora_rank(kv_lora_rank),
        num_heads(num_heads),
        nope_size(nope_size),
        rope_size(rope_size) {}
};

struct QuantConfig {
  std::string quant_method = "";
  int bits = 0;
  int group_size = 0;
  bool zero_point = false;
  bool per_channel = false;  // Per-channel quantization (GLM-4.7-FP8 style)
};

// ── Prism partial geometry types (계약: sglang prism CONTRACTS.md ②) ────
// 함께 다니는 값들을 구조로 묶는다 — GeneralMOEConfig의 평평한 필드 사이에
// 섞이지 않도록 config.partial.* 네임스페이스를 형성.

// contraction 축의 연속 밴드 [offset, offset + rows) — N shard 전용
struct KRange {
  int offset = 0;
  int rows = 0;
};

// contraction 축에서 이 티어가 소유하는 행 (sglang CONTRACTS.md ① 2026-08-25).
// 모드는 정확히 둘이다:
//
//  - **퇴화형** (`row_off` 비었음): 전 expert 공통 연속 밴드
//    [offset, offset + rows). 밴드 시절과 비트 동일 경로이고, gather도 하지
//    않는다 (포인터 산술만).
//  - **인덱스형**: `row_off`[E+1]이 expert별 구간을, `idx`가 원본 K행 번호를
//    준다. 길이가 expert마다 달라도 되고 순서도 임의다. 둘은 **항상 짝**으로
//    온다 — 연속인 인덱스도 idx를 싣는다(2 B/행). 모드를 셋으로 늘려
//    "row_off는 있고 idx는 없는" 경우를 만들면 gather 경로가 갈라지는데,
//    그 절약(행당 2 B)이 분기 하나만 못하다.
//
// `row_base`는 weight 주입 오프셋이기도 하다: expert e의 블록은 ckpt 방향
// [N, k(e)]이므로 시작 원소가 N * row_base(e)다. 하나의 테이블이 인덱스와
// 스토어를 함께 서비스한다 (계약 ③).
struct KIndex {
  int offset = 0;
  int rows = 0;
  std::vector<int32_t> row_off;   // [E+1]
  std::vector<uint16_t> idx;      // [row_off.back()]
  // [E] — 패딩 전의 실제 행 수. 비어 있으면 패딩이 없다는 뜻(k(e)와 같다).
  //
  // plan/자산이 지켜야 하는 정렬은 페어(%2)뿐인데(sglang 계약 ①) packed 저장은
  // 커널 타일(K_STEP)의 배수를 요구하므로, 로더가 k_real(e)를 타일 경계까지
  // 올리고 그 만큼의 **0 행**을 weight에 채워 넣는다. dense 경로는 0을 곱해
  // 무해하지만, sparse 경로는 tail 비트를 명시적으로 꺼야 한다 — thr == 0이면
  // imp = 0 >= 0이 참이라 비트가 서고 쓸모없는 B 로드가 생긴다.
  std::vector<int32_t> real_rows;

  inline bool indexed() const { return !row_off.empty(); }

  inline int k(int e) const { return indexed() ? row_off[e + 1] - row_off[e] : rows; }

  inline int k_real(int e) const { return real_rows.empty() ? k(e) : real_rows[e]; }

  inline int64_t row_base(int e) const {
    return indexed() ? (int64_t)row_off[e] : (int64_t)e * rows;
  }

  inline const uint16_t* index_of(int e) const {
    return indexed() ? idx.data() + row_off[e] : nullptr;
  }

  // 버퍼 산정용 최악치. 풀은 전 expert가 공유하므로 max로 잡아야 한다.
  inline int k_max() const {
    if (!indexed()) return rows;
    int m = 0;
    for (size_t e = 0; e + 1 < row_off.size(); ++e) {
      m = std::max(m, row_off[e + 1] - row_off[e]);
    }
    return m;
  }

  inline bool same_as(const KIndex& o) const {
    return offset == o.offset && rows == o.rows && row_off == o.row_off && idx == o.idx &&
           real_rows == o.real_rows;
  }
};

struct PartialGeometry {
  bool enabled = false;
  // gate와 up은 **독립**이다 (2026-08-25). 기하가 같으면 pack을 공유하고,
  // 다르면 activation을 두 번 pack한다 (dual-pack) — A 풀이 2배가 되는 것이
  // 이 자유도의 값이다. 같은 값을 넣으면 이전과 완전히 같은 경로를 탄다.
  KIndex gate;  // hidden 축
  KIndex up;    // hidden 축
  KIndex down;  // intermediate 축, global 좌표. full이면 {0, intermediate_size} 명시
  int n_total = 0;  // gate/up 출력축(intermediate)의 full 크기

  // ── K2: NUMA N-shard — 전 proj가 각자의 출력축(N)으로 노드 분할 ──────
  // (down도 N(hidden) 분할: act가 GPU rejoin을 거쳐 full-width로 돌아오므로
  //  kt의 act-locality 결합이 없고, 노드들은 서로소 열에 직접 쓴다)
  //
  // top-level 테이블 (Plan이 주입; 노드 i의 shard). 비어 있으면 TP ctor가
  // 균등 분할로 채운다. 불균등 비율 = warm 소켓의 cold 몫 축소 노브.
  std::vector<int> node_gateup_n_offset;  // inter 축
  std::vector<int> node_gateup_n_rows;
  std::vector<int> node_down_n_offset;  // hidden 축
  std::vector<int> node_down_n_rows;

  // node-scope: TP ctor가 각 tp_config에 기록하는 "이 인스턴스의 N shard".
  // (gateup_n.rows는 노드 intermediate_size와 항상 같아야 함 — init이 검증)
  KRange gateup_n;
  KRange down_n;

  // ── 입력기반 sparsity (k2wl2, 계약 ①) ────────────────────────────────
  // 점수 재료는 weight와 같은 수명이다 (로드 시 주입, host 상주 NUMA-local).
  // 전부 이 인스턴스의 **cold K 밴드로 절단된** fp32 테이블이고, expert 축이
  // 앞이다: *_wn_sq = [E, rows] (= wn², a), *_pair_dot = [E, rows/2] (= c).
  // gate와 up은 K 밴드를 공유하지만 노름·내적·threshold가 모두 다르므로
  // 테이블도 따로 받는다.
  //
  // threshold 곡선과 예산도 여기 상주한다 (2026-08-24). step마다 오는 것은
  // 라우터 가중뿐이고, 예산 → s → 격자 조회 → 판정이 전부 이쪽에서 끝난다:
  //
  //   s   = clip(p - lam*(g_e - gbar), 0, pmax)   # renorm_it회 재정규화
  //   thr = thr_*[expert * ng + round(s / grid)]
  //
  // GPU가 thr을 소비하지 않으므로(warm은 dense) 계산해서 내려보내던 왕복을
  // 없앤 것이다. thr_* 는 [E, ng] fp32 host 테이블.
  struct SparsityGeometry {
    bool enabled = false;
    // void*는 GeneralMOEConfig의 weight 포인터들과 같은 관례 (DEF_PTR_PROPERTY).
    void* gate_wn_sq = nullptr;
    void* gate_pair_dot = nullptr;
    void* up_wn_sq = nullptr;
    void* up_pair_dot = nullptr;
    void* down_wn_sq = nullptr;
    void* down_pair_dot = nullptr;
    void* thr_gate = nullptr;
    void* thr_up = nullptr;
    void* thr_down = nullptr;
    // 예산: proj별 (p, lam) + 격자 공통값.
    float p_gate = 0.0f, lam_gate = 0.0f;
    float p_up = 0.0f, lam_up = 0.0f;
    float p_down = 0.0f, lam_down = 0.0f;
    float pmax = 0.0f;
    float grid = 0.0f;
    int ng = 0;
    int renorm_it = 0;
  };
  SparsityGeometry sparsity;
};

struct GeneralMOEConfig {
  // Basic Config
  int expert_num;
  int num_experts_per_tok;
  int hidden_size;
  int intermediate_size;

  int layer_idx = 0;
  WorkerPool* pool = nullptr;

  // SGLang offload
  int num_gpu_experts = 0;              // Computed from gpu_experts_mask
  uint8_t* gpu_experts_mask = nullptr;  // Bool mask: true = expert on GPU
  void* physical_to_logical_map = nullptr;

  // Compute num_gpu_experts from gpu_experts_mask
  void compute_num_gpu_experts() {
    num_gpu_experts = 0;
    if (gpu_experts_mask) {
      for (int i = 0; i < expert_num; i++) {
        if (gpu_experts_mask[i]) num_gpu_experts++;
      }
    }
  }

  // Check if expert should be skipped (invalid, out of range, or on GPU)
  inline bool should_skip_expert(int64_t expert_id) const {
    return expert_id < 0 || expert_id >= expert_num || (gpu_experts_mask && gpu_experts_mask[expert_id]);
  }

  // 단계별 skip: cold 몫이 **0행**인 expert는 이 노드가 계산할 것이 없다.
  // gpu_experts_mask와 의미론이 같고(계산 없음 + export가 0을 쓴다) 시점만
  // 다르다 — 마스크는 step마다 정해지고, 이쪽은 기하가 로드 때 정한다.
  // 예산이 expert별로 갈리는 plan(전량 GPU 티어인 expert)에서 나온다.
  // 단계로 나눈 이유는 K가 proj별이기 때문이다: gate/up이 비어도 down은
  // 남을 수 있다(그 반대도). gate와 up은 한 단계에서 융합되므로 **둘 다**
  // 비어야 건너뛴다 — 한쪽만 비는 기하는 로더가 거부한다(validate_kindex).
  inline bool skip_gateup(int64_t expert_id) const {
    return should_skip_expert(expert_id) ||
           (partial.enabled && gate_k((int)expert_id) == 0 && up_k((int)expert_id) == 0);
  }
  inline bool skip_down(int64_t expert_id) const {
    return should_skip_expert(expert_id) || (partial.enabled && down_k((int)expert_id) == 0);
  }

  void* gate_proj = nullptr;
  void* up_proj = nullptr;
  void* down_proj = nullptr;

  void* gate_scale = nullptr;
  void* up_scale = nullptr;
  void* down_scale = nullptr;

  void* gate_zero = nullptr;
  void* up_zero = nullptr;
  void* down_zero = nullptr;

  QuantConfig quant_config;

  // for amx
  int max_len = 0;
  std::vector<std::vector<void*>> gate_projs;
  std::vector<std::vector<void*>> up_projs;
  std::vector<std::vector<void*>> down_projs;
  std::vector<std::vector<void*>> gate_scales;
  std::vector<std::vector<void*>> up_scales;
  std::vector<std::vector<void*>> down_scales;
  std::vector<std::vector<void*>> gate_zeros;
  std::vector<std::vector<void*>> up_zeros;
  std::vector<std::vector<void*>> down_zeros;

  // Pre-quantized backward weights (transposed, in BufferB format) [tp_count][expert_id]
  std::vector<std::vector<void*>> gate_bwd_projs;
  std::vector<std::vector<void*>> up_bwd_projs;
  std::vector<std::vector<void*>> down_bwd_projs;
  std::vector<std::vector<void*>> gate_bwd_scales;
  std::vector<std::vector<void*>> up_bwd_scales;
  std::vector<std::vector<void*>> down_bwd_scales;

  std::string path;
  bool save = false;
  bool load = false;
  bool share_backward_bb = false;
  bool share_cache_pool = false;

  // for llamafile
  int m_block = 4;
  int group_min_len = 0;
  int group_max_len = 0;
  int gate_type;
  int up_type;
  int down_type;
  int hidden_type;

  int max_cache_depth = 1;

  // SwiGLU asymmetric clamp applied to gate/up before silu*up. 0.0f =
  // disabled (default for all non-MXFP4 paths). Set to e.g. 10.0f for
  // DeepSeek V4-Flash 2604B routed experts, matching the trtllm
  // `gemm1_clamp_limit` and the sglang deep_gemm path's
  // `_apply_swiglu_limit`:
  //   gate = clamp(gate, max=limit)            // one-sided (silu input)
  //   up   = clamp(up, min=-limit, max=limit)  // symmetric
  // Read by `act_fn` in la/amx.hpp; non-zero only for MXFP4 today.
  // Origin: kt-sglang 耦合 (carries the V4-2604B limit set by sglang side).
  float swiglu_limit = 0.0f;

  // MiniMax M3 "swigluoai" activation: gate * sigmoid(gate * alpha) * (up + 1).
  // When alpha > 0, act_fn uses the swigluoai formula with symmetric clamp on
  // both gate and up (±swiglu_limit). 0.0f = disabled (standard silu path).
  float swiglu_alpha = 0.0f;

  // ── Prism partial (K-split tier) geometry ─────────────────────────────
  // 계약: sglang prism CONTRACTS.md ②. partial.enabled == false(기본)이면
  // 기존 full-K 동작과 완전히 동일하다. K-축 밴드 기하는 로드 시점에
  // 인스턴스에 구워지며(호출 인자에 없음), 입력 activation은 항상
  // full-width로 받아 인스턴스가 자기 offset에서 읽는다.
  // enabled == true이면 모든 range와 n_total은 명시값이어야 한다
  // ("0 = full" 센티널 없음 — down이 full이면 {0, intermediate_size}로 적음).
  PartialGeometry partial;

  inline bool is_partial() const { return partial.enabled; }
  // contraction 축 행 수 — **expert별**. 무인자 판을 남기지 않은 것은 의도다:
  // 가변 K에서 "이 자리가 per-expert인가 최악치인가"를 컴파일러가 강제로 묻게
  // 한다 (틀리면 조용한 오답이다).
  inline int gate_k(int e) const { return partial.enabled ? partial.gate.k(e) : hidden_size; }
  inline int up_k(int e) const { return partial.enabled ? partial.up.k(e) : hidden_size; }
  inline int down_k(int e) const { return partial.enabled ? partial.down.k(e) : intermediate_size; }
  // 패딩 전의 실제 행 수 — sparse 마스크의 tail을 끄는 데만 쓰인다.
  inline int gate_k_real(int e) const { return partial.enabled ? partial.gate.k_real(e) : hidden_size; }
  inline int up_k_real(int e) const { return partial.enabled ? partial.up.k_real(e) : hidden_size; }
  inline int down_k_real(int e) const {
    return partial.enabled ? partial.down.k_real(e) : intermediate_size;
  }
  // 풀 산정용 최악치. gate/up 풀은 하나를 나눠 쓰므로 둘의 최대다.
  inline int gateup_k_max() const {
    return partial.enabled ? std::max(partial.gate.k_max(), partial.up.k_max()) : hidden_size;
  }
  inline int down_k_max() const { return partial.enabled ? partial.down.k_max() : intermediate_size; }
  // gate와 up의 기하가 달라 activation을 두 번 pack해야 하는가.
  inline bool dual_pack() const { return partial.enabled && !partial.gate.same_as(partial.up); }
  // 주입 weight에서 expert e 블록의 시작 행 (열 수는 호출부가 곱한다)
  inline int64_t gate_row_base(int e) const {
    return partial.enabled ? partial.gate.row_base(e) : (int64_t)e * hidden_size;
  }
  inline int64_t up_row_base(int e) const {
    return partial.enabled ? partial.up.row_base(e) : (int64_t)e * hidden_size;
  }
  inline int64_t down_row_base(int e) const {
    return partial.enabled ? partial.down.row_base(e) : (int64_t)e * intermediate_size;
  }
  // gate/up 출력축(intermediate)의 full 크기 — TP 분할 후 노드 config의
  // intermediate_size는 shard가 되므로 full-width out 열 쓰기에 필요
  inline int n_total() const { return partial.enabled ? partial.n_total : intermediate_size; }
  // down의 출력축(hidden) 중 이 인스턴스의 shard 크기 (K2: down N-분할)
  inline int down_n() const { return partial.enabled ? partial.down_n.rows : hidden_size; }

  GeneralMOEConfig() {}

  GeneralMOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size)
      : expert_num(expert_num),
        num_experts_per_tok(routed_expert_num),
        hidden_size(hidden_size),
        intermediate_size(intermediate_size) {}

  int max_possible_qlen() { return std::max(max_len, group_max_len); }
};

// SFT (Supervised Fine-Tuning) configuration for MoE with LoRA
struct MOESFTConfig : public GeneralMOEConfig {
  // LoRA configuration
  int lora_rank = 16;
  float lora_alpha = 32.0f;
  float lora_dropout = 0.0f;
  float lora_scaling() const { return lora_alpha / lora_rank; }

  // LoRA weight pointers (directly pointing to Python tensor memory, zero-copy)
  // Layout: [expert_num, lora_rank, in_dim] for A, [expert_num, out_dim, lora_rank] for B
  void* gate_lora_a = nullptr;  // [expert_num, lora_rank, hidden_size]
  void* gate_lora_b = nullptr;  // [expert_num, intermediate_size, lora_rank]
  void* up_lora_a = nullptr;    // [expert_num, lora_rank, hidden_size]
  void* up_lora_b = nullptr;    // [expert_num, intermediate_size, lora_rank]
  void* down_lora_a = nullptr;  // [expert_num, lora_rank, intermediate_size]
  void* down_lora_b = nullptr;  // [expert_num, hidden_size, lora_rank]

  // Full weight gradient configuration
  bool full_weight_grad = false;

  // Opt in to the C++-authoritative optimizer-gradient lifecycle. Keep this
  // disabled by default so existing callers retain legacy buffer semantics.
  bool authoritative_optimizer_grads = false;

  // Base weight gradient buffer pointers (directly pointing to Python tensor memory, zero-copy)
  // Only used when full_weight_grad == true
  void* grad_gate_proj = nullptr;  // [expert_num, intermediate_size, hidden_size]
  void* grad_up_proj = nullptr;    // [expert_num, intermediate_size, hidden_size]
  void* grad_down_proj = nullptr;  // [expert_num, hidden_size, intermediate_size]

  MOESFTConfig() : GeneralMOEConfig() {}

  MOESFTConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size)
      : GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size) {}

  // Conversion constructor from GeneralMOEConfig (for MOE_TP_PART concept satisfaction)
  explicit MOESFTConfig(const GeneralMOEConfig& base) : GeneralMOEConfig(base) {
    // LoRA fields use default values (already initialized in struct definition)
  }
};

struct GeneralGateConfig {
  size_t hidden_size;
  size_t num_experts_per_tok;
  size_t n_routed_experts;
  size_t n_group;
  size_t topk_group;

  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;

  std::string scoring_func = "sigmoid";
  std::string topk_method = "noaux_tc";

  int layer_idx = 0;
  WorkerPool* pool = nullptr;

  void* weight = nullptr;
  ggml_type weight_type;
  void* e_score_correction_bias = nullptr;
  ggml_type e_score_correction_bias_type;

  size_t max_seqlen = 25600;

  GeneralGateConfig() = default;

  GeneralGateConfig(int hidden_size, int num_experts_per_tok, int n_routed_experts, int n_group, int topk_group)
      : hidden_size(hidden_size),
        num_experts_per_tok(num_experts_per_tok),
        n_routed_experts(n_routed_experts),
        n_group(n_group),
        topk_group(topk_group) {}
};

class MLA_Interface {
 public:
  virtual void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
                       const void* input, void* output) = 0;
};

class MoE_Interface {
 public:
  virtual void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output, bool incremental = false) = 0;
};
inline void init_ggml() {
  static bool inited = false;
  if (inited) {
    return;
  }
  struct ggml_init_params params = {
      0,
      NULL,
      true,
  };

  auto ctx_eval = ggml_init(params);

  if (!ctx_eval) {
    throw std::runtime_error("Failed to create ggml context");
  }
  inited = true;
}

template <typename A, typename B>
void convert_or_copy(A* dst, const B* src, size_t count) {
  if constexpr (std::is_same_v<A, B>) {
    // printf("Direct copy\n");
    memcpy(dst, src, sizeof(A) * count);
  } else {
    if constexpr (std::is_same_v<A, float>) {
      if constexpr (std::is_same_v<B, ggml_bf16_t>) {
        // printf("Converting ggml_bf16_t to float\n");
        ggml_bf16_to_fp32_row(src, dst, count);
      } else if constexpr (std::is_same_v<B, ggml_fp16_t>) {
        ggml_fp16_to_fp32_row(src, dst, count);
      } else {
        throw std::runtime_error("Unsupported conversion");
      }
    } else if constexpr (std::is_same_v<A, ggml_bf16_t>) {
      if constexpr (std::is_same_v<B, float>) {
        // printf("Converting float to ggml_bf16_t\n");
        ggml_fp32_to_bf16_row(src, dst, count);
      } else {
        throw std::runtime_error("Unsupported conversion");
      }
    }

    else {
      throw std::runtime_error("Unsupported conversion");
    }
  }
}

template <typename A>
void convert_or_copy(A* dst, void* src, ggml_type type, size_t count) {
  switch (type) {
    case GGML_TYPE_BF16: {
      auto src_bf16 = (ggml_bf16_t*)src;
      convert_or_copy(dst, src_bf16, count);
      break;
    }
    case GGML_TYPE_F16: {
#if defined(__aarch64__) && defined(CPU_USE_KML)
      auto src_fp16 = (float16_t*)src;
      convert_or_copy(dst, src_fp16, count);
#else
      throw std::runtime_error("GGML_TYPE_F16 is not supported on this platform");
#endif
      break;
    }
    case GGML_TYPE_F32: {
      auto src_f32 = (float*)src;
      convert_or_copy(dst, src_f32, count);
      break;
    }
    default:
      throw std::runtime_error("Unsupported type for conversion");
  }
}

template <typename A>
void check_numerics(A* data, size_t count) {
  for (size_t i = 0; i < count; i++) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      printf("Numerics check failed at index %zu: value = %f\n", i, data[i]);
      throw std::runtime_error("Numerics check failed");
    }
  }
  printf("Numerics check passed for %zu elements.\n", count);
}

inline void debug_bf16(ggml_bf16_t* x) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", ggml_bf16_to_fp32(x[i]));
  }
  printf("\n");
}
inline void debug_f32(float* x) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", x[i]);
  }
  printf("\n");
}

inline void debug_f32(float* x, size_t count) {
  if (count < 10) {
    for (size_t i = 0; i < count; i++) {
      printf("%f ", x[i]);
    }
  } else {
    for (size_t i = 0; i < 3; i++) {
      printf("%f ", x[i]);
    }
    printf("...");
    for (size_t i = count - 3; i < count; i++) {
      printf("%f ", x[i]);
    }
    printf("\n");
  }
}

#endif
