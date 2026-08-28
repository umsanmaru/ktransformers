/**
 * @Description  : Common AMX MoE base class extracted from K2 implementation.
 * @Author       : oql, Codex and Claude
 * @Date         : 2025-12-09
 * @Version      : 0.1.0
 * @LastEditors  : oql, Codex and Claude
 * @LastEditTime : 2025-12-09
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MOE_BASE_H
#define CPUINFER_OPERATOR_AMX_MOE_BASE_H

// #define FORWARD_TIME_PROFILE

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "la/act_plan.hpp"
#include "pair_mask.hpp"
#include "la/amx.hpp"
#include "llama.cpp/ggml.h"

// Prism partial prefill 단계 타이머 — env KT_PARTIAL_TIMING=1 이면 호출마다
// 단계별 ms를 stderr에 찍는다 (진단용, 기본 꺼짐 — 비용은 분기 하나).
struct PartialTimer {
  static bool enabled() {
    static const bool on = [] { const char* e = getenv("KT_PARTIAL_TIMING"); return e && e[0] == '1'; }();
    return on;
  }
  const char* phase; int part; int qlen;
  std::chrono::steady_clock::time_point t0, last;
  std::string line;
  PartialTimer(const char* ph, int p, int q) : phase(ph), part(p), qlen(q) {
    if (enabled()) { t0 = last = std::chrono::steady_clock::now(); }
  }
  void lap(const char* name) {
    if (!enabled()) return;
    auto now = std::chrono::steady_clock::now();
    char buf[64];
    snprintf(buf, sizeof(buf), " %s %.2f", name, std::chrono::duration<double, std::milli>(now - last).count());
    line += buf; last = now;
  }
  ~PartialTimer() {
    if (!enabled()) return;
    auto now = std::chrono::steady_clock::now();
    fprintf(stderr, "[partial %s tp%d qlen=%d] total %.2f ms:%s\n", phase, part, qlen,
            std::chrono::duration<double, std::milli>(now - t0).count(), line.c_str());
  }
};

// Prism cold GPU view — 한 (numa node, proj)의 packed B slab 기술자.
struct PrismColdSlabView {
  int node = 0;
  int proj = 0;                     // 0=gate, 1=up, 2=down
  uintptr_t ptr = 0;                // 페이지 정렬 slab 시작
  size_t bytes = 0;                 // 페이지 올림된 slab 크기
  std::vector<size_t> expert_off;   // [E+1] 바이트 오프셋 (expert 블록 시작)
  int n = 0;                        // 이 노드의 N 행 수 (packed 논리 B의 n)
  std::vector<int> k;               // [E] expert별 K (packed = 타일 올림된 값)
  int n_step = 0, k_step = 0, n_block = 0, k_block = 0;
};

template <class T, class Derived>
class AMX_MOE_BASE {
 public:
  int tp_part_idx = 0;

  ggml_bf16_t* m_local_input_ = nullptr;
  // dual-pack 전용: up의 gather된 activation. gate와 기하가 같으면 pack을
  // 공유하므로 할당조차 하지 않는다 (침습성 상한 — 기존 구성은 경로가 불변).
  ggml_bf16_t* m_local_input_up_ = nullptr;
  std::vector<ggml_bf16_t*> m_local_input_up_ptr_;
  ggml_bf16_t* m_local_gate_output_ = nullptr;
  ggml_bf16_t* m_local_up_output_ = nullptr;
  ggml_bf16_t* m_local_down_output_ = nullptr;

  std::vector<std::vector<int>> m_local_pos_;
  std::vector<int> m_local_num_;
  std::vector<int> m_expert_id_map_;
  std::vector<ggml_bf16_t*> m_local_input_ptr_;
  std::vector<ggml_bf16_t*> m_local_gate_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_up_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_down_output_ptr_;

  // Prism sparsity: (expert, proj)별 타일 비트마스크. 타일당 K_STEP/2(=16)
  // 페어, 비트 1 = 유지. A를 dense로 두고 커널이 로드를 건너뛴다 (gate/up이
  // gate_up_ba_를 공유하므로 마스크를 A에 구울 수 없다 — 계약 ①).
  // decode(qlen==1)에서만 채워진다: prefill은 토큰별 마스크가 B 재사용을
  // 깨서 이득이 없다 (prefill-dense).
  // 이 커널이 Seam B 인코더를 갖는가. 없으면 plan 관련 코드가 전부
  // 컴파일아웃되어 기존 operator는 무변경이다. (CRTP 파생이 소비하므로
  // 멤버들과 같은 접근 구역에 둔다.)
  static constexpr bool kHasPlanEncoder = requires { T::plan_arena_bytes(1); };

  void* plan_arena_for(int e, int slot) const {
    return (char*)m_plan_arena_ + (size_t)slot * m_plan_off_.back() + m_plan_off_[e];
  }

  // Seam B: 커널이 자기 blocking으로 인코딩한 per-expert 계획. 인코더가 없는
  // 커널에서는 비어 있고 아무도 읽지 않는다.
  ggml_bf16_t* m_plan_arena_ = nullptr;
  std::vector<amx::ActivationPlan> m_plan_gate_, m_plan_up_, m_plan_down_;
  std::vector<size_t> m_plan_off_;  // [E+1] expert별 arena 오프셋 (바이트)

  std::vector<std::vector<uint16_t>> m_pair_mask_gate_;
  std::vector<std::vector<uint16_t>> m_pair_mask_up_;
  std::vector<std::vector<uint16_t>> m_pair_mask_down_;
  bool m_sparse_active_ = false;  // 이 호출이 sparse 경로인가
  // s 벡터의 스택 상한 (num_experts_per_tok). 넘으면 ctor에서 즉사한다.
  static constexpr int kMaxTopK = 64;

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  // dual-pack 전용 (gate와 up의 K 기하가 다를 때만 채워진다)
  std::vector<std::shared_ptr<typename T::BufferA>> up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  // proj(0=gate,1=up,2=down)별 packed B slab — expert 블록은 64 B 정렬로 이어진다.
  void* cold_slab_[3] = {nullptr, nullptr, nullptr};
  size_t cold_slab_bytes_[3] = {0, 0, 0};
  std::vector<size_t> cold_slab_off_[3];  // [E+1] 바이트 오프셋

  void alloc_cold_slabs() {
    const size_t E = config_.expert_num;
    for (int proj = 0; proj < 3; ++proj) {
      auto& off = cold_slab_off_[proj];
      off.assign(E + 1, 0);
      for (size_t i = 0; i < E; ++i) {
        size_t bytes = proj == 0   ? buffer_b_required_size(config_.intermediate_size, config_.gate_k(i))
                       : proj == 1 ? buffer_b_required_size(config_.intermediate_size, config_.up_k(i))
                                   : buffer_b_required_size(config_.down_n(), config_.down_k(i));
        off[i + 1] = off[i] + ((bytes + 63) / 64) * 64;
      }
      size_t total = ((off.back() + 4095) / 4096) * 4096;
      if (total == 0) total = 4096;
      cold_slab_[proj] = std::aligned_alloc(4096, total);
      if (cold_slab_[proj] == nullptr) throw std::runtime_error("cold slab allocation failed");
      cold_slab_bytes_[proj] = total;
    }
  }
  void* cold_slab_ptr(int proj, size_t expert) const {
    return static_cast<char*>(cold_slab_[proj]) + cold_slab_off_[proj][expert];
  }

 public:
  // Prism cold hybrid: 이 노드 인스턴스만의 expert skip 마스크 (TP 공용 포인터 대신).
  // 마스크가 1인 expert는 이 노드가 계산하지 않고 partial 행을 0으로 채운다.
  void set_gpu_experts_mask(const uint8_t* mask) { config_.gpu_experts_mask = const_cast<uint8_t*>(mask); }

 public:
  // Prism cold GPU view (sglang prism이 cudaHostRegister + grouped GEMM으로 소비).
  // 레이아웃은 T::BufferB의 packed 6D (n_block → k_block → n_step → k_step → tile).
  PrismColdSlabView cold_slab_view(int proj) const {
    PrismColdSlabView v;
    v.node = tp_part_idx;
    v.proj = proj;
    v.ptr = reinterpret_cast<uintptr_t>(cold_slab_[proj]);
    v.bytes = cold_slab_bytes_[proj];
    v.expert_off = cold_slab_off_[proj];
    v.n = proj == 2 ? config_.down_n() : config_.intermediate_size;
    v.k.resize(config_.expert_num);
    for (size_t i = 0; i < config_.expert_num; ++i)
      v.k[i] = proj == 0 ? config_.gate_k(i) : proj == 1 ? config_.up_k(i) : config_.down_k(i);
    v.n_step = T::N_STEP;
    v.k_step = T::K_STEP;
    v.n_block = T::N_BLOCK;
    v.k_block = T::K_BLOCK;
    return v;
  }

 public:
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  size_t pool_count_ = 0;
  size_t gate_up_ba_pool_bytes_ = 0;
  size_t gate_bc_pool_bytes_ = 0;
  size_t up_bc_pool_bytes_ = 0;
  size_t down_ba_pool_bytes_ = 0;
  size_t down_bc_pool_bytes_ = 0;
  void* gate_up_ba_pool_ = nullptr;
  void* up_ba_pool_ = nullptr;  // dual-pack 전용
  void* gate_bc_pool_ = nullptr;
  void* up_bc_pool_ = nullptr;
  void* down_ba_pool_ = nullptr;
  void* down_bc_pool_ = nullptr;

  GeneralMOEConfig config_;
  // config_.dual_pack()의 값을 생성 시점에 한 번 굳힌 것.
  //
  // 그 메서드는 플래그 조회처럼 보이지만 gate와 up의 KIndex를 **전수 비교**
  // 한다 (`same_as` → `std::vector<uint16_t>` operator==). 인덱스가 Σₑ k[e]개라
  // E=128·k=1792이면 호출당 458 KB를 양쪽 읽고, 흔한 경우가 "같다"라서 조기
  // 종료도 없다 — 빠른 경로가 가장 비싼 경로다. 그런데 호출부가 활성 expert
  // 루프와 GEMM 태스크 안에 있어(층당 A회 이상 × 40층 × 스텝마다) 단일 스레드
  // memcmp 대역폭이 그대로 벽시계에 드러났다 (E=128에서 실측 gather 델타의
  // 상당분).
  //
  // 값은 로드 후 안 바뀌는 기하 속성이므로 스텝마다 답이 달라질 수 없다.
  // 지연 memoize(mutable 캐시) 대신 생성자에서 굳히는 이유는 워커 스레드들이
  // 동시에 읽기 때문이다 — 값이 늘 같아 실害는 없어도 형식상 레이스이고,
  // 단일 스레드 지점에서 미리 계산하면 그 문제가 아예 없다.
  bool dual_pack_ = false;
  using input_t = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  AMX_MOE_BASE(GeneralMOEConfig config, int tp_part_idx_) : tp_part_idx(tp_part_idx_), config_(config) {
    init();
    derived()->derived_init();
  }

  void set_staged_weight_pointers(void* gate, void* up, void* down, void* gate_scale, void* up_scale,
                                  void* down_scale) {
    config_.gate_proj = gate;
    config_.up_proj = up;
    config_.down_proj = down;
    config_.gate_scale = gate_scale;
    config_.up_scale = up_scale;
    config_.down_scale = down_scale;
  }

  void clear_staged_weight_pointers() {
    set_staged_weight_pointers(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  void init() {
    if (config_.load && config_.path == "") {
      config_.load = false;
    }

    // Prism partial: 기하 정합을 생성 시점에 검증 (조용한 오답 방지).
    // enabled면 모든 값이 명시여야 하고, 노드 shard는 TP ctor가 채운 상태다.
    if (config_.is_partial()) {
      const auto& pg = config_.partial;
      if (pg.n_total <= 0) {
        throw std::runtime_error("partial: enabled but n_total not explicit");
      }
      validate_kindex(pg.gate, config_.hidden_size, "gate");
      validate_kindex(pg.up, config_.hidden_size, "up");
      validate_kindex(pg.down, pg.n_total, "down");
      if (pg.gateup_n.rows != config_.intermediate_size) {
        throw std::runtime_error("partial: node gateup_n.rows != node intermediate_size");
      }
      if (pg.down_n.rows < 0 || pg.down_n.offset < 0 ||  // rows == 0: 이 노드는 down 열이 없다 (warm-kt 원격 노드)
          pg.down_n.offset + pg.down_n.rows > config_.hidden_size) {
        throw std::runtime_error("partial: node down_n shard out of hidden range");
      }
    }

    // 아래 버퍼 산정이 이미 이 값을 쓰므로 여기서 굳혀야 한다.
    dual_pack_ = config_.dual_pack();

    MemoryRequest mem_requests;
    const size_t ml = config_.max_len;
    const size_t k_tok = config_.num_experts_per_tok;
    const size_t H = config_.hidden_size;
    const size_t I = config_.intermediate_size;
    mem_requests.append_pointer(&m_local_input_, sizeof(ggml_bf16_t) * k_tok * ml * H);
    if (dual_pack_) {
      mem_requests.append_pointer(&m_local_input_up_, sizeof(ggml_bf16_t) * k_tok * ml * H);
    }
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(ggml_bf16_t) * k_tok * ml * I);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(ggml_bf16_t) * k_tok * ml * I);
    mem_requests.append_pointer(&m_local_down_output_, sizeof(ggml_bf16_t) * k_tok * ml * H);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    if (config_.partial.sparsity.enabled) {
      if (config_.num_experts_per_tok > kMaxTopK) {
        throw std::runtime_error("prism sparsity: num_experts_per_tok exceeds kMaxTopK");
      }
      // 타일 수는 expert별이다 (k(e)/K_STEP). 균일 plan에서는 전부 같은 값.
      m_pair_mask_gate_.resize(config_.expert_num);
      m_pair_mask_up_.resize(config_.expert_num);
      m_pair_mask_down_.resize(config_.expert_num);
      for (int e = 0; e < config_.expert_num; e++) {
        m_pair_mask_gate_[e].assign(config_.gate_k(e) / T::K_STEP, 0);
        m_pair_mask_up_[e].assign(config_.up_k(e) / T::K_STEP, 0);
        m_pair_mask_down_[e].assign(config_.down_k(e) / T::K_STEP, 0);
      }
    }
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_input_up_ptr_.resize(config_.expert_num);
    if constexpr (kHasPlanEncoder) {
      // per-expert K 때문에 arena도 expert별이다 — seg 길이가 k/K_BLOCK + 2라
      // expert마다 다르다 (KT_MERGE_CONTRACT §8.3 마지막 항).
      m_plan_off_.assign(config_.expert_num + 1, 0);
      for (int e = 0; e < config_.expert_num; e++) {
        const size_t need = std::max({T::plan_arena_bytes(config_.gate_k(e)),
                                      T::plan_arena_bytes(config_.up_k(e)),
                                      T::plan_arena_bytes(config_.down_k(e))});
        m_plan_off_[e + 1] = m_plan_off_[e] + ((need + 63) & ~(size_t)63);
      }
      // gate/up/down 세 벌 (한 레이어 안에서 동시에 살아 있다)
      mem_requests.append_pointer(&m_plan_arena_, 3 * m_plan_off_.back());
      m_plan_gate_.resize(config_.expert_num);
      m_plan_up_.resize(config_.expert_num);
      m_plan_down_.resize(config_.expert_num);
    }
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    // Prism partial: contraction(K) 축 치수는 접근자를 통해 읽는다 —
    // 기하 필드가 0(기본)이면 full 치수를 돌려주므로 기존 동작과 동일.
    // (gate/up의 K = gateup_k(), down의 K = down_k(); N 축은 불변)
    alloc_cold_slabs();
    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(make_buffer_a(config_.max_len, config_.gate_k(i), nullptr));
      if (dual_pack_) {
        up_ba_.push_back(make_buffer_a(config_.max_len, config_.up_k(i), nullptr));
      }
      gate_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(make_buffer_a(config_.max_len, config_.down_k(i), nullptr));
      down_bc_.push_back(make_buffer_c(config_.max_len, config_.down_n(), nullptr));

      // expert B 버퍼는 proj별 **slab 하나**에 연속 배치한다 (아래 alloc_cold_slabs).
      // 이유는 Prism의 cold GPU 읽기다 (prism-partial-entrypoints.md §14): GPU가
      // 이 packed 버퍼를 UVA로 제자리 읽으려면 cudaHostRegister를 해야 하는데,
      // expert마다 따로 aligned_alloc하면 인접 버퍼가 페이지를 공유해 등록이
      // 겹친다. slab은 페이지 정렬이고 (node, proj)당 1회 등록으로 끝난다.
      gate_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.gate_k(i),
                                       cold_slab_ptr(0, i)));
      up_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.up_k(i), cold_slab_ptr(1, i)));
      down_bb_.push_back(make_buffer_b(config_.down_n(), config_.down_k(i), cold_slab_ptr(2, i)));
    }
    // TODO: need update to all *.hpp
    // (config_.expert_num * T::M_STEP) in pool_count_ is to ensure padding for each experts.
    pool_count_ = (size_t)config_.max_len * config_.num_experts_per_tok + config_.expert_num * T::M_STEP;

    // 풀은 전 expert가 공유하므로 최악치로 잡는다 (가변 K에서 유일한 안전값).
    gate_up_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.gateup_k_max()) + pool_count_ * 64;
    gate_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    up_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.down_k_max()) + pool_count_ * 64;
    down_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.down_n()) + pool_count_ * 64;

    mem_requests.append_pointer(&gate_up_ba_pool_, gate_up_ba_pool_bytes_);
    if (dual_pack_) {
      // A 풀 2배 — gate≠up의 대가다 (doc §10.3).
      mem_requests.append_pointer(&up_ba_pool_, gate_up_ba_pool_bytes_);
    }
    mem_requests.append_pointer(&gate_bc_pool_, gate_bc_pool_bytes_);
    mem_requests.append_pointer(&up_bc_pool_, up_bc_pool_bytes_);
    mem_requests.append_pointer(&down_ba_pool_, down_ba_pool_bytes_);
    mem_requests.append_pointer(&down_bc_pool_, down_bc_pool_bytes_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  ~AMX_MOE_BASE() { shared_mem_buffer_numa.dealloc(tp_part_idx, this); }

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (config_.is_partial()) {
      // partial 인스턴스는 fused forward를 지원하지 않는다 — 잘못 호출되면
      // K-슬라이스 weight로 full expert를 계산해 조용히 틀린 값이 나온다.
      throw std::runtime_error("AMX_MOE_BASE::forward called on a Prism partial instance");
    }
    if (qlen > 1) {
      forward_prefill(qlen, k, expert_ids, weights, input, output);
    } else {
      forward_decode(k, expert_ids, weights, input, output);
    }
  }

  // ── Prism partial 진입점 (계약: sglang prism CONTRACTS.md ②) ──────────
  // forward_decode의 1~4단계(라우팅 준비 → 풀 carving → x pack → gate/up
  // GEMM)까지만 수행하고 pre-activation partial을 out에 export한다.
  // act / down / router 가중 / expert 합산 없음 — 전부 GPU rejoin 소관.
  //
  //   input: bf16 [qlen × hidden_FULL] — full-width. 이 인스턴스는
  //          [gateup_k_offset, +gateup_k_rows) 열만 읽는다 (기하는 로드
  //          시점에 구워졌고 호출 인자에 없다).
  //   out  : bf16 [qlen × k × 2·n_total], 열 [0,n_total)=gate,
  //          [n_total,2·n_total)=up. TP 노드는 자기 inter shard 열 구간만
  //          쓴다 (gate/up은 열 분할이라 노드 간 합산 없음). overwrite
  //          의미론 — skip된 expert slot은 미기록(undefined).
  //
  // decode(qlen==1)는 x에서 직접 pack하는 빠른 경로, qlen>1은 kt
  // forward_prefill과 동형의 토큰 그룹핑(gather가 밴드 슬라이싱을 흡수 —
  // strided pack 불필요) 경로를 탄다. stage/export는 두 경로가 공유한다
  // (decode는 m_local_pos_ == 0의 퇴화형).
  void forward_gateup_partial(int qlen, int k, const int64_t* expert_ids, const void* input, void* out,
                              const float* weights = nullptr) {
    if (qlen == 1) {
      // full-width 입력을 그대로 넘긴다 — 어느 열을 읽을지는 인스턴스에
      // 구워진 기하가 안다 (계약 ②-3). 단일 행이라 row stride 무관.
      const ggml_bf16_t* x_full = (const ggml_bf16_t*)input;
      int activated_expert = prepare_gateup_routing(k, expert_ids);
      if (activated_expert == 0) {
        return;
      }
      carve_gateup_decode(activated_expert, qlen);
      prep_gateup_decode(k, expert_ids, x_full, weights);
      run_gateup_stage(activated_expert, qlen);
      m_sparse_active_ = false;
      export_gateup_partial(qlen, k, expert_ids, (ggml_bf16_t*)out);
      return;
    }
    // prefill: 토큰별 마스크는 B 재사용을 깨서 이득이 없다 (prefill-dense).
    m_sparse_active_ = false;
    PartialTimer tm("gateup", tp_part_idx, qlen);
    int activated_expert = prepare_prefill_routing(qlen, k, expert_ids);
    if (activated_expert == 0) {
      return;
    }
    tm.lap("prep");
    carve_gateup_prefill();
    tm.lap("carve");
    gather_and_pack_gateup_prefill(qlen, k, expert_ids, (const ggml_bf16_t*)input, activated_expert);
    tm.lap("pack");
    run_gateup_stage(activated_expert, qlen);
    tm.lap("gemm");
    export_gateup_partial(qlen, k, expert_ids, (ggml_bf16_t*)out);
    tm.lap("export");
  }

 private:
  // ── K 인덱스 (sglang CONTRACTS.md ① 2026-08-25) ────────────────────────
  //
  // 티어 멤버십이 밴드에서 **가변 per-expert 인덱스**로 바뀌면서 이 인스턴스가
  // 하는 일 하나가 늘었다: pack 직전에 activation을 인덱스로 모아 dense를
  // 만든다. 그 뒤 코드(from_mat, 마스크 빌더, GEMM)는 dense만 보므로 무변경이다
  // — packed B의 행 순서가 곧 gather된 순서라 마스크 비트 ↔ 타일 대응이
  // 그대로 유지된다.
  //
  // 퇴화형(연속 밴드)은 gather를 건너뛰고 포인터만 넘긴다. 밴드 plan에서
  // 비트 동일 동작이 유지되는 것이 이 분기의 존재 이유다.

  static void validate_kindex(const KIndex& ki, int axis_len, const char* what) {
    if (!ki.indexed()) {
      if (ki.rows <= 0) {
        throw std::runtime_error(std::string("partial: ") + what + " rows not explicit");
      }
      if (ki.offset < 0 || ki.offset + ki.rows > axis_len) {
        throw std::runtime_error(std::string("partial: ") + what + " K band out of range");
      }
      return;
    }
    if (ki.idx.size() != (size_t)ki.row_off.back()) {
      throw std::runtime_error(std::string("partial: ") + what + " idx length != row_off.back()");
    }
    for (size_t e = 0; e + 1 < ki.row_off.size(); ++e) {
      const int kr = ki.row_off[e + 1] - ki.row_off[e];
      if (kr < 0) {
        throw std::runtime_error(std::string("partial: ") + what + " row_off not monotone");
      }
      // packed 저장이 타일 단위라 각 expert의 행 수가 K_STEP의 배수여야 한다.
      // plan 쪽 정렬 요구는 페어(%2)뿐이고, 여기까지 올림하는 것은 로더의 몫이다.
      if (kr % T::K_STEP != 0) {
        throw std::runtime_error(std::string("partial: ") + what +
                                 " per-expert rows must be a multiple of K_STEP");
      }
    }
    for (uint16_t v : ki.idx) {
      if ((int)v >= axis_len) {
        throw std::runtime_error(std::string("partial: ") + what + " index out of axis range");
      }
    }
    if (!ki.real_rows.empty()) {
      if (ki.real_rows.size() + 1 != ki.row_off.size()) {
        throw std::runtime_error(std::string("partial: ") + what + " real_rows size != E");
      }
      for (size_t e = 0; e < ki.real_rows.size(); ++e) {
        const int kr = ki.row_off[e + 1] - ki.row_off[e];
        // 패딩은 타일 경계까지의 올림이므로 [kr - K_STEP, kr] 안에 있어야 하고,
        // 페어 단위여야 한다 (마스크 tail이 페어 경계에서 끊기므로).
        if (ki.real_rows[e] <= 0 || ki.real_rows[e] > kr || ki.real_rows[e] % 2 != 0 ||
            kr - ki.real_rows[e] >= T::K_STEP) {
          throw std::runtime_error(std::string("partial: ") + what +
                                   " real_rows must be an even count within one tile of the padded rows");
        }
      }
    }
  }

  // **항상 dst에 쓴다.** prefill은 토큰마다 자기 슬롯에 담아야 하므로 퇴화형
  // 에서도 복사가 필요하다 (여기서 zero-copy를 하면 prefill이 조용히 빈
  // 버퍼를 pack한다 — 실제로 한 번 그렇게 틀렸다).
  void gather_rows(const KIndex& ki, int64_t e, const ggml_bf16_t* row, ggml_bf16_t* dst) const {
    const int kr = ki.k((int)e);
    if (!ki.indexed()) {
      std::memcpy(dst, row + ki.offset, (size_t)kr * sizeof(ggml_bf16_t));
      return;
    }
    const uint16_t* idx = ki.index_of((int)e);
    for (int r = 0; r < kr; r++) {
      dst[r] = row[idx[r]];
    }
  }

  // decode용: dense를 가리키는 포인터를 얻는다. 퇴화형이면 복사 없이
  // 원본을 가리키므로 밴드 경로의 비용이 그대로 0이다.
  const ggml_bf16_t* dense_rows(const KIndex& ki, int64_t e, const ggml_bf16_t* row,
                                ggml_bf16_t* scratch) const {
    if (!ki.indexed()) {
      return row + ki.offset;
    }
    gather_rows(ki, e, row, scratch);
    return scratch;
  }

  // ── prep 태스크용 소프트웨어 prefetch ────────────────────────────────
  //
  // prep의 스칼라 루프(gather, build_pair_mask, build_plan)는 명령 수가 아니라
  // cold miss의 지연으로 시간을 쓴다 — expert별 점수 테이블은 매 스텝 다른
  // expert라 L2에 없고, 루프가 한 라인씩 순서대로 찍으니 MLP가 1~2에 머문다.
  // HW prefetcher가 켜진 소켓은 순차 스캔을 앞서 가져와 대부분 숨기지만, 꺼진
  // 소켓에서는 miss가 하나씩 노출된다 (2026-08-26: prep 잔여 ~17 vs ~55 µs).
  // 태스크가 만질 입력(합쳐 ~16-24 KB, L2에 넉넉히 든다)을 시작 시 한꺼번에
  // 발행해 두면 SQ가 꽉 찬 채로 채워지고, 뒤따르는 루프는 L2에서 돈다.
  // export 전용: 행을 복사하면서 앞선 캐시라인을 SW 프리페치한다. HW 프리페치가
  // 꺼진 환경에서 partial export(8× 볼륨의 산개 memcpy)가 라인당 DRAM 지연으로
  // 직렬화되는 것을 막는다 — 프리페치가 MLP를 소프트웨어로 복원한다. n은 bf16 원소
  // 수(64B=32원소 배수 가정, export 행은 항상 그렇다). PF_AHEAD는 512B(8라인).
  static inline void copy_rows_pf(ggml_bf16_t* dst, const ggml_bf16_t* src, size_t n) {
    constexpr size_t AHEAD = 512;  // bytes
    const char* s = reinterpret_cast<const char*>(src);
    char* d = reinterpret_cast<char*>(dst);
    const size_t bytes = n * sizeof(ggml_bf16_t);
    size_t i = 0;
    for (; i + 64 <= bytes; i += 64) {
      if (i + AHEAD < bytes) _mm_prefetch(s + i + AHEAD, _MM_HINT_T0);
      avx512_copy_32xbf16(reinterpret_cast<__m512i*>(const_cast<char*>(s + i)),
                          reinterpret_cast<__m512i*>(d + i));
    }
    if (i < bytes) std::memcpy(d + i, s + i, bytes - i);
  }

  static inline void prefetch_bytes(const void* p, size_t bytes) {
    const char* c = reinterpret_cast<const char*>(p);
    const char* end = c + bytes;
    for (; c < end; c += 64) {
      _mm_prefetch(c, _MM_HINT_T1);
    }
  }

  // (proj, expert) 하나의 prep 입력: x — 밴드는 그 구간만, 인덱스형은 인덱스
  // 배열과 x 행 전체(gather가 x를 임의 순서로 찍으므로) — 와 점수 테이블
  // a(k×4B)/c(k/2×4B). sparse가 아니면 x만.
  void prefetch_prep_inputs(const KIndex& ki, int64_t e, const ggml_bf16_t* row, int axis_len, const float* a,
                            const float* c, bool sparse) const {
    const int kr = ki.k((int)e);
    if (ki.indexed()) {
      prefetch_bytes(ki.index_of((int)e), (size_t)kr * sizeof(uint16_t));
      prefetch_bytes(row, (size_t)axis_len * sizeof(ggml_bf16_t));
    } else {
      prefetch_bytes(row + ki.offset, (size_t)kr * sizeof(ggml_bf16_t));
    }
    if (sparse) {
      prefetch_bytes(a, (size_t)kr * sizeof(float));
      prefetch_bytes(c, (size_t)(kr / 2) * sizeof(float));
    }
  }

  // ── k2wl2 페어 마스크 (계약 ①) ────────────────────────────────────────
  // imp_j = sqrt(max(a[2j]·x0² + a[2j+1]·x1² + 2·c[j]·x0·x1, 0)) >= thr
  //
  // sqrt를 피하고 e >= thr² 로 비교한다 — thr >= 0이 보장되므로(threshold
  // 테이블이 음이 아닌 분포의 분위수) 판정이 동일하고, 밴드당 928회의
  // sqrt가 사라진다. thr==0이면 e>=0이 항상 참이라 전량 통과 — GPU 측
  // (imp>=0)과 같은 결론이다.
  //
  // 마스크는 **절대 k의 성질**이다: 페어 p = k/2가 워드 p/16의 비트 p%16이고,
  // 이 대응은 어떤 B packing에도 영향받지 않는다 (VNNI 짝짓기가 (짝수 k, 홀수
  // k)로 하드웨어가 고정하므로). 이전 판이 `T::K_STEP`으로 이 격자를 유도한
  // 것은 수치적으로 같았지만 커널 타입을 마스크 정의에 새게 했다 —
  // n-contig B 레이아웃 도입 검토에서 드러난 누수라 여기서 끊는다.
  //
  // k_real < k_rows이면 그 차이는 타일 경계까지 올린 **패딩 행**이다. 해당
  // 페어의 비트는 끈다: weight가 0이라 수치에는 영향이 없지만, thr == 0일 때
  // 비트가 서면 쓸모없는 B 로드가 생긴다.
  // 구현은 pair_mask.hpp (AVX-512, 16 페어 = zmm 하나). 연산 순서는 이 자리에
  // 있던 스칼라 루프가 -ffast-math 아래에서 컴파일된 결과와 동일하게 고정돼
  // 있어(그 파일은 fast-math를 끈다) 마스크가 비트 동일하다.
  static void build_pair_mask(const ggml_bf16_t* src, int k_rows, int k_real, const float* a_sq, const float* c,
                              float thr, uint16_t* out) {
    amx::build_pair_mask(src, k_rows, k_real, a_sq, c, thr, out);
  }

  // 예산 → s 벡터 (계약 ①). w는 [k] 라우터 가중으로, 정규화되지 않았을 수
  // 있으므로 여기서 한 번 더 나눈다 (멱등) — lam과 gbar의 스케일이 곧 마스크
  // 세기이므로 calib(twn = tw / tw.sum)과 어긋나면 안 된다.
  //
  // 재정규화가 s의 **평균**을 쓰므로 slot 하나만 따로 구할 수 없다: k개를 한
  // 번에 채운다 (k는 num_experts_per_tok, 보통 8).
  static void slot_sparsity(const float* w, int k, float p, float lam, float pmax, int renorm_it,
                            float* s) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
      sum += w[i];
    }
    const float inv = 1.0f / (sum > 1e-9f ? sum : 1e-9f);
    float gbar = 0.0f;
    for (int i = 0; i < k; i++) {
      gbar += w[i] * inv;
    }
    gbar /= (float)k;
    auto clip = [pmax](float v) { return v < 0.0f ? 0.0f : (v > pmax ? pmax : v); };
    for (int i = 0; i < k; i++) {
      s[i] = clip(p - lam * (w[i] * inv - gbar));
    }
    for (int it = 0; it < renorm_it; it++) {
      float mean = 0.0f;
      for (int i = 0; i < k; i++) {
        mean += s[i];
      }
      mean /= (float)k;
      if (mean < 1e-6f) {
        mean = 1e-6f;
      }
      const float scale = p / mean;
      for (int i = 0; i < k; i++) {
        s[i] = clip(s[i] * scale);
      }
    }
  }

  // 격자 조회. lrint는 기본 반올림 모드가 round-half-to-even이라 torch
  // .round()와 같은 결정을 낸다 (경계에서 갈리면 마스크가 조용히 달라진다).
  static float thr_of(const float* tab, int64_t e, int ng, float grid, float s) {
    long idx = std::lrint(s / grid);
    if (idx < 0) {
      idx = 0;
    }
    if (idx > (long)ng - 1) {
      idx = (long)ng - 1;
    }
    return tab[e * (int64_t)ng + idx];
  }

  // ── decode(qlen==1) 준비 단계: expert별 gather → pack → 마스크 → plan ──
  //
  // 포인터 carving(carve_*_decode: 누적 커서라 순서 의존이지만 산술만이라 싸다)과
  // 데이터 작업을 분리해, 후자를 subpool의 work-stealing job으로 돌린다. 슬롯 j
  // 하나가 태스크 하나다 — gate와 up을 한 태스크에 두는 이유는 dual_pack이 아닐
  // 때 up 마스크가 gate의 gather 결과를 읽기 때문이다 (태스크 간 의존 0).
  //
  // 2026-08-26 측정 (sglang cold_bench_results/a2a): 이 구간이 호출 스레드
  // 하나에서 순차로 돌 때 스텝당 ~75 µs(HW pf on) / ~280 µs(HW pf off) — 커널
  // 본체(~290 µs)와 같은 자릿수였고 그동안 나머지 워커는 spin이었다. expert별
  // 점수 테이블(wn_sq/pair_dot, fp32 [E·k])이 스텝마다 cold라 스칼라 루프가
  // 지연 바운드가 되기 때문이다. 워커 수만큼의 MLP를 얻는 것이 이 분해의 목적.
  //
  // 마스크·plan의 산술과 순서는 그대로다 (마스크 수치 계약 ①은 손대지 않는다).
  // 포인터 유효성은 호출 스레드(TP_MOE::forward_*_partial)가 이미 검증했다 —
  // worker 안의 throw는 uncaught → std::terminate이므로 여기서는 던지지 않는다.
  void prep_gateup_decode(int k, const int64_t* expert_ids, const ggml_bf16_t* x_full, const float* weights) {
    const int qlen = 1;
    const int H = config_.hidden_size;
    const auto& sp = config_.partial.sparsity;
    const bool sparse = sp.enabled && weights != nullptr;
    const float* g_a = (const float*)sp.gate_wn_sq;
    const float* g_c = (const float*)sp.gate_pair_dot;
    const float* u_a = (const float*)sp.up_wn_sq;
    const float* u_c = (const float*)sp.up_pair_dot;
    const float* g_t = (const float*)sp.thr_gate;
    const float* u_t = (const float*)sp.thr_up;

    float s_gate[kMaxTopK], s_up[kMaxTopK];
    if (sparse) {
      slot_sparsity(weights, k, sp.p_gate, sp.lam_gate, sp.pmax, sp.renorm_it, s_gate);
      slot_sparsity(weights, k, sp.p_up, sp.lam_up, sp.pmax, sp.renorm_it, s_up);
    }

    // dual_pack이면 gate와 up이 각자 gather/pack하므로 태스크를 둘로 나눈다
    // (병렬도 A → 2A). 아니면 up 마스크가 gate의 gather 결과를 읽으므로 한
    // 태스크에 둔다 — 태스크 간 의존은 어느 경우에도 없다.
    const bool dual = dual_pack_;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    pool->do_work_stealing_job(
        dual ? 2 * k : k, nullptr,
        [&](int t) {
          const int j = dual ? (t >> 1) : t;
          const bool do_up = dual && (t & 1);
          const int64_t e = expert_ids[j];
          if (config_.should_skip_expert(e)) {
            return;
          }
          // 점수 테이블도 weight와 같은 인덱스로 gather되어 들어오므로, expert
          // 오프셋은 행 기준 row_base다 (균일 밴드에서는 e*rows와 같다).
          const size_t g_off = (size_t)config_.gate_row_base(e);
          const size_t u_off = (size_t)config_.up_row_base(e);
          const float* ga = g_a + g_off;
          const float* gc = g_c + g_off / 2;
          const float* ua = u_a + u_off;
          const float* uc = u_c + u_off / 2;

          if (do_up) {
            prefetch_prep_inputs(config_.partial.up, e, x_full, H, ua, uc, sparse);
            const ggml_bf16_t* xu = dense_rows(config_.partial.up, e, x_full, m_local_input_up_ptr_[e]);
            up_ba_[e]->from_mat(qlen, const_cast<ggml_bf16_t*>(xu), 0, 1);
            if (!sparse) {
              return;
            }
            build_pair_mask(xu, config_.up_k(e), config_.up_k_real(e), ua, uc,
                            thr_of(u_t, e, sp.ng, sp.grid, s_up[j]), m_pair_mask_up_[e].data());
            if constexpr (kHasPlanEncoder) {
              T::build_plan(m_pair_mask_up_[e].data(), xu, config_.up_k(e), plan_arena_for((int)e, 1),
                            &m_plan_up_[e]);
            }
            return;
          }

          prefetch_prep_inputs(config_.partial.gate, e, x_full, H, ga, gc, sparse);
          if (!dual && sparse) {
            // up은 gate의 행을 공유한다 — 테이블만 따로 든다.
            prefetch_bytes(ua, (size_t)config_.up_k(e) * sizeof(float));
            prefetch_bytes(uc, (size_t)(config_.up_k(e) / 2) * sizeof(float));
          }
          // gather(인덱스형) + pack. 퇴화형(밴드)은 원본을 가리켜 복사 0.
          const ggml_bf16_t* xg = dense_rows(config_.partial.gate, e, x_full, m_local_input_ptr_[e]);
          gate_up_ba_[e]->from_mat(qlen, const_cast<ggml_bf16_t*>(xg), 0, 1);
          if (!sparse) {
            return;
          }
          build_pair_mask(xg, config_.gate_k(e), config_.gate_k_real(e), ga, gc,
                          thr_of(g_t, e, sp.ng, sp.grid, s_gate[j]), m_pair_mask_gate_[e].data());
          if constexpr (kHasPlanEncoder) {
            // 마스크(정책, 절대 k)를 커널의 자료구조로 인코딩한다 — expert당
            // 1회, 결과는 read-only로 전 워커가 공유한다.
            T::build_plan(m_pair_mask_gate_[e].data(), xg, config_.gate_k(e), plan_arena_for((int)e, 0),
                          &m_plan_gate_[e]);
          }
          if (!dual) {
            build_pair_mask(xg, config_.up_k(e), config_.up_k_real(e), ua, uc,
                            thr_of(u_t, e, sp.ng, sp.grid, s_up[j]), m_pair_mask_up_[e].data());
            if constexpr (kHasPlanEncoder) {
              T::build_plan(m_pair_mask_up_[e].data(), xg, config_.up_k(e), plan_arena_for((int)e, 1),
                            &m_plan_up_[e]);
            }
          }
        },
        nullptr);
    m_sparse_active_ = sparse;
  }

  void prep_down_decode(int k, const int64_t* expert_ids, const ggml_bf16_t* act, const float* weights) {
    const int qlen = 1;
    const auto& sp = config_.partial.sparsity;
    const bool sparse = sp.enabled && weights != nullptr;
    const float* d_a = (const float*)sp.down_wn_sq;
    const float* d_c = (const float*)sp.down_pair_dot;
    const float* d_t = (const float*)sp.thr_down;
    const int act_row = config_.n_total();  // act의 slot 행 stride = down K 축 길이

    float s_down[kMaxTopK];
    if (sparse) {
      slot_sparsity(weights, k, sp.p_down, sp.lam_down, sp.pmax, sp.renorm_it, s_down);
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);
    pool->do_work_stealing_job(
        k, nullptr,
        [&](int j) {
          const int64_t e = expert_ids[j];
          if (config_.should_skip_expert(e)) {
            return;
          }
          const size_t off = (size_t)config_.down_row_base(e);
          const float* da = d_a + off;
          const float* dc = d_c + off / 2;
          const ggml_bf16_t* row = act + (size_t)j * act_row;
          prefetch_prep_inputs(config_.partial.down, e, row, act_row, da, dc, sparse);
          const ggml_bf16_t* src = dense_rows(config_.partial.down, e, row, m_local_input_ptr_[e]);
          down_ba_[e]->from_mat(qlen, const_cast<ggml_bf16_t*>(src), 0, 1);
          if (!sparse) {
            return;
          }
          build_pair_mask(src, config_.down_k(e), config_.down_k_real(e), da, dc,
                          thr_of(d_t, e, sp.ng, sp.grid, s_down[j]), m_pair_mask_down_[e].data());
          if constexpr (kHasPlanEncoder) {
            T::build_plan(m_pair_mask_down_[e].data(), src, config_.down_k(e), plan_arena_for((int)e, 2),
                          &m_plan_down_[e]);
          }
        },
        nullptr);
    m_sparse_active_ = sparse;
  }

  // 라우팅 준비: forward_decode 1단계와 동일한 멤버 상태(m_expert_id_map_,
  // m_local_num_, m_local_{gate,up}_output_ptr_)를 만든다. 반환 = 활성 expert 수.
  int prepare_gateup_routing(int k, const int64_t* expert_ids) {
    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) {
        continue;
      }
      m_local_pos_[0][i] = 0;  // export가 pos 기반이라 decode도 채운다 (퇴화값)
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_num_[expert_ids[i]] = 1;
      activated_expert++;
    }
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      offset += 1;
    }
    return activated_expert;
  }

  // 풀 carving (decode): 활성 expert에만 gate_bc/up_bc/gate_up_ba 구간과 gather
  // 목적지를 배정한다 (forward_decode 2단계의 gateup 부분집합). 포인터 산술만
  // — 데이터(gather/pack/마스크)는 prep_gateup_decode가 워커에서 채운다.
  void carve_gateup_decode(int activated_expert, int qlen) {
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    // gather 목적지 커서 (인덱스형에서만 쓰인다). decode의 gateup phase에서
    // m_local_input_에 다른 소비자가 없어 prefill 버퍼를 그대로 재사용한다 —
    // 신규 할당 0.
    ggml_bf16_t* gather_cursor = m_local_input_;
    ggml_bf16_t* gather_cursor_up = m_local_input_up_;
    void* up_ba_pool_ptr = up_ba_pool_;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr +
                                 align64(buffer_c_required_size(max_m, config_.intermediate_size)));

      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      up_bc_pool_ptr =
          (void*)((uintptr_t)up_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));

      gate_up_ba_[expert_idx]->max_m = max_m;
      gate_up_ba_[expert_idx]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr +
                                    align64(buffer_a_required_size(max_m, config_.gate_k(expert_idx))));
      m_local_input_ptr_[expert_idx] = gather_cursor;
      gather_cursor += config_.gate_k(expert_idx);

      if (dual_pack_) {
        up_ba_[expert_idx]->max_m = max_m;
        up_ba_[expert_idx]->set_data(up_ba_pool_ptr);
        up_ba_pool_ptr = (void*)((uintptr_t)up_ba_pool_ptr +
                                 align64(buffer_a_required_size(max_m, config_.up_k(expert_idx))));
        m_local_input_up_ptr_[expert_idx] = gather_cursor_up;
        gather_cursor_up += config_.up_k(expert_idx);
      }
    }
  }

  // gate/up GEMM 스테이지: forward_decode 4단계와 동일한 work-stealing 루프.
  // CRTP seam(do_gate_up_gemm)을 그대로 지나므로 dtype/sparse 클래스 교체에
  // 중립이고, GEMM의 K 치수는 config_.gateup_k()가 공급한다.
  void run_gateup_stage(int activated_expert, int qlen) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);
  }

  // ── prefill(qlen>1) 전용: kt forward_prefill의 1~3단계 동형 ─────────────
  // 라우팅: 토큰을 expert 그룹으로 배치 (m_local_pos_[m][j] = 그룹 내 행).
  int prepare_prefill_routing(int qlen, int k, const int64_t* expert_ids) {
    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }
    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }
    return activated_expert;
  }

  // carve (prefill, gateup 부분집합): 전 expert의 m_local ptr을 그룹 크기
  // 누적으로 배정하고, 활성 expert에만 버퍼 구간을 준다. m_local_input의
  // 행 폭은 gateup_k() — gather가 밴드만 담기 때문.
  void carve_gateup_prefill() {
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* up_ba_pool_ptr = up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    size_t offset = 0, input_cursor = 0, input_cursor_up = 0;
    size_t used_pool_m = 0, used_pool_bytes_a = 0, used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      // 행 폭이 expert마다 다르므로 offset×폭이 아니라 누적 커서다.
      m_local_input_ptr_[i] = m_local_input_ + input_cursor;
      input_cursor += (size_t)m_local_num_[i] * config_.gate_k(i);
      if (dual_pack_) {
        m_local_input_up_ptr_[i] = m_local_input_up_ + input_cursor_up;
        input_cursor_up += (size_t)m_local_num_[i] * config_.up_k(i);
      }
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      offset += m_local_num_[i];
      if (m_local_num_[i] == 0) {
        continue;
      }
      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.gate_k(i)));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);
      if (dual_pack_) {
        up_ba_[i]->max_m = max_m;
        up_ba_[i]->set_data(up_ba_pool_ptr);
        up_ba_pool_ptr = (void*)((uintptr_t)up_ba_pool_ptr +
                                 align64(buffer_a_required_size(max_m, config_.up_k(i))));
      }

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      used_pool_m += max_m;
      used_pool_bytes_a += ba_size;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
    }
    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_a <= gate_up_ba_pool_bytes_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
  }

  // gather + pack (prefill): gather의 memcpy가 full-width x에서 밴드만 담아
  // strided pack을 불필요하게 만든다 — from_mat은 연속 [m_e, gateup_k]를 읽음.
  void gather_and_pack_gateup_prefill(int qlen, int k, const int64_t* expert_ids, const ggml_bf16_t* x,
                                      int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        const int64_t e = expert_ids[i * k + j];
        if (config_.should_skip_expert(e)) {
          continue;
        }
        const ggml_bf16_t* row = x + (size_t)i * config_.hidden_size;
        const size_t pos = (size_t)m_local_pos_[i][j];
        gather_rows(config_.partial.gate, e, row,
                    m_local_input_ptr_[e] + pos * (size_t)config_.gate_k(e));
        if (dual_pack_) {
          gather_rows(config_.partial.up, e, row,
                      m_local_input_up_ptr_[e] + pos * (size_t)config_.up_k(e));
        }
      }
    });
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
      if (dual_pack_) {
        up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_up_ptr_[expert_idx], 0, 1);
      }
    });
  }

  // export: 내부 expert-뭉치 배치(m_local_*)를 계약 slot 배치로 복사.
  // kt가 act를 적용하던 자리를 대체하는 단계다. 이 노드는 full-width out의
  // 자기 shard 열 [n_off, n_off+I)만 쓴다.
  void export_gateup_partial(int qlen, int k, const int64_t* expert_ids, ggml_bf16_t* out) {
    const size_t inter = (size_t)config_.intermediate_size;
    const size_t n_total = (size_t)config_.n_total();
    const size_t n_off = (size_t)config_.partial.gateup_n.offset;  // TP ctor가 Plan/균등에서 기록
    if (n_off + inter > n_total) {
      throw std::runtime_error("forward_gateup_partial: node shard exceeds n_total");
    }
    const size_t row_elems = 2 * n_total;
    // 토큰 블록 단위로 subpool 전체가 복사한다. 단일 스레드 루프였을 때 prefill
    // M=2688에서 44 MB에 6.4 ms(down은 88 MB에 14.5 ms — 층당 21 ms, 전체의 1/3)가
    // 여기 있었다 (2026-08-27 KT_PARTIAL_TIMING 실측). decode(qlen==1)는 블록 하나.
    export_rows_parallel(qlen, [&](int m) {
      for (int j = 0; j < k; j++) {
        int64_t e = expert_ids[m * k + j];
        ggml_bf16_t* dst = out + ((size_t)m * k + j) * row_elems;
        if (config_.should_skip_expert(e)) {
          // 이 expert는 다른 소비자(GPU cold hybrid)가 계산한다 — 우리 열 구간을
          // 0으로 채워 rejoin 합에서 무해하게 한다 (staging은 재사용 버퍼라 stale).
          std::memset(dst + n_off, 0, inter * sizeof(ggml_bf16_t));
          std::memset(dst + n_total + n_off, 0, inter * sizeof(ggml_bf16_t));
          continue;
        }
        const size_t pos = (size_t)m_local_pos_[m][j];  // expert 그룹 내 이 토큰의 행
        copy_rows_pf(dst + n_off, m_local_gate_output_ptr_[e] + pos * inter, inter);
        copy_rows_pf(dst + n_total + n_off, m_local_up_output_ptr_[e] + pos * inter, inter);
      }
    });
  }

  // export 공통: 토큰 [0, qlen)을 블록으로 나눠 subpool에 뿌린다. 블록 하나면
  // (decode) 호출 스레드가 직접 돈다 — work-stealing 진입 고정비를 피한다.
  template <typename Fn>
  void export_rows_parallel(int qlen, Fn&& per_token) {
    constexpr int kBlock = 16;
    const int nblocks = (qlen + kBlock - 1) / kBlock;
    if (nblocks <= 1) {
      for (int m = 0; m < qlen; m++) per_token(m);
      return;
    }
    auto pool = config_.pool->get_subpool(tp_part_idx);
    pool->do_work_stealing_job(nblocks, [&](int b) {
      const int m_end = std::min(qlen, (b + 1) * kBlock);
      for (int m = b * kBlock; m < m_end; m++) per_token(m);
    });
  }

 public:
  // ── Prism down partial (계약 ②, K2) — gateup과 동형의 절단 ────────────
  // forward_decode의 down 구간(입력 pack → down GEMM)만 수행하고 partial을
  // export한다. router 가중 / expert 합산 없음 — GPU rejoin#2 소관.
  //
  //   act: bf16 [qlen × k × n_total(inter_FULL)] — GPU rejoin#1의 act 결과가
  //        D2H로 내려온 것. slot j의 행에서 down K 밴드
  //        [down.offset, +down.rows)만 읽는다.
  //   out: bf16 [qlen × k × hidden_FULL]. down은 N(hidden) 축 노드 분할이라
  //        이 노드는 자기 shard 열 [down_n.offset, +down_n.rows)만 쓴다.
  //        overwrite 의미론 — skip slot 미기록(undefined).
  void forward_down_partial(int qlen, int k, const int64_t* expert_ids, const void* act, void* out,
                            const float* weights = nullptr) {
    if (qlen == 1) {
      int activated_expert = prepare_down_routing(k, expert_ids);
      if (activated_expert == 0) {
        return;
      }
      carve_down_decode(k, expert_ids, qlen);
      prep_down_decode(k, expert_ids, (const ggml_bf16_t*)act, weights);
      run_down_stage(activated_expert, qlen);
      m_sparse_active_ = false;
      export_down_partial(qlen, k, expert_ids, (ggml_bf16_t*)out);
      return;
    }
    m_sparse_active_ = false;  // prefill-dense
    PartialTimer tm("down", tp_part_idx, qlen);
    int activated_expert = prepare_prefill_routing(qlen, k, expert_ids);
    if (activated_expert == 0) {
      return;
    }
    tm.lap("prep");
    carve_down_prefill();
    tm.lap("carve");
    gather_and_pack_down_prefill(qlen, k, expert_ids, (const ggml_bf16_t*)act, activated_expert);
    tm.lap("pack");
    run_down_stage(activated_expert, qlen);
    tm.lap("gemm");
    export_down_partial(qlen, k, expert_ids, (ggml_bf16_t*)out);
    tm.lap("export");
  }

 private:
  // 라우팅 준비 (down): m_local_down_output_ptr_ 배정. stride = down_n().
  int prepare_down_routing(int k, const int64_t* expert_ids) {
    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) {
        continue;
      }
      m_local_pos_[0][i] = 0;  // export가 pos 기반 (decode 퇴화값)
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_num_[expert_ids[i]] = 1;
      activated_expert++;
    }
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.down_n();
      offset += 1;
    }
    return activated_expert;
  }

  // 풀 carving (decode, down): slot j의 expert에 down_ba/down_bc 구간과 gather
  // 목적지를 배정한다. (qlen==1이라 slot마다 expert가 유일 — slot 순회가 곧
  // expert 순회) 데이터는 prep_down_decode가 워커에서 채운다.
  void carve_down_decode(int k, const int64_t* expert_ids, int qlen) {
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    ggml_bf16_t* gather_cursor = m_local_input_;  // 인덱스형 gather 목적지
    for (int j = 0; j < k; j++) {
      int64_t e = expert_ids[j];
      if (config_.should_skip_expert(e)) {
        continue;
      }
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      down_ba_[e]->max_m = max_m;
      down_ba_[e]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr +
                                 align64(buffer_a_required_size(max_m, config_.down_k(e))));

      down_bc_[e]->max_m = max_m;
      down_bc_[e]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr =
          (void*)((uintptr_t)down_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.down_n())));

      m_local_input_ptr_[e] = gather_cursor;
      gather_cursor += config_.down_k(e);
    }
  }

  // down GEMM 스테이지: CRTP seam(do_down_gemm) 경유, N = down_n().
  void run_down_stage(int activated_expert, int qlen) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int nth = T::recommended_nth(config_.down_n());
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);
  }

  // carve (prefill, down 부분집합): m_local_input(act gather용, 행 폭 down_k)과
  // m_local_down(행 폭 down_n) ptr 배정 + down_ba/down_bc 구간.
  void carve_down_prefill() {
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    size_t offset = 0, input_cursor = 0;
    size_t used_pool_m = 0, used_pool_bytes_ba_down = 0, used_pool_bytes_bc_down = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + input_cursor;
      input_cursor += (size_t)m_local_num_[i] * config_.down_k(i);
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.down_n();
      offset += m_local_num_[i];
      if (m_local_num_[i] == 0) {
        continue;
      }
      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.down_k(i)));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_size);

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      size_t bc_size = align64(buffer_c_required_size(max_m, config_.down_n()));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_size);

      used_pool_m += max_m;
      used_pool_bytes_ba_down += ba_size;
      used_pool_bytes_bc_down += bc_size;
    }
    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);
  }

  // gather + pack (prefill, down): act의 (m, slot) 행에서 down K 밴드만
  // expert 그룹으로 모은다 — m_local_input을 act gather 버퍼로 재사용
  // (gateup gather와 down gather는 시간상 서로 다른 phase라 충돌 없음).
  void gather_and_pack_down_prefill(int qlen, int k, const int64_t* expert_ids, const ggml_bf16_t* act,
                                    int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };
    const size_t act_row = (size_t)config_.n_total();
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        const int64_t e = expert_ids[i * k + j];
        if (config_.should_skip_expert(e)) {
          continue;
        }
        const size_t band = (size_t)config_.down_k(e);
        gather_rows(config_.partial.down, e, act + ((size_t)i * k + j) * act_row,
                    m_local_input_ptr_[e] + (size_t)m_local_pos_[i][j] * band);
      }
    });
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });
  }

  // export: 자기 hidden shard 열 [down_n.offset, +down_n.rows)에 직접 쓰기.
  void export_down_partial(int qlen, int k, const int64_t* expert_ids, ggml_bf16_t* out) {
    const size_t rows = (size_t)config_.down_n();
    const size_t h_total = (size_t)config_.hidden_size;
    const size_t n_off = (size_t)config_.partial.down_n.offset;
    if (n_off + rows > h_total) {
      throw std::runtime_error("forward_down_partial: node shard exceeds hidden_size");
    }
    export_rows_parallel(qlen, [&](int m) {
      for (int j = 0; j < k; j++) {
        int64_t e = expert_ids[m * k + j];
        ggml_bf16_t* dst = out + ((size_t)m * k + j) * h_total;
        if (config_.should_skip_expert(e)) {
          std::memset(dst + n_off, 0, rows * sizeof(ggml_bf16_t));  // GPU cold hybrid 몫
          continue;
        }
        const size_t pos = (size_t)m_local_pos_[m][j];
        copy_rows_pf(dst + n_off, m_local_down_output_ptr_[e] + pos * rows, rows);
      }
    });
  }

 public:
  template <typename... Args>
  void load_weights(Args&&... args) {
    derived()->load_weights(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void write_weights_to_buffer(Args&&... args) const {
    derived_const()->write_weights_to_buffer(std::forward<Args>(args)...);
  }

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    PartialTimer tm("native", tp_part_idx, qlen);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;
#endif

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
#ifdef FORWARD_TIME_PROFILE
        max_local_num = std::max(max_local_num, m_local_num_[i]);
#endif
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_a = 0, used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0, used_pool_bytes_ba_down = 0,
           used_pool_bytes_bc_down = 0;

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) {
        continue;
      }

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_a += ba_size;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }

    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_a <= gate_up_ba_pool_bytes_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);

    tm.lap("prep");
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });

    tm.lap("cpy_input");
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      cpy_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

    tm.lap("packA");
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

    tm.lap("gateup_gemm");
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      up_gate_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    apply_activation(activated_expert, nth, qlen);

    tm.lap("act");
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      act_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    tm.lap("packA_down");
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    tm.lap("down_gemm");
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              if (config_.should_skip_expert(expert_ids[i * k + j])) {
                continue;
              }
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32((__m512i*)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
                                                   m_local_pos_[i][j] * config_.hidden_size + e),
                                        &down_output0, &down_output1);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            auto f32out = (__m512*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
          }
        },
        nullptr);

    tm.lap("weight_sum");
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us, max_local_num: "
        "%d, qlen: %d\n",
        tp_part_idx, activated_expert, prepare_time, cpy_input_time, q_input_time, up_gate_time, act_time, q_down_time,
        down_time, weight_time, forward_total_time, max_local_num, qlen);
#endif
  }

  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    int qlen = 1;
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    long q_input_time = 0, up_gate_time = 0, act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
#endif

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) {
        continue;
      }
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_pos_[0][i] = 0;
      m_local_num_[expert_ids[i]] = qlen;
      activated_expert++;
    }

    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.hidden_size;
      offset += qlen;
    }

    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0, used_pool_bytes_ba_down = 0,
           used_pool_bytes_bc_down = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[expert_idx]->max_m = max_m;
      down_ba_[expert_idx]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[expert_idx]->max_m = max_m;
      down_bc_[expert_idx]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }
    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);

    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[expert_idx]->max_m = max_m;
      gate_up_ba_[expert_idx]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);
      gate_up_ba_[expert_idx]->from_mat(qlen, (ggml_bf16_t*)input, 0, 1);
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      up_gate_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    apply_activation(activated_expert, nth, qlen);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      act_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    for (int e = 0; e < config_.hidden_size; e += 32) {
      __m512 x0 = _mm512_setzero_ps();
      __m512 x1 = _mm512_setzero_ps();
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[j])) {
          continue;
        }
        __m512 weight = _mm512_set1_ps(weights[j]);
        __m512 down_output0, down_output1;
        avx512_32xbf16_to_32xfp32(
            (__m512i*)(m_local_down_output_ptr_[expert_ids[j]] + m_local_pos_[0][j] * config_.hidden_size + e),
            &down_output0, &down_output1);
        x0 = _mm512_fmadd_ps(down_output0, weight, x0);
        x1 = _mm512_fmadd_ps(down_output1, weight, x1);
      }
      auto f32out = (__m512*)((float*)output + e);
      f32out[0] = x0;
      f32out[1] = x1;
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us\n",
        tp_part_idx, activated_expert, q_input_time, up_gate_time, act_time, q_down_time, down_time, weight_time,
        forward_total_time);
#endif
  }

 protected:
  Derived* derived() { return static_cast<Derived*>(this); }
  const Derived* derived_const() const { return static_cast<const Derived*>(this); }

  // ============================================================================
  // Derived class initialization hook
  // Called after base class init() completes, allows derived classes to perform
  // their own initialization that depends on base class being fully initialized
  // ============================================================================
  void derived_init() {
    // Default implementation does nothing - derived classes can override
  }

  // ============================================================================
  // Virtual points for buffer creation and size calculation
  // Default implementations use group_size (for KGroup quantization like K2)
  // Derived classes (like moe.hpp) can override to not use group_size
  // ============================================================================

  size_t buffer_a_required_size(size_t m, size_t k) const { return derived_const()->buffer_a_required_size_impl(m, k); }
  size_t buffer_b_required_size(size_t n, size_t k) const { return derived_const()->buffer_b_required_size_impl(n, k); }
  size_t buffer_c_required_size(size_t m, size_t n) const { return derived_const()->buffer_c_required_size_impl(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a(size_t m, size_t k, void* data) const {
    return derived_const()->make_buffer_a_impl(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b(size_t n, size_t k, void* data) const {
    return derived_const()->make_buffer_b_impl(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c(size_t m, size_t n, void* data) const {
    return derived_const()->make_buffer_c_impl(m, n, data);
  }

  void apply_activation(int activated_expert, int nth, int qlen) {
    apply_activation_to(activated_expert, nth, qlen, m_local_gate_output_ptr_);
  }

  void apply_activation_to(int activated_expert, int nth, int qlen,
                           const std::vector<ggml_bf16_t*>& destination_ptrs) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto fn = [this, nth, &destination_ptrs](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* destination_ptr = &destination_ptrs[expert_idx][i * config_.intermediate_size];
        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0, gate_val1, up_val0, up_val1;
          avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
          avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
          __m512 result0 = amx::act_fn(gate_val0, up_val0, config_.swiglu_limit, config_.swiglu_alpha);
          __m512 result1 = amx::act_fn(gate_val1, up_val1, config_.swiglu_limit, config_.swiglu_alpha);
          avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(destination_ptr + j));
        }
      }
    };

    if (activated_expert == 0) {
      return;
    }

    if (qlen < 10) {
      for (int task_id = 0; task_id < nth * activated_expert; task_id++) {
        fn(task_id);
      }
    } else {
      pool->do_work_stealing_job(nth * activated_expert, nullptr, fn, nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_MOE_BASE derived classes
// ============================================================================

template <class T, class Derived>
class TP_MOE<AMX_MOE_BASE<T, Derived>> : public TP_MOE_Common<AMX_MOE_BASE<T, Derived>> {
 public:
  using TP_MOE_Common<AMX_MOE_BASE<T, Derived>>::TP_MOE_Common;

  // Default load_weights implementation - can be overridden by derived TP_MOE classes
  void load_weights() override { throw std::runtime_error("Not Implemented"); }

  void write_weight_scale_to_buffer(int gpu_tp_count, int gpu_experts_num,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    throw std::runtime_error("Not Implemented");
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 32) {
          __m512 x0, x1;
          avx512_32xbf16_to_32xfp32((__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e), &x0, &x1);
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), x0);
          *((__m512*)(merge_to + e + 16)) = _mm512_add_ps(*((__m512*)(merge_to + e + 16)), x1);
        }
      }
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), *((__m512*)(merge_from + e)));
        }
      }
      for (int e = 0; e < config.hidden_size; e += 32) {
        __m512 x0 = *(__m512*)(merge_to + e);
        __m512 x1 = *(__m512*)(merge_to + e + 16);
        avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
      }
    };

    auto pool = config.pool;

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, merge_fn);
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

#endif  // CPUINFER_OPERATOR_AMX_MOE_BASE_H
