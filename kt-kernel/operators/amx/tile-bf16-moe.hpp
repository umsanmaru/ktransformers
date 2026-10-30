/**
 * @file tile-bf16-moe.hpp
 * @brief bf16 MoE operator — Prism cold 티어의 **k=1 타일** 전용 (per-k sparse decode).
 *
 * `AMX_BF16_MOE_TP`(bf16-moe.hpp)와 같은 입력(bf16 행우선 weight), 같은 수치지만 저장과
 * 커널이 다르다: `GemmKernelTileK1BF16`(la/tile_k1_bf16.hpp)의 k1 타일 레이아웃 +
 * 프리페치 커서 GEMV, 그리고 Prism partial의 Seam B/C(per-k 마스크 → plan → gemv_slab).
 *
 * **왜 operator를 따로 두는가.** k2 형제(`GemmKernelTileK2BF16`)는 NC pack을 그대로 써서
 * `AMX_BF16_MOE_TP`에 그냥 꽂혔다 — 상속받은 AMX prefill이 같은 바이트를 읽기 때문이다.
 * k1은 pack이 다르므로 그 경로에 꽂으면 **prefill이 같은 바이트를 다른 순열로 읽어 조용히
 * 쓰레기를 낸다** (예외도 안 난다). 그래서 이 operator는 `mat_mul`을 아예 갖지 않고
 * fp8/mxfp4 타일 형제와 같은 구조로 돈다:
 *
 *   m == 1 (decode) — sparse면 moe_base가 만든 plan(m_plan_*), dense면 즉석 dense plan
 *   1 < m (소배치)   — 토큰 루프 (dense plan per token)
 *
 * 대가는 prefill이 M배라는 것이다. fp8 타일은 그 자리를 GPU(제자리 slab 읽기)가 메우는데
 * bf16 dense 레인에는 그 배관이 아직 없다 — 그래서 이 커널은 **decode 전용**으로 쓰고,
 * 긴 prompt가 필요하면 k2 커널(`kt_tile_k2_bf16`)로 돌리는 것이 지금의 선택이다.
 */
#ifndef CPUINFER_OPERATOR_TILE_BF16_MOE_H
#define CPUINFER_OPERATOR_TILE_BF16_MOE_H

#include <cstring>

#include "bf16-moe.hpp"
#include "la/tile_k1_bf16.hpp"

template <class T = amx::GemmKernelTileK1BF16>
class TILE_BF16_MOE_TP : public AMX_MOE_BASE<T, TILE_BF16_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, TILE_BF16_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::dual_pack_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::m_plan_down_;
  using Base::m_plan_gate_;
  using Base::m_plan_up_;
  using Base::m_sparse_active_;
  using Base::tp_part_idx;
  using Base::up_ba_;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using kernel_t = T;
  using typename Base::input_t;
  using typename Base::output_t;

  TILE_BF16_MOE_TP() = default;
  TILE_BF16_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    // 타일 super = 256 n: 노드 N shard가 256 배수가 아니면 pack이 부분 super를 만들 수 없다.
    if (config_.intermediate_size % T::N_BLOCK || config_.down_n() % T::N_BLOCK)
      throw std::runtime_error("TILE_BF16_MOE_TP: node N shards (inter " +
                               std::to_string(config_.intermediate_size) + ", down_n " +
                               std::to_string(config_.down_n()) + ") must be multiples of " +
                               std::to_string(T::N_BLOCK));
    printf("Creating TILE_BF16_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~TILE_BF16_MOE_TP() = default;

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k); }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // m == 1: plan 경로 (sparse는 moe_base가 만든 plan, dense는 즉석). m > 1: 토큰 루프 (dense).
  void run_gemv(int m, int n, int k, typename T::BufferA* ba, typename T::BufferB* bb,
                typename T::BufferC* bc, int ith, int nth, const amx::ActivationPlan* sparse_plan) {
    auto [n0, n1] = T::split_range_n(n, ith, nth);
    if (n0 >= n1) return;
    amx::ActivationPlan tmp;
    if (m == 1) {
      const amx::ActivationPlan* pl = sparse_plan ? sparse_plan : T::dense_plan(ba->a, k, &tmp);
      T::gemv_slab(1, n, k, n0, n1, ba, bb, bc, pl);
      return;
    }
    for (int t = 0; t < m; ++t) {
      T::dense_plan(ba->a + (size_t)t * k, k, &tmp);
      T::gemv_row(m, t, n, k, n0, n1, bb, bc, &tmp);
    }
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    const int m = m_local_num_[expert_idx];
    auto& ba = (do_up && dual_pack_) ? up_ba_[expert_idx] : gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    const int gu_k = do_up ? config_.up_k(expert_idx) : config_.gate_k(expert_idx);
    const amx::ActivationPlan* pl =
        m_sparse_active_ ? &(do_up ? m_plan_up_ : m_plan_gate_)[expert_idx] : nullptr;
    run_gemv(m, config_.intermediate_size, gu_k, ba.get(), bb.get(), bc.get(), ith, nth, pl);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    const int m = m_local_num_[expert_idx];
    const amx::ActivationPlan* pl = m_sparse_active_ ? &m_plan_down_[expert_idx] : nullptr;
    run_gemv(m, config_.down_n(), config_.down_k(expert_idx), down_ba_[expert_idx].get(),
             down_bb_[expert_idx].get(), down_bc_[expert_idx].get(), ith, nth, pl);
  }

  // per-TP pack: 행우선 bf16 → k1 타일 순열. 배율이 없어 fp8 판보다 짧다.
  void load_weights() {
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (config_.gate_proj == nullptr) throw std::runtime_error("TILE_BF16_MOE_TP: needs bf16 weights");
    const size_t I = config_.intermediate_size, DN = config_.down_n();

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, I](int task_id) {
          const uint64_t e = task_id / nth;
          const uint64_t le = expert_map(physical_to_logical_map, e);
          const int ith = task_id % nth;
          gate_bb_[e]->from_mat((const ggml_bf16_t*)config_.gate_proj + I * (size_t)config_.gate_row_base(le),
                                ith, nth);
          up_bb_[e]->from_mat((const ggml_bf16_t*)config_.up_proj + I * (size_t)config_.up_row_base(le), ith,
                              nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.down_n());
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, DN](int task_id) {
          const uint64_t e = task_id / nth;
          const uint64_t le = expert_map(physical_to_logical_map, e);
          const int ith = task_id % nth;
          down_bb_[e]->from_mat((const ggml_bf16_t*)config_.down_proj + DN * (size_t)config_.down_row_base(le),
                                ith, nth);
        },
        nullptr);
  }

  // GPU writeback(kt 하이브리드)은 partial 인스턴스에서 쓰지 않는다.
  void write_weights_to_buffer(int, int, int, const GeneralMOEConfig&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&) const {
    throw std::runtime_error("TILE_BF16_MOE_TP: write_weights_to_buffer is not supported");
  }
};

// bf16 타일 weight staging (Prism partial 전용): full 텐서 → per-TP N-shard 사본 → 각 TP가 pack.
// `TP_MOE<AMX_BF16_MOE_TP<K>>::load_weights`의 partial 분기와 같은 복사이고, 비-partial(열
// 슬라이스) 분기는 이 operator가 지원하지 않으므로 즉사시킨다.
template <class TPM>
inline void bf16_tile_tp_load_weights(TPM& self) {
  auto& config = self.config;
  auto& tps = self.tps;
  auto pool = config.pool;
  const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

  if (config.gate_proj == nullptr) throw std::runtime_error("TILE_BF16_MOE_TP: needs bf16 weights");
  if (!config.is_partial())
    throw std::runtime_error(
        "TILE_BF16_MOE_TP: Prism partial only (the non-partial bf16 path is AMX_BF16_MOE_TP)");

  pool->dispense_backend()->do_numa_job([&](int i) {
    auto& tpc = tps[i]->config_;
    const size_t I = tpc.intermediate_size, DN = tpc.down_n();
    const size_t g_tot = I * (size_t)tpc.gate_row_base(tpc.expert_num);
    const size_t u_tot = I * (size_t)tpc.up_row_base(tpc.expert_num);
    const size_t d_tot = DN * (size_t)tpc.down_row_base(tpc.expert_num);
    tpc.gate_proj = new ggml_bf16_t[g_tot];
    tpc.up_proj = new ggml_bf16_t[u_tot];
    tpc.down_proj = new ggml_bf16_t[d_tot];
    const size_t gu_n_off = (size_t)tpc.partial.gateup_n.offset;
    const size_t dn_n_off = (size_t)tpc.partial.down_n.offset;
    const size_t I_full = config.intermediate_size, H_full = config.hidden_size;
    pool->get_subpool(i)->do_work_stealing_job(
        tpc.expert_num, nullptr,
        [&](int expert_id_) {
          const size_t e = expert_map(physical_to_logical_map, expert_id_);
          const size_t g_k = config.gate_k(e), u_k = config.up_k(e), d_k = config.down_k(e);
          // 소스(full N) expert 블록 시작 + shard 행 시작 (n_off × k(e)).
          const ggml_bf16_t* g_src =
              (const ggml_bf16_t*)config.gate_proj + I_full * (size_t)config.gate_row_base(e) + gu_n_off * g_k;
          const ggml_bf16_t* u_src =
              (const ggml_bf16_t*)config.up_proj + I_full * (size_t)config.up_row_base(e) + gu_n_off * u_k;
          const ggml_bf16_t* d_src =
              (const ggml_bf16_t*)config.down_proj + H_full * (size_t)config.down_row_base(e) + dn_n_off * d_k;
          std::memcpy((ggml_bf16_t*)tpc.gate_proj + I * (size_t)tpc.gate_row_base(e), g_src,
                      I * g_k * sizeof(ggml_bf16_t));
          std::memcpy((ggml_bf16_t*)tpc.up_proj + I * (size_t)tpc.up_row_base(e), u_src,
                      I * u_k * sizeof(ggml_bf16_t));
          std::memcpy((ggml_bf16_t*)tpc.down_proj + DN * (size_t)tpc.down_row_base(e), d_src,
                      DN * d_k * sizeof(ggml_bf16_t));
        },
        nullptr);
    printf("TP %d (partial bf16 tile) load weight done.\n", i);
  });

  pool->dispense_backend()->do_numa_job([&](int numa_id) {
    tps[numa_id]->config_.physical_to_logical_map = config.physical_to_logical_map;
    tps[numa_id]->load_weights();
  });

  pool->dispense_backend()->do_numa_job([&](int i) {
    auto& tpc = tps[i]->config_;
    delete[] (ggml_bf16_t*)(tpc.gate_proj);
    delete[] (ggml_bf16_t*)(tpc.up_proj);
    delete[] (ggml_bf16_t*)(tpc.down_proj);
  });

  self.weights_loaded = true;
}

template <typename K>
class TP_MOE<TILE_BF16_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, TILE_BF16_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, TILE_BF16_MOE_TP<K>>>;
  using kernel_t = K;
  using Base::Base;
  template <class TPM>
  friend void bf16_tile_tp_load_weights(TPM& self);

  void set_node_gpu_experts_mask(int node, uintptr_t mask_ptr) {
    if (node < 0 || node >= this->tp_count || !this->tps[node]) throw std::runtime_error("bad node index");
    this->tps[node]->set_gpu_experts_mask(reinterpret_cast<const uint8_t*>(mask_ptr));
  }

  // 레이아웃은 K::BufferB — k1 타일 bf16 코드 (sglang slab 태그 `kt_bf16k1`).
  std::vector<PrismColdSlabView> cold_slab_views() const {
    std::vector<PrismColdSlabView> out;
    for (int i = 0; i < this->tp_count; ++i) {
      if (!this->tps[i]) continue;
      for (int proj = 0; proj < 3; ++proj) out.push_back(this->tps[i]->cold_slab_view(proj));
    }
    return out;
  }

  void load_weights() override { bf16_tile_tp_load_weights(*this); }

  void write_weight_scale_to_buffer(int, int, const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                                    const std::vector<uintptr_t>&, const std::vector<uintptr_t>&) {
    throw std::runtime_error("TILE_BF16_MOE_TP: write_weight_scale_to_buffer is not supported");
  }
};

#endif  // CPUINFER_OPERATOR_TILE_BF16_MOE_H
