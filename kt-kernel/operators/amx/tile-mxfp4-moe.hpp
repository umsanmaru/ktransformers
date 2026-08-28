/**
 * @file tile-mxfp4-moe.hpp
 * @brief MXFP4 MoE operator over the fp4 **tile layout** — Prism cold 티어용 (sparse decode).
 *
 * `AMX_FP4_MOE_TP`(fp4-moe.hpp)와 같은 입력(nibble 행 + bf16 배율), 같은 W4A16 수치지만 저장과
 * decode 커널이 다르다: `GemmKernelTileK2MXFP4`(la/tile_k2_mxfp4.hpp)의 타일 레이아웃 +
 * 프리페치 커서 GEMV, 그리고 Prism partial의 Seam B/C(마스크 → plan → gemv_slab). cpu-mm
 * FINDINGS §13: HW 프리페처 off 소켓에서 kt fp4 원본 대비 5.3×.
 *
 * 경로:
 *   m == 1 (decode)   — sparse면 moe_base가 만든 plan(m_plan_*), dense면 즉석 dense plan → gemv_slab
 *   1 < m (소배치)     — 토큰 루프 (dense plan per token). prefill(m ≫ 1)은 sglang이 GPU(KT_TILE4
 *                        로더)로 보낸다 — 이 operator의 `mat_mat`은 존재하지 않는다.
 * TP_MOE 로더는 fp4-moe.hpp의 `mxfp4_tp_load_weights`를 공유한다 (per-TP `load_weights`만 다르다:
 * 배율을 fp32 d가 아니라 전치 E8M0으로 pack).
 */
#ifndef CPUINFER_OPERATOR_TILE_MXFP4_MOE_H
#define CPUINFER_OPERATOR_TILE_MXFP4_MOE_H

#include "fp4-moe.hpp"
#include "la/tile_k2_mxfp4.hpp"

template <class T = amx::GemmKernelTileK2MXFP4>
class TILE_MXFP4_MOE_TP : public AMX_MOE_BASE<T, TILE_MXFP4_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, TILE_MXFP4_MOE_TP<T>>;
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
  using typename Base::input_t;
  using typename Base::output_t;

  TILE_MXFP4_MOE_TP() = default;
  TILE_MXFP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size != T::K_STEP || quant_config.zero_point)
      throw std::runtime_error("TILE_MXFP4_MOE_TP: MXFP4 g32 only (group_size must be 32, no zero point)");
    // 타일 super = 256 n: 노드 N shard가 256 배수가 아니면 pack이 부분 super를 만들 수 없다.
    if (config_.intermediate_size % T::N_BLOCK || config_.down_n() % T::N_BLOCK)
      throw std::runtime_error("TILE_MXFP4_MOE_TP: node N shards (inter " + std::to_string(config_.intermediate_size) +
                               ", down_n " + std::to_string(config_.down_n()) + ") must be multiples of 256");
    printf("Creating TILE_MXFP4_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~TILE_MXFP4_MOE_TP() = default;

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // m == 1: plan 경로 (sparse는 moe_base가 만든 plan, dense는 즉석). m > 1: 토큰 루프 (dense).
  void run_gemv(int m, int n, int k, typename T::BufferA* ba, typename T::BufferB* bb, typename T::BufferC* bc,
                int ith, int nth, const amx::ActivationPlan* sparse_plan) {
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
    run_gemv(m, config_.down_n(), config_.down_k(expert_idx), down_ba_[expert_idx].get(), down_bb_[expert_idx].get(),
             down_bc_[expert_idx].get(), ith, nth, pl);
  }

  // per-TP pack: codes는 fp4 원본과 같은 from_raw_mat(타일 순열), 배율은 bf16 → 전치 E8M0.
  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (quant_config.group_size != T::K_STEP || quant_config.zero_point)
      throw std::runtime_error("TILE_MXFP4_MOE_TP: MXFP4 g32 only");
    if (config_.gate_scale == nullptr) throw std::runtime_error("TILE_MXFP4_MOE_TP: scales required");

    const size_t I = config_.intermediate_size, DN = config_.down_n(), gs = quant_config.group_size;
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, I](int task_id) {
          const uint64_t e = task_id / nth;
          const uint64_t le = expert_map(physical_to_logical_map, e);
          const int ith = task_id % nth;
          gate_bb_[e]->from_raw_mat((uint8_t*)config_.gate_proj + ((I * (size_t)config_.gate_row_base(le)) >> 1), ith,
                                    nth);
          up_bb_[e]->from_raw_mat((uint8_t*)config_.up_proj + ((I * (size_t)config_.up_row_base(le)) >> 1), ith, nth);
        },
        nullptr);
    nth = T::recommended_nth(config_.down_n());
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, DN](int task_id) {
          const uint64_t e = task_id / nth;
          const uint64_t le = expert_map(physical_to_logical_map, e);
          const int ith = task_id % nth;
          down_bb_[e]->from_raw_mat((uint8_t*)config_.down_proj + ((DN * (size_t)config_.down_row_base(le)) >> 1), ith,
                                    nth);
        },
        nullptr);
    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map, I, DN, gs](int e) {
          const uint64_t le = expert_map(physical_to_logical_map, e);
          gate_bb_[e]->from_scales_bf16((const ggml_bf16_t*)config_.gate_scale +
                                        (I * (size_t)config_.gate_row_base(le)) / gs);
          up_bb_[e]->from_scales_bf16((const ggml_bf16_t*)config_.up_scale + (I * (size_t)config_.up_row_base(le)) / gs);
          down_bb_[e]->from_scales_bf16((const ggml_bf16_t*)config_.down_scale +
                                        (DN * (size_t)config_.down_row_base(le)) / gs);
        },
        nullptr);
  }

  // GPU writeback(kt 하이브리드)은 partial 인스턴스에서 쓰지 않는다.
  void write_weights_to_buffer(int, int, int, const GeneralMOEConfig&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&) const {
    throw std::runtime_error("TILE_MXFP4_MOE_TP: write_weights_to_buffer is not supported");
  }
};

template <typename K>
class TP_MOE<TILE_MXFP4_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, TILE_MXFP4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, TILE_MXFP4_MOE_TP<K>>>;
  using Base::Base;
  template <class TPM>
  friend void mxfp4_tp_load_weights(TPM& self);

  void set_node_gpu_experts_mask(int node, uintptr_t mask_ptr) {
    if (node < 0 || node >= this->tp_count || !this->tps[node]) throw std::runtime_error("bad node index");
    this->tps[node]->set_gpu_experts_mask(reinterpret_cast<const uint8_t*>(mask_ptr));
  }

  // 레이아웃은 GemmKernelTileK2MXFP4::BufferB (codes 타일 + 전치 E8M0) — sglang KT_TILE4 로더.
  std::vector<PrismColdSlabView> cold_slab_views() const {
    std::vector<PrismColdSlabView> out;
    for (int i = 0; i < this->tp_count; ++i) {
      if (!this->tps[i]) continue;
      for (int proj = 0; proj < 3; ++proj) out.push_back(this->tps[i]->cold_slab_view(proj));
    }
    return out;
  }

  void load_weights() override { mxfp4_tp_load_weights(*this); }

  void write_weight_scale_to_buffer(int, int, const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                                    const std::vector<uintptr_t>&, const std::vector<uintptr_t>&) {
    throw std::runtime_error("TILE_MXFP4_MOE_TP: write_weight_scale_to_buffer is not supported");
  }
};

#endif  // CPUINFER_OPERATOR_TILE_MXFP4_MOE_H
