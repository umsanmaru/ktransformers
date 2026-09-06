/**
 * @file tile-fp8-moe.hpp
 * @brief FP8(e4m3, 128×128 블록 배율) MoE operator — Prism cold 티어용 (sparse decode).
 *
 * `AMX_FP8_MOE_TP`(fp8-moe.hpp)와 같은 입력(e4m3 바이트 행 + fp32 `scale_inv`), 같은 W8A16
 * 수치지만 저장과 decode 커널이 다르다: `GemmKernelTileK2FP8B128`(la/tile_k2_fp8b128.hpp)의
 * 타일 레이아웃 + 프리페치 커서 GEMV, 그리고 Prism partial의 Seam B/C(마스크 → plan →
 * gemv_slab). cpu-mm FINDINGS §15: kt 슬랩 크기(256-n)에서 flat fp8 대비 dense 1.8×.
 *
 * 경로 (tile-mxfp4-moe.hpp와 동형):
 *   m == 1 (decode)  — sparse면 moe_base가 만든 plan(m_plan_*), dense면 즉석 dense plan → gemv_slab
 *   1 < m (소배치)    — 토큰 루프 (dense plan per token). prefill(m ≫ 1)은 sglang이 GPU(KT_TILE8
 *                       로더)로 보낸다 — 이 operator의 `mat_mat`은 존재하지 않는다.
 *
 * 템플릿 인자 T로 `GemmKernelTileK1FP8B128`(k=1 타일, per-k 마스크 — la/tile_k1_fp8b128.hpp)도 그대로
 * 인스턴스화된다. 두 커널은 BufferB 크기·배율 pack·slab 기하가 같고 그룹 안 바이트 순열과 마스크
 * 종류만 다르다 — 마스크 분기는 moe_base(`kMaskPerK`)가 한다.
 *
 * 이 operator는 **Prism partial 전용**이다: 타일 레이아웃은 GPU가 제자리 읽는 cold slab의
 * 형태이고, 비-partial(kt 자체 MoE) 경로는 원본 `AMX_FP8_MOE_TP`가 담당한다.
 */
#ifndef CPUINFER_OPERATOR_TILE_FP8_MOE_H
#define CPUINFER_OPERATOR_TILE_FP8_MOE_H

#include "fp8-moe.hpp"
#include "la/tile_k1_fp8b128.hpp"
#include "la/tile_k2_fp8b128.hpp"
#include "la/tile_k2_fp8pt.hpp"

/// 커널이 per-tensor 배율(`kPerTensorScale`, SCALE_BLK == 0)을 쓰는가. 없으면 128×128 블록.
template <class K>
constexpr bool fp8_tile_per_tensor_f() {
  if constexpr (requires { K::kPerTensorScale; }) return K::kPerTensorScale;
  else return false;
}
template <class K>
inline constexpr bool fp8_tile_per_tensor_v = fp8_tile_per_tensor_f<K>();

template <class T = amx::GemmKernelTileK2FP8B128>
class TILE_FP8_MOE_TP : public AMX_MOE_BASE<T, TILE_FP8_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, TILE_FP8_MOE_TP<T>>;
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

  TILE_FP8_MOE_TP() = default;
  TILE_FP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    // group_size는 커널의 배율 기하와 일치해야 한다: b128 = 128, per-tensor = 0.
    if (quant_config.group_size != T::SCALE_BLK || quant_config.zero_point)
      throw std::runtime_error("TILE_FP8_MOE_TP<" + T::name() + ">: group_size must be " +
                               std::to_string(T::SCALE_BLK) + " (got " + std::to_string(quant_config.group_size) +
                               "), no zero point");
    // 타일 super = 256 n: 노드 N shard가 256 배수가 아니면 pack이 부분 super를 만들 수 없다.
    if (config_.intermediate_size % T::N_BLOCK || config_.down_n() % T::N_BLOCK)
      throw std::runtime_error("TILE_FP8_MOE_TP: node N shards (inter " +
                               std::to_string(config_.intermediate_size) + ", down_n " +
                               std::to_string(config_.down_n()) + ") must be multiples of 256");
    printf("Creating TILE_FP8_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~TILE_FP8_MOE_TP() = default;

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

  // 배율 소스 포인터: b128은 n-major [N/128][k/128] 블록표의 expert 시작, per-tensor는 [E]의 e번째.
  const float* scale_src(const void* base, size_t n, size_t row_base_le, uint64_t le) const {
    if constexpr (fp8_tile_per_tensor_v<T>) {
      (void)n; (void)row_base_le;
      return (const float*)base + le;
    } else {
      const size_t B = T::SCALE_BLK;
      return (const float*)base + (n / B) * (row_base_le / B);
    }
  }

  // per-TP pack: codes는 행우선 e4m3 → 타일 순열, 배율은 n-major fp32 → 전치 fp32 (b128) / 스칼라 1개 (pt).
  void load_weights() {
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (config_.gate_scale == nullptr) throw std::runtime_error("TILE_FP8_MOE_TP: scales required");
    const size_t I = config_.intermediate_size, DN = config_.down_n();

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, I](int task_id) {
          const uint64_t e = task_id / nth;
          const uint64_t le = expert_map(physical_to_logical_map, e);
          const int ith = task_id % nth;
          gate_bb_[e]->from_raw_mat((const uint8_t*)config_.gate_proj + I * (size_t)config_.gate_row_base(le), ith,
                                    nth);
          up_bb_[e]->from_raw_mat((const uint8_t*)config_.up_proj + I * (size_t)config_.up_row_base(le), ith, nth);
          gate_bb_[e]->from_scales(scale_src(config_.gate_scale, I, (size_t)config_.gate_row_base(le), le), ith, nth);
          up_bb_[e]->from_scales(scale_src(config_.up_scale, I, (size_t)config_.up_row_base(le), le), ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.down_n());
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, DN](int task_id) {
          const uint64_t e = task_id / nth;
          const uint64_t le = expert_map(physical_to_logical_map, e);
          const int ith = task_id % nth;
          down_bb_[e]->from_raw_mat((const uint8_t*)config_.down_proj + DN * (size_t)config_.down_row_base(le), ith,
                                    nth);
          down_bb_[e]->from_scales(scale_src(config_.down_scale, DN, (size_t)config_.down_row_base(le), le), ith, nth);
        },
        nullptr);
  }

  // GPU writeback(kt 하이브리드)은 partial 인스턴스에서 쓰지 않는다.
  void write_weights_to_buffer(int, int, int, const GeneralMOEConfig&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                               const std::vector<uintptr_t>&) const {
    throw std::runtime_error("TILE_FP8_MOE_TP: write_weights_to_buffer is not supported");
  }
};

// FP8 타일 weight staging (Prism partial 전용): full 텐서 → per-TP N-shard 사본 → 각 TP가 pack.
// mxfp4의 `mxfp4_tp_load_weights` partial 분기와 동형이고, 다른 것은 원소 크기(1 B)와 배율
// 기하뿐이다: b128은 fp32 128×128 블록표(N-shard로 잘라 복사), per-tensor는 proj당 fp32 [E]
// (shard와 무관 — expert별 스칼라를 그대로 복사).
template <class TPM>
inline void fp8_tile_tp_load_weights(TPM& self) {
  using K = typename TPM::kernel_t;
  constexpr bool kPT = fp8_tile_per_tensor_v<K>;
  auto& config = self.config;
  auto& tps = self.tps;
  auto pool = config.pool;
  const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
  const size_t B = kPT ? 1 : (size_t)K::SCALE_BLK;  // pt에서는 나눗셈 회피용 더미 (아래 분기에서 안 쓴다)

  if (config.gate_proj == nullptr || config.gate_scale == nullptr)
    throw std::runtime_error("TILE_FP8_MOE_TP: needs packed e4m3 weights + fp32 block scales");
  if (!config.is_partial())
    throw std::runtime_error(
        "TILE_FP8_MOE_TP: Prism partial only (the non-partial fp8 path is AMX_FP8_MOE_TP)");

  pool->dispense_backend()->do_numa_job([&](int i) {
    auto& tpc = tps[i]->config_;
    const size_t I = tpc.intermediate_size, DN = tpc.down_n();
    const size_t g_tot = I * (size_t)tpc.gate_row_base(tpc.expert_num);
    const size_t u_tot = I * (size_t)tpc.up_row_base(tpc.expert_num);
    const size_t d_tot = DN * (size_t)tpc.down_row_base(tpc.expert_num);
    tpc.gate_proj = new uint8_t[g_tot];
    tpc.up_proj = new uint8_t[u_tot];
    tpc.down_proj = new uint8_t[d_tot];
    if constexpr (kPT) {
      tpc.gate_scale = new float[tpc.expert_num];
      tpc.up_scale = new float[tpc.expert_num];
      tpc.down_scale = new float[tpc.expert_num];
    } else {
      tpc.gate_scale = new float[g_tot / (B * B)];
      tpc.up_scale = new float[u_tot / (B * B)];
      tpc.down_scale = new float[d_tot / (B * B)];
    }
    const size_t gu_n_off = (size_t)tpc.partial.gateup_n.offset;
    const size_t dn_n_off = (size_t)tpc.partial.down_n.offset;
    const size_t I_full = config.intermediate_size, H_full = config.hidden_size;
    pool->get_subpool(i)->do_work_stealing_job(
        tpc.expert_num, nullptr,
        [&](int expert_id_) {
          const size_t e = expert_map(physical_to_logical_map, expert_id_);
          const size_t g_k = config.gate_k(e), u_k = config.up_k(e), d_k = config.down_k(e);
          // 소스(full N) 블록 시작: N_full × row_base(e); shard 행 시작 = n_off × k(e).
          const uint8_t* g_src =
              (const uint8_t*)config.gate_proj + I_full * (size_t)config.gate_row_base(e) + gu_n_off * g_k;
          const uint8_t* u_src =
              (const uint8_t*)config.up_proj + I_full * (size_t)config.up_row_base(e) + gu_n_off * u_k;
          const uint8_t* d_src =
              (const uint8_t*)config.down_proj + H_full * (size_t)config.down_row_base(e) + dn_n_off * d_k;
          memcpy((uint8_t*)tpc.gate_proj + I * (size_t)tpc.gate_row_base(e), g_src, I * g_k);
          memcpy((uint8_t*)tpc.up_proj + I * (size_t)tpc.up_row_base(e), u_src, I * u_k);
          memcpy((uint8_t*)tpc.down_proj + DN * (size_t)tpc.down_row_base(e), d_src, DN * d_k);
          if constexpr (kPT) {
            // per-tensor: expert(슬롯)당 스칼라 하나, N-shard와 무관.
            ((float*)tpc.gate_scale)[e] = ((const float*)config.gate_scale)[e];
            ((float*)tpc.up_scale)[e] = ((const float*)config.up_scale)[e];
            ((float*)tpc.down_scale)[e] = ((const float*)config.down_scale)[e];
            return;
          }
          // 배율은 n-major [N/128][k/128]이라 shard 행 시작이 (n_off/128) 블록행이다.
          const float* gs_src = (const float*)config.gate_scale +
                                (I_full / B) * ((size_t)config.gate_row_base(e) / B) + (gu_n_off / B) * (g_k / B);
          const float* us_src = (const float*)config.up_scale +
                                (I_full / B) * ((size_t)config.up_row_base(e) / B) + (gu_n_off / B) * (u_k / B);
          const float* ds_src = (const float*)config.down_scale +
                                (H_full / B) * ((size_t)config.down_row_base(e) / B) + (dn_n_off / B) * (d_k / B);
          memcpy((float*)tpc.gate_scale + (I / B) * ((size_t)tpc.gate_row_base(e) / B), gs_src,
                 sizeof(float) * (I / B) * (g_k / B));
          memcpy((float*)tpc.up_scale + (I / B) * ((size_t)tpc.up_row_base(e) / B), us_src,
                 sizeof(float) * (I / B) * (u_k / B));
          memcpy((float*)tpc.down_scale + (DN / B) * ((size_t)tpc.down_row_base(e) / B), ds_src,
                 sizeof(float) * (DN / B) * (d_k / B));
        },
        nullptr);
    printf("TP %d (partial fp8 tile%s) load weight done.\n", i, kPT ? ", per-tensor" : "");
  });

  pool->dispense_backend()->do_numa_job([&](int numa_id) {
    tps[numa_id]->config_.physical_to_logical_map = config.physical_to_logical_map;
    tps[numa_id]->load_weights();
  });

  pool->dispense_backend()->do_numa_job([&](int i) {
    auto& tpc = tps[i]->config_;
    delete[] (uint8_t*)(tpc.gate_proj);
    delete[] (uint8_t*)(tpc.up_proj);
    delete[] (uint8_t*)(tpc.down_proj);
    delete[] (float*)(tpc.gate_scale);
    delete[] (float*)(tpc.up_scale);
    delete[] (float*)(tpc.down_scale);
  });

  self.weights_loaded = true;
}

template <typename K>
class TP_MOE<TILE_FP8_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, TILE_FP8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, TILE_FP8_MOE_TP<K>>>;
  using kernel_t = K;
  using Base::Base;
  template <class TPM>
  friend void fp8_tile_tp_load_weights(TPM& self);

  void set_node_gpu_experts_mask(int node, uintptr_t mask_ptr) {
    if (node < 0 || node >= this->tp_count || !this->tps[node]) throw std::runtime_error("bad node index");
    this->tps[node]->set_gpu_experts_mask(reinterpret_cast<const uint8_t*>(mask_ptr));
  }

  // 레이아웃은 K::BufferB — b128: codes 타일 + 전치 fp32 배율표 (sglang KT_TILE8 로더),
  // pt: 같은 codes 타일 + 64 B 슬롯의 fp32 배율 1개 (sglang kt_tile8pt).
  std::vector<PrismColdSlabView> cold_slab_views() const {
    std::vector<PrismColdSlabView> out;
    for (int i = 0; i < this->tp_count; ++i) {
      if (!this->tps[i]) continue;
      for (int proj = 0; proj < 3; ++proj) out.push_back(this->tps[i]->cold_slab_view(proj));
    }
    return out;
  }

  void load_weights() override { fp8_tile_tp_load_weights(*this); }

  void write_weight_scale_to_buffer(int, int, const std::vector<uintptr_t>&, const std::vector<uintptr_t>&,
                                    const std::vector<uintptr_t>&, const std::vector<uintptr_t>&) {
    throw std::runtime_error("TILE_FP8_MOE_TP: write_weight_scale_to_buffer is not supported");
  }
};

#endif  // CPUINFER_OPERATOR_TILE_FP8_MOE_H
