"""Prism partial MoE의 Python wrapper (계약: sglang prism CONTRACTS.md ②/③).

BaseMoEWrapper(experts_base.py)의 자매 — 대응 관계:
  load_weights_from_tensors ↔ 동명 (K-슬라이스 텐서 주입)
  submit_forward / sync_forward ↔ submit_forward_gateup·down / sync
  forward ↔ forward_gateup·down (동기 편의)

의도적 편차 (기록: doc/prism-partial-entrypoints.md §6): kt wrapper는
staging 버퍼를 self-serve하지만, 이 wrapper는 **caller-provide** — pinned
staging(x/act/expert_ids/partial out)은 sglang prism ExecutionResources가
소유하고 여기는 포인터만 받는다. 이 모듈은 Plan 어휘를 모른다 (config의
숫자만 소비 — Plan→config 번역은 sglang 쪽 번역기의 몫).

수명 규칙 (계약 ③): 주입 텐서는 pack 완료(동기)까지 이 객체가 참조를
들고, 완료 후 해제한다. C++ MOE 객체가 packed NUMA 메모리를 소유하며,
이 wrapper 핸들의 수명 = cold weight의 수명.

**sparsity 테이블은 수명 규칙이 다르다**: weight는 pack되어 C++ 소유로
넘어가지만, 점수 재료(wn_sq/pair_dot)와 threshold 곡선(thr_*)은 C++가
**매 step 원시 포인터로 읽는다**. 따라서 이 wrapper가 wrapper 수명 내내
참조를 들고 있어야 한다 (놓으면 dangling → segfault).

step마다 오는 것은 라우터 가중 하나뿐이다 — 예산 → s → 격자 조회 → 마스크
판정이 전부 C++에서 끝난다 (계약 ④).
"""

from __future__ import annotations

from typing import Optional

import torch

from kt_kernel import kt_kernel_ext


def _k_total(ki, expert_num: int) -> int:
    """이 티어가 소유하는 총 K 행 수. 밴드형이면 E*rows, 인덱스형이면 row_off[-1]."""
    return int(ki.row_off[-1]) if len(ki.row_off) else expert_num * int(ki.rows)

# cpu_cold 커널 키 → ext 클래스. (키의 유효성 검증은 sglang prism kernels.py,
# 실제 클래스 선택은 여기 — 라이브러리가 자기 클래스 목록을 소유한다)
_PARTIAL_MOE_CLASSES = {
    "kt_amx_bf16": lambda: kt_kernel_ext.moe.AMXBF16_MOE,
    # n-contiguous weight layout: prefill은 상속받은 AMX amx_kernel, sparse
    # decode는 tile_k2의 gemv_slab — 두 경로가 weight 한 벌을 공유한다.
    "kt_tile_k2_bf16": lambda: kt_kernel_ext.moe.TileK2BF16_MOE,
    # 그 no-SW-pf 형제 (tile_k2.hpp `GemmKernelTileK2BF16NoPF`): 소비 커서·prefill B-panel·
    # prep/export 보조 프리페치가 컴파일 타임에 빠진 같은 커널. 레이아웃·plan·자산·정렬이
    # 전부 같아서 plan의 커널 키만 바꿔 "SW 프리페처 없음" ablation을 돌린다.
    "kt_tile_k2_bf16_nopf": lambda: kt_kernel_ext.moe.TileK2BF16NoPF_MOE,
    # MXFP4 g32 (e2m1 nibble 행 [N, k/2] + bf16 배율 [N, k/32]) — dense 전용 (sparse plan 거절).
    # 저장은 kt BufferBInt4KGroupImpl(행우선 nibble + fp32 배율)이고 GPU cold 읽기는 그 레이아웃을
    # 제자리 해석한다 (sglang prism_grouped_mxfp4.cuh KT_FP4 로더).
    "kt_amx_fp4": lambda: kt_kernel_ext.moe.AMXFP4_KGroup_MOE,
    # MXFP4 타일 레이아웃 + 프리페치 커서 sparse decode (cpu-mm tile_k2_mxfp4 포팅). 노드 N shard는
    # 256 배수. prefill은 GPU KT_TILE4 로더가 slab을 읽는다.
    "kt_tile_k2_mxfp4": lambda: kt_kernel_ext.moe.TileK2MXFP4_MOE,
    # FP8 e4m3 타일 레이아웃 + 프리페치 커서 sparse decode (cpu-mm tile_k2_fp8b128 포팅).
    # 배율은 128×128 블록당 fp32 하나(`scale_inv`, n-major). 노드 N shard 256 배수, K 128 배수.
    # prefill은 GPU KT_TILE8 로더가 slab을 읽는다.
    "kt_tile_k2_fp8b128": lambda: kt_kernel_ext.moe.TileK2FP8B128_MOE,
    # k=1 형제 (tile_k1_fp8b128): 64 B 라인 = k 한 행, **per-k 마스크**(score k1: |x| >= thr).
    # 같은 입력 텐서·같은 slab 기하; sparsity 테이블은 thr 곡선 셋만 받는다 (wn_sq/pair_dot 불필요).
    # prefill은 GPU KT_TILE8_K1 로더가 slab을 읽는다.
    "kt_tile_k1_fp8b128": lambda: kt_kernel_ext.moe.TileK1FP8B128_MOE,
    # FP8 e4m3 **per-tensor** 배율 (Mistral-Medium-3.5 static-tensor FP8: weight_block_size null).
    # 코드 타일 레이아웃은 b128과 같고 배율은 proj당 fp32 [E] (expert = dense 슬롯당 스칼라 1개).
    # 노드 N shard 256 배수, K 32 배수 (128 제약 없음).
    "kt_tile_k2_fp8pt": lambda: kt_kernel_ext.moe.TileK2FP8PT_MOE,
}

# 커널 키 → (weight dtype, 원소당 바이트 분모, 배율 dtype 또는 None, 배율 원소당 weight 원소 수,
#            quant group_size). 배율 개수는 "weight 원소 수 / scale_div"다 — mxfp4는 32-k 블록
# 하나에 배율 1개(행마다)라 32, fp8 블록은 128×128이라 16384다.
_WEIGHT_FORMAT = {
    "kt_amx_bf16": (torch.bfloat16, 1, None, 1, 0),
    "kt_tile_k2_bf16": (torch.bfloat16, 1, None, 1, 0),
    "kt_tile_k2_bf16_nopf": (torch.bfloat16, 1, None, 1, 0),
    "kt_amx_fp4": (torch.uint8, 2, torch.bfloat16, 32, 32),
    "kt_tile_k2_mxfp4": (torch.uint8, 2, torch.bfloat16, 32, 32),
    "kt_tile_k2_fp8b128": (torch.uint8, 1, torch.float32, 128 * 128, 128),
    "kt_tile_k1_fp8b128": (torch.uint8, 1, torch.float32, 128 * 128, 128),
    # scale_div None = "expert당 배율 1개" (weight 원소 수와 무관하게 [E]). group_size 0 = per-tensor.
    "kt_tile_k2_fp8pt": (torch.uint8, 1, torch.float32, None, 0),
}
FP4_GROUP_SIZE = 32


class PartialMoEWrapper:
    """레이어당 1개. C++ partial MOE 인스턴스의 수명·호출을 관리한다."""

    def __init__(self, config, cpuinfer, kernel_key: str = "kt_amx_bf16"):
        """config: kt_kernel_ext.moe.MOEConfig — partial 기하까지 채워진 것.
        weight 포인터는 비워두고, load_weights_from_tensors가 채운 뒤
        C++ 인스턴스를 생성한다 (config는 생성 순간 C++로 복사·불변화).
        """
        if kernel_key not in _PARTIAL_MOE_CLASSES:
            raise ValueError(f"unknown partial kernel key: {kernel_key!r}")
        if not config.partial.enabled:
            raise ValueError("PartialMoEWrapper requires config.partial.enabled")
        self._config = config
        self._cpuinfer = cpuinfer
        self._kernel_key = kernel_key
        self._moe_class = _PARTIAL_MOE_CLASSES[kernel_key]()
        self._moe = None
        self._sparsity_tables = None  # 설치되면 wrapper 수명 내내 보관

    # ── Stage 2: 로딩 (kt SFT 텐서 주입 경로와 동형) ─────────────────────
    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,  # bf16 [E, inter, gateup_k]  (K-슬라이스, ckpt 방향)
        up_proj: torch.Tensor,    # bf16 [E, inter, gateup_k]
        down_proj: torch.Tensor,  # bf16 [E, hidden, down_k]
        physical_to_logical_map_cpu: Optional[torch.Tensor] = None,
        sparsity_tables: Optional[dict] = None,
        gate_scale: Optional[torch.Tensor] = None,  # kt_amx_fp4: bf16 [E·inter·gateup_k/32] (2^e)
        up_scale: Optional[torch.Tensor] = None,
        down_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """sparsity_tables: 주면 sparse decode가 켜진다 (없으면 dense).

        양자화 키면 배율 셋이 필수다: kt_amx_fp4/kt_tile_k2_mxfp4는 weight가 uint8 nibble
        (원소 2개/바이트) + 배율 bf16(32-k 블록당 1, C++가 fp32 또는 전치 E8M0으로 변환),
        kt_tile_k2_fp8b128은 weight가 e4m3 바이트 + 배율 fp32(128×128 블록당 1, n-major
        `scale_inv`), kt_tile_k2_fp8pt는 e4m3 바이트 + 배율 fp32 [E] (expert당 스칼라 1개,
        per-tensor `weight_scale_inv`).

        페어 마스크 커널(k2wl2)은 키 9개 전부, per-k 커널(`mask_per_k`, score k1)은
        thr_gate/thr_up/thr_down 셋만 필요하다 (나머지가 와도 무시). 전부 fp32 contiguous CPU:
          gate_wn_sq/up_wn_sq       [E, gateup_k]    = wn^2
          gate_pair_dot/up_pair_dot [E, gateup_k/2]  = 인접열 내적
          down_wn_sq [E, down_k] / down_pair_dot [E, down_k/2]
          thr_gate/thr_up/thr_down  [E, ng]          = threshold 곡선
        점수 재료는 이 인스턴스의 K 밴드로 절단된 것이어야 하고, thr 곡선은
        밴드와 무관하므로 절단하지 않는다. 예산(p/lam/pmax/grid/ng/renorm_it)은
        config.partial.sparsity에 미리 채워져 있어야 한다.
        """
        if self._moe is not None:
            raise RuntimeError("weights already loaded")
        cfg = self._config
        # 원소 수로 검증한다 (shape가 아니라): 밴드형은 [E, N, rows]지만
        # 인덱스형은 expert 블록 [N, k(e)]를 이어 붙인 flat이라 rank가 다르다.
        # 두 경우 모두 "N × Σₑ k(e)"라는 하나의 불변식으로 덮인다.
        w_dtype, w_div, s_dtype, s_div, group_size = _WEIGHT_FORMAT[self._kernel_key]
        expect = {
            "gate": cfg.intermediate_size * _k_total(cfg.partial.gate, cfg.expert_num),
            "up": cfg.intermediate_size * _k_total(cfg.partial.up, cfg.expert_num),
            "down": cfg.hidden_size * _k_total(cfg.partial.down, cfg.expert_num),
        }

        def _check(name, t, numel, dtype):
            if (t is None or t.dtype != dtype or t.numel() != numel
                    or not t.is_contiguous() or t.device.type != "cpu"):
                # device 검증 필수: CUDA 텐서의 data_ptr()는 device 주소라
                # C++ host memcpy가 그대로 segfault한다 (조용한 죽음 방지)
                raise ValueError(
                    f"{name} must be contiguous CPU {dtype} with {numel} elements, got "
                    f"{None if t is None else (t.dtype, tuple(t.shape), t.device, t.is_contiguous())}"
                )

        for name, t in (("gate_proj", gate_proj), ("up_proj", up_proj), ("down_proj", down_proj)):
            _check(name, t, expect[name.split("_")[0]] // w_div, w_dtype)
        scales = (gate_scale, up_scale, down_scale)
        if s_dtype is not None:
            for name, t in zip(("gate_scale", "up_scale", "down_scale"), scales):
                want = cfg.expert_num if s_div is None else expect[name.split("_")[0]] // s_div
                _check(name, t, want, s_dtype)
            cfg.quant_config.group_size = group_size
            cfg.quant_config.zero_point = False
        elif any(t is not None for t in scales):
            raise ValueError(f"{self._kernel_key} takes no scale tensors")
        if physical_to_logical_map_cpu is None:
            physical_to_logical_map_cpu = torch.arange(cfg.expert_num, dtype=torch.int64)
        phys_map = physical_to_logical_map_cpu.to(dtype=torch.int64, device="cpu").contiguous()

        # pack이 읽는 동안 생존 보장 (계약 ③ 수명 규칙)
        self._pending = (gate_proj, up_proj, down_proj, phys_map) + scales
        cfg.gate_proj = gate_proj.data_ptr()
        cfg.up_proj = up_proj.data_ptr()
        cfg.down_proj = down_proj.data_ptr()
        if s_dtype is not None:
            cfg.gate_scale = gate_scale.data_ptr()
            cfg.up_scale = up_scale.data_ptr()
            cfg.down_scale = down_scale.data_ptr()

        if sparsity_tables is not None:
            self._install_sparsity(cfg, sparsity_tables)

        self._moe = self._moe_class(cfg)  # config가 C++로 복사되는 순간
        self._cpuinfer.submit(self._moe.load_weights_task(phys_map.data_ptr()))
        self._cpuinfer.sync()  # pack 동기 완료
        self._pending = None  # 소스 텐서 해제 — 이후 소유권은 C++ (계약 ③)

    # (키, 축 종류, 나눔) — 축 "grid"는 밴드와 무관한 threshold 곡선이다.
    _THR_KEYS = (
        ("thr_gate", "grid", 1),
        ("thr_up", "grid", 1),
        ("thr_down", "grid", 1),
    )
    _SCORE_KEYS = (
        ("gate_wn_sq", "gate", 1),
        ("gate_pair_dot", "gate", 2),
        ("up_wn_sq", "up", 1),
        ("up_pair_dot", "up", 2),
        ("down_wn_sq", "down", 1),
        ("down_pair_dot", "down", 2),
    )
    _SPARSITY_KEYS = _SCORE_KEYS + _THR_KEYS  # 페어 마스크 커널의 전체 집합 (하위 호환 이름)

    @property
    def mask_per_k(self) -> bool:
        """이 커널의 Seam B 마스크가 per-k(score k1)인가. C++ 클래스가 소유한 사실을 그대로 읽는다."""
        return bool(getattr(self._moe_class, "mask_per_k", False))

    def sparsity_keys(self):
        """이 커널이 요구하는 sparsity 테이블 키 표 (키, 축, 나눔)."""
        return self._THR_KEYS if self.mask_per_k else self._SPARSITY_KEYS

    def _install_sparsity(self, cfg, tables: dict) -> None:
        """테이블 검증 + config 포인터 설치. 참조는 self가 영구 보관한다."""
        keys = self.sparsity_keys()
        missing = [n for n, _, _ in keys if n not in tables]
        if missing:
            raise ValueError(f"sparsity_tables missing keys: {missing}")
        sp = cfg.partial.sparsity
        if sp.ng <= 1 or sp.grid <= 0.0:
            raise ValueError(
                f"config.partial.sparsity grid not set (ng={sp.ng}, grid={sp.grid}) "
                f"— 예산은 테이블 설치 전에 채워져야 한다"
            )
        # 점수 테이블도 weight와 같은 인덱스로 잘려 들어오므로 총 행 수가 기준
        # 이다 (밴드형이면 E*rows와 같다). threshold 곡선만 축이 다르다.
        rows = {"gate": _k_total(cfg.partial.gate, cfg.expert_num),
                "up": _k_total(cfg.partial.up, cfg.expert_num),
                "down": _k_total(cfg.partial.down, cfg.expert_num),
                "grid": cfg.expert_num * sp.ng}
        kept = {}
        for name, band, div in keys:
            t = tables[name]
            want = rows[band] // div
            if (t.dtype != torch.float32 or t.numel() != want
                    or not t.is_contiguous() or t.device.type != "cpu"):
                raise ValueError(
                    f"{name} must be contiguous CPU fp32 {want}, got "
                    f"{t.dtype} {tuple(t.shape)} device={t.device} "
                    f"contiguous={t.is_contiguous()}"
                )
            kept[name] = t
        # C++가 매 step 원시 포인터로 읽으므로 wrapper 수명 내내 붙잡는다.
        self._sparsity_tables = kept
        for name in kept:
            setattr(sp, name, kept[name].data_ptr())
        sp.enabled = True

    def _require_loaded(self):
        if self._moe is None:
            raise RuntimeError("load_weights_from_tensors first")
        return self._moe

    # ── task 경로: (fn, args)를 CPUInfer로 제출 (graph-호환 배관) ─────────
    # 모든 포인터 인자는 정수 주소(intptr). qlen_ptr가 가리키는 int는
    # pinned·주소고정이어야 하며 값은 step마다 in-place 갱신 (계약 ④).
    # weights_ptr: fp32 [qlen][k] 라우터 가중 pinned. 0 = dense.
    # sparse는 decode 전용이므로 qlen != 1에 주면 C++가 즉사시킨다.
    def set_node_gpu_experts_mask(self, node: int, mask: "torch.Tensor | None") -> None:
        """노드 인스턴스별 expert skip 마스크 (uint8 [E], pinned 권장; C++가 매 호출 읽으므로
        호출자가 참조를 유지해야 한다). None = 마스크 해제."""
        if self._moe is None:
            raise RuntimeError("weights not loaded")
        self._moe.set_node_gpu_experts_mask(int(node), 0 if mask is None else mask.data_ptr())

    def cold_slab_views(self) -> list:
        """(node, proj)별 packed B slab 기술자 — sglang prism의 cold GPU 읽기용.
        load_weights_from_tensors 이후에만 유효하다."""
        if self._moe is None:
            raise RuntimeError("weights not loaded")
        return self._moe.cold_slab_views()

    def submit_forward_gateup(self, qlen_ptr: int, k: int, expert_ids_ptr: int,
                              x_ptr: int, out_ptr: int, cuda_stream: Optional[int] = None,
                              weights_ptr: int = 0) -> None:
        task = self._require_loaded().forward_gateup_partial_task(
            qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr, weights_ptr)
        self._submit(task, cuda_stream)

    def submit_forward_down(self, qlen_ptr: int, k: int, expert_ids_ptr: int,
                            act_ptr: int, out_ptr: int, cuda_stream: Optional[int] = None,
                            weights_ptr: int = 0) -> None:
        task = self._require_loaded().forward_down_partial_task(
            qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr, weights_ptr)
        self._submit(task, cuda_stream)

    def _submit(self, task, cuda_stream: Optional[int]) -> None:
        if cuda_stream is None:
            self._cpuinfer.submit(task)
        else:
            self._cpuinfer.submit_with_cuda_stream(cuda_stream, task)

    def sync(self, cuda_stream: Optional[int] = None) -> None:
        """완료 계약 (②-4): 이 호출(또는 sync host node)이 반환/실행된 시점에
        out은 완전히 쓰여 있다."""
        if cuda_stream is None:
            self._cpuinfer.sync()
        else:
            self._cpuinfer.sync_with_cuda_stream(cuda_stream)

    # ── 동기 편의 (테스트/eager) — kt BaseMoEWrapper.forward의 대응물 ─────
    def forward_gateup(self, expert_ids: torch.Tensor, x: torch.Tensor, out: torch.Tensor,
                       weights: Optional[torch.Tensor] = None) -> None:
        qlen_t = torch.tensor([x.shape[0]], dtype=torch.int32)
        self._require_loaded().forward_gateup_partial(
            qlen_t.data_ptr(), expert_ids.shape[1], expert_ids.data_ptr(), x.data_ptr(), out.data_ptr(),
            0 if weights is None else weights.data_ptr()
        )

    def forward_down(self, expert_ids: torch.Tensor, act: torch.Tensor, out: torch.Tensor,
                     weights: Optional[torch.Tensor] = None) -> None:
        qlen_t = torch.tensor([act.shape[0]], dtype=torch.int32)
        self._require_loaded().forward_down_partial(
            qlen_t.data_ptr(), expert_ids.shape[1], expert_ids.data_ptr(), act.data_ptr(), out.data_ptr(),
            0 if weights is None else weights.data_ptr()
        )
