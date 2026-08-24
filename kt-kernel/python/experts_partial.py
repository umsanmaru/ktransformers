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
넘어가지만, wn_sq/pair_dot은 C++가 **매 step 원시 포인터로 읽는다**.
따라서 이 wrapper가 wrapper 수명 내내 참조를 들고 있어야 한다 (놓으면
dangling → segfault). threshold는 반대로 step마다 갱신되는 호출자 소유
pinned 버퍼이고 포인터만 인자로 온다 (계약 ④).
"""

from __future__ import annotations

from typing import Optional

import torch

from kt_kernel import kt_kernel_ext

# cpu_cold 커널 키 → ext 클래스. (키의 유효성 검증은 sglang prism kernels.py,
# 실제 클래스 선택은 여기 — 라이브러리가 자기 클래스 목록을 소유한다)
_PARTIAL_MOE_CLASSES = {
    "kt_amx_bf16": lambda: kt_kernel_ext.moe.AMXBF16_MOE,
}


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
    ) -> None:
        """sparsity_tables: 주면 sparse decode가 켜진다 (없으면 dense).

        키 6개 전부 필요하고 전부 fp32 contiguous CPU여야 한다:
          gate_wn_sq/up_wn_sq   [E, gateup_k]      = wn^2
          gate_pair_dot/up_pair_dot [E, gateup_k/2] = 인접열 내적
          down_wn_sq [E, down_k] / down_pair_dot [E, down_k/2]
        전부 이 인스턴스의 K 밴드로 이미 절단된 것이어야 한다.
        """
        if self._moe is not None:
            raise RuntimeError("weights already loaded")
        cfg = self._config
        expect = {
            "gate": (cfg.expert_num, cfg.intermediate_size, cfg.partial.gateup.rows),
            "up": (cfg.expert_num, cfg.intermediate_size, cfg.partial.gateup.rows),
            "down": (cfg.expert_num, cfg.hidden_size, cfg.partial.down.rows),
        }
        for name, t, shape in (
            ("gate", gate_proj, expect["gate"]),
            ("up", up_proj, expect["up"]),
            ("down", down_proj, expect["down"]),
        ):
            if (t.dtype != torch.bfloat16 or tuple(t.shape) != shape
                    or not t.is_contiguous() or t.device.type != "cpu"):
                # device 검증 필수: CUDA 텐서의 data_ptr()는 device 주소라
                # C++ host memcpy가 그대로 segfault한다 (조용한 죽음 방지)
                raise ValueError(
                    f"{name}_proj must be contiguous CPU bf16 {shape}, got "
                    f"{t.dtype} {tuple(t.shape)} device={t.device} "
                    f"contiguous={t.is_contiguous()}"
                )
        if physical_to_logical_map_cpu is None:
            physical_to_logical_map_cpu = torch.arange(cfg.expert_num, dtype=torch.int64)
        phys_map = physical_to_logical_map_cpu.to(dtype=torch.int64, device="cpu").contiguous()

        # pack이 읽는 동안 생존 보장 (계약 ③ 수명 규칙)
        self._pending = (gate_proj, up_proj, down_proj, phys_map)
        cfg.gate_proj = gate_proj.data_ptr()
        cfg.up_proj = up_proj.data_ptr()
        cfg.down_proj = down_proj.data_ptr()

        if sparsity_tables is not None:
            self._install_sparsity(cfg, sparsity_tables)

        self._moe = self._moe_class(cfg)  # config가 C++로 복사되는 순간
        self._cpuinfer.submit(self._moe.load_weights_task(phys_map.data_ptr()))
        self._cpuinfer.sync()  # pack 동기 완료
        self._pending = None  # 소스 텐서 해제 — 이후 소유권은 C++ (계약 ③)

    _SPARSITY_KEYS = (
        ("gate_wn_sq", "gateup", 1),
        ("gate_pair_dot", "gateup", 2),
        ("up_wn_sq", "gateup", 1),
        ("up_pair_dot", "gateup", 2),
        ("down_wn_sq", "down", 1),
        ("down_pair_dot", "down", 2),
    )

    def _install_sparsity(self, cfg, tables: dict) -> None:
        """테이블 검증 + config 포인터 설치. 참조는 self가 영구 보관한다."""
        missing = [n for n, _, _ in self._SPARSITY_KEYS if n not in tables]
        if missing:
            raise ValueError(f"sparsity_tables missing keys: {missing}")
        rows = {"gateup": cfg.partial.gateup.rows, "down": cfg.partial.down.rows}
        kept = {}
        for name, band, div in self._SPARSITY_KEYS:
            t = tables[name]
            want = (cfg.expert_num, rows[band] // div)
            if (t.dtype != torch.float32 or tuple(t.shape) != want
                    or not t.is_contiguous() or t.device.type != "cpu"):
                raise ValueError(
                    f"{name} must be contiguous CPU fp32 {want}, got "
                    f"{t.dtype} {tuple(t.shape)} device={t.device} "
                    f"contiguous={t.is_contiguous()}"
                )
            kept[name] = t
        # C++가 매 step 원시 포인터로 읽으므로 wrapper 수명 내내 붙잡는다.
        self._sparsity_tables = kept
        sp = cfg.partial.sparsity
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
    # thr_ptr: fp32 [qlen][k][3] (gate, up, down) pinned. 0 = dense.
    # sparse는 decode 전용이므로 qlen != 1에 thr을 주면 C++가 즉사시킨다.
    def submit_forward_gateup(self, qlen_ptr: int, k: int, expert_ids_ptr: int,
                              x_ptr: int, out_ptr: int, cuda_stream: Optional[int] = None,
                              thr_ptr: int = 0) -> None:
        task = self._require_loaded().forward_gateup_partial_task(
            qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr, thr_ptr)
        self._submit(task, cuda_stream)

    def submit_forward_down(self, qlen_ptr: int, k: int, expert_ids_ptr: int,
                            act_ptr: int, out_ptr: int, cuda_stream: Optional[int] = None,
                            thr_ptr: int = 0) -> None:
        task = self._require_loaded().forward_down_partial_task(
            qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr, thr_ptr)
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
                       thr: Optional[torch.Tensor] = None) -> None:
        qlen_t = torch.tensor([x.shape[0]], dtype=torch.int32)
        self._require_loaded().forward_gateup_partial(
            qlen_t.data_ptr(), expert_ids.shape[1], expert_ids.data_ptr(), x.data_ptr(), out.data_ptr(),
            0 if thr is None else thr.data_ptr()
        )

    def forward_down(self, expert_ids: torch.Tensor, act: torch.Tensor, out: torch.Tensor,
                     thr: Optional[torch.Tensor] = None) -> None:
        qlen_t = torch.tensor([act.shape[0]], dtype=torch.int32)
        self._require_loaded().forward_down_partial(
            qlen_t.data_ptr(), expert_ids.shape[1], expert_ids.data_ptr(), act.data_ptr(), out.data_ptr(),
            0 if thr is None else thr.data_ptr()
        )
