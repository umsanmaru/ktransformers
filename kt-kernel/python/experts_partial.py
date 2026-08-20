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

    # ── Stage 2: 로딩 (kt SFT 텐서 주입 경로와 동형) ─────────────────────
    def load_weights_from_tensors(
        self,
        gate_proj: torch.Tensor,  # bf16 [E, inter, gateup_k]  (K-슬라이스, ckpt 방향)
        up_proj: torch.Tensor,    # bf16 [E, inter, gateup_k]
        down_proj: torch.Tensor,  # bf16 [E, hidden, down_k]
        physical_to_logical_map_cpu: Optional[torch.Tensor] = None,
    ) -> None:
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
            if t.dtype != torch.bfloat16 or tuple(t.shape) != shape or not t.is_contiguous():
                raise ValueError(
                    f"{name}_proj must be contiguous bf16 {shape}, got "
                    f"{t.dtype} {tuple(t.shape)} contiguous={t.is_contiguous()}"
                )
        if physical_to_logical_map_cpu is None:
            physical_to_logical_map_cpu = torch.arange(cfg.expert_num, dtype=torch.int64)
        phys_map = physical_to_logical_map_cpu.to(dtype=torch.int64, device="cpu").contiguous()

        # pack이 읽는 동안 생존 보장 (계약 ③ 수명 규칙)
        self._pending = (gate_proj, up_proj, down_proj, phys_map)
        cfg.gate_proj = gate_proj.data_ptr()
        cfg.up_proj = up_proj.data_ptr()
        cfg.down_proj = down_proj.data_ptr()

        self._moe = self._moe_class(cfg)  # config가 C++로 복사되는 순간
        self._cpuinfer.submit(self._moe.load_weights_task(phys_map.data_ptr()))
        self._cpuinfer.sync()  # pack 동기 완료
        self._pending = None  # 소스 텐서 해제 — 이후 소유권은 C++ (계약 ③)

    def _require_loaded(self):
        if self._moe is None:
            raise RuntimeError("load_weights_from_tensors first")
        return self._moe

    # ── task 경로: (fn, args)를 CPUInfer로 제출 (graph-호환 배관) ─────────
    # 모든 포인터 인자는 정수 주소(intptr). qlen_ptr가 가리키는 int는
    # pinned·주소고정이어야 하며 값은 step마다 in-place 갱신 (계약 ④).
    def submit_forward_gateup(self, qlen_ptr: int, k: int, expert_ids_ptr: int,
                              x_ptr: int, out_ptr: int, cuda_stream: Optional[int] = None) -> None:
        task = self._require_loaded().forward_gateup_partial_task(qlen_ptr, k, expert_ids_ptr, x_ptr, out_ptr)
        self._submit(task, cuda_stream)

    def submit_forward_down(self, qlen_ptr: int, k: int, expert_ids_ptr: int,
                            act_ptr: int, out_ptr: int, cuda_stream: Optional[int] = None) -> None:
        task = self._require_loaded().forward_down_partial_task(qlen_ptr, k, expert_ids_ptr, act_ptr, out_ptr)
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
    def forward_gateup(self, expert_ids: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
        qlen_t = torch.tensor([x.shape[0]], dtype=torch.int32)
        self._require_loaded().forward_gateup_partial(
            qlen_t.data_ptr(), expert_ids.shape[1], expert_ids.data_ptr(), x.data_ptr(), out.data_ptr()
        )

    def forward_down(self, expert_ids: torch.Tensor, act: torch.Tensor, out: torch.Tensor) -> None:
        qlen_t = torch.tensor([act.shape[0]], dtype=torch.int32)
        self._require_loaded().forward_down_partial(
            qlen_t.data_ptr(), expert_ids.shape[1], expert_ids.data_ptr(), act.data_ptr(), out.data_ptr()
        )
