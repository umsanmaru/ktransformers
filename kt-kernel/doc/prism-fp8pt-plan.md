# per-tensor FP8 cold 커널 (branch `fp8-per-tensor`)

작성 2026-09-05. 대상: **Mistral-Medium-3.5-128B**(dense 88층, hidden 12288, ffn 28672)의
체크포인트 FP8 포맷 — HF `quantization_config`:

```json
{"quant_method": "fp8", "activation_scheme": "static", "weight_block_size": null,
 "modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector", "lm_head"]}
```

safetensors 실측: 선형층 weight는 `F8_E4M3 [N, K]`, `weight_scale_inv`와 `activation_scale`은
**BF16 스칼라(shape `[]`)**. 88층 × 7 proj(q/k/v/o/gate/up/down) = 616 텐서가 각각 배율 쌍을
가진다. DeepSeek/GLM/Qwen의 `[N/128, K/128]` fp32 블록표(`kt_tile_k2_fp8b128`)와 다르다.

## 1. kt-kernel 쪽 — 이 브랜치에서 한 것

| 파일 | 내용 |
|---|---|
| `operators/amx/la/tile_k2_fp8pt_port.hpp` (새) | `ktf8pt_gemv` — b128 포트의 순열/pack/plan/커서/디코더를 그대로 include, 소비 커널만 다르다: super(256 n) 누산기 16개를 전 K에 걸쳐 레지스터에 두고 마지막에 스칼라 배율 1회. 컬럼당 flush·배율표 로드·y memset 제거. zmm 스필 0 (asm 확인). |
| `operators/amx/la/tile_k2_fp8pt.hpp` (새) | `GemmKernelTileK2FP8PT` — `SCALE_BLK = 0`, `kPerTensorScale = true`, BufferB = codes 타일(b128과 같은 순열) + 64 B 배율 슬롯. 기하: n % 256, **k % 32** (128 제약 해제). |
| `operators/amx/tile-fp8-moe.hpp` | `TILE_FP8_MOE_TP<T>`를 배율 기하에 대해 일반화: `fp8_tile_per_tensor_v<K>` 트레이트, `derived_init`의 `group_size == T::SCALE_BLK` 검사(pt = 0), `scale_src()`(b128: 블록표 expert 시작 / pt: `[E]`의 e번째), `fp8_tile_tp_load_weights`의 pt 분기(proj당 `float[E]`, shard 절단 없음). b128 경로는 산술 동일. |
| `ext_bindings.cpp` | `TileK2FP8PT_MOE` 등록. |
| `python/experts_partial.py` | 키 `kt_tile_k2_fp8pt`; `_WEIGHT_FORMAT`에 `scale_div=None`(= 배율 `[E]`), `group_size=0`. |
| `operators/amx/test/tile_k2_fp8pt-gate.cpp` (새) | 헤더 전용 게이트: dense/mask1/sparse × k ∈ {32, 96, 160, 1024, 544, 12288, 2048}, 2^n·임의 배율. **전부 err/mag ≤ 5e-8**. |
| `operators/amx/test/tile_k2_fp8pt-bench.cpp` (새) | b128 vs pt 단일 스레드 마이크로벤치 (Mistral shard 형상). |
| `test/prism/test_partial_fp8pt.py` (새) | wrapper 경로: dequant fp64 레퍼런스, qlen 1/5, rows 416/224(비-128 정렬), expert별 배율 부착, 배율 모양 위반 즉사, b128 대조 거절, sparse thr=0 비트일치 / 큰 thr 0 / 중간 thr 마스크 레퍼런스. |
| `operators/amx/la/tile_k2_fp8b128.hpp` | include 순서만 교정(`amx_raw_buffers.hpp` → `act_plan.hpp`). 문서에 적힌 standalone 게이트 빌드가 `ggml_bf16_t` 미정의로 깨져 있었다. |

`activation_scale`(static per-tensor 활성 배율)은 **CPU 커널이 쓰지 않는다** — W8A16(bf16 활성 ×
정확 e4m3)이므로 활성을 e4m3로 재양자화하지 않는다. 프리즘의 GPU 티어 커널(`prism_gemv_fp8`,
`prism_grouped_fp8`)도 같은 W8A16이라 **티어 간 비트일치 계약은 per-tensor에서도 성립한다**(정확표현
입력, 2^n 배율). static 활성 양자화를 쓰는 것은 plan에 안 든 층의 sglang 원본 `Fp8LinearMethod`뿐이다.
GPU/sglang 확장 계획: `prism-fp8pt-gpu-plan.md`.

### 1.1 마이크로벤치 (2026-09-05, Xeon 6534 코어 1개 pin, HW pf 상태 불문)

| 형상(shard) | 조건 | b128 | pt | pt/b128 |
|---|---|---|---|---|
| gateup 3584×3072 | DRAM(24사본) dense / sp50 | 957 / 514 µs | 943 / 501 µs | 0.986 / 0.974 |
| down 1536×7168 | DRAM dense / sp50 | 960 / 506 µs | 958 / 506 µs | 0.999 / 1.001 |
| o_proj 1536×3072 | DRAM dense / sp50 | 406 / 221 µs | 414 / 215 µs | 1.018 / 0.974 |
| 셋 | 2사본(L3 근접) | — | — | 0.975~0.996 |

**결론: 속도는 동률(−1~−2.5 %, 잡음 수준).** 코어당 11~19 GB/s로 메모리 바운드라 flush/배율표
제거는 보이지 않는다. 이 커널의 값어치는 (a) 포맷이 맞다는 것(블록표 broadcast 없이 체크포인트
그대로), (b) **K 정렬이 32로 풀려** 티어 절단이 bf16처럼 자유롭다는 것이다. b128 커널에 배율을
broadcast해서 돌려도 속도는 같았을 것이고, 그 경우 k_align=128이 남는다.

### 1.2 kt 경로 다중 스레드 (2026-09-05, PartialMoEWrapper + CPUInfer, HW pf ON 상태, 6슬롯 회전 = L3 밖)

Mistral cold 형상: gateup K=3072(hidden/4)·N=28672(gate+up 176 MB/슬롯), down K=7168(ffn/4)·N=12288(88 MB/슬롯).
벤치 스크립트: `test/prism/bench_fp8pt_mt.py --threads 28 --experts 6`. **주의: 같은 시간에 이 박스에서
다른 sglang 서버(328 GB RES, ~8코어)가 돌고 있었다** — 절대치는 그 영향을 받았고, 비율은 인터리브 측정이라 유효.

| 스레드 | 진입점 | 조건 | b128 | pt | pt/b128 | 유효 GB/s |
|---|---|---|---|---|---|---|
| 28 (14/NUMA) | gateup | dense | 1138 µs | 1133 µs | 0.995 | 155 |
| 28 | gateup | sp50 | 575 | 572 | 0.994 | 154 |
| 28 | down | dense | 659 | 652 | 0.991 | 135 |
| 28 | down | sp50 | 331 | 329 | 0.993 | 134 |
| 14 (7/NUMA) | gateup | dense | 1295 | 1278 | 0.987 | 137 |
| 14 | gateup | sp50 | 670 | 658 | 0.982 | 133 |
| 14 | down | dense | 718 | 705 | 0.981 | 124 |
| 14 | down | sp50 | 356 | 370 | 1.04 | 120 |

14스레드 down sp50은 두 번 재서 0.715 / 1.04가 나왔다 — p10과 중앙값이 100 µs 벌어진 잡음(공유 박스). 나머지는
**0.98~1.00으로 동률**. 28스레드가 14보다 10~15 % 빠르고(HT 활용), 두 소켓 합 155 GB/s가 이 형상의 상한이다.
sp50에서 시간이 정확히 절반 → 유효 대역 불변: 페어 skip이 두 커널에서 같게 작동하고 pf-ON 과다 fetch 세금도 같다.

단일 코어 조사(`bw_probe`): 커널 자체의 계산 상한은 L2 상주에서 **20 GB/s/코어**(fp8→bf16 디코드 + dpbf16 + 커서),
같은 코어의 순수 스트리밍 읽기도 SW pf 128라인일 때 20 GB/s(pf 없으면 8.9) — 단일 코어 DRAM 11.7 GB/s는
둘의 곱이 아니라 디코드 스트림에 지연이 겹친 결과다. 코어 8개×2소켓이면 이 상한이 소켓 대역 위로 올라가
결국 메모리 바운드가 된다. b128의 flush 제거는 어느 조건에서도 안 보인다.

## 2. sglang 쪽 — 남은 일 (별도 worktree; `sglang-dense`/`prism-dense` 레인)

dense 레인 TODO.md §6이 이미 적어둔 것: "Mistral-Medium은 per-tensor라 별도 포맷이 필요하고,
`Fp8LinearMethod`는 `WEIGHT_LOADER_V2_SUPPORTED`에 있어 래핑하면 로더가 조용히 v1으로 떨어진다".

1. `srt/layers/prism/kernels.py` — `_CPU_COLD_KERNELS += kt_tile_k2_fp8pt`, slab layout 태그
   `kt_tile8pt`, `n_align 256`, `tile_rows 32`.
2. `srt/layers/prism/linear/formats.py` — `Fp8PerTensorLinearFormat`: `k_align = PAIR_GROUP`,
   `has_scales = True`, `check`: weight e4m3 `[N, K]`, scale **스칼라**(BF16 → fp32 변환 허용),
   `gather`: codes만 자르고 scale은 그대로, `cold_flat`: k_pad를 32로 올리고 scale `[1]`.
   wrapper에 넘길 때 그룹의 슬롯(=expert)마다 `[E]`로 stack.
3. `linear/cold_backend.py::_load_group` — 배율 텐서를 `load_weights_from_tensors(..., gate_scale=,
   up_scale=, down_scale=)`로 전달 (지금은 bf16만 가정하고 배율을 넘기지 않는다). 더미 슬롯(gate/up
   또는 down)의 배율은 1.0.
4. GPU 티어: hot/warm은 sglang의 per-tensor fp8 GEMM(`Fp8LinearMethod`, static activation) 그대로.
   cold prefill을 GPU가 slab을 제자리 읽게 하려면 `prism_grouped_fp8.cuh`에 `Layout::KT_TILE8_PT`
   (코드 주소식 = KT_TILE8, 배율 = expert 블록 끝 64 B 슬롯의 float 1개) — 코드 로더 공유, 배율
   조회만 다르다.
5. 스토어 정합 검사(`store_format`)에 `gpu_store_format_tag` ↔ `kt_tile8pt` 쌍 추가.
6. calib 자산 `assets/mistral_med35.pt`, `mistral_med_128B.pt`가 이미 있다 — `k2wl2` 점수 재료가
   들어 있는지 확인 후 sparse 배선.

### 2.1 진행 (2026-09-05 밤)

- sglang worktree `/home/um3maru/prism-sglang/sglang-fp8pt` (branch `fp8-per-tensor`, **prism-orchestration** 994a7f8e3f에서 분기 —
  사용자 결정; 위 "prism-dense 레인" 대신). 이 worktree의 kt는 prism-e2e env로 CUDA 포함 `build_ext --inplace`
  (`python/kt_kernel_ext*.so`, `submit_with_cuda_stream` 있음), shim `/tmp/claude-20048/fp8ptshim/kt_kernel -> …/python`.
- §2-1 완료: `srt/layers/prism/kernels.py` 네 표에 `kt_tile_k2_fp8pt` (layout `kt_tile8pt`, n_align 256, tile_rows 32).
  §2-2~5(포맷·`_load_group` 배율·GPU 로더·정합 검사)는 미착수.
- **프로파일러 확장** (`srt/layers/moe/prism/profile`): `Store("fp8pt")`(cold 전용, `gpu=False`, 배율 `[E]`, k_align 2/타일 32),
  `N_ALIGN` 키, `ColdCpuProfiler(sparsity=None)` = kt dense 경로(테이블 없음), `m` 토큰 축(qlen; sparse는 decode 전용이라 m>1은 dense만),
  `cold_sparse_gemv(..., proj=, m=)` — dense 레인의 cold(expert=슬롯, top_k=1, 테이블 없음)를 그대로 흉내. `bench_cold_cpu.py --dtype fp8pt
  --sparsity none --m 64`. 테스트 `test_profile_dtype.py` fp8pt 4건, 전체 84 passed.
- **실측 (Xeon 6534 소켓0 16 thr, E=8 슬롯 회전, K16384×N5120 down, 공유 머신이라 절대치 흔들림):**
  m=1 fp8 1480 / fp8pt 1455 / bf16 2831 µs; **m=64 fp8·fp8pt ≈ 1107 µs/tok vs bf16 113 µs/tok**. fp8 타일 커널은 m>1이
  **토큰 루프**다(`tile-fp8-moe.hpp:12,107` — prefill은 GPU KT_TILE8 로더가 맡는 설계). 즉 dense 레인에서 CPU cold로 prefill 청크를
  돌리면 fp8pt는 bf16 대비 토큰당 ~10×이고, 이는 fp8pt 고유가 아니라 fp8 타일 계열 공통. Mistral dense 레인의 cold prefill은
  GPU 제자리 읽기(§2-4 `KT_TILE8_PT`)가 전제여야 한다.
- **GPU 티어 연결 (2026-09-06):** `Store("fp8pt")`가 `common.Fp8PtGpu`(gemv_fp8pt_* 8개, MoE 포맷 아님)를 들어 hot/warm/full_layer도 됨; k_align 32(GPU row_off·kt 타일), GPU 스토어 배율 `[E]`. `test_profile_dtype`/`test_full_layer` 34 passed. E16 topk4 H3072 I1024 sp0.5 8 thr: gateup warm_only b128 39 / pt 41 µs, cold_only 146 / 142, combined 204 / 150(노이즈); down warm 28 / 34, cold 69 / 68; rel err 3e-3 양쪽.
- 노이즈 주의: 같은 측정이 1.4~6.0 ms로 튀는 구간이 있었다(다른 사용자 프로세스). 비교는 인터리브·같은 세션에서만.

## 3. 검증 순서

1. `tile_k2_fp8pt-gate` ✅ (헤더 전용)
2. `test/prism/test_partial_fp8pt.py` — env `ktk-fp8pt`(ktk-prism 클론)에 이 worktree를 설치해 실행
3. `test/prism/test_partial_moe.py` 회귀(bf16 경로 무변경 확인)
4. sglang 배선 후 `test/prism/test_linear_cold.py`에 fp8pt 케이스, 실모델 Mistral-Medium-3.5 e2e
