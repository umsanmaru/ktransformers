# per-tensor FP8 — GPU 티어(hot/warm/cold-prefill) + sglang dense 레인 확장 계획

작성 2026-09-05. 전제: kt 쪽 `kt_tile_k2_fp8pt`는 끝났다(`prism-fp8pt-plan.md` §1). 지금 상태로는
plan에 이 커널을 적어도 **hot/warm 티어가 계산할 GPU 커널이 없고**, dense 레인 `method.py`가
`Fp8LinearMethod`를 감싸는 것 자체를 거절한다. 이 문서는 그 두 공백을 메우는 순서다.

작업 트리: sglang 포크 `sglang/`(source of truth) → `scripts/port_prism_to_upstream.sh`로
`sglang-dense/`에 이식. GPU 커널은 `python/sglang/jit_kernel/csrc/moe/`, 레인 코드는
`python/sglang/srt/layers/prism/{kernels.py, linear/}`. 별도 worktree(`sglang-fp8pt`)를 권장.

## 0. 조사에서 나온 사실 (설계를 정하는 것들)

**(a) 프리즘 안에서는 모든 티어가 W8A16이다.** hot/warm GPU gemv·grouped 커널도 `bf16(x) ×
e4m3_val(code) × scale`을 fp32로 누산한다(`prism_fp8.cuh`). 체크포인트의 static `activation_scale`
(`input_scale`)은 sglang 원본 `Fp8LinearMethod`만 쓴다. 따라서 **per-tensor 판에서도 티어 간
비트일치 계약(⑤)은 그대로 성립한다** — 정확표현 입력에서 CPU cold ≡ GPU gemv ≡ grouped.
(앞서 kt 문서에 "GPU static-fp8 티어와 다르다"고 적은 것은 plan에 안 든 층에 대한 말이다.)

**(b) sglang per-tensor `Fp8LinearMethod`의 파라미터 형태.** 블록 양자화는 `weight_scale_inv
[N/128, K/128]`이지만, per-tensor는 `weight_scale = PerTensorScaleParameter(len(output_partition_sizes))`
— **fused 모듈(qkv_proj, gate_up_proj)은 shard당 스칼라 하나씩**이다(`fp8.py:431` vs
`PerTensorScaleParameter`). 즉 qkv_proj는 `[3]`, gate_up_proj는 `[2]`, o_proj/down_proj는 `[1]`.
그리고 `process_weights_after_loading`이 fused 모듈에서 cutlass/marlin이면 `convert_to_channelwise`
(무손실, 배율만 `[N]`으로 펼침), 아니면 `requantize_with_max_scale`(**코드를 다시 양자화 — 손실**).
프리즘은 이 후처리 **전**의 원시 코드 + shard별 스칼라를 가져가야 한다. 지금 `method.py:243`은
`layer.weight.data`만 읽고 배율은 안 읽는다.

**(c) v2 로더 문제의 실체.** `ColumnParallelLinear.__init__`이
`self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED`로 로더를 고른다. 래퍼 이름이
"PrismLinearMethod"라 v1로 떨어지고, v2 전용 파라미터(`ModelWeightParameter`, `PerTensorScaleParameter`)의
shard 로딩이 어긋난다. 해법은 두 가지 중 하나 — ① `linear.py`의 `WEIGHT_LOADER_V2_SUPPORTED`에
"PrismLinearMethod"를 추가(레인 트리는 이미 `port_prism_to_upstream.sh`가 linear.py에 앵커를 심으므로
같은 방식으로 한 줄), ② 래퍼 클래스를 inner 이름으로 동적 생성. ①을 택한다 — 이름 흉내는 다른
`__name__` 판정에도 걸릴 수 있다. 단 v2 로더는 파라미터가 v2 타입이어야 하므로 **`create_weights`를
inner(`Fp8LinearMethod`)에 위임**해야 한다(지금은 bf16용 자체 생성).

**(d) K 정렬.** fp8 gemv는 32-k 청크 단위(`kChunk=32`, `cnt % 32 == 0` 전제), grouped는 kBK=64이되
`k < kr` 가드가 있어 32 단위 부분 타일을 이미 처리한다(ROWMAJOR·KT_TILE8 둘 다). kt는 k % 32. →
**per-tensor 포맷의 `k_align = 32`**, cold `tile_rows = 32`. (bf16의 2까지 내리려면 gemv 청크를
바꿔야 하는데, 얻는 게 없다.)

**(e) Mistral 체크포인트 이름.** safetensors는 `weight_scale_inv`(스칼라)·`activation_scale`인데 sglang
파라미터는 `weight_scale`·`input_scale`이다. 어디서 이름이 바뀌는지 아직 못 찾았다(`ministral3.py`에는
없음) — **구현 첫날 확인 항목**. 못 찾으면 `PrismLinearMethod`가 `weight_loader`에서 두 이름을 다 받게 한다.

## 1. GPU 커널 — **완료 (2026-09-05, worktree `sglang-fp8pt`, 미커밋)**

구현 결과: `prism_gemv_fp8.cuh`에 `PT` 템플릿 인자(진입점 `gemv_fp8pt_*` 8개, 배율 [E], 합산 뒤 1회 곱),
`prism_grouped_fp8.cuh`에 `ROWMAJOR_PT=2`·`KT_TILE8_PT=3`(진입점 `grouped_fp8pt_*` 4개, `grouped_fp8_cold`가
layout 3 수용). 테스트 `test/prism/test_fp8pt_kernels.py` 37개 통과(비-128 정렬 k_rows 96/160, 부분 K 타일,
pt≡b128 균일배율 비트일치, 비-2^n 배율, kt_tile8pt slab 로더), 기존 `test_fp8_kernels.py` 21개 회귀 통과.
`fp8_ref.py`에 `*_pt` 헬퍼 추가. 아래 §1.1~1.3은 설계 기록으로 남긴다.

### 1.1 warm/hot decode — `prism_gemv_fp8.cuh`

배율 조회가 청크마다 `scales_e[(krow/128)·nblk + nblock]`인 것을 per-tensor에서는 슬롯당 스칼라 1개로
바꾸고, 곱은 **누산 끝에 1회**로 옮긴다(kt 커널과 같은 구조 → 같은 값).

- 템플릿 인자 `bool PER_TENSOR` 추가. `F8Slot.scales`의 의미: 블록판 `[Σk/128, N/128]` / pt판 `[E]`
  (expert = 슬롯 = N-part). 커널 본체 분기는 두 줄: 청크 루프 안 `acc += accb·sc` → `acc += accb`,
  루프 뒤 `acc *= scales_e[0]`(pt만).
- **정확도 주의**: 블록판은 `Σ_chunk (accb·sc)`, pt판은 `(Σ accb)·sc` — 배율이 2의 거듭제곱이면
  둘이 비트동일이고, 임의 fp32면 마지막 라운딩 하나가 다르다. grouped/kt와의 비트일치 테스트는
  exact 픽스처(2^n 배율)로만 건다 — 기존 `test_fp8_kernels.py`의 `exact` 축 그대로.
- `verify_f8_store`에 pt 분기: scales `[E]` fp32(device 또는 pinned), `kidx` 행 수 `% 32 == 0`.
- 진입점 8개 복제: `gemv_fp8pt_indexed[_pinned][_sparse][_gateup]`. 정적 배열 `_WRAPPERS`에 추가하고
  `prism_gemv_fp8.py`에 같은 팩토리로 노출 (별 파일 `prism_gemv_fp8pt.py` 대신 같은 모듈에 두면 JIT
  1회).
- SPARSE 경로(`pair_energy`, `keep[]`)는 배율과 무관 — 무변경.
- 타일 선택 launcher(`PRISM_F8_LAUNCH`, kV/kNX/kNY)는 공유. 제약 `kNCol ≤ 128, 128 | kNCol`은 pt에선
  불필요하지만 유지(코드 공유가 더 값짐).

### 1.2 prefill — `prism_grouped_fp8.cuh`

`enum Layout { ROWMAJOR=0, KT_TILE8=1, ROWMAJOR_PT=2, KT_TILE8_PT=3 }`.

- `load_b`의 `sreg`: ROWMAJOR_PT → `s.scales[e]`(상수, K 타일과 무관); KT_TILE8_PT →
  `*reinterpret_cast<const float*>(blk + align64(n_cols·kr))` (kt `BufferB::scale_offset` = 코드 끝 64 B
  슬롯의 첫 float). 코드 주소식(`t8_super`, `kColBytes`, 16 B 조각 매핑)과 `store_b`는 KT_TILE8과
  **완전히 같다** — 배율 소스만 갈린다. 구현은 `constexpr bool kTile = LAYOUT & 1, kPT = LAYOUT >= 2`.
- `kr` 정렬 검사 "128 배수" → pt는 32 배수. 부분 K 타일(kr % 64 == 32)은 기존 가드가 처리하나
  **테스트로 못 박을 것**(§3-2).
- host 진입점 `grouped_fp8_impl`의 `layout` 분기와 `launch(...)` 4종, `grouped_fp8_cold[_gateup]`의
  layout 검사 확장. `prism_grouped_fp8.py`: `COLD_LAYOUTS["kt_tile8pt"] = 3`, ROWMAJOR_PT용 wrapper
  `grouped_fp8pt_indexed[_pinned][_gateup]`.

### 1.3 공용 — `prism_fp8.cuh`

디코드(`e4m3_val`)는 그대로. 주석의 "128×128 블록" 문구를 "블록 또는 per-tensor" 로 고치고
`kBlk`는 블록판 전용임을 적는다.

## 2. sglang dense 레인

### 2.1 `srt/layers/prism/kernels.py`
- `_GPU_WARM_KERNELS += "gemv_worklist_fp8pt"`, `_GPU_STORE_FORMAT["gemv_worklist_fp8pt"] = "fp8pt"` — **남음**.
- CPU 쪽 키(`kt_tile_k2_fp8pt` / `kt_tile8pt` / 256 / 32)는 다른 세션이 이미 넣었다(2026-09-05, 미커밋).
  그 주석의 "GPU 로더(KT_TILE8_PT)는 아직 없다"는 이제 틀렸다 — 갱신할 것.

### 2.2 `linear/formats.py` — `Fp8PerTensorLinearFormat` (name `"fp8pt"`)
- `k_align = 32`, `has_scales = True`, `cold_layouts = ("kt_tile8pt",)`.
- `check(weight, scale, k, n)`: weight e4m3/u8 `[N, K]`, `k % 32 == 0`(n은 GPU gemv 열 타일 정렬만),
  scale **`[1]` fp32**(호출자가 shard 스칼라를 fp32 `[1]`로 만들어 넘긴다; BF16 스칼라는 여기서 변환).
- `_scale_rows` 없음. `gather` = codes만 `_gather_km`, scale은 그대로 `[1]`. `cold_flat`: `k_pad`를
  `tile(32)`로 올리고 scale `[1]` 유지(패딩 코드 0x00 = +0.0이라 배율 무관).
- `store_args` → `(w_flat, s_flat)` 그대로(gemv 규약 동일). `gemv()`/`grouped()`는 `*_fp8pt_*` 진입점.
- `store_format(gpu_warm, cpu_cold)` 정합: `"fp8pt"` ↔ `kt_tile8pt`만 허용. `fp8`(블록) ↔ `kt_tile8pt`
  조합은 즉사 — 배율 표와 스칼라를 섞으면 조용히 틀린다.
- 모듈 docstring의 표에 열 추가: `fp8 (per-tensor)`: 코드 `[N, K]` e4m3, 배율 `[1]`(shard당), K 정렬 32,
  cold 커널 `kt_tile_k2_fp8pt`.

### 2.3 `linear/weights.py`
- `prepare_linear_weights(..., scale=)`: 지금은 `[N/128, K/128]`를 기대하고 `_prepare_part`가
  `scale[part.n_start//128 : part.n_end//128]`로 N-part를 자른다. pt에서는 **scale이 `[n_parts]`**(shard당
  스칼라)이고 part i가 `scale[i:i+1]`을 받는다. `halves`(N-part) 경계가 `output_partition_sizes`와 일치해야
  배율이 한 part 안에서 하나다 — `plan.check_partition`의 "포함" 완화는 pt에서 **"일치"로 되돌려야** 한다
  (part가 shard 둘을 가로지르면 배율이 둘). 포맷에 `requires_shard_aligned_parts = True` 플래그를 두고
  `check_partition`이 그것을 본다.
- `PreparedPart.scale` 주석 갱신.

### 2.4 `linear/method.py`
- `_V2_LOADER_UNSUPPORTED`에서 `Fp8LinearMethod` 제거 + linear.py `WEIGHT_LOADER_V2_SUPPORTED`에
  "PrismLinearMethod" 추가(포크 + `port_prism_to_upstream.sh` 앵커).
- `create_weights`: inner가 `Fp8LinearMethod`면 `inner.create_weights(layer, ...)`에 위임 → `weight`,
  `weight_scale [n_shards]`, `input_scale [n_shards]`가 v2 파라미터로 생긴다. bf16은 기존 경로 유지.
- `process_weights_after_loading`: inner 후처리를 **호출하지 않고** `layer.weight.data`(원시 코드)와
  `layer.weight_scale.data`(shard 스칼라, CPU fp32)를 `prepare_linear_weights`에 넘긴다. `input_scale`은
  버린다(프리즘 티어는 전부 W8A16). 완료 후 `weight_scale`/`input_scale`도 빈 텐서로 소멸.
- 블록 fp8(`weight_scale_inv`)과 per-tensor(`weight_scale`)를 `inner.block_quant`로 구분해 각 포맷에
  맞는 scale 인자를 만든다.
- (e)의 이름 매핑 확인: `weight_scale_inv`가 `weight_scale`로 들어오는 경로를 찾아 적는다.

### 2.5 `linear/cold_backend.py`
- `_load_group`: bf16 전제라 배율을 안 넘긴다. `key.kernel`의 포맷이 `has_scales`면 그룹 슬롯별로
  `[E]` fp32를 stack해 `load_weights_from_tensors(..., gate_scale=, up_scale=, down_scale=)`. 더미 슬롯
  (gateup 그룹의 down / down 그룹의 gate·up)의 배율은 `torch.ones(E)`.
- `_dummy_weight`: dtype이 u8이면 코드 0x00.
- `GroupKey`에 포맷이 이미 커널 키로 함의되므로 변경 없음.

### 2.6 `linear/executor.py` / `tiers.py`
- `LinearGpuTier.run`은 `s.fmt.gemv/grouped`를 통해 호출하므로 무변경. cold GPU prefill(kt slab을
  `cudaHostRegister`로 읽기)은 dense 레인에 **아직 없다**(TODO §6 "cold의 prefill"). 지금 dense cold
  prefill은 CPU 토큰 루프다. `KT_TILE8_PT` 로더는 그 경로가 생길 때 쓰이므로 §1.2의 KT_TILE8_PT는
  **2순위**(ROWMAJOR_PT가 1순위).

### 2.7 plan / 컨버터
- `plan.py`: `kernels.gpu_warm = "gemv_worklist_fp8pt"`, `cpu_cold = "kt_tile_k2_fp8pt"`. `expected_calib`
  등은 포맷 무관.
- `scripts/compiled_plan_to_prism.py`: Mistral 프리셋(`COLD_N_ALIGN["kt_tile_k2_fp8pt"] = 256`, K 정렬
  32, dense 레인 좌표 `(layer, proj)`).

## 3. 테스트 (게이트 순서)

1. `test/prism/fp8_ref.py`: `random_expert_ckpt_pt(N, K, exact)`(스칼라 배율), `row_store_pt`, `tile_block_pt`
   (`ktf8_off` 재사용 + 64 B 슬롯 float).
2. `test/prism/test_fp8_kernels.py` pt 케이스: gemv ≡ dequant 레퍼런스(inexact tol), gemv pinned ≡ device
   비트일치, gateup fused 비트일치, sparse thr0 ≡ dense 비트일치, **grouped ROWMAJOR_PT ≡ gemv 비트일치
   (exact)**, `kr % 64 == 32`인 부분 K 타일 케이스, KT_TILE8_PT 로더 ≡ gemv.
3. `test_linear_weights.py`: `Fp8PerTensorLinearFormat` check/gather/cold_flat, `store_format` 정합 즉사,
   shard 경계 안 맞는 halves 거절.
4. `test_linear_cold.py`: fp8pt 그룹 로딩 + M ∈ {1, 2, 5, 17} 비트일치(정수 픽스처) — kt는 이 worktree
   빌드를 `prism-dense` env에 설치해야 한다(`scripts/build_prism_dense_env.sh`의 kt 경로를
   `ktransformers-fp8pt`로).
5. `test_linear_registry.py`: `Fp8LinearMethod` 래핑 허용 + v2 로더 선택 확인(가짜 `PerTensorScaleParameter`
   shard 로딩이 올바른 슬롯에 들어가는지).
6. e2e: Mistral-Medium-3.5-128B. **이 박스에 모델이 없다**(134 GB, 디스크 2개 한도) — 다운로드 자리부터
   확인. plan 예시: hot 0 / warm 15 % / cold 85 %. 기대치: cold 155 GB/s(28스레드) 기준 85 %×128 GB ≈
   0.7 s/tok, warm 15 %×128 GB / 47 GB/s ≈ 0.4 s → 2-phase max라 **~0.7 s/tok, sparse 없이**. sparse
   자산 `assets/mistral_med35.pt`/`mistral_med_128B.pt` 확인 후 k2wl2 배선.

## 4. 순서와 규모

| 단계 | 파일 | 규모 | 게이트 |
|---|---|---|---|
| G1 ✅ | `prism_gemv_fp8.cuh/.py` PER_TENSOR | 완료 | 테스트 2 (gemv 계열) 통과 |
| G2 ✅ | `prism_grouped_fp8.cuh/.py` ROWMAJOR_PT | 완료 | grouped ≡ gemv exact 통과 |
| L1 | `kernels.py`, `formats.py` Fp8PerTensorLinearFormat | ~150줄 | 테스트 3 |
| L2 | `method.py` v2 로더 + create_weights 위임 + scale 인출, linear.py 앵커 | ~80줄 | 테스트 5 |
| L3 | `weights.py` shard 스칼라 파트 매핑, `check_partition` 일치 모드 | ~60줄 | 테스트 3 |
| L4 | `cold_backend.py` 배율 전달 | ~40줄 | 테스트 4 (kt 재빌드 선행) |
| G3 ✅ | `KT_TILE8_PT` 로더 | 완료(G2와 함께) | slab 로더 ≡ gemv 통과; dense cold GPU prefill 경로는 여전히 없음 |
| E | plan/컨버터 프리셋, 모델 확보, e2e | — | tok/s, 출력 정합 |

G1·G2와 L1은 독립이라 병행 가능. L2가 가장 위험한 지점(sglang 로더 내부와 맞물림) — 먼저 가짜
레이어로 v2 shard 로딩을 검증하고 나서 나머지를 붙일 것.

## 5. 열린 것

- (e) 체크포인트 배율 이름 매핑 위치.
- `check_partition`을 pt에서만 "일치"로 되돌릴지, 전체를 되돌릴지(TODO §6에 이미 "포함 완화의 구멍"이
  적혀 있다 — 전체 되돌림이 맞아 보이나 Qwen3.8 plan이 깨지는지 확인).
- Mistral vision tower / projector는 BF16이고 plan 좌표(`split_prefix`)가 `model.language_model.layers.N`
  접두를 알아보는지 — dense 좌표 파서 확인.
- 5090 32 GB에서 KV 256K 컨텍스트는 불가능 — e2e는 짧은 컨텍스트로.
