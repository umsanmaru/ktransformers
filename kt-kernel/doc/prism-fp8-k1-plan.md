# blocked-fp8 k=1 커널 계획 (branch `blocked-fp8-k1`)

작성 2026-09-05. 대상: GLM 5.2 fp8 / GLM 5.3 flash가 cold 티어로 쓰는
`kt_tile_k2_fp8b128`(e4m3 + 128×128 fp32 `scale_inv`)의 **k=1 변종**.

"k=1"은 cpu-mm-platform FINDINGS §11/§12/§15의 용어다 — 64 B 라인 하나가 **k 한
행 × 64 n**이고, 활성값을 fp32로 broadcast해 FMA로 누산하며, **skip 단위가 k 한
행**인 커널. 지금 커널(k=2)은 라인이 VNNI 페어(k, k+1)를 나르고 `dpbf16`으로
누산하며 skip 단위가 페어다.

## 0. 왜 하는가 / 어디서 이기는가 (측정 근거)

| 조건 (fp8, N=4096, cpu-mm dev box) | k1 | k2 | 출처 |
|---|---|---|---|
| dense | 0.718 | **0.601** | §15 (HW pf ON, nb=N) |
| 페어 단위 sp50 | 0.369 | **0.318** | §15 |
| **per-k sp50** | **0.371** | 0.468 | §15 |
| 같은 바이트(survival-matched) | k2가 0~6 % 우세 | | §11 |

결론(§11): **sparsity가 k 단위로 정해지면** k1이 `(1−s²)/(1−s)`만큼 이기고, 페어
단위로 정해지면 k2가 이긴다. 즉 k1 커널의 값어치는 **per-k 마스크**와 함께 올 때만
있다. 현재 Prism 계약은 페어 마스크(`k2wl2`)라, k1 커널만 넣으면 dense·sparse 모두
손해다(§15 2026-08-28 결정 "fp8은 k2 유지, k1은 per-k 소비자 전용").

→ 이 브랜치의 범위는 **커널 + per-k 마스크 경로** 둘이다. 페어 마스크 그대로 k1
레이아웃만 바꾸는 건 의미가 없다.

## 1. 지금 코드의 계약 지도 (무엇이 어디에 고정돼 있나)

### 1.1 kt-kernel 쪽 — Seam A/B/C (`operators/amx/la/act_plan.hpp`)

| Seam | 소유 | 현재 형태 | k1에서 |
|---|---|---|---|
| A (슬랩 진입) | operator | `run_gemv(m,n,k,ba,bb,bc,ith,nth,plan)` → `T::gemv_slab` / `T::gemv_row` | 시그니처 유지 |
| B (인코더) | 커널 | `T::plan_arena_bytes(k)`, `T::build_plan(const uint16_t* mask, x, k, arena, out)` → `ActivationPlan{koff,pval,seg,npairs,nseg}` | 마스크 타입·`pval` 의미가 바뀐다 |
| C (소비) | 커널 | `ktf8_gemv(W,S,koff,pval,seg,npairs,N,K,y,n0,n1,pf)` | 새 본체 |

moe_base가 커널에 요구하는 정적 인터페이스(덕 타이핑, `if constexpr`/`requires`):

- `T::M_STEP, N_STEP(32), K_STEP(32), N_BLOCK(256), K_BLOCK, SCALE_BLK(128), ELEMENT_SIZE, name(), config()`
- `T::recommended_nth(n)`, `T::split_range_n(n, ith, nth)` — 워커 슬라이스는 whole super
- `T::BufferA/BufferB/BufferC` + `required_size` — `BufferB::from_raw_mat/from_scales/from_mat`
- `kHasPlanEncoder = requires { T::plan_arena_bytes(1); }` (moe_base.hpp:110) → plan arena를 expert별로 잡고 `build_plan_for<T>` 호출
- `plan_takes_slab_n<K>` (act_plan.hpp) — 6인자 `build_plan`이면 슬랩 n을 넘김
- `cold_slab_view(proj)`가 `T::N_STEP/K_STEP/N_BLOCK/K_BLOCK`을 GPU에 내보낸다 — sglang `cold_gpu.py:122`가 **(32,32) 고정 검사**, tile 계열은 `n_block==256` 검사

### 1.2 마스크 계약 ① — 페어에 박혀 있는 곳

`k2wl2`: `imp_j = sqrt(a[2j]x0² + a[2j+1]x1² + 2c[j]x0x1) ≥ thr`, 마스크 `uint16 / 32-k`, 비트 j = 페어 j.

kt-kernel:
- `operators/amx/pair_mask.hpp` `build_pair_mask` — AVX-512, 연산 순서가 수치 계약(비트 동일성 골든)
- `moe_base.hpp:122-124` `std::vector<std::vector<uint16_t>> m_pair_mask_*` (크기 `k/K_STEP`), `:312-314` 할당
- `moe_base.hpp:704-707, 796-797` `sp.*_wn_sq`(a), `sp.*_pair_dot`(c) 포인터 소비; `:741,757,818 prefetch_prep_inputs(..., a, c, sparse)`
- `moe_base.hpp:747,769,779,824` `build_pair_mask(...)` → `build_plan_for<T>(mask, ...)`
- `operators/common.hpp:331-351` `PartialGeometry::SparsityGeometry` — `*_wn_sq`, `*_pair_dot`, `thr_*`
- `operators/moe-tp.hpp:331,374` — `pair_dot` 포인터 null 검사(필수 취급)
- `ext_bindings.cpp:873-877` 파이썬 속성, `python/experts_partial.py:178-182` 테이블 주입 표(`(name, proj, div)`)
- `tests/prism_partial_golden.py`, `test/prism/test_partial_moe.py:396-413,566-605` 페어 에너지 레퍼런스

sglang prism:
- `CONTRACTS.md` ① "score 변종은 `k2wl2` 하나로 고정 (2026-08-24 사용자 결정)", `PAIR_GROUP=2` 정렬 규칙의 **근거 두 개**가 (a) k2wl2 재구성, (b) cold 커널 skip 단위 = VNNI 페어
- `plan.py:78 KNOWN_SPARSITY_SCORES=("k2wl2",)`, `:148 pair_dot` shape, `:420 K % PAIR_GROUP`
- `calib.py:43-57` 논리명→자산 키 (`tg2l/tu2l/td2l`, `wn_*`, `cg/cu/cd`), `_SUPPORTED_SCORES`
- `cold_backend.py:229-241, 296-299` 테이블 주입(cold, warm-kt 둘)
- GPU 측 `jit_kernel/csrc/moe/prism_sparse_common.cuh pair_energy` — warm worklist(bf16/mxfp4/fp8)가 같은 식을 쓴다. 단 ①에 따르면 **마스킹은 cold에만** 적용되므로, warm 커널의 SPARSE 경로는 계약상 비활성(코드는 남아 있음).

### 1.3 GPU prefill(KT_TILE8) — 레이아웃이 박힌 곳

fp8/mxfp4 prefill은 AMX 없이 GPU가 cold slab을 제자리 읽는다(사용자 결정 2026-08-28). 따라서 **레이아웃을 바꾸면 GPU 로더도 바뀐다** — 이것이 "prefill 쪽도 달라져야" 하는 이유의 전부다.

- `sglang/python/sglang/jit_kernel/csrc/moe/prism_grouped_fp8.cuh:41-45,58-70` 레이아웃 주석·상수·`enum Layout {ROWMAJOR, KT_TILE8}`, `:103-106` 스레드→(컬럼,그룹,16 B) 매핑, `:157-165 load_b`, `:180-200 store_b`(페어 → Bs 두 행)
- `jit_kernel/prism_grouped_fp8.py:24 COLD_LAYOUTS={"kt_tile8":1}`
- `srt/layers/prism/kernels.py:27-60` `_CPU_COLD_KERNELS / _SLAB_LAYOUT / _N_ALIGN / _TILE_ROWS`
- `srt/layers/moe/prism/formats.py:477 Fp8Format.cold_kernels`
- `srt/layers/moe/prism/cold_gpu.py:122-126` view 검사
- `test/prism/fp8_ref.py:75 ktf8_off`, `tile_block`; `test/prism/test_fp8_kernels.py:316 test_grouped_fp8_cold_tile_matches_gemv`
- `scripts/compiled_plan_to_prism.py:78,94` `COLD_N_ALIGN`, GLM 프리셋 `cpu_kernel="kt_tile_k2_fp8b128"`; `prism-lab/plans/glm53f/*_fp8.json`

## 2. 설계 결정 (제안값 — 사용자 확정 필요 항목은 ★)

### 2.1 k1 타일 레이아웃 (바이트 오프셋)

k2 스택 `k:1/1 | n:6/2 | k:4/128 | n:2/2048 | k-cols/8192 | supers`에서 k 페어 비트만 위로 올린다:

```
off_k1(n,k) = (n & 63) + (k & 31)*64 + ((n>>6)&3)*2048 + (k>>5)*8192 + (n>>8)*8192*(K/32)
```

- 청크 = **k 한 행 × 64 n = 64 B = 라인 하나** (k2는 128 B 두 라인)
- 타일 컬럼 8 KB, 그룹 2 KB, super 256 n, 배율 전치 `[K/128][N/128]` fp32 — 전부 **그대로**
- `BufferB::required_size`, 배율 pack, GPU `blk_off`/배율 테이블 위치 불변. 바뀌는 건 코드 순열 하나(`ktf8k1_off`)뿐.
- 64-n 그룹이 128-n 배율 블록을 넘지 않으므로 flush는 여전히 스칼라 broadcast 4 FMA.

### 2.2 per-k 마스크와 plan (2026-09-05 사용자 확정)

- **점수 = 순수 활성값**: `keep_k = |x_k| ≥ thr`. 가중치 통계(`wn_sq`, `pair_dot`)를 **전혀 쓰지 않는다**. 마스크 빌더 입력은 x(bf16)와 thr 하나 — bf16 → fp32는 `<<16`, 비교는 `andnot(sign) ≥ thr` (bf16 비트 비교와 동치라 fp 연산 순서 이슈가 없다; 스칼라/벡터 비트 동일이 자명).
- **score 이름 `k1`**: `sglang/.../prism/linear/calib.py:53 _THR_SUFFIX = {"k2wl2": "2l", "k1": ""}`에 이미 어휘가 있다 — thr 자산 키는 접미사 없는 `tg/tu/td`. MoE `calib.py`에도 같은 매핑을 넣는다.
- **threshold 곡선**: 사용자가 per-k 분포로 새로 캘리브해 준다(`tg/tu/td`, shape `[L, E, ng]`). `wn_*`/`c*` 테이블은 k1 자산에 없어도 된다.
- **마스크 타입**: `KMask = uint32_t`, 워드 c = 32-k 컬럼, 비트 j = k `32c + j`. `PairMask=uint16`과 병존; 커널이 `static constexpr bool kMaskPerK = true`로 선언.
- **plan 인코딩**: `ActivationPlan`은 레이아웃 무관 struct라 그대로 쓴다 — `koff[K]` = 유지된 k의 super 안 바이트 오프셋 `(k>>5)*8192 + (k&31)*64`, `pval[K]` = **x_k의 fp32 비트**(이름은 pval 유지, 문서화), `seg[K/32+1]`, `npairs` = 유지 k 수. arena = `(K*2 + K/32+1)*4 + 64` B (k2의 2배).
- 패딩 행(`k_real < k_rows`)은 `k_real` 이상 비트를 끈다 (기존 규칙 그대로).

### 2.3 커널 본체 (Seam C)

```
for super: for col c: for group g(4):  a0..a3 = 0
  for kept k in seg[c]:  pf cursor 1 line ahead (pf_dist lines, 이번엔 청크=라인이라 /2 없음)
     v = set1_ps(pval[j]);  line = W + koff[j] + g*2048
     4 × { fp8→fp32 16개 decode; a_i = fmadd(v, w_i, a_i) }
  flush: y[64g..] += a_i * S[c>>2][nb0 + (g>>1)]
```

- decode 제안: `cvtepu8_epi32` → `(w&0x7f)<<20` + rebias `(120<<23)` masked nonzero + `sign<<24` — fp32 직행 6 op/16. (k2의 fp8→bf16 6 op/32 + dpbf16 대비 라인당 명령 ~2배: §12의 dense 페널티 −9 % 근거.)
- 대안 ★: fp8→bf16(k2 디코더 재사용) 후 lo/hi 16-bit shift로 fp32 widen — 명령 수 비슷, 코드 공유 이점. 벤치로 결정.
- 프롤로그 프리페치(`KTF8_PROLOGUE_PF`) 유지.

### 2.4 이름/키

- kt: `GemmKernelTileK1FP8B128` (`la/tile_k1_fp8b128.hpp` + `_port.hpp`), operator는 `TILE_FP8_MOE_TP<T>`를 그대로 인스턴스화(템플릿 인자만), 바인딩 `TileK1FP8B128_MOE`, 파이썬 키 `kt_tile_k1_fp8b128`.
- sglang: slab layout 태그 `kt_tile8k1`, C++ `Layout::KT_TILE8_K1 = 2`, `COLD_LAYOUTS["kt_tile8k1"]=2`.

## 3. 구현 항목 (파일별)

### 3.1 kt-kernel — 커널 (헤더 전용, 게이트 테스트로 검증)

1. `operators/amx/la/tile_k1_fp8b128_port.hpp` (새)
   - `ktf8k1_off`, `ktf8k1_pack_codes[_range]` (배율 pack은 `ktf8_pack_scales_range` 재사용 — include)
   - `ktf8k1_build_plan(const uint32_t* mask, const uint16_t* x, K, koff, pval, seg)` — bf16→fp32 비트 (`<<16`)
   - `ktf8k1_cur_*` (4 그룹, 라인 단위) — k2의 커서와 동일 구조, 스텝 단위만 다름
   - `ktf8k1_fp8_16_ps` decode, `ktf8k1_gemv(...)`
2. `operators/amx/la/tile_k1_fp8b128.hpp` (새) `GemmKernelTileK1FP8B128`
   - 상수는 k2와 동일(N_STEP 32, K_STEP 32, N_BLOCK 256, SCALE_BLK 128, K_BLOCK 1<<30)
   - `static constexpr bool kMaskPerK = true;` `using MaskWord = uint32_t;`
   - `BufferB` — k2 것을 복사, `from_raw_mat`만 `ktf8k1_pack_codes_range`
   - `plan_arena_bytes`, `build_plan(const uint32_t*, ...)`, `gemv_row`, `gemv_slab`, `dense_plan`
3. `operators/amx/test/tile_k1_fp8-gate.cpp` (새, `tile_k2_fp8-gate.cpp` 복사) — dense / all-ones / 실제 per-k 마스크 vs fp64 dequant 레퍼런스. 빌드 명령은 k2 테스트 상단 주석과 같음.

### 3.2 kt-kernel — moe_base의 마스크 다형성 (핵심 계약 작업)

4. `operators/amx/la/act_plan.hpp`
   - `using KMask = uint32_t;` 추가
   - trait `mask_per_k<K>` (`K::kMaskPerK` 있고 true면) ; `build_plan_for<K>`의 mask 인자를 `const void*` 대신 **두 오버로드**(PairMask*/KMask*)로 — 호출부는 `if constexpr (mask_per_k<T>)`로 갈린다
5. `operators/amx/k_mask.hpp` (새) `build_k_mask(const ggml_bf16_t* src, int k_rows, int k_real, float thr, uint32_t* out)` — 32 bf16(64 B) 한 zmm → `|x| ≥ thr` 마스크 32비트 = 워드 하나. 패딩(`k ≥ k_real`) 비트 끔. 스칼라 폴백.
6. `operators/amx/moe_base.hpp`
   - `:122-124` 마스크 벡터 원소 타입을 `std::conditional_t<mask_per_k<T>, uint32_t, uint16_t>`로
   - `:312-314` 크기 `k / K_STEP`은 그대로(워드 수 동일: 32 k/워드)
   - `:627` `build_pair_mask` 래퍼 옆에 `build_k_mask` 래퍼; `:747,769,779,824` 호출을 `if constexpr (mask_per_k<T>)` 분기 — per-k 분기는 `a/c` 테이블을 읽지 않음
   - `:704-707,796-797` `g_a/g_c/u_a/u_c/d_a/d_c` 포인터는 per-k에서 nullptr — `:741,757,818 prefetch_prep_inputs(..., a, c, sparse)`와 `:760-761 prefetch_bytes(ua/uc)`가 nullptr을 건너뛰게
   - `cold_slab_view`는 무변경
7. `operators/moe-tp.hpp:331,374` — `wn_sq/pair_dot` null 검사를 페어 마스크 커널에만 적용 (`MoeTP::kMaskPerK` 같은 정적 플래그로 분기); thr/ng/grid 검사는 공통
8. `operators/common.hpp` `SparsityGeometry` — 필드 추가 없음(thr_*·예산만 쓴다). `ext_bindings.cpp` `bind_moe_module`에 `mask_per_k` 읽기 전용 속성 노출 → `experts_partial.py`가 요구 테이블 집합을 커널 클래스에서 읽어 결정(플래그 하드코딩 대신)

### 3.3 kt-kernel — 등록

9. `operators/amx/tile-fp8-moe.hpp` — 변경 최소: `derived_init`의 `group_size` 검사 그대로; `fp8_tile_tp_load_weights`의 `SCALE_BLK` 참조를 `K::SCALE_BLK`로(현재 `GemmKernelTileK2FP8B128::SCALE_BLK` 하드코딩 :192) 
10. `ext_bindings.cpp:57` include, `:1055` 옆 `bind_moe_module<TILE_FP8_MOE_TP<amx::GemmKernelTileK1FP8B128>, true>(moe_module, "TileK1FP8B128_MOE")`
11. `python/experts_partial.py:57,68` 두 표에 `kt_tile_k1_fp8b128`; `:176-186 _SPARSITY_KEYS`를 커널별로 — per-k면 `thr_gate/thr_up/thr_down` 셋만 필수(`:189 missing` 검사, `:198-212` shape 검사 모두 그 집합으로)
12. `tests/prism_partial_golden.py:87` 커널 축에 추가(k1은 thr만 생성), `test/prism/test_partial_moe.py:396-413,566-605`에 `|x|≥thr` 레퍼런스 케이스

### 3.4 sglang — GPU prefill 로더 + 레지스트리

13. `jit_kernel/csrc/moe/prism_grouped_fp8.cuh`
    - `enum Layout { ROWMAJOR, KT_TILE8, KT_TILE8_K1 }`
    - `load_b`: KT_TILE8과 **같은 주소식**(컬럼 c·그룹 g·16 B 조각 q, +1 KB 두 번째) — 청크 배치가 같으므로 로드는 공유
    - `store_b`: k1 분기 — 16 B = **k 한 행 × n 16개**: `krow = t8_cc*32 + (b0>>6)`, `ncol = t8_gg*64 + (b0&63)`, 한 행에 16 bf16
    - host 진입점 `grouped_fp8_cold[_gateup]`의 `layout` 검사 확장, `launch(prism_grouped_gemm_fp8<KT_TILE8_K1>)`
14. `jit_kernel/prism_grouped_fp8.py:24 COLD_LAYOUTS["kt_tile8k1"]=2`
15. `srt/layers/prism/kernels.py` 네 표에 키 추가 (`layout kt_tile8k1`, `n_align 256`, `tile_rows 128`)
16. `srt/layers/moe/prism/formats.py:477 Fp8Format.cold_kernels += ("kt_tile_k1_fp8b128",)`
17. `srt/layers/moe/prism/cold_gpu.py:124` tile 계열 검사에 `kt_tile8k1` 포함
18. `test/prism/fp8_ref.py` `ktf8k1_off`, `tile_block_k1`; `test/prism/test_fp8_kernels.py` k1 로더 ≡ GEMV 테스트
19. `srt/layers/prism/linear/formats.py:215 _FP8_COLD_LAYOUTS`에 `kt_tile8k1` (dense linear는 sparsity·fp8 cold 미배선이지만 `store_format` 정합 검사가 layout 표를 본다)
19b. 레인 트리(`sglang-glm`, `sglang-dense`, `sglang-merged`)는 `scripts/port_prism_to_upstream.sh`가 `srt/layers/moe/prism`, `srt/layers/prism`, `jit_kernel`, `test/prism`을 통째로 복사한다 — 원본 `sglang/`만 고치고 재이식하면 된다. kt-kernel은 `scripts/build_kt_for_glm_env.sh`(env `prism-glm`)로 재빌드

### 3.5 sglang — score `k1` 배선 (사용자 확정: 순수 |x|, calib 신규 제공)

20. `moe/prism/plan.py:78 KNOWN_SPARSITY_SCORES = ("k2wl2", "k1")`; `:142-149 expected_calib_shapes` — `k1`이면 `thr_*`만 요구(`wn_*/pair_dot_*` 항목 없음); `:420 K % PAIR_GROUP` 검사는 유지(인덱스 정렬 규칙은 별개)
21. `moe/prism/calib.py:45-57 _ASSET_KEYS/_SUPPORTED_SCORES` — score별 키 표(`k1`: `tg/tu/td`만), `:108-113 CalibTables.__init__ missing` 검사·`:150-165 load` 루프를 그 표로; `:176-194 slice_band`·`:196-234 gather_index`는 `k1`에서 wn/pair_dot 없이 `CalibBand/CalibShard`를 만들거나 None 반환. `linear/calib.py:53`의 `_THR_SUFFIX` 어휘와 일치시킬 것
22. `moe/prism/cold_backend.py:216-241 load_layer`, `:292-300 warm-kt` — `k1`이면 `tables = {thr_gate, thr_up, thr_down}`만; `:244-247`에서 **`plan.kernels.cpu_cold`의 마스크 종류(kt `mask_per_k`)와 `plan.sparsity.score` 정합 검사**(페어 커널 + k1, per-k 커널 + k2wl2 모두 즉사)
23. `moe/prism/weights.py` — `band.calib`(CalibBand) 생성 경로가 `k1`에서 wn/pair_dot 없음을 허용하는지 확인(`:371-490` cold_tile_rows 경로 옆); `method.py:380-400`은 키만 바뀌므로 무변경 예상
24. `CONTRACTS.md` ① 개정 메모: score 목록 `{k2wl2, k1}`, "마스크는 k/2 비트" 문장을 커널별로; `PAIR_GROUP=2` 정렬 규칙은 fp8이 `k_align=128`이라 실질 영향 없음
25. 테스트 갱신: `test/prism/test_calib.py:40,62,88-92,128`, `test_weights.py:395-427,462-512`, `test_sparsity.py:102-130,319-330` — k2wl2 케이스는 유지하고 `k1` 케이스 추가
26. `scripts/compiled_plan_to_prism.py:614-617` — `"score"`를 프리셋/인자로(`--score k1`, k1이면 새 calib 경로·sha256), `:78 COLD_N_ALIGN`에 새 키; `prism-lab/plans/glm53f/*_fp8.json` k1 판 생성
27. calib 자산: 사용자 제공(per-k `|x|` 분위수 `tg/tu/td` `[L, E, ng]`). 메모: k1은 gate/up 입력이 같은 x라 `tu == tg` — §3.2-6의 per-k 마스크/plan을 (token, expert)당 한 번 빌드해 gate/up이 공유할 수 있다

## 4. 검증 순서

1. 게이트 테스트(3.1-3) — 커널 단독, fp64 레퍼런스, err/mag ≤ 1e-7급
2. `tests/prism_partial_golden.py dump/compare` — **k2 경로가 한 비트도 안 바뀌었는지**(moe_base 다형성 작업의 회귀 검사)
3. `test/prism/test_partial_moe.py` per-k 케이스 — 정수 입력 비트일치(계약 ⑤-5)
4. sglang `test_fp8_kernels.py` k1 로더 ≡ GEMV(exact/inexact)
5. 마이크로벤치: `sglang/test/prism/bench_cold_cpu.py` 류로 실모델 shape(GLM 5.3 flash gate/up/down)에서 dense / per-k sp25/50/75 — k2 대비. §11 기대: dense −9 %, per-k sp50 +20~25 %. 이 표가 프리셋 교체 여부를 정한다.
6. e2e: GLM 5.3 flash plan을 `--cpu-kernel kt_tile_k1_fp8b128`로 재생성 → tok/s, 그리고 nnz 실측(①: cold nnz는 실측 대상)

## 5. 리스크 / 열린 질문

- **score 계약 개정 확정(2026-09-05)**: `k1` = 순수 `|x_k| ≥ thr`, 가중치 통계 없음. ①의 "k2wl2 고정" 문장을 `{k2wl2, k1}`로 개정하고, 커널의 마스크 종류와 plan score의 정합을 startup에서 검사한다.
- **calib 자산**: 사용자가 per-k 곡선(`tg/tu/td`)을 새로 만들어 준다. 그 전까지 정확성 테스트는 합성 thr로 가능. (자산 어휘 실측 메모는 `prism-mask-stats.md` §2.)
- dense 회귀: per-k 마스크가 있어도 sparsity가 낮은 층/expert에서는 k2보다 느리다(−9 %). 층별 kernel 선택은 plan에 없으니(`kernels.cpu_cold` model-global) 전체 교체 아니면 유지다.
- **GPU warm 티어의 마스킹**: `tiers.py SPARSE_TIERS = {WARM}` — plan에 sparsity가 있으면 GPU warm은 sparse 커널을 쓴다 (CONTRACTS ①의 "cold에만" 문장은 옛 것). score k1의 per-k 마스크는 **fp8 worklist 커널에 구현**했다(2026-09-05 저녁, 아래 §6): `prism_gemv_fp8.cuh` `MODE = kPerK`, 진입점 `*_sparsek1[_gateup]`, `SparseSpec.per_k`. bf16/mxfp4 worklist와 warm-kt(bf16 packed slab)에는 per-k 진입점이 없다 — k1 plan은 fp8 스토어 전용이므로 실제 조합에서는 문제가 없고, 그 조합이 오면 `_sparse_spec`이 NotImplementedError, warm-kt는 dense로 둔다.
- HW 프리페처 ON 환경에서는 64 B 청크 skip이 128 B보다 더 over-fetch에 노출된다(§15 (3)). **nutella3는 2026-08-30 리부팅 뒤 두 소켓 모두 pf ON**(root-less perf 확인 2026-09-05·09-06; MSR 0x1a4는 리부팅에 안 남고 재적용은 root 필요, 마스크 0x2f). 2026-09-06 GLM-5.3-Flash shape 스윕(cold_cpu, 2노드 14 thr, 아래 §6)은 이 조건이고 k1이 전 sparsity에서 k2보다 느렸다 — pf OFF 재측정이 결정 실험이다.

## 6. 진행 상태 (2026-09-05, worktree `ktransformers-k1` / `sglang-k1`, 미커밋)

| 항목 | 상태 | 검증 |
|---|---|---|
| 3.1 커널 (`tile_k1_fp8b128[_port].hpp`, 게이트) | 완료 | `tile_k1_fp8-gate.cpp` 양 디코드 경로 err/mag ≤ 9e-8; 레이아웃 전단사 |
| 디코드 경로 | **bf16 경유 기본**(`KTF8K1_DECODE_VIA_BF16=1`) | 마이크로벤치(소켓 1, 8 thr, N4096 K2048 E16): dense 0.107 vs 0.116 ms, per-k sp50 0.054 vs 0.059 |
| k1 vs k2 (같은 벤치) | dense k1 +8 %(0.107 vs 0.100), per-k sp25/50/75 k1 −17/−28/−40 % (0.080/0.054/0.028 vs 0.096/0.075/0.047) | `tile_k1_fp8-bench.cpp` |
| 3.2 moe_base 마스크 다형성 (`act_plan.hpp KMask/mask_per_k`, `k_mask.hpp`, `moe_base.hpp build_mask`, `moe-tp.hpp` null 검사) | 완료 | `tests/prism_partial_golden.py` main 빌드 vs k1 빌드 **identical**; `test_partial_moe.py` 39 passed |
| 3.3 등록 (`ext_bindings` `TileK1FP8B128_MOE` + `mask_per_k` 속성, `experts_partial.py` 키·`sparsity_keys()`) | 완료 | `test/prism/test_tile_k1_fp8.py` 10 passed (dense ref, thr-only 테이블, sparse == masked dense 비트일치, k2는 thr-only 거절) |
| 3.4 GPU `KT_TILE8_K1` + 레지스트리 (`kernels.py` 4표+`_CPU_COLD_SCORE`, formats, cold_gpu, linear formats, jit wrapper) | 완료 | `test_fp8_kernels.py::test_grouped_fp8_cold_tile_matches_gemv[kt_tile8k1]` exact/inexact 통과 (RTX 5090, prism-e2e) |
| 3.5 score `k1` (`plan.py`, `calib.py` score별 자산 키, `cold_backend._sparsity_tables` + 커널/score 정합 검사, `tiers.py` GPU warm dense, CONTRACTS ①) | 완료 | `test_calib.py` k1 4건 추가; calib/weights/sparsity/plan/kernels 92 passed |
| e2e (`sglang-k1/test/prism/test_fp8_prefill.py`, worktree kt 빌드 심 + prism-e2e GPU) | 완료 | k1 3-tier == bf16 dequant (m=1/40), GPU cold prefill(KT_TILE8_K1) == CPU cold k1, k1 sparse thr=0 == dense 비트일치·thr=1e9 방향, 커널/score 불일치 즉사 — 14 passed |
| 3.5-26 컨버터 `--score k1` | **미착수** — `scripts/compiled_plan_to_prism.py`는 메인 트리에서 수정 중인 파일 | |
| 3.5-27 calib 자산 | act-stats 수집기(별도 문서) | |
| GPU warm per-k 마스크 (`prism_gemv_fp8.cuh` kPerK, `sparsek1` 진입점 4개, `SparseSpec.per_k`, `_sparse_spec` k1 분기) | 완료 | `test_fp8_kernels.py` per-k 3건(thr0 ≡ dense 비트, thr>0 ≡ dense(|x|<thr→0) 비트, 융합 ≡ 2 launch, 혼합 모드 거절) 포함 27 passed; e2e 14 passed (warm도 per-k 마스킹) |
| §4-5 프로파일러 (`sglang-k1 .../prism/profile`) | 완료 (2026-09-06 개정) — 마스크를 **레벨 x + expert별 thr(밴드 안 순서통계)** 로 실현(`common.per_k_levels`/`per_k_thr`, bf16 레벨 512개 [0.125,2)); cold(kt)·warm(GPU `*_sparsek1`, `SparseSpec.per_k`) 모두 per-k, `check()`는 Σ x_k·W 레퍼런스라 마스크 유실을 잡음; expert별 다른 sparsity·티어 경계(full_layer)·split_index 전부 허용, 실현 keep은 레벨 동률만큼(≤1/512) 어긋나 `keep_frac`이 진짜 값 | `test_profile_dtype.py` k1 6건 + `test_full_layer` 등 32 passed (shim). E16 topk4 H2048 I1024 fp8 소켓0 8 thr, warm12.5/cold87.5: sp0 cold_only k2 246 / k1 231 µs, sp50 k2 97 / k1 113 µs; combined sp50 gateup k2 109 / k1 120; warm_only 동률(PCIe 천장); rel err ~3e-3 |

빌드/실행 레시피 (worktree, 메인 env 설치본 무손상):
`python setup.py build_ext --inplace` (ktk-prism env, 메모리의 CMAKE_ARGS) → `python/kt_kernel_ext*.so`;
**주의:** 그 빌드는 `CPUINFER_USE_CUDA=0`이라 `CPUInfer.submit_with_cuda_stream`이 없다 → 프로파일러의 `combined`/graph 변형과
`full_layer`가 `AttributeError`로 죽는다(`cold_only`/`warm_only`/`cold_cpu`는 무관). CUDA 포함 빌드는 prism-e2e env에서
`scripts/build_kt_for_lane_env.sh`의 플래그(`CPUINFER_USE_CUDA=1`, `CUDA_HOME=/usr/local/cuda-12.8`)로 같은 `build_ext --inplace`.
2026-09-05 그 CUDA 빌드로 교체됨(`python/kt_kernel_ext*.so` 16:46, `submit_with_cuda_stream` 있음). 이때 드러난 버그: `k_mask.hpp`가
`_pdep_u32`(BMI2)를 써서 `-march=native` 없는 AVX512 플래그 빌드가 깨졌다 → 16 bf16씩 `cvtepu16_epi32<<16`으로 k 순서 그대로 비교하는
pdep 없는 구현으로 교체(`test_tile_k1_fp8.py` 10 passed, 게이트 `-mno-bmi2`도 OK). 프로파일 `combined`(graph) k1 실측 위 표 참조.
`PYTHONPATH=<shim>` 에서 `shim/kt_kernel -> ktransformers-k1/kt-kernel/python` 심볼릭 링크로 덮어쓴다.
sglang 테스트는 `prism-e2e` env + `PYTHONPATH=sglang-k1/python`.

### 6.1 warm(GPU pinned fp8 worklist) per-k 벤치 (2026-09-05, RTX 5090, E=32 TOPK=8 N=4096 k_rows=2048, dense 67 MB/step)

| 케이스 | keep | min ms | GB/s |
|---|---|---|---|
| dense | 1.00 | 1.324 | 50.7 |
| per-k k1 / pair k2 (같은 keep 0.75) | 0.75 | 0.998 / 0.993 | 50.5 / 50.7 |
| per-k k1 / pair k2 (같은 keep 0.50) | 0.50 | 0.670 / 0.667 | 50.1 / 50.3 |
| per-k k1 / pair k2 (같은 keep 0.25) | 0.25 | 0.348 / 0.345 | 48.3 / 48.6 |
| pair k2, per-k sparsity를 페어로 받은 경우 (keep 1−s²: 0.94 / 0.75 / 0.44) | | 1.242 / 0.991 / 0.588 | 50.6 / 50.8 / 49.9 |

warm은 PCIe 천장(~50 GB/s)에 붙어 있고 per-k와 페어 커널이 같은 바이트에서 1 % 안에서 같다 — cold(CPU)와
같은 결론: **시간 = 유지 바이트 / 대역폭**이고, per-k의 이득은 (1−s) 대 (1−s²)의 바이트 차이 그 자체다.
