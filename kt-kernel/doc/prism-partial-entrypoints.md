# Prism partial 진입점 구현 노트 (K1 착수 전 조사 결과)

sglang 쪽 계약: `sglang/python/sglang/srt/layers/moe/prism/CONTRACTS.md` ②.
이 문서는 moe_base.hpp 정독(2026-08-20, 핀 시점 main 41e54fd)에서 확정한
절단면과 구현 시 주의점을 기록한다.

> **2026-08-25 개정.** sglang CONTRACTS.md가 같은 날 개정되면서 이 문서의 전제
> 둘이 바뀌었다: (a) K축 기하가 **밴드에서 가변 per-expert 인덱스**로, (b) gate와
> up이 **독립**으로(→ dual-pack). **sparsity 배치는 그대로다** — 예산·곡선이 kt
> config에 남고, step마다 경계를 건너는 것은 라우터 가중 하나이며, `s → thr →
> 점수 → 마스크 → masked GEMV`가 전부 이쪽에서 끝난다 (threshold 조회를 prism으로
> 되돌리는 안은 검토 후 폐기 — 마스크 기계가 `AMX_MOE_BASE`에 있어 CRTP 파생
> 커널은 어느 쪽이든 그것을 물려받는다). **진입점 시그니처는 diff 0이다.**
> §1~§9는 as-built 기록으로 그대로 두고, 바뀌는 것은 **§10**에 모았다. §2-3과
> §2-5에는 그 자리에서 거짓이 된 문장에 정오표를 붙였다.

## 1. forward_decode의 8단계와 절단 위치

```
moe_base.hpp forward_decode (:455)
 1. 라우팅 준비        :464-483  m_expert_id_map_, m_local_*_output_ptr_ 배정
 2. BufferC/A 풀 분배  :485-528  gate_bc/up_bc/down_ba/down_bc set_data
 3. x pack             :530-539  gate_up_ba->from_mat(input)
 4. gate/up GEMM       :549-565  do_gate_up_gemm → to_mat → m_local_{gate,up}_output
 ──────────────── 절단선 #1 (gateup partial은 여기서 종료·export) ────────────────
 5. act                :575      apply_activation (m_local_gate_output에 in-place)
 6. down 입력 pack     :585-591  down_ba->from_mat(m_local_gate_output_ptr_)
 7. down GEMM          :601-610  do_down_gemm → to_mat → m_local_down_output
 ──────────────── 절단선 #2 (down partial은 여기서 종료·export) ──────────────────
 8. router 가중 합     :620-638  weights[j] fmadd → fp32 output
```

- `forward_gateup_partial` = 1~4 + bf16 export (§2-1). 5~8 없음.
- `forward_down_partial`   = 1~2 + (act 텐서에서 6의 pack) + 7 + bf16 export.
  입력 act가 [qlen × k × inter]이므로 6의 from_mat 소스가
  m_local_gate_output이 아니라 호출 인자의 slot별 행이 된다.
- 8(가중 합)은 대응물 없음 — GPU rejoin 소관 (계약 ② overwrite 의미론).

## 2. 주의점 (조사에서 발견)

1. **export는 기존 `to_mat`(fp32 누산 → bf16 재료화)을 무변경 재사용한다.**
   (계약 ⑤ 2026-08-20 개정: wire dtype = bf16, 합산 정밀도는 GPU rejoin의
   fp32 누산이 보장.) kt의 정밀도 프로파일 — m_local_* 전부 bf16
   (moe_base.hpp:44-46), GEMM 누산은 AMX 타일 fp32, act도 fp32 계산 후
   bf16 저장(:708-710) — 을 그대로 계승하므로 신규 export 코드가 없다.
   초판 노트의 "fp32 export 변형 필요"는 이 개정으로 폐기됨.
2. **out 레이아웃 [qlen × k × N]의 slot 순서**: 내부 버퍼는
   expert_idx 키의 m_local_*이지만 export는 호출자의 expert_ids slot 순서.
   skip된 expert slot(should_skip_expert)은 미기록 → undefined (계약 ②-4).
3. **치수 비대칭** (기하 표현은 §10.3에서 `KIndex`로 대체됨 — 아래 서술은 K1
   시점 기록): partial 인스턴스의 config는 gateup K=|cold hidden rows|,
   N=inter(full) / down K=|cold inter rows|, N=hidden(full)로, 기존
   MOEConfig의 (hidden, inter) 한 쌍으로 표현 불가. `PartialMOEConfig`에
   proj별 (k_offset, k_rows) 4필드를 추가한다. full-width 입력에서
   k_offset만큼 건너뛰어 pack하는 것은 from_mat의 열 시작점 문제
   (stride는 full K) — from_mat 호출부의 포인터 산술로 해결 가능한지
   BufferA::from_mat 시그니처 확인 필요 (K1 첫 작업).
4. **NUMA 분할 (2026-08-20 최종 결정: 전 proj N축 분할)**: kt TP는
   inter 축 단일 분할(moe-tp.hpp:130)로 gate/up N과 down K를 결합하지만
   — 이는 "노드가 자기 act 열만 소비"하는 locality 결합의 산물 — 우리는
   act가 GPU rejoin#1을 거쳐 full-width로 되돌아오므로 그 결합이 없다.
   따라서 **down도 N(hidden)축으로 분할**한다 (사용자 결정):
   - gateup partial: inter 열 분할, 서로소 direct write ✓ (K1 구현됨).
   - down partial(K2): hidden 열 분할, gateup과 동형의 서로소 direct
     write — **노드-합 없음** (중간에 검토했던 "무가중 node-sum" 안은
     이 결정으로 폐기). 필요 작업: config down N-shard 기하 +
     do_down_gemm N 접근자화 + TP ctor 이중 독립 분할 + down 스테이징
     방향 변경(자기 hidden 행 × cold inter 전체) + 노드별 full cold-inter
     act pack (중복 소량). TP load의 down guard는 이 구현으로 해제.
5. **타일 상수 확정**: N_STEP=32, K_STEP=32, M_STEP=TILE_M×2
   (amx_raw_kernels.hpp:26-27). → plan.py의 COL_GROUP=32 유지,
   ROW_GROUP=64는 K_STEP 배수라 안전.

   > **정오표 (2026-08-25)**: 뒷문장이 plan 쪽 제약으로 굳어버렸다. 실제 요구는
   > `amx_raw_buffers.hpp:63,132`의 `assert(k % K_STEP == 0)` 하나뿐이고 그것도
   > **packed 저장**에 걸리는 것이지 논리적 K에 걸리는 게 아니다. `ROW_GROUP=64`는
   > 폐기됐고(sglang ① 2026-08-25), plan/자산이 지켜야 하는 정렬은 페어(%2)뿐이다.
   > 타일 올림은 인스턴스 내부 사정이며 그 책임은 §10.6에 적었다.
6. **버퍼 풀 재사용**: 1~3의 풀 분배·pack 코드는 gateup/down partial이
   각자 필요한 부분만 복제하되, 풀 자체(gate_up_ba_pool_ 등)는 기존
   인스턴스 멤버 구조를 그대로 계승한다 (partial 인스턴스도 자기 풀 소유).

## 3. K1 커밋 범위 (기존 계획 유지, 위 발견 반영)

- `PartialMOEConfig` (proj별 k_offset/k_rows + N-shard) — sglang CONTRACTS ②에
  필드 목록 동기화 커밋 동반
- `forward_gateup_partial` (decode 경로 먼저; prefill은 같은 절단을
  forward_prefill에 반복). export는 기존 to_mat 재사용 (위 §2-1)
- CPU-only 테스트: torch fp64 레퍼런스 대비 tolerance + **정확 표현 가능
  입력(작은 정수)으로 비트일치** 검증 (계약 ⑤-5), skip-expert/slot 순서 포함

## 4. kt 버퍼 할당 기계 조사 (2026-08-20) — K1이 따를 것/바꿀 것

kt의 할당은 4계층:
1. **SharedMemBuffer 전 레이어 공유 스크래치** (shared_mem_buffer.{h,cpp},
   moe_base.hpp:161 `shared_mem_buffer_numa.alloc(tp_part_idx, this, ...)`):
   모든 레이어 객체의 m_local/풀이 물리적으로 같은 버퍼 1벌 (레이어 직렬
   전제). grow 시 free→재할당→**전 객체 포인터 재패치** (append_pointer가
   재바인딩 람다를 저장하는 이유). NUMA는 node별 버퍼.
2. **forward-time carving**: BufferA/C는 init 때 nullptr 껍데기, 매 forward
   활성 expert만 풀에서 set_data (:485-539).
3. **weight(BufferB)**: expert×proj당 aligned_alloc 개별 영구 (:135-142).
4. **경계 pinned**: Python(KExpertsCPUBuffer) 소유, C++는 포인터만.

partial 인스턴스: 1·2·3·4 전부 그대로 계승. 다른 점은 (a) mem_requests
치수가 K' 기준 (gateup: cold hidden rows / down: cold inter rows),
(b) m_local_gate/up 용도가 act 입력 → export 소스, (c) down pack 소스가
호출 인자 act (K2), (d) fp32 가중합 출력 버퍼 부재, (e) act staging은
prism ColdStaging에 신설(kt 대응물 없음).

주의 2건:
- 스크래치 주소는 grow 시 이사한다 → **C++ 스크래치 주소를 밖(GPU 쪽)에
  절대 노출 금지.** 경계는 pinned staging뿐.
- deferral(P1+)이 공유 스크래치와 안전한 이유 = 지연 누산은 pinned에
  하기 때문 (kt 동일). partial export가 pinned이므로 우리도 같은 방식.

## 5. K1 버퍼 인벤토리 (경계 버퍼는 sglang resources.py에 이미 커밋됨)

C++ partial 인스턴스 소유 (mem_requests/carving 패턴):
- gateup: gate_up_ba 풀(K'), gate_bc/up_bc 풀, m_local_gate/up(export 소스)
- down(K2): down_ba 풀(K'), down_bc 풀, m_local_down
- 불필요: 최종 fp32 output (merge_results 부재)

prism ExecutionResources 소유 (C++는 포인터만 받음):
- ColdStaging: _x [M,h] / _expert_ids [M,k] / _act [M,k,inter] /
  _partial_gateup·down (bf16) — out 인자가 partial pinned를 직접 가리킴

## 6. K1 텐서 주입 — kt SFT 경로와의 1:1 (2026-08-20 조사)

주입의 실체 = config에 data_ptr 싣기 (sft/amx.py:276-287). 단계 대응:
① load_weights_from_tensors: 시그니처 동일, 텐서만 K-슬라이스 [E,N,k_cold].
   contiguous 보관 = pack 완료까지 생존 보장 (수명 규칙 — K3 wrapper가
   동기 완료 후 해제 순서 명시).
② config 구성: PartialMOEConfig = 기존 + proj별 (k_offset, k_rows).
   config.gate_proj = data_ptr (kt 방식 그대로). 텐서 k차원 == k_rows 검증.
③ init: 같은 기계, 크기 산정만 K' 기준 (buffer_b_required_size(N, k_rows),
   mem_requests 치수).
④ pack 루틴 무변경 — K' % K_STEP(32) == 0 이 ROW_GROUP=64 정렬로 보장되어
   pack은 슬라이스 여부를 모름. K1의 pack쪽 diff = 0.
⑤ pack 완료 → PendingColdTensors 해제 → cold = 핸들 (계약 ③).

staging 조달 편차 (의도적, 1:1 유일 예외): kt는 wrapper가 KExpertsCPUBuffer를
self-serve, 우리는 prism ExecutionResources가 소유하고 wrapper에 포인터
주입(caller-provide). 근거: 형상이 Plan 어휘에서 나옴 / 영구 메모리 단일
소유자(graph identity 등록 지점 1곳) / graph에 굳는 주소의 관리 주체 일치.

## 7. K1 구현 형태 (as-built, 2026-08-20)

채택한 구조 = **config 접근자 주입** (별도 클래스 없음):
- `GeneralMOEConfig` += gateup_k_offset/rows, down_k_offset/rows,
  partial_n_total + 접근자 gateup_k()/down_k()/n_total() (common.hpp).
  전부 0(기본)이면 기존 동작과 비트 동일 — 침습성 상한.
- `AMX_MOE_BASE`의 init 버퍼 치수와 bf16 do_*_gemm의 K 치수가 접근자를
  경유 → CRTP seam이 기하-인지형이 됨. **partial 진입점을 base에 한 번
  구현하면 모든 dtype 클래스가 상속** (단, do_*_gemm K 치수를 갱신한
  클래스만 안전 — 바인딩은 WithPartial opt-in, 현재 BF16만).
- `forward_gateup_partial` (moe_base.hpp): prepare_gateup_routing →
  carve_and_pack_gateup → run_gateup_stage(CRTP seam 경유) →
  export_gateup_partial. 단일 책임 4함수. K1은 qlen==1만 (guard).
- fused `forward`는 partial 인스턴스에서 throw (조용한 오답 방지 guard).
- TP: TP_MOE_Common::forward_gateup_partial — 노드 디스패치만, merge 없음.
- 테스트: test/prism/test_partial_moe.py (레퍼런스 tolerance + 정수
  비트일치 + fused-forward 거부).

## 8. K2 구현 형태 (as-built, 2026-08-20)

- **down N(hidden)-split 완성**: forward_down_partial = prepare_down_routing
  → carve_and_pack_down(slot별 act 행에서 down K 밴드 pack) → run_down_stage
  (CRTP do_down_gemm, N=down_n()) → export_down_partial(자기 hidden shard 열
  direct write). 노드 합산 없음. gateup과 완전 동형.
- **per-node shard 테이블**: PartialGeometry에 top-level 테이블 4 vector
  (Plan 주입) + node-scope KRange 2개(TP ctor 기록). TP ctor가 테이블
  검증(크기/합/범위, 호출 스레드) 후 주입하거나, 없으면 균등 분할로 채움.
  K1의 균등 전제 export 공식(tp_part_idx×I)은 gateup_n.offset 읽기로 대체.
- **TP 스테이징**: down은 is_partial 분기 — partial이면 행 슬라이스
  (연속 memcpy 1회, [down_n.offset, +rows) × down_k), 아니면 기존 열 슬라이스
  루프 (kt 무변경). gate/up src offset도 shard offset 기반(불균등 대응).
- init에 partial 기하 정합 검증 추가 (enabled면 전 값 명시·범위·노드
  gateup_n.rows==intermediate_size).
- 테스트 8종 green (기존 4 + down 레퍼런스/정수 비트일치/gateup→act→down
  체인/불균등 테이블 96·32/192·64) + AVX2 bf16·AMX int4 회귀 green.
- 남은 것: partial prefill(qlen>1, §TODO), K3 배관.

## 9. partial prefill (as-built, 2026-08-20)

- 두 진입점 모두 qlen==1(decode 빠른 경로) / qlen>1(prefill) 디스패치.
- prefill = kt forward_prefill 1~3단계 동형: prepare_prefill_routing
  (m_local_pos_ 장부 그대로 재사용) → carve_*_prefill(전 expert 누적
  offset + kt pool assert 유지) → gather_and_pack_*_prefill.
- **strided from_mat 불필요 확정**: gather의 per-행 memcpy가 full-width
  입력에서 밴드만 담으므로 (gateup: x+i*H+off, down: act+(i*k+j)*n_total
  +off) from_mat은 연속 [m_e, band]를 그대로 읽음. amx_buffers 무변경.
- down의 act gather는 m_local_input_을 재사용 (gateup gather와 phase가
  달라 충돌 없음; 용량 k_tok*ml*H ≥ 필요량).
- stage/export는 decode와 공유: to_mat이 m_local_num_[e] 사용, export가
  m_local_pos_[m][j] 기반 (decode는 pos=0 퇴화형 — prepare가 채움).
- TP guard: qlen ∈ [1, max_len] (batch>1 eager decode 해제).
- 테스트: 6종 × qlen {1, 16} 파라미터화 + guard 2종 = 14 green.
  qlen=16 × k=2 / E=8 → 토큰 간 중복 expert 필연 — 정수 비트일치가
  (m, slot) 좌표 뒤섞임의 exact 검출기.


## 10. 2026-08-25 개정 — 인덱스 기하

sglang CONTRACTS.md ①③④⑤의 같은 날 개정에 대응하는 kt 쪽 작업 목록.
커밋 단위는 sglang 쪽 계획의 K3/K4/K5에 대응한다. **②는 시그니처 불변**이라
진입점 커밋이 따로 없다 — `PartialGeometry`의 내용물만 바뀐다.

### 10.1 KIndex + per-expert 접근자 (K3) — **as-built 2026-08-25**

```cpp
struct KIndex {
  std::vector<int32_t>  row_off;   // [E+1] — weight/idx/calib이 공유
  std::vector<uint16_t> idx;       // [row_off[E]] — 원본 K행 번호
  bool contiguous;                 // 연속 퇴화형이면 gather 스킵 → 밴드와 비트일치
};
PartialGeometry { bool enabled; KIndex gate, up, down; int n_total; ... }
```

`gateup_k()`/`down_k()` → `gate_k(e)`/`up_k(e)`/`down_k(e)`
(`= row_off[e+1] − row_off[e]`). 치수를 읽는 곳 전부가 expert 인자를 갖게 된다:

| 위치 | 지금 | 바뀜 |
|---|---|---|
| `bf16-moe.hpp:111,117,120` | `config_.gateup_k()` | `gate_k(e)` / `up_k(e)` |
| `bf16-moe.hpp:127,132,135` | `config_.down_k()` | `down_k(e)` |
| `bf16-moe.hpp:201,207` | 주입 stride `logical_id * I * gateup_k()` | `row_off[logical_id]` 기반 (주입 텐서가 flat) |
| `bf16-moe.hpp:225` | 〃 down | 〃 |
| `moe_base.hpp:459` carve | `buffer_a_required_size(max_m, gateup_k())` | expert별 크기 |
| `moe_base.hpp:696` carve down | 〃 | 〃 |
| `moe_base.hpp:162-164` | `gu_tiles`/`dn_tiles` 균일 | expert별 `k(e)/K_STEP` |
| `moe_base.hpp:370,404` `build_*_masks` | 테이블 오프셋 `e * rows` | `row_off[e]` (`pair_dot`은 `row_off[e]/2`) |
| `moe_base.hpp:118-136` init 검증 | 밴드 범위 | `row_off` 단조 + 합 정합 + 페어 무결성 |

`wn_sq`/`pair_dot`은 K축이므로 **weight와 같은 인덱스로 gather된 것**이 주입된다
(로드 타임). `thr_*` 곡선은 `[E, ng]`라 인덱스와 무관하므로 그대로다.

### 10.2 gather-to-dense (K4) — **as-built 2026-08-25**

pack 직전에 activation을 인덱스로 모아 dense를 만들고, **그 뒤 코드는 무변경**
으로 그 dense를 읽는다 (planir `PackLhsCold`와 같은 형태). 마스크 코드가 손대지
않아도 되는 이유가 이것이다 — `build_*_masks`가 읽는 x가 packed B의 행 순서와
같은 순서가 되므로 비트 ↔ 타일 대응이 그대로다.

| 위치 | 변경 |
|---|---|
| `moe_base.hpp:459` `carve_and_pack_gateup` | `from_mat(qlen, x_band, …)` 앞에 gather. decode는 expert 루프에서 스크래치 1벌 재사용 |
| `moe_base.hpp:580` `gather_and_pack_gateup_prefill` | per-token `memcpy` → gather 루프. 목적지는 기존 `m_local_input_ptr_[e]` (신규 버퍼 0) |
| `moe_base.hpp:696` `carve_and_pack_down` | 동형 (`act + j*act_row + down.offset` → 인덱스 gather) |
| `moe_base.hpp:778` `gather_and_pack_down_prefill` | 동형 |
| `moe_base.hpp:370,404` `build_*_masks` | 인자를 gather된 dense로 (`x_band` → 스크래치) |

#### as-built 노트

- **모드는 둘로 고정했다**: `row_off`가 비면 퇴화형(전 expert 공통 밴드),
  채우면 인덱스형이고 이때 `idx`는 **항상 함께 온다**. "row_off는 있고 idx는
  없는" 세 번째 모드(연속·가변)를 두면 gather 경로가 갈라지는데, 그 절약이
  행당 2 B라 분기 하나만 못하다.
- **무인자 접근자를 남기지 않았다.** `gateup_k()` → `gateup_k(e)` /
  `gateup_k_max()`로 갈라, "이 자리가 per-expert인가 최악치인가"를 컴파일러가
  전 호출부에서 강제로 묻게 했다 (틀리면 조용한 오답이다). 실제로 44곳이
  걸렸고 분류는 기계적이었다: 풀 크기만 `_max`, 나머지는 전부 per-expert.
- **gather와 dense를 분리해야 했다.** 처음엔 한 함수가 "dense 포인터를
  돌려준다(퇴화형이면 zero-copy)"를 겸했는데, prefill은 토큰마다 **자기
  슬롯에 반드시 써야** 하므로 퇴화형에서도 복사가 필요하다. 겸하게 두었더니
  prefill이 조용히 빈 버퍼를 pack했다 (테스트 7개가 잡았다). 지금은
  `gather_rows`(항상 dst에 쓴다) / `dense_rows`(decode용, 퇴화형 zero-copy)다.
- **gather 목적지는 `m_local_input_`을 재사용한다** — decode의 gateup phase
  에서 이 버퍼에 다른 소비자가 없어 prefill 버퍼를 그대로 쓴다. 신규 할당 0.
  마스크 빌더도 같은 버퍼를 읽으므로(`rows_of`) 비트 ↔ 타일 대응이 유지된다.
- **점수 테이블 오프셋**은 `e * rows`에서 `row_base(e)`로 바뀌었다.
  `pair_dot`은 그 절반이고, 모든 `k(e)`가 K_STEP(32)의 배수라 짝수가 보장된다.
- **TP staging**(bf16-moe.hpp의 노드별 weight 복사)도 per-expert 원소 수로
  바뀌었다. 총량은 `row_base(E)`, expert 시작은 `row_base(e)`다.
- 검증은 init에서: `row_off` 단조, `idx` 길이 정합, **expert별 행 수가 K_STEP
  배수**, 인덱스가 축 범위 안. 마지막 둘은 조용한 오답의 원천이라 즉사시킨다.

`contiguous`(연속 인덱스)면 gather 커널이 아니라 memcpy로 떨어지고, 퇴화형이면
아예 포인터만 넘긴다 — 기존 밴드 plan과의 비트일치가 전환기의 검증 기준이다.

### 10.3 dual-pack (K5)

gate ≠ up이면 packed activation이 달라지므로 `gate_up_ba_` 하나가
`gate_ba_`/`up_ba_` 둘로 갈리고 A 풀이 2배가 된다. 2026-08-20에 "폐기"로
적었던 항목이며(§TODO 및 sglang TODO.md) 2026-08-25에 되살아났다.
**선행 확인**: prefill에서 A 풀 2배가 `SharedMemBuffer` grow에 걸리는지.

### 10.4 타일 올림과 tail 책임

논리적 `k_real[e]`는 페어(%2)만 지키면 되고, packed 저장은 `k_pad[e] =
ceil32(k_real[e])`로 올림한다. 버퍼 기계에는 `k_pad`를 넘겨 `assert(k % K_STEP)`을
그대로 통과시키고, 커널이 두 가지를 책임진다:

1. **dense 경로**(prefill `mat_mul`, AMX `amx_kernel`)를 위해 패딩 행은 0.
   packed B는 로드 시 0-fill, A는 gather 후 tail memset(≤62 B).
2. **sparse 경로**는 tail 비트를 `k_real/2` 이상에서 0으로 유지하면 그만이다 —
   `avx_kernel_4_sparse`(`amx_raw_kernels.hpp:248`)의 `while(mask)`가 세워진
   비트만 순회하므로 **패딩 행의 내용조차 읽히지 않는다**. 단 `build_pair_mask`가
   tail 비트를 명시적으로 클리어해야 한다: calib을 0으로 패딩해 자연히 떨어지길
   기대하면 `thr == 0`일 때 `imp = 0 >= 0`이 참이라 비트가 서고 쓸모없는 로드가
   생긴다.

새 CPU 커널이 ragged k를 네이티브로 다루면 저장 올림 자체가 불필요하다 —
그래서 sglang 계약에는 32가 적혀 있지 않다.

### 10.5 바뀌지 않는 것

- **진입점 시그니처 전부.** 계약 ②-3("K축 기하는 호출에 없다")이 이 전환에서
  실제로 값을 했다.
- **sparsity의 배치와 경계** — 예산·곡선은 kt config, 경계를 건너는 것은 라우터
  가중, `s → thr → 점수 → 마스크 → masked GEMV`는 전부 이쪽. 인덱스 때문에
  바뀌는 것은 점수 테이블의 **오프셋**뿐이다 (10.1의 마지막 줄).
- **pack 루틴(BufferB::from_mat) diff 0** — 슬라이스/인덱싱 여부를 모른다.
- `export_*_partial`, NUMA N-shard 테이블, overwrite 누산 의미론, 완료 계약.

### 10.6 검증

- 연속 인덱스(균일 길이 + shuffle 없음)에서 기존 밴드 경로와 **비트일치**
  (as-built: `test_indexed_contiguous_matches_band`).
- 셔플 인덱스 + expert마다 다른 `k[e]`에서 **정수 비트일치** (정확 표현 가능한
  작은 정수 입력 — 계약 ⑤-5). 좌표 뒤섞임의 유일한 exact 검출기다.
  as-built: `test_indexed_{gateup,down}_bitexact_on_integer_inputs`, decode·
  prefill 파라미터화. 밴드에서는 이 테스트가 순서를 검증하지 못한다 — 어느
  순서로 더해도 같은 행이기 때문이고, 셔플에서 비로소 검출기가 된다.
- gate ≠ up 조합, `k[e] == 0`인 expert가 섞인 구성, 불균등 NUMA shard × prefill.
