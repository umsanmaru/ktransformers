# effective sparsity 실측 도구 (`mask_stats`) — 2026-09-05

사용자 요구: "특정 prompt에 대해 지금 돌아가는 k2wl2 마스크의 effective sparsity만 알고 싶다."
calib 자산 생성기 같은 큰 것은 범위 밖(그 계획은 폐기).

## 1. 무엇을 만들었나 (sglang `srt/layers/moe/prism`)

- `mask_stats.py` — `MaskStats`: executor의 decode 경로(`masking = sparse and m == 1`)에서 커널이 받는
  같은 입력(hidden / act, topk_ids, topk_weights)으로 계약 ①의 식(`slot_sparsity` → `thr_of` → 페어 에너지
  `e ≥ thr²`)을 CPU torch로 재계산해 (layer, proj, tier)별 keep 페어 수를 누적. `every` 토큰마다 json 덤프 + atexit.
- `executor.py` — 생성자 `mask_stats=`, `register_layer`에서 층 등록, Phase 1 직전 `observe_gateup`,
  rejoin#1 직후 `observe_down`. graph 경로(`flow.graph_flow`)에서는 관측하지 않는다(파이썬이 replay되지
  않으므로 **eager `--disable-cuda-graph`로 띄워야 값이 쌓인다**).
- `method.py` — env `SGLANG_PRISM_MASK_STATS=<out.json>`, `SGLANG_PRISM_MASK_STATS_EVERY`(기본 16).
- `test/prism/test_mask_stats.py` — 스칼라 루프(C++ 그대로 이식) 레퍼런스와 (layer, proj, tier) keep 수 정수 일치,
  thr=0/∞ 극단. CPU 전용, 3 passed.
- 드라이버 `scripts/mask_stats_prompt.py` — prompt 1개 전송 후 json 요약 표.
- 트리: `sglang-k1`에 원본, 같은 패치를 `sglang-glm`(실행 레인)에도 적용.

보고 json: `total.{gate,up,down,all}.{masked,warm,cold}.{rows, keep_frac, effective_sparsity}`,
`total.all.hot_if_masked`(hot은 dense — 참고값), `layers.<l>.<proj>.{hot,warm,cold,masked_keep_frac,s_mean}`.

## 2. 자산 어휘 실측 메모 (k1 calib 만들 때 참고)

- `thr[l,e,i]`는 `s_i = i·grid`의 **drop 분율 분위수**(keep = imp ≥ thr). `thr[...,0] = 0`, 단조증가,
  `thr[...,200] == SC[kmode+proj]`(전역 cap; GLM 17.698 = `SC["2lg"]`). 런타임은 pmax 0.9라 idx ≤ 180만 읽는다.
- k1 키는 `tg/tu/td`(접미사 없음), dense `qwen38_27b.pt`에 이미 있음(`KSET=['k1','k2wl2']`, `SC.k1_*` per-layer).
- `lam0` 산출식은 자산에 없다(메타 `TARGET_STD=0.1146`만). 마스킹은 decode에서만 → prefill 관측은 오염 없음.
- 기존 생성기: accuracy-eval, wikitext seq 1024, cal_tok 300032.
