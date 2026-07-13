다음 strict matched-budget 실험 결과를 기존 Notion 페이지에 업데이트해주세요.

## 작성 역할

실험 결과를 옮겨 적는 요약문이 아니라, 이후 모델 선택 근거로 다시 읽을 수 있는
한국어 연구 기록을 작성해주세요. 아래에 제공된 수치와 artifact만 근거로 사용하고,
확인되지 않은 원인이나 추가 성능을 추정하지 마세요.

## 문체 원칙

- 사람이 실험 직후 정리한 기술 연구노트처럼 담백하게 작성합니다.
- 첫 문단에서 실험을 반복 설명하지 말고, 비교 조건과 최종 판정을 바로 밝힙니다.
- 표에 있는 수치를 문장으로 전부 반복하지 않습니다. 본문은 수치가 의미하는 차이,
  남아 있는 예외, 다음 판단에 필요한 내용만 설명합니다.
- 짧은 문장과 설명 문장을 섞되, 모든 문단을 같은 길이나 같은 문장 구조로 맞추지
  않습니다. 한 문단은 보통 2~4문장으로 제한합니다.
- `종합적으로`, `주목할 만한`, `유의미한`, `괄목할 만한`, `강력한 성능`,
  `명확하게 입증`, `혁신적인`, `이를 통해 알 수 있듯이`, `결론적으로` 같은
  상투적이거나 홍보성인 표현은 쓰지 않습니다.
- `~을 확인할 수 있었습니다`, `~라고 볼 수 있습니다`를 반복하지 않습니다.
  근거가 직접 확인된 경우에는 `감소했다`, `통과했다`, `종료됐다`처럼 바로 씁니다.
- 개선을 설명할 때는 형용사 대신 수치를 씁니다. 예: `크게 개선됐다`만 쓰지 말고
  `quantity MAE가 49.086% 감소했다`라고 씁니다.
- 해석은 단정하지 않습니다. artifact가 직접 뒷받침하지 않는 원인은
  `가능성이 있다`, `후속 실험에서 분리해 확인해야 한다` 수준으로 제한합니다.
- 한글과 영어 용어를 불필요하게 번갈아 쓰지 않습니다. 코드 옵션, metric 이름,
  model variant는 아래 표기 그대로 유지하고 백틱으로 표시합니다.
- emoji, 장식용 구분선, 과도한 callout, 질문형 소제목은 사용하지 않습니다.
- `핵심 해석`, `최종 판정`, `다음 액션`을 각기 다른 말로 반복하는 결론 문단을
  만들지 않습니다. 최종 결정은 한 곳에서 한 번만 정리합니다.

## 근거와 편집 원칙

- confirmed fact와 interpretation을 섞지 않습니다. 결과 수치 다음에 해석과 제한을
  별도 문단으로 둡니다.
- total NLL, marker NLL, time NLL을 분리해 적고, total NLL만으로 모델 전체가
  좋아졌다고 표현하지 않습니다.
- V2 seed 62의 quantity outlier를 숨기지 않되, 이 값만 제거하거나 별도 계산한
  결과를 새로 만들지 않습니다.
- V2 best epoch가 `48-50`이라는 사실과 e50 이후 개선 가능성을 함께 기록합니다.
- Taxi 결과만으로 V3b가 모든 데이터셋에서 V2보다 우월하다고 일반화하지 않습니다.
- 기존 `TitanTPP V3b Taxi Multi-Seed Confirmation e50` 페이지의 e200 비교는
  삭제하지 말고, 이번 e50 matched comparison이 기존 한계를 보완했다는 관계만
  명시합니다.
- 새 수치, 원인, citation을 추가하지 않습니다. 제공된 값끼리 재계산해야 할 경우
  원문과 일치하는지 확인한 뒤 사용합니다.

## 권장 페이지 구성

아래 순서는 유지하되, 각 섹션을 불필요하게 길게 확장하지 마세요.

1. 상태와 판정: `completed`, strict matched-budget confirmation `PASS`
2. 비교 목적과 통제 조건: 왜 e50 matched comparison이 필요했는지 1개 문단
3. 실행 정보와 artifact: 접을 수 있는 상세 블록 또는 간단한 표
4. 평균 held-out test 결과: 핵심 비교표 1개
5. Seed 및 scale 확인: 필요한 표와 1~2개 해석 문단
6. 안정성과 제한: variance, time NLL, best epoch 경계
7. 모델 결정과 다음 작업: V3b Taxi 전용 승격 및 후속 실험

페이지 상단 callout에는 아래 문장 수준으로만 판정을 요약해주세요.

> 동일 e50 조건에서 V3b는 세 seed 모두 NLL, marker NLL, quantity MAE,
> mark accuracy를 함께 개선했다. Taxi 전용 모델 강화안으로 유지하되,
> Intermittent를 포함한 공통 baseline은 V2로 남긴다.

Notion 위치:
- `2. Confirm and Refine Topic > Model Validation`
- 같은 제목의 페이지를 먼저 검색하고, 있으면 해당 페이지를 업데이트합니다.
- 같은 제목의 페이지가 없을 때만 아래 제목으로 새 페이지를 만듭니다.
- 유사한 기존 실험 페이지를 새 comparison 페이지로 이름만 바꾸거나 덮어쓰지 않습니다.

대상 페이지 제목:
- `TitanTPP V2/V3b Taxi Strict Matched-Budget Comparison e50`

연결할 기존 페이지:
- `TitanTPP V3b Taxi Multi-Seed Confirmation e50`와 상호 링크
- `5. Model Design Enhancement`의 V3b 설계 페이지에 최종 판정 2~3문장과 링크 추가

실험 상태:
- `completed`

실험 시간:
- 실험 시작 시각: `2026-07-10 16:42:36 KST`
- 실험 종료 시각: `2026-07-10 17:03:17 KST`
- 총 소요시간: `20분 41초`
- 실행 서버: `5090` (`192.168.0.71:22`)
- tmux session: `taxi_v2_multiseed_e50_0710`
- conda env: `/opt/miniconda3/envs/ai_env`

실험 목적:
- 기존 V3b Taxi multi-seed 결과의 핵심 한계였던 V2 e200 대 V3b e50 epoch budget mismatch를 제거
- V2와 V3b를 동일한 e50, seeds `42,52,62`, data split 및 학습 조건에서 비교
- V3b의 NLL, marker, quantity 개선과 seed 안정성이 strict matched condition에서도 유지되는지 확인

결과 artifact:
- V2 local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710`
- V3b local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`
- V2 server: `/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710`

실행 완료 및 무결성:
- V2 seeds `42,52,62`의 `3/3` run 완료
- NaN, Traceback, ERROR 없음
- dataset `yellow_trip_hourly`, candidate `mid_lmm`, epochs `50`
- V2: `value_head_mode=shared`, `qty_mark_gradient_mode=coupled`
- V3b: `value_head_mode=mark_conditioned_experts`, `qty_mark_gradient_mode=detached`
- 공통 조건: lr `1e-3`, batch size `128`, lookback `168`, max sequence length `256`
- 공통 조건: fixed split, target-only, hybrid loss, residual input, best-validation-NLL selection

Artifact 분석 순서:
- manifest
- log
- summary
- test_summary
- histories
- validation scale-wise metrics
- test scale-wise metrics
- report
- plots

V2 validation 결과:
- mean best validation NLL: `1.587014 ± 0.008673`
- best epoch: seed 42=`50`, seed 52=`48`, seed 62=`49`
- mean best validation quantity MAE: `67.381351 ± 33.722702`
- mean best validation mark accuracy: `91.630%`
- mean final validation NLL: `1.602461`
- mean final validation quantity MAE: `72.192574`

`best_val_nll` held-out test 평균 비교:

| Metric | V2 e50 | V3b e50 | V3b change |
| --- | ---: | ---: | ---: |
| Total NLL | `1.650430` | `1.611892` | `-2.335%` |
| Marker NLL | `0.249688` | `0.208618` | `-16.448%` |
| Time NLL | `1.400742` | `1.403274` | `+0.181%` |
| Quantity MAE | `75.249345` | `38.312184` | `-49.086%` |
| Value MAE | `0.208378` | `0.151485` | `-27.303%` |
| Mark accuracy | `91.173%` | `91.902%` | `+0.729%p` |

V2 seed별 `best_val_nll` held-out test:

| Seed | Total NLL | Marker NLL | Quantity MAE | Mark accuracy | Value MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `1.638664` | `0.240351` | `48.613045` | `91.317%` | `0.202651` |
| 52 | `1.653831` | `0.249928` | `59.549083` | `91.005%` | `0.204201` |
| 62 | `1.658794` | `0.258784` | `117.585907` | `91.197%` | `0.218282` |

Seed-matched V3b 변화:

| Seed | Total NLL | Marker NLL | Quantity MAE | Mark accuracy | Value MAE | 동시개선 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | `-1.664%` | `-12.924%` | `-16.446%` | `+0.540%p` | `-24.182%` | PASS |
| 52 | `-2.545%` | `-16.633%` | `-30.574%` | `+0.877%p` | `-27.704%` | PASS |
| 62 | `-2.788%` | `-19.543%` | `-71.956%` | `+0.769%p` | `-29.824%` | PASS |

Scale-wise held-out test quantity MAE:

| Scale | Share | V2 e50 | V3b e50 | V3b change |
| --- | ---: | ---: | ---: | ---: |
| `1-9` | `54.23%` | `1.646924` | `1.561491` | `-5.187%` |
| `10-99` | `24.33%` | `12.996297` | `11.434228` | `-12.019%` |
| `100-999` | `13.89%` | `167.416043` | `81.922968` | `-51.066%` |
| `1000-9999` | `7.54%` | `635.561996` | `308.954200` | `-51.389%` |

Stability:
- test NLL std: V2 `0.010487`, V3b `0.000590`, `-94.378%`
- quantity MAE std: V2 `37.070037`, V3b `4.635932`, `-87.494%`
- test NLL CV: V2 `0.635%`, V3b `0.037%`
- quantity MAE CV: V2 `49.263%`, V3b `12.100%`
- V2 seed 62 quantity MAE `117.585907`이 V2 평균과 분산을 크게 높임
- V3b는 outlier seed 62뿐 아니라 세 seed 각각에서 모든 핵심 동시개선 gate를 통과

Strict confirmation gate:

| Gate | Result |
| --- | --- |
| V2/V3b 동일 e50, seed, split, optimizer 및 data condition | PASS |
| 3/3 완료, NaN/Traceback/ERROR 없음 | PASS |
| Mean total NLL 악화 <= 0.5% | PASS, `-2.335%` |
| Mean marker NLL 악화 <= 2% | PASS, `-16.448%` |
| Mean mark accuracy regression <= 0.25%p | PASS, `+0.729%p` |
| Mean quantity MAE 개선 | PASS, `-49.086%` |
| Seed-matched NLL/marker/quantity/accuracy 동시개선 2/3 이상 | PASS, `3/3` |
| Share >= 5% bucket의 quantity regression 없음 | PASS, 모든 bucket 개선 |

핵심 해석:
- V3b의 Taxi 개선은 V2/V3b epoch budget 차이나 단일 seed 효과가 아닙니다.
- total NLL 개선은 marker NLL `-16.448%`가 주도하며 time NLL은 `+0.181%`로 소폭 악화됐습니다.
- quantity MAE는 평균 `49.086%` 개선됐고 모든 scale에서 개선됐습니다.
- 특히 `100-999`, `1000-9999` 대수요 bucket에서 약 `51%` 개선되어 mark-conditioned expert 구조의 효과가 scale 편향 구간에서 크게 나타났습니다.
- V2의 quantity 결과는 seed 62에서 크게 흔들렸지만 V3b는 NLL 및 quantity 분산을 동시에 줄였습니다.
- V2 best epoch가 `48-50`에 위치하므로 e50 이후 추가 개선 가능성은 남아 있으며, 장기학습 비교에는 early stopping과 동일 budget을 적용해야 합니다.

최종 판정:
- strict matched-budget confirmation `PASS`
- V3b를 Taxi 전용 `confirmed model enhancement`로 유지
- Taxi 후속 본실험의 기본 value head를 `mark_conditioned_experts + detached`로 설정
- V2는 Intermittent를 포함한 전체 데이터셋 공통 baseline으로 유지
- V3b를 모든 데이터셋의 공통 replacement로 승격하지 않음

다음 액션:
- Taxi V2/V3b 동일 e100 또는 e200 + early stopping 장기 안정성 비교
- V3b의 time NLL 소폭 악화를 줄이는 time-head 비간섭 설계 검토
- Intermittent는 V2 baseline을 유지하고 V3c shared-encoder gradient routing 설계 진행

## 작성 후 자체 점검

Notion 업데이트를 마치기 전에 아래 항목을 확인해주세요.

- 같은 결론을 callout, 본문, 마지막 요약에서 세 번 반복하지 않았는가
- 표의 모든 숫자를 본문에서 다시 읽어주고 있지 않은가
- 직접 확인된 결과와 원인에 대한 추론이 구분되어 있는가
- V3b를 전 데이터셋 공통 우위로 과장하지 않았는가
- time NLL `+0.181%`와 V2 best epoch `48-50` 제한을 빠뜨리지 않았는가
- `유의미한`, `주목할 만한`, `종합적으로` 같은 자동 생성 문구가 남아 있지 않은가
- 기존 페이지를 갱신하고 관련 페이지를 상호 링크했으며 중복 페이지를 만들지 않았는가

업데이트가 끝나면 새 문서 본문을 다시 길게 재출력하지 말고, 수정한 페이지 제목,
추가한 섹션, 연결한 관련 페이지, 남겨둔 한계만 짧게 보고해주세요.
