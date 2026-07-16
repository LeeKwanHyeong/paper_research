## 2026-07-17 | TitanTPP Model Status와 Baseline 재정리

### 목적

구현 완료, integration 통과, 성능 승격을 분리하고 현재 데이터셋별 incumbent와
후속 비교 기준을 하나로 고정한다. 이 lock은 다음 구조 실험을 위한 기준이며 논문
최종 모델 확정을 뜻하지 않는다.

### 현재 Baseline

| Dataset | Common control | Active incumbent | 후속 비교 |
| --- | --- | --- | --- |
| Intermittent | V2 `small_lmm` | V2 `small_lmm` | fresh matched V2와 비교 |
| Yellow Trip Hourly / Taxi | V2 `mid_lmm` | V3b `mid_lmm` | replacement는 V3b, attribution은 V2와 비교 |
| Instacart | V2 `small_lmm` | V2 `small_lmm` | fresh matched V2와 비교 |

V2는 residual value input, shared value/time head, coupled gradient routing,
plain CE, hybrid quantity objective, target-only supervision을 사용한다. Taxi V3b는
V2에서 value head만 mark-conditioned experts로 바꾸고 quantity-to-mark gate를
detach한 모델이다.

### 모델 상태

| Model | 상태 | 판단 |
| --- | --- | --- |
| R0 RMTPP / L0 legacy TitanTPP / V1 | reference only | 외부·과거·보조 비교 기준 |
| V2 | active baseline | 전 데이터셋 common control |
| V3a | 미승격 | quantity 이득은 있었지만 marker guardrail 실패 |
| V3b | Taxi 전용 승격 | strict matched e50, seeds 42/52/62 동시 통과 |
| V3c | 미승격 | Intermittent gate 실패, V2 유지 |
| V4a/V4b | 미승격 | Taxi time NLL primary gate 미달 |
| V5a | 미승격 | Intermittent ordinal objective gate 실패 |
| V5b | 보류 | class-prior fallback idea만 유지 |
| M0 | 미승격 | log-domain direct/global negative ablation |
| M1-M4 | 종료 | M0 prerequisite 실패 후 미실행 |
| Q0/Q1/Q2 | 미승격 | raw 또는 short-history 이득과 marker/low-scale safety 동시 충족 실패 |
| Q3a/Q3b/Q3c | 종료 | factorial screening 후 V2 유지 |
| V6 | 보류 | `series_lmm` scaffold만 있고 runner memory injection과 품질 실험 없음 |

### 해석

- 현재 공통 TitanTPP baseline은 V2다.
- 현재 구조적으로 승격된 모델은 Taxi의 V3b 하나다.
- strict Q2 e3 A/B exact pass는 재현성 infrastructure 결과이며 Q2 승격 근거가 아니다.
- Instacart `dataset_best=mid_lmm`는 일반 추천이고, enhancement baseline lock은
  V1/V2 e200에서 사용한 `small_lmm`이다.
- CUDA model-test와 top-20 e1 smoke는 구현·integration gate이며 모델 품질 순위를
  만들지 않는다.

### 후속 비교 규칙

첫 품질 비교는 strict fixed-split validation-only로 수행한다. 사전 gate를 통과한
후보만 multi-seed로 확장하고, 구조와 coefficient를 freeze한 뒤 held-out test를
한 번 연다. Intermittent와 Instacart는 V2, Taxi는 V3b를 primary incumbent로
사용하며 Taxi에서는 V2를 attribution control로 함께 둔다.

### Notion 직접 반영 결과

- 반영 시각: `2026-07-17 06:29:42 KST`
- 전략 페이지: `TitanTPP Model Enhancement Strategy`
  (`https://app.notion.com/p/394bbe4056138046bf3bfbc6f4c8c31a`)
- 상위 페이지: `5. Model Design Enhancement`
  (`https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`)
- 전략 페이지에서 2026-07-17 baseline lock, Taxi V3b, Q3 종료,
  validation-first/held-out lock, V5a 최종 상태를 재조회 확인했다.
- 상위 페이지에서 공통 V2, Taxi V3b, Q3a/Q3b/Q3c 미승격,
  strict Q2 `22/22` infrastructure-only 문구를 재조회 확인했다.
