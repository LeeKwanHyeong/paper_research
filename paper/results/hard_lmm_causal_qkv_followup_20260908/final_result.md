# Causal QKV 탐색적 후속 평가 — 최종 결과

**완료·감사 통과, 추가 seed 검토 기준 미달.** Taxi의 수량 개선은 확인됐으나,
Intermittent의 RMSE와 두 데이터셋의 시간 loss가 악화했다. 고정한 기준에 따라
이 후보의 탐색을 종료한다. 원래 Instacart의 기준 미달 판정도 보존한다.

## 종료와 선택 checkpoint

| 서버 | 데이터셋 | 종료 시각 KST, 2026-09-08 | 종료 epoch | 선택 epoch |
| --- | --- | --- | ---: | ---: |
| RTX 5080 | Taxi | 08:52:56 | 153 | 113 |
| RTX 5090 | Intermittent | 09:16:26 | 47 | 7 |

두 실험 모두 최대300/min40/patience40 계약에 따른 최초 허용 시점에서 정상
조기 종료됐다. 선택 epoch는 validation raw RMSE의 가장 이른 엄격한 최솟값이다.
Intermittent의 선택 epoch7은 최소 학습 epoch40 규칙과 충돌하지 않는다.
최소40은 종료 가능 시점을 제한하며, checkpoint 후보에서 초기 epoch를 제외하지 않는다.

## B 대비 validation 결과

아래 수량 변화율은 음수가 개선이다. 시간은 현재 legacy clamped time loss이며,
다른 정상화된 likelihood의 NLL과 직접 비교하지 않는다.

| 데이터셋 | 전체 MAE 변화 | raw RMSE 변화 | body MAE 변화 | >p99 MAE 변화 | 시간 loss B → 후보 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Taxi | −5.7566% | −2.6737% | −13.5229% | −2.4492% | 1.473391 → 25.407218 |
| Intermittent | −2.8499% | +3.2568% | −7.1191% | +1.5624% | −3.483794 → −1.602208 |
| Instacart, 기존 결과 | +0.0460% | +0.0858% | −0.2997% | +1.2426% | 3.215523 → 3.215251 |

Taxi의 raw RMSE는 88.194997에서 85.836912, 전체 MAE는 28.674020에서
27.023365로 개선됐다. 그러나 시간 loss가 +23.933828 증가해 +0.01 guardrail을
초과했다. Intermittent의 raw RMSE는 1.499555에서 1.548392로 3.2568% 악화했고,
시간 loss도 +1.881585 증가했다.

기존 데이터셋별 gate에서 Taxi는 시간 항목에, Intermittent는 RMSE·시간 항목에
미달했다. 별도로 고정한 추가 seed 검토 기준도 Taxi 시간, Intermittent의 RMSE
악화1% 초과와 시간 항목 때문에 실패했다. Taxi의 RMSE 1% 이상 개선은 충족하지만
다른 실패 항목을 상쇄하지 않는다. 판정은 `stop_candidate_after_exploratory_followup`이다.

## 완료한 최종 감사

- 실행 소스 `84668e207d5121f211a6a93af12ca2f96068a25e`, 원래 모델 구현
  `a7795355920d4bdc5fb63dc40deccbfcbd4daec0`의 405개 Python 파일과
  데이터·split checksum을 두 서버에서 종료 후 다시 검증했다.
- 실제 전송한 archive를 별도 임시 경로에 추출해 로컬 최종 감사를 실행했다.
  현재 작업 폴더의 관련 없는 모델 수정은 보존하고 감사에 포함하지 않았다.
- 모든 epoch의 train 처리 건수는 Taxi38,393, Intermittent393,824로 일치한다.
  Validation 건수는 각각8,268/86,285이며 identity·수량 hash와 구간별 건수도 일치한다.
- 선택/마지막 checkpoint, 모델·optimizer의 finite 상태와 strict 복원,
  새 kernel lag의 학습, 선택 최솟값과 초기 중단 조건을 검증했다.
- 독립 JSON/history/구간별 검산에서 선택 summary와 history 지표 차이는0,
  구간 재조립 차이는 최대2.22e−16이었다. 로컬/원격 checkpoint·summary hash도 일치한다.
- 상세 숫자는 `final_decision.json`, 복원 감사는 `final_local_audit.json`,
  독립 수치 감사는 `final_independent_metric_review.json`에 보관했다.

## 종료 후 상태와 남은 작업

**완료 / 5080·5090·로컬 — 승인된 두 평가와 최종 증적 보관**
- 두 GPU 학습은 종료됐고, 마지막 확인에서 학습 프로세스가 없다.
- 시간별 heartbeat를 PAUSED로 변경했다. 추가 seed와 held-out 평가는 실행하지 않았다.

**다음 작업 / 로컬 — 연구 코드·증적의 통합 범위 검토**
- `paper_research/codex/hard-lmm-causal-qkv`에 결과를 보존한다.
  `master` 병합과 기본 모델 전환은 수행하지 않는다.
- 수량의 부분적 개선과 시간/RMSE의 악화를 함께 보고하며, 세 데이터셋 공통 개선이나
  최종 채택·통계적 유의성을 주장하지 않는다. 이 고정 후보의 후속 학습은 없다.
