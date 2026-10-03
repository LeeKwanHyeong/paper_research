# GPT-6 Astra 적용 상태와 연구 작업 검증

> 후속 변경: 사용자의 전환 승인 후 프로젝트 역할 설정을 Astra 4개·Terra 8개로
> 변경하고 Codex의 설정 연결을 검증했다. 아래 모델 점검은 **전환 전 기록**이며,
> 현재 구성은 [역할 모델 전환 기록](role_model_migration.md)을 따른다.

2026-09-10 KST. 대상: `paper_research`, 브랜치 `codex/hard-lmm-causal-qkv`.
작업 시작 HEAD: `1de2c31` (전체 SHA는 `model_audit.json` 참조).

**결론: 현재 주 작업은 실제로 `gpt-6-astra` / `high`를 사용한다.** 기존 연구
자료를 읽고 수치를 대조하는 이번 검증은 통과했다. 역할별 에이전트 12개는 별도로
GPT-5.4에 고정되어 있어, 모든 역할의 Astra 전환이 끝난 상태는 아니다.

## 1. 실제 모델과 예외 설정 — 완료

| 확인 대상 | 관찰 결과 | 판단 |
| --- | --- | --- |
| 사용자 `~/.codex/config.toml` | `gpt-6-astra`, `high` | 기본값이 목표와 일치 |
| 현재 실행 turn | 실행 기록 86행: `gpt-6-astra`, `high`, `default` | 설정 추정이 아닌 실행 기록으로 확인 |
| 프로젝트 trust | `trusted` | 프로젝트 설정을 읽을 수 있는 상태 |
| 프로젝트 및 검사한 상위 경로 `.codex/config.toml` | 별도 파일 없음 | 해당 경로에서 덮어쓰기 없음 |
| 사용자 `*.config.toml` 프로필 | 0개 | 별도 프로필 파일 발견되지 않음 |
| `/etc/codex/`의 config, managed_config, requirements | 검사한 파일 없음 | 해당 파일 기반 덮어쓰기 발견되지 않음 |
| macOS Codex managed preference 두 키 | `defaults read` 종료 코드 1 | 값 확인되지 않음; MDM 부재 전체를 단정하지 않음 |
| 다른 작업·원격 호스트·cloud 설정 payload | 조사하지 않음 | 현재 작업 결과를 전체 환경으로 일반화하지 않음 |

세부 경로와 선택한 설정 필드만 [model_audit.json](model_audit.json)에 보존했다.
인증정보와 전체 설정·대화 내용은 복사하지 않았다. 설정 변경은 없었다.
일반 설정 우선순위와 legacy managed 설정의 우선순위는 다르므로, 최종 판정은
현재 turn의 실제 모델 기록을 사용했다.
[공식 일반 설정 문서](https://learn.chatgpt.com/docs/config-file/config-basic#configuration-precedence),
[공식 managed 설정 문서](https://learn.chatgpt.com/docs/enterprise/managed-configuration#managed-defaults-managed_configtoml).

### 역할별 예외와 전환 범위 제안

아래 12개는 모두 `~/.codex/agents/`의 명시적 `gpt-5.4` 설정이다.
파일 머리말에 oh-my-agent가 생성한 설정이라고 표시되어 있다. 전역 기본 모델을
바꾸는 것만으로 이 명시적 예외가 Astra로 바뀌지는 않는다.

| 역할 | 현재 추론 수준 | 이 프로젝트의 전환 우선순위 제안 |
| --- | --- | --- |
| architecture-reviewer, debug-investigator, qa-reviewer | high | 구조·오류·검증 판단을 위한 Astra/high 비교 우선 |
| research-explorer | medium | 논문 근거 대조를 Astra/medium으로 비교; 현재 effort 유지 |
| docs-curator, pm-planner | medium | 문서·계획 작업의 품질과 사용량을 비교한 뒤 결정 |
| refactor-engineer | medium | 실제 리팩터링 작업이 필요할 때 비교 |
| backend-engineer, db-engineer, frontend-engineer, mobile-engineer, tf-infra-engineer | medium | 이번 연구 검증 범위에서 변경하지 않음 |

위 분류는 역할의 목적에 따른 제안이며 성능·비용 실험 결과가 아니다. 이번에는
서브에이전트를 실행하지 않았으므로 GPT-5.4 역할의 실제 가용성도 확인하지 않았다.
역할별 동일 과제의 사용량과 계정 과금 방식이 확인되지 않아 비용 절감률이나 달러
비용을 산정하지 않았다. 생성 파일을 바꾸기 전에는 OMA 설정 생성 경로도 확인해야 한다.

## 2. 연구 지침 충돌 정리 — 완료

[프로젝트 AGENTS.md](../../AGENTS.md)를 추가해 후속 작업이
[TEST_SESSION_PROTOCOL.md](../../TEST_SESSION_PROTOCOL.md)의 관련 규칙을 읽도록 했다.

| 기존 혼동 지점 | 적용한 정리 |
| --- | --- |
| 5090 기본 안내와 5080 사용자 예외가 함께 존재 | Model Enhancement의 5080 예외 우선, 5090은 명시적 요청 대상임을 본문·접속 제목에 명시 |
| 과거 서버 사용 상태가 현재 사실처럼 표현됨 | 당시 상황과 현재 실행 시 확인할 상태를 분리 |
| 일반 명령 예시와 실제 실행 권한의 구분 부족 | 사용자 작업 범위·승인 우선, 예시는 승인 자체가 아님을 명시 |
| DB·Runtime 변경 승인 경계가 실험 절차에 없음 | 사용자가 제공한 명시적 승인 대상을 0.1절에 반영 |
| validation-only 규칙과 test 결과 열람 템플릿이 혼재 | 0.2·9·11절에서 held-out lock 우선 및 조건부 열람으로 정리 |
| 과거 marked TPP 해석이 현재 v0.7에도 적용될 수 있음 | mark-free frozen 기준선과 과거·후속 실험 해석을 분리 |
| 현재 코드 검사와 frozen 결과 재현의 혼동 | cap 30과 과거 cap 300의 차이를 명시 |
| 문서 점검에도 GPU·Notion 절차가 확대 적용될 여지 | 실제 연구 실험 범위에만 적용하도록 명시 |
| 지속 polling 금지와 `tail -f` 예시가 병존 | 초기 확인 또는 요청된 결과 확인용 예시임을 명시 |

Astra 공식 문서가 안내하는 지침 충돌 검토를 이 프로젝트의 실제 규칙에 적용했다.
[공식 Astra 지침](https://developers.openai.com/api/docs/guides/latest-model#instruction-following).
사용자·역할 모델 설정, 실험 코드, checkpoint, 기존 논문과 결과표는 수정하지 않았다.

## 3. 대표 연구 작업 품질 검증 — 완료

### A. 코드 경로 설명과 CPU 계약 확인

현재 count-aware 실행 경로는 다음과 같다.

1. [실험 runner](../../paper/scripts/run_count_aware_tpp_backbone_control.py)는
   `train_one`을 호출한다.
2. [training.py](../../paper/scripts/count_aware_tpp_backbone/training.py)의
   `train_one`은 727행에서 공통 factory로 모델을 만든다.
3. [CountAwareFactory.py](../../models/TPPs/CountAwareFactory.py)의
   `build_count_aware_model`은 `titantpp`를 static hard memory 경로의
   `CountAwareTitanTPP`로 연결한다.
4. [CountAwareTPP.py](../../models/TPPs/CountAwareTPP.py)의 `continuous_features`는
   시간 간격과 수량의 `log1p`를 사용한다. `MemoryEncoder`는 causal attention과
   persistent memory를 사용하고, `HardLocalMemoryMatcher`는 64개 prototype 중
   cosine similarity 상위 4개를 선택해 평균 residual을 더한다.
5. [core.py](../../paper/scripts/count_aware_tpp_backbone/core.py)의
   `target_outputs`는 목표 수량을 가리고 목표 직전 hidden state에서 시간·수량을
   예측한다. frozen log-regression 계약에서는 joint loss가 시간 손실과 log-count
   MSE의 합이다.

합성 CPU 입력으로 목표 수량을 7에서 999로 바꾸어도 예측 수량이 같음을 확인했다.
평가 전후 model state가 동일하고, 수량 예측이 유한하며, mark head가 없고,
메모리 크기·top-k·joint loss가 설명과 일치했다.
이 경로는 과거 marked runner나 별도 TitansMAC 후보와 구분했다.

현재 cap 30 동작을 확인했지만 과거 cap 300 checkpoint를 재현한 것은 아니다.
논문도 이 차이를 이미 명시한다:
[manuscript Method](../../paper/titantpp_short_paper_draft_v0_7_manuscript.md).

### B. 결과표 재계산과 주장 대조

기존 독립 verifier의 읽기·검증 함수만 호출하고 기존 보고서 writer는 실행하지 않았다.
검증 입력 경로와 SHA-256은 [research_verification.json](research_verification.json)에 있다.

| 검증 대상 | 결과 |
| --- | --- |
| 3 datasets × 3 models × 3 seeds | validation 27행, 중복·누락·평가 범위 확인 |
| Taxi·Instacart의 원본 validation export 대조 | 18 seed행의 주요 5개 지표 대조 |
| T3 집계표 / T4 paired 비교 / T5 수량 구간표 | 9 / 18 / 45행 검증 |
| T5 수량 구간 seed 자료 | 135행 재계산 |
| manuscript와 README | 표 수치·표본 표준편차·링크 정규화 후 내용 일치 |
| 자동 검증 합계 | 1,692 / 1,692 통과, 수치 비교 851회 |
| 기존 count-aware 계약 테스트 | 8 / 8 통과 |

대표 주장의 재계산은 다음과 같다. 개선율은 `(기준값 - TitanTPP) / abs(기준값)`이다.

| validation 비교 | 재계산 | 논문 표현 | 판정 |
| --- | --- | --- | --- |
| Intermittent MAE vs RMTPP | 74.2666% 개선 | 74.3% 개선 | 일치 |
| Intermittent RMSE vs RMTPP | 81.8549% 개선 | 81.9% 개선 | 일치 |
| Intermittent RMSE vs THP | 10.7507% 개선 | 10.8% 개선 | 일치 |
| Intermittent MAE vs THP | 12.0859% 악화 | 12.1% 악화 | 일치 |
| Taxi RMSE vs RMTPP | 0.5674% 개선 | 0.57% 개선 | 일치 |
| Instacart 집계 지표 | 주요 5개 지표에서 RMTPP 최저 | RMTPP 우세 | 일치 |

각 주장은 [T3 validation 집계표](../../paper/tables/T3_v0_7_backbone_validation.csv)와
[논문](../../paper/titantpp_short_paper_draft_v0_7_manuscript.md)을 대조했다.
원본 `time_nll` 명칭은 provenance로 보존하며 해석은 clamped time loss로 제한한다.
위 결과는 전체 데이터셋에서 TitanTPP가 우월하다는 주장을 지지하지 않는다.

검증 JSON의 최대 절대 차이 0.0493255는 THP 대비 RMSE 개선율을 소수 첫째 자리로
반올림한 10.8과 원값의 차이(퍼센트포인트)다. 모든 비교를 같은 허용오차로
판정한 것이 아니며, 기존 수치 비교 허용오차와 논문 표시 정밀도를 각각 적용했다.

### 검증 범위와 제한

- held-out 예측·성능 파일을 열거나 평가하지 않았다. 논문 표의 train/validation/test
  행 수는 이미 공개된 CSV와 대조했으며, 원본 데이터 전체 통계는 재계산하지 않았다.
- GPU·원격 Runtime·DB·Notion 작업, 재학습, checkpoint replay는 수행하지 않았다.
- 전체 publication manifest와 모든 문헌의 외부 진위는 이번 검증 대상이 아니다.
- 이번 실제 Astra 작업의 정확성 점검이며, 다른 모델과의 통제된 품질·속도 비교는 아니다.
- Python 3.12.10 / Torch 2.14.0 / CPU. 문서·수치·합성 입력 검증 10.063초,
  pytest 9.006초. 두 명령은 독립적으로 실행했으므로 시간을 합산하지 않는다.
- Matplotlib의 기본 캐시 경로 쓰기 경고 후 임시 경로를 사용했고, pytest에서는
  `pkg_resources` 폐기 예정 경고가 있었다. 검증 실패는 없었다.

## 4. 실행 증적과 사용량

재실행 명령(저장소 루트 기준):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 reports/gpt6_astra_readiness_20260910/verify_research.py
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider simple_lab_test/search/tests/test_count_aware_tpp_contract.py --junitxml=reports/gpt6_astra_readiness_20260910/contract_tests.xml
```

[usage_snapshot.json](usage_snapshot.json)은 현재 turn의 실행 기록에서 추출한
중간 사용량이다. 09:45:07 KST 시작부터 09:51:33 KST 스냅샷까지 386.042초이며,
해당 시점의 누적 입력 1,699,707 tokens(캐시 입력 1,640,960 포함), 출력
11,574 tokens로 기록되었다. 입력에는 여러 요청에서 반복 전송된 context가 포함된다.
이는 고유 문서 길이나 신규 과금 토큰 수가 아니며, 최종 문서 작성·응답까지 포함한
최종 사용량도 아니다. 계정 사용 제한 퍼센트나 달러 비용으로 환산하지 않았다.

## 5. 초기 점검 종료 시 남은 작업 순서

아래는 초기 점검 시점의 순서다. 이후 승인·완료된 역할 전환과 현재 남은 확인은
[후속 기록](role_model_migration.md)에 정리했다.

**역할별 모델 전환 범위 결정 — 다음 작업**
- 현재 주 작업의 Astra/high 적용은 완료된 기준선이다.
- 연구에 사용하는 역할부터 동일 과제의 품질·사용량·가용성을 비교하고, OMA 설정
  생성 경로를 확인한 뒤 필요한 역할만 전환한다. 이번에는 설정을 변경하지 않았다.

**변경사항 검토와 버전 관리 — 다음 작업**
- `paper_research`의 `codex/hard-lmm-causal-qkv` 작업 트리에 이번 지침과 검증
  증적만 추가했다. 기존 미추적 연구 산출물은 보존했다.
- 커밋·Push·MR은 수행하지 않았다. 별도 정리가 필요하면 저장소와 대상 브랜치를
  확정하고 이번 변경만 포함한다.
