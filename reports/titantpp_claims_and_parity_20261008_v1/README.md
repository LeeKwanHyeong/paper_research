# TitanTPP 주장 계약과 동일 환경 기준선 검사

현재 상태: 승인된1·2단계를 완료했다. 과학적 질문·비교군·예산·선택 계약을 정했고, 2026-10-08 09:11:52KST의5080 native 검사에서 CNN+GRU54와 mean이 총83개 bitwise 비교를 통과했다. 앞으로의 동일 환경 비교에 사용할 기준선이 확보됐으며, 이 결과는 수량 성능 우월성이나 과거 A100 학습 경로 동등성의 증명이 아니다. 이 문서는 연구 실행과 검증 증적이며 투고용 본문이 아니다.

**논문의 중심 질문과 보조 질문을 분리한다 — 완료**

- 중심 질문은 인코더의 이력 요약 위에 직전 관측 상태를 명시적으로 연결하는 보정이 수량 예측을 추가로 개선하는가이다. 대표 구조는 동결 CNN+GRU54이며 보정 제거와 같은 활성 예산의 현재 상태 전용 보정을 비교한다.
- 검색부는 시간·수량 점수를 분리한 정적 prototype 검색이 mean·weighted·같은 활성512개 예산의 일반 검색보다 나은가라는 별도 질문이다. fixed는 후속 후보이며 기존 generic521개는 정확한 예산 대조가 아니다.
- 전체 데이터에 같은 모델 버전을 적용한다. full Validation raw 전체 수량 RMSE의 최초 strict finite 최소 epoch를 선택하고 같은 epoch의 전체/큰 수량 RMSE·MAE·TimeNLL을 함께 보고한다.
- 예정 데이터3개·seed42/52/62의 학습 예산과 비용 기준, 개발 단계 판정, 독립 평가 경계를 [주장 계약](CLAIM_CONTRACT.md)과 [현재 JSON 계약 v2](/Users/igwanhyeong/PycharmProjects/paper_research/paper/contracts/titantpp_claims_validation_20261008_v2.json)에 기록했다. 이 계약은 새 확대 학습의 실행 승인이 아니다.
- 시간 단위는 Taxi=hour, Intermittent=week, RAF=month다. 최초 계약의 공통 hour 표기는 v2에서 정정했고, native 계산이 참조한 v1과 rawSHA는 그대로 보존했다. 실제 Taxi 검사는 동결 데이터 계약을 사용했으며 이 문서 표기 정정으로 계산 결과가 변경되지 않았다. 서로 다른 데이터의 TimeNLL을 하나로 합산하지 않는다.

**선행 구성요소와 추가 검증 대상을 확인한다 — 완료 / 신규성 미확정**

- [표적 문헌 검토](LITERATURE_REVIEW.md)는7편의 방법 절을 확인했다. 이력 요약·residual·정적 학습 기억·top-k 가중 검색은 기존 구성요소다. 특정 보정 경로와 수량 예측상의 추가 효과, 검색 정보 분리의 제약 효과를 검증해야 한다.
- 고정 두 점수 합이 일반 내적 검색으로 재표현되는지와 선택·결합 효과를 분리하는 대조가 필요하다. S2P2와 최근 count-aware TPP의 포괄적 대조는 후속 문헌 작업이다.

**새 독립 평가에 사용할 자료의 확보 가능성을 확인한다 — 후보 확인 완료 / 미노출 인증 대기**

- 이후 기간 Taxi는 공식 TLC 목록의2015년2–3월 자료를 후보로 확인했다. 새 원본을 내려받지 않았고 기존 접근 이력도 완전히 확인하지 않았으므로 외부 자료 확보와 미노출 감사가 남았다.
- Online Retail II는 로컬 준비 이력이 있지만 Train/Validation의 과거 사용이 있다. 예약할 Test 기간의 접근 이력 확인 전에는 untouched 평가로 인증하지 않는다.
- 기존 Taxi·Intermittent·RAF와 Instacart의 Test 노출 증거를 새 독립 평가와 구분했다. 이번 조사는 [자료 계보 metadata](evidence/independent_evaluation_inventory.json)만 다뤘으며 새 성과 평가를 하지 않았다.

**하나의 5080 Runtime에서 두 구현의 동등성을 검증한다 — 완료**

- 대상: GPU `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, 기존5080 `ai_env`, 과학 revision `4274043717cf24640d9a18fb35231a1f7ba50c8e`의 동결126파일. 완료 원격 경로는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_baseline_parity_20261008_v1_5080_attempt3`다.
- 두 구현은 seed42에서 각각 자연 초기화한다. 구현 간 파라미터를 복사하지 않는다. 합성 B128/T64,T256에서 eval과1회 업데이트씩, 실제 Taxi 첫 Train 배치를 반복한3회 업데이트를 비교한다. 각 case의 모델 쌍은 fresh 초기화한다.
- 초기 tensor·constructor RNG·첫 예측·손실·전후 clipping gradient·AdamW state·업데이트 이후 파라미터·종료 RNG를 bitwise로 검증한다. mean의 식별 buffer만 허용된 추가 상태다.
- 900초·GPU8GiB·RSS16GiB 상한, 배타적 intent/claim, source/input rawSHA, 실제 Runtime·GPU PID, 종료 후 GPU PID 부재를 검증한다. 새 full fit·Validation/Test 추론·checkpoint 선택을 하지 않는다.
- 과거 A100 학습의 초기화·학습 경로 동등성은 이 새 환경 검사로 증명되지 않는다. 전체 학습 성능 차이나 고정 검색의 기존 강한 모델 대비 개선도 이번 검사의 결론이 아니다.

| 검사 | 입력 | 각 구현 업데이트 | bitwise 비교 | 결과 |
|---|---|---:|---:|---|
| 합성 짧은 입력 | B128/T64 | 1 | 21 | PASS |
| 합성 긴 입력 | B128/T256 | 1 | 21 | PASS |
| 실제 첫 Train 배치 반복 | B128/T256 | 3 | 41 | PASS |

- 두 모델 모두 학습 가능한 파라미터115,167개다. mean의 전체 state SHA는 추가 식별 buffer 때문에 다르지만, 그 buffer를 제외한 공통 초기 state SHA는 CNN+GRU54와 동일한 `592be61554bfc0731ce67abce774550d462770d022825c8db4d119c60289388c`다. buffer 자체도 업데이트 전후 그대로 유지됐다.
- 실제 Train만38,524행 materialize했고 eligible target38,393개·Train 수량/시간 통계·target order를 동결 계약과 대조했다. 독립 로더2개의 첫 배치 byte digest도 일치했다. 같은 첫 배치를3번 사용한 짧은 진단이며3개 epoch나 전체 학습이 아니다.
- 실제 Runtime은 Python3.12.13, torch2.11.0+cu130/CUDA13.0/cuDNN91900, driver595.84다. 엄격한 deterministic mode·TF32 off를 적용했다. 전체 bounded native 검사5.01초, 최대 GPU allocation2.12GB·reservation2.21GB, 최대 RSS2.67GB였다. 이 시간은 모델 성능/구조 효율 벤치마크로 사용하지 않는다.
- wrapper exit0·실제 소유 GPU PID binding·종료 후 GPU compute PID 부재를 확인했다. 동결 source126·공유117·입력 파일 SHA가 전후 동일했다. [native 원본](evidence/attempt3/result_5080.json), [실제 종료](evidence/attempt3/execution_receipt.json), [완료 증적](evidence/attempt3/completion_receipt.json)을 보존했다.
- 앞선2번은 각각 GPU UUID 접두사 표기와 CUDA 메모리 API의 장치 번호 요구로 모델 생성 전0회 업데이트에서 종료됐다. UUID의 원 표기와 정규화 값을 함께 기록하고 같은 장치`cuda:0`을 명시해 해결했다. [첫 실패](evidence/attempt1/result_5080.json)와 [두 번째 실패](evidence/attempt2/result_5080.json), 기존 claim·원본은 보존했다. 과학 소스·데이터·손실·공용 Runtime·학습 예산은 바꾸지 않았다.
- CPU 단위/배포 경계22테스트와4subtest가 통과했다. 파라미터·RNG·입력·optimizer drift, signed-zero byte 차이, 실제 업데이트 부재, 잘못된 GPU UUID, 범위 확대와 intent 덮어쓰기를 거부하는 검사를 포함한다. [독립 QA](evidence/qa_review.json)는83개 비교와 공통 tensor 증거1,976개, 실제 업데이트·Runtime·종료·원본SHA를 대조해 PASS로 판정했으며 미해결 발견사항은 없었다.

**코드와 증적을 저장소에 보존한다 — 완료**

- 대상 저장소는 `paper_research`, 브랜치는 `codex/contextual-retrieval-pilot-20261007`이다. 준비 코드·계약은`a6e1874`, UUID 운영 수정은`013ff6c`, 명시적 CUDA 장치 실행은`51cbfd1`에 보존했다. 최종 결과와 문서는 별도 로컬 커밋으로 기록한다.

**남은 작업을 의존관계에 맞춰 진행한다 — 다음 작업**

1. 이력 보정의 추가 효과를 확인할 보정 제거·활성 예산 현재 상태 대조의 구체적 구현 계약을 정한다 — 다음 작업. 공통 구조와 비교 기준이 확정된 뒤 구현·계약 테스트를 수행한다.
2. 검색부 fixed와 같은 활성512개 예산의 일반 검색, score 정규화·선택/결합 분리 대조를 구체화한다 — 중심 보정 대조 이후의 조건부 작업. 중심 근거와 섞어 독립 기여로 먼저 확정하지 않는다.
3. 새 GPU 대조 학습은 별도 실행 범위와 운영 예산의 승인 이후 수행한다 — 승인 필요. 선행문헌 보강과 독립 평가 후보의 접근 이력 감사는 병렬로 진행할 수 있다 — 다음 작업/외부 자료 확보 대기.
4. 개발 단계 근거가 확보되면 모델·selector를 동결하고 예약한 미노출 자료에서 독립 평가한다 — 후속 작업. 현재 Test를 새 독립 근거로 다시 표시하지 않는다.
