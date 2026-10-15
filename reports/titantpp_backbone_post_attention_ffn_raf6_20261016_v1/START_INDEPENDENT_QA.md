# RAF FFN 실험 시작 관측의 독립 로컬 검산 — 완료

검산 결과는 `PASS_local_start_evidence`이며 열린 발견은 없습니다. 대상은 **2026-10-16 07:51:48.499060 KST에 저장된 관측 한 회차**입니다. 새 원격 조회나 학습 상태 갱신은 하지 않았습니다.

## 관측 당시 학습 상태

| 후보 | seed | 상태 | 저장 / best epoch | GPU PID |
|---|---:|---|---|---:|
| Current FFN | 42 | 진행 | 저장 전 / 미생성 | 3057360 |
| Current FFN | 52 | 진행 | 저장 전 / 미생성 | 3057362 |
| Current FFN | 62 | 진행 | 저장 전 / 미생성 | 3057364 |
| Temporal FFN | 42 | 대기 | 미생성 / 미생성 | — |
| Temporal FFN | 52 | 대기 | 미생성 / 미생성 | — |
| Temporal FFN | 62 | 대기 | 미생성 / 미생성 | — |

완료 0 · 진행 3 · 대기 3 · 미확정 0입니다. 완료 epoch 시간 표본은 0개이므로 ETA를 산출하지 않았습니다.

## 검증한 연결 관계

- 1,155개 로컬 봉인 파일의 SHA를 다시 계산했습니다. source 147개 중 부모의 145개가 동일 바이트이며 신규 모델·runner 두 개만 추가됐습니다. 배포 archive의 파일 158개와 manifest도 검산했습니다.
- 원문 transport의 실제 exit 0, 길이·SHA와 길이 구분자까지 검증했습니다. payload를 읽은 값과 round가 동일하며, round에서 다시 만든 summary도 저장값과 같습니다. latest의 경로와 round·summary SHA도 일치합니다.
- round에 담긴 원문 35개의 base64·바이트 수·SHA를 각각 검증했습니다. 계약·승인·시작 허가·native qualification·학습 허가·pipeline·dispatcher·worker·lease를 연결했습니다. 각 학습 PID의 argv, start ticks, 부모·process group과 GPU PID 집합이 일치합니다.
- native qualification은 두 후보 모두 실제 GPU의 합성 검증을 통과한 기록이며, 별도 3개 admission worker의 실제 exit 0과 두 동시 실행 장벽의 PID·GPU·메모리 증적을 확인했습니다. Python 3.12.13, torch 2.11.0+cu130, cuDNN 92000, 4 threads 및 계약의 GPU UUID가 일치합니다.
- RAF 실제 문맥 길이 84, 합성 batch 128, 활성 가중치 80,579개·등록 105,311개, 추가 FFN 1,152개·기존 QKV 지역 필터 576개, gradient·target 차단·평가 상태 불변 검증 기록이 계약과 일치합니다. 입력 SHA 및 Train 25,779 / Validation 6,690 대상 수를 확인했습니다.
- 원래 절대 host 기한 1792679647.915156과 각 fit의 36시간 제한이 유지됩니다. Current 세 개가 진행하고 Temporal 세 개에는 시작 claim·worker 기록이 없습니다.

독립 stdlib 검산은 3,357개 확인 항목을 통과했고 실제 종료 코드는 0입니다. 검산 프로그램·stdout/stderr 원문과 SHA는 JSON에 포함했습니다. 학습·observer·prepare·평가 helper를 호출하지 않았습니다.

## 원문 보존과 해석 범위

배포 transport의 exit는 0입니다. stderr 352바이트의 tmux `libtinfo` 버전 경고 두 줄은 원문 그대로 보존돼 있으며 학습 수치 실패로 분류하지 않았습니다. 관측 transport stderr는 0바이트입니다.

동시 admission의 참조 목록 22개 중 이번 round에 포함된 원문 11개는 독립 재해시했습니다. 나머지 11개는 qualification receipt에 SHA로 연결돼 있지만 이번 시작 관측에는 원문이 없어 독립 재해시했다고 주장하지 않습니다. 두 후보의 native 합성 보고, 3개 worker의 실제 exit와 ready/trained 동시 실행 장벽은 원문으로 검증했습니다.

이 결과는 해당 과거 관측의 검산입니다. 현재 GPU 유휴, 학습 종료, scientific worker·dispatcher·pipeline exit 0, checkpoint 선택, 원본 회수 또는 Validation/Test 완료를 증명하지 않습니다. supervisor wrapper 자신의 OS exit는 직접 관측하지 않았습니다. 기존 CNN을 재학습·재추론하지 않았습니다.

## 남은 작업 — 진행 중

승인된 기존 관측 일정으로 여섯 학습의 자연 종료를 확인합니다. 모든 종료·원본 증적이 검증된 뒤 고정 checkpoint의 native Validation 재현과 Test 비교를 이어갑니다.
