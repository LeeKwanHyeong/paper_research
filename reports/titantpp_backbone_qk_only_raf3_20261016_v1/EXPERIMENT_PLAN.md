# RAF Q/K-only CNN 구조 축소 실험 계획

- Origin Skill: academic-research-suite / experiment-agent
- Mode: plan
- Date: 2026-10-16T03:35:32.809591+09:00
- Version: code_plan_v1
- Verification Status: UNVERIFIED — 과학 명세와 중복 감사만 작성했으며, 새 구현·native 검증·학습은 아직 완료하지 않았다.

**현재 기준선과 진단 근거를 고정한다 — 완료**
- 기존 CNN 모델·checkpoint·결과는 읽기 검증 후 저장 결과만 비교에 재사용한다. 기존 CNN의 재학습과 재추론은 하지 않는다.
- 현재 V 차분 모델의 고정 가중치에서 V 잔차를 끄면 Validation MAE는 세 seed 모두 개선했지만, Train RMSE는 세 seed 모두 악화했다. Validation RMSE도 행 가중에서는 일관되지 않았다. 큰수량·시간 지표의 악화도 보존한다.
- 따라서 V가 항상 해롭다고 결론 내리지 않는다. 이번 질문은 **V 추가 CNN 없이 Q/K와 공통 표현을 처음부터 함께 학습했을 때도 개선되는가**이다. 기존 on/off 진단은 이 질문에 답하지 못한다.
- 근거 원본과 SHA는 SCIENCE_SPEC.json의 evidence_pins에 고정했다. 이 계획 작성 과정에서 Test 결과·기존 CNN 예측·checkpoint를 열거나 모델을 실행하지 않았다.

**과거 구조와 중복되는지 확인한다 — 완료**
- 현재 기존 CountAwareTitan 소스 32개와 관련 학습 계약 7개의 arm 선언을 좁게 조사했다. 명시적인 QK-only local CNN은 확인되지 않았다.
- 이전 Query-only 비선형 후보는 원래 V CNN을 유지했으며, V 차분 후보도 학습 가능한 V 차분 계수를 유지했다. Adaptive·MultiLag·LevelHistory는 적용 위치와 제거 범위가 다르다.
- 이는 조사한 범위의 결론이다. 모든 과거 실행이나 외부 연구에 대한 부재 증명 또는 새 원리 주장이 아니다.
- DUPLICATE_AUDIT.json SHA: `91621fea33cd9d807af5cab0135e8d945706205b72a7827e1502f24cd66045a0`

**Q/K의 시간 관계 처리만 남기는 구조를 고정한다 — 다음 작업**
- RAF seed 42·52·62에 `titantpp_cnn_qk_only_v1` 한 구조를 새로 학습한다. Q/K는 기존 관측 3-tap CNN을 자유롭게 학습하고, 공통 Q/K/V 투영도 정상 학습한다. V의 추가 local CNN만 물리적으로 제거한다.
- 원 모델을 같은 초기화 순서로 만든 뒤 공통 tensor와 Q/K filter를 그대로 유지해 교체한다. 추가 난수 소비·기존 checkpoint warmstart·dummy V 가중치는 없다.
- local CNN 계수는 576에서 384로 줄어든다. 따라서 결과는 V 경로 제거와 용량·최적화 변화가 결합된 구조 축소 효과로 해석한다.
- 공통 입력/log1p·손실·head·메모리·예산·선택을 유지한다. RevIN·GRU·gate·새 비선형·새 정규화는 추가하지 않는다.
- SCIENCE_SPEC.json SHA: `0393da551a4933d3ed8105910fc3859d8aadcaf008b19e786cd177282a56db14`

| 선택지 | 구현·운영 부담 | 해석과 향후 변경 부담 | 이번 결정 |
|---|---|---|---|
| Q/K-only를 같은 조건으로 새로 학습 | 작은 모델/runner 변경, 3개 학습과 고정 평가 | 용량 감소가 혼재하지만 조작 범위가 명확함 | 고정 모델 probe에 남은 공동 학습 질문을 직접 검증 |
| gate·감쇠 계수·추가 필터를 도입 | 새 장치와 하이퍼파라미터·대조군 필요 | 지금 조건별 원인이 확정되지 않아 임의 탐색 위험 | 채택하지 않음 |
| 같은 예측을 추가로 잘게 나눠 분석 | GPU 부담은 작음 | 사후 구간 선택이 늘고 공동 학습 효과는 확인 불가 | 이번 구조 결정의 선행 조건으로 확대하지 않음 |

**새 구현과 운영 계약을 검증한다 — 다음 작업**
- 모델 담당자는 신규 모델/runner만, 운영 담당자는 신규 계약·실행 경로만 다룬다. 독립 파일 구현은 병렬로 진행하되 공통 명세 확정·통합 검사·봉인·native 시작은 직렬로 진행한다.
- 초기화 전후 공통 tensor 동일성·추가 RNG 소비 없음, Q/K와 공통 V 투영의 학습 연결, V local 가중치 부재, 관측 경계·padding·미래 target 차단을 확인한다.
- 최종 local 계수 384와 기존 비활성 GRU 등록 보존을 확인한다. 기존 CNN·실패·claim·source·공용 Runtime은 변경하지 않는다.
- 검증된 prepare 반환과 실제 integrity SHA만 운영 담당자가 기록한다. 이 문서에서는 실행 명령이나 SHA를 추측하지 않는다.

**5090에서 동일한 세 seed를 실행한다 — 다음 작업**
- 대상: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_qk_only_raf3_20261016_v1_5090`.
- GPU UUID `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`, Python 3.12.13, torch 2.11.0+cu130, cuDNN 92000, 학습 4 threads를 유지한다. 최대 3개 병렬이며 총 3개만 시작한다.
- source·입력·승인·계약 SHA와 실제 소유·유휴 상태 및 3개 동시 합성 검증을 통과한 뒤 exclusive permit으로 한 번 시작한다. 사용자 사전승인 범위이며 별도 승인 대기를 만들지 않는다.
- 상속 host 절대 deadline `1792679647.915156`을 유지한다. 자동 retry/resume·중복 시작·기한 연장은 없다.

**학습 종료와 고정 평가를 검증한다 — 다음 작업**
- 세 worker와 dispatcher/pipeline 실제 종료, 소유 PID/GPU 부재, 선택·종료 binary 및 회수 archive/manifest SHA를 확인한다. wrapper 자체 OS exit는 직접 관측한 경우에만 확인했다고 기록한다.
- checkpoint는 전체 fullValidation raw RMSE의 첫 strict finite 최소 epoch로 고정한다. 세 모델의 native Validation 대상과 여섯 지표가 모두 재현돼야 Test3를 수행한다.
- 기존 CNN3와 외부 모델은 저장 결과만 재사용한다. Test를 보고 checkpoint를 다시 고르지 않는다.

**사용자 기준과 구조의 한계를 함께 판정한다 — 다음 작업**
- Validation/Test 각각의 RMSE·MAE 3seed 평균이 기존 CNN과 같은 대상의 외부 지표별 최우수 평균보다 모두 엄격히 낮아야 통과한다.
- 표본 SD, 모든 seed별 절대·상대 차이, 큰수량 q>200 RMSE/MAE와 전체·큰수량 TimeNLL, 단순 기준선 및 불리한 결과를 보존한다. 서로 다른 데이터셋 raw 오류를 평균내지 않는다.
- 기존 CNN/외부 5080·cuDNN91900 대비 신규 RAF 5090·cuDNN92000 차이는 한계다. 이를 없애려고 기존 기준선을 다시 실행하지 않는다.
- 같은 Test를 참고한 반복 탐색이므로 독립 일반화·통계적 유의성·V 경로의 보편적 해로움을 주장하지 않는다.
- RAF 통과 뒤에만 같은 구조의 Taxi5090 및 실제 유휴가 새로 확인된 Intermittent5080 확장으로 이어간다. 미달하면 결과와 실패를 보존하고 다음 가설 전에 원인을 검토한다.

**남은 순서**
1. 신규 모델·runner·운영 도구의 독립 구현과 계약 검증을 마친다.
2. 통합 검증·봉인 후 5090 native 조건을 검증하고 RAF3를 한 번 시작한다.
3. 종료·회수 검증 → Validation3 재현 → 고정 Test3 → 저장 기준선 비교 순으로 진행한다.
4. 결과·한계·판정을 기록하고 같은 구조 확장 또는 다음 근거 검토를 결정한다. 이 문서는 scheduler/state나 다른 자동화를 변경하지 않는다.
