# 현재 QK-only CNN의 Q/K 지역 보정 단일 민감도 진단

- Origin Skill: academic-research-suite / experiment-agent
- Mode: plan
- Date: 2026-10-16T04:17:49.556154+09:00
- Verification Status: UNVERIFIED — 명세 작성 단계이며 독립 설계 검토·구현 검증·native 실행은 미완료다.

**현재 결과와 확인할 질문을 고정한다 — 완료**
- 루프4 QK-only는 Validation RMSE가 세 seed 모두 기존 CNN보다 높았다. MAE는 seed42만 낮았다. 전체/큰수량 오류의 상충 방향도 seed마다 달라 다음 필터 구조를 바로 정하지 않는다.
- 현재 모델에서 Q/K의 추가 지역 보정을 동시에 끄면 어떻게 달라지는지 확인한다. loop3의 V 보정 probe와 달리, 이번은 V 보정이 없는 모델의 Q/K 관계 계산 경로를 묻는다.
- 선택 checkpoint는 seed42/52/62의 epoch5/5/22로 고정한다. 기존 CNN 추론·학습·Test·checkpoint 재선택은 없다.

**한 가지 개입만 구현하고 독립 검증한다 — 다음 작업**
- 새 operation adapter에서 현재 후보의 Q/K residual 결과만 함께 on1/off0으로 둔다. 기존 scientific source143개와 binary는 그대로 두고 parameter를 직접 바꾸지 않는다.
- off도 원래 shape·mask·index·nonfinite 검사를 보존한다. 성공·예외 모두 finally에서 원 함수 참조를 복구한다. 조건별 새 프로세스와 동일한 eval·batch·초기 context를 사용하고 parameter/buffer/context 불변을 검증한다.
- 독립 설계 검토와 로컬 합성·계약 검사를 통과하기 전에는 native 실행하지 않는다. 모델·분석 구현은 독립 파일에서 병렬 가능하지만 공통 명세·봉인·단계 gate는 직렬이다.
- SCIENCE_SPEC.json SHA `10371800c7e0f4095defad1380344a5bef19a9b31578dd599bc163779b1af7ee`
- ACCEPTANCE.json SHA `5bc011e4473f71db5fd520af8ff3763753c767ceeb98d12cfdc1a6f2c4063193`

**같은 5090에서 on 기준을 먼저 확인한다 — 다음 작업**
- 신규 root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_qk_branch_probe_20261016_v1_5090`.
- 같은 GPU UUID·Python3.12.13·torch2.11.0+cu130·cuDNN92000·4threads를 사용한다. 이전 평가의 종료와 실제 GPU 유휴를 새로 확인한 뒤 진입한다.
- 상속 host deadline1792679647.915156은 연장하지 않는다. 새 경로의 한 번 실행이며 실패·미완료 attempt를 자동 재시도하지 않는다.
- 먼저 on Validation3의 동일 대상과 여섯 지표를 저장 기준에 재현한다. 기존 native 결과는 raw_predictions_written=false이며 개별 행이 없다. 따라서 과거 개별 행 재현은 요구하거나 주장하지 않고, 원래 population의 6690개 대상·수량·대상 SHA와 집계 여섯 지표를 재현한다. 고정 허용오차는 앞 진단과 같은 상대/절대1e-5다.
- 이 gate를 통과한 뒤 on Train3를 추출한다. 이어 off Validation3, off Train3 순서로 각각 새 프로세스에서 실행한다. 총 실데이터 worker12개, 최대 동시3개다. 합성 qualification은 이12개와 별도로 센다.

**같은 대상의 차이를 고정 조건으로 분석한다 — 다음 작업**
- 주 비교는 동일 checkpoint의 off−on이다. 기존 15개 이력 조건, Train 변화폭 경계0.4054651081081645/1.252762968495368, 행 가중·조건별 공통개체 가중을 그대로 사용한다.
- raw RMSE·MAE, 부호 있는 오차, AE·SE, q>200 큰수량과 recorded TimeNLL을 seed별·평균±표본SD로 보고한다. 악화·예외·제외대상을 모두 남긴다.
- 현재 on 대 기존 CNN 저장 행, 현재 on 대 loop3off 저장 행은 별도 보조 비교다. 기존 모델을 다시 실행하지 않으며 이 보조 비교를 같은 모델 내 개입과 혼동하지 않는다.
- 입력 SHA·split/개체/순번·정답/이력을 결속하고 Train25779/Validation6690개 대상의 중복·누락을 막는다. native의 parent-input SHA·float32 이력평균 target 규약과 exporter의 pure-input SHA·float64 이력평균 규약은 다르므로 digest를 직접 같다고 비교하지 않는다. 원래 population identity 감사 digest는 별도로 재현하고, 새 on/off 행끼리는 동일 exporter 규약을 강제한다.

**진단의 결론과 다음 학습을 구분한다 — 다음 작업**
- off가 두 split·raw지표·seed·가중에서 일관되게 좋아지면 Q/K 자유 보정의 영향을 제약할 연구 질문이 강화된다. 그렇더라도 DC 보존 또는 다중 범위 필터를 자동 채택하지 않는다.
- Validation만 좋아지거나 지표·seed가 엇갈리면 공동 적응과 상충을 그대로 기록한다. off가 나빠지면 현재 Q/K 경로의 유용성을 함께 인정한다.
- Q/K를 함께 바꾸면 사건·메모리 사이의 attention 비중도 바뀐다. 이번 진단은 Q와K, 크기와방향, DC와관측범위의 원인을 분리하지 않는다. off는 새로 학습한 무보정 모델도 기존 CNN도 아니다.

**남은 순서**
1. 독립 설계 검토 → 신규 adapter·분석 구현 → 합성·계약 검증 → 증적 봉인.
2. 이전 평가 실제 종료·유휴와 native 고정 조건 확인 → on Validation3 gate → on Train3 → off6.
3. 실제 종료·원본 archive/manifest SHA·행별 재구성 검증 → 모든 이득과 손해 보고.
4. 증거를 검토한 후에만 다음 단일 backbone 가설을 결정한다. 이번 명세에는 새 학습 자동 규칙이 없다.
