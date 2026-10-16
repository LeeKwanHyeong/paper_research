## 루프6 — RAF6 Validation·Test 수동 완료와 채택 판정
<mention-date start="2026-10-16"/>
**반복 자동화 종료 후 현재 실험을 마무리했다 — 완료**
- 사용자 요청으로 cnn-backbone-raf 자동화를 삭제했다. 이어 “다음작업 진행하자”에 따라 이미 시작한 RAF6의 종료·회수·평가를 수동으로 마쳤다. 자동화를 재생성하거나 다음 루프·Taxi/Intermittent 학습을 시작하지 않았다. 이전 자동 관측 지침은 당시 기록이다.
- 학습 종료·원본 회수 관측08:32:31.831268 KST, 고정 checkpoint Validation6 전부 통과 뒤 Test6 완료08:34:11.800520 KST다. 학습172개·평가339개 파일SHA를 검증했다. 평가 native worker12개/pipeline2개 actualexit0이며 wrapper 자신의 OSexit는 직접 관측하지 않았다. 과거 종료 관측을 현재 GPU 유휴 증명으로 사용하지 않는다.
- 기존 CNN 모델/checkpoint/결과는 변경·재학습·재추론·재평가하지 않았다. 기존 Q/K/V576과 공통 입력/log1p·손실·head·메모리 규칙을 유지한 마지막 FFN 확장 비교이며 두 arm은 각1152개 파라미터를 추가했다.
**RAF 3seed 평균±표본SD(ddof1) — 완료**
- Validation 기존 CNN: RMSE 34.431408 ± 0.073201, MAE 9.218271 ± 0.028002.
- Validation Current 대조군: RMSE 34.466041 ± 0.071568, MAE 9.249838 ± 0.012293.
- Validation Temporal 이력 결합: RMSE 34.472074 ± 0.066691, MAE 9.223596 ± 0.015190.
- Test 기존 CNN: RMSE 39.600663 ± 0.023654, MAE 9.972138 ± 0.036428.
- Test Current 대조군: RMSE 39.551056 ± 0.035158, MAE 10.037737 ± 0.061708.
- Test Temporal 이력 결합: RMSE 39.544125 ± 0.045950, MAE 9.992155 ± 0.053091.
**사용자 기준과 방법론 판단 — 완료**
- 두 후보 모두 Test RMSE만 기존 CNN과 외부 최우수보다 낮다. Validation RMSE·MAE와 Test MAE는 모두 미달이며 통과 후보/확장 대상은 없다. 외부 지표별 최우수는 Validation RMSE S2P2 matched-head33.994704, MAE RMTPP9.136205, Test RMSE S2P2 matched-head39.772935, MAE RMTPP9.906347이다. 동일 대상의 저장 구현 결과이며 논문 원문 수치로 바꾸지 않는다.
- Temporal−Current 평균: Validation RMSE+0.006033/MAE−0.026242, Test RMSE−0.006931/MAE−0.045582. MAE는 양 split 모든 seed에서 개선했지만 RMSE는 seed42만 개선하고52·62는 악화했다. 사전의 Validation 두 지표 동시 개선 기준은 미달이다.
- Temporal−CNN 평균: Validation RMSE+0.040666/MAE+0.005325, Test RMSE−0.056538/MAE+0.020016. Test seed42는 두 지표 개선,52는 두 지표 악화,62는 RMSE 개선/MAE 악화다. 일부 이득을 확인했지만 채택 가능한 전체 개선으로 결론 내리지 않는다.
- Temporal의 큰수량q>200 RMSE/MAE는 양 split에서 Current보다 악화하며 전체/큰수량 TimeNLL은 개선한다. Current의 Test 큰수량 TimeNLL은 CNN보다 악화한다. 모든 여섯지표·seed별 절대/상대차이·예외와 단순 기준선(TimeNLL N/A)을 원 평가 JSON/Markdown에 보존했다.
- 기존 CNN/외부5080·cuDNN91900 대 신규5090·cuDNN92000 한계가 남는다. Current도 attention 이력을 쓰며 동일 파라미터 수는 동일 함수공간·최적화가 아니다. 공통 가중치와 메모리 검색이 달라질 수 있다. 반복 Test 참고 탐색으로 독립 일반화·인과관계·통계 유의성을 주장하지 않는다.
**검산과 다음 작업 — 완료/다음 작업**
- 별도 stdlib 검산에서1884회 SHA 검사(고유1702파일), 평균36·표본SD36·여섯지표 seed 절대/상대차이108개·사용자 조건8개를 재계산해 통과했다. 이 검산의 원격 호출0이다.
- 학습 receipt `8a1d34eeb80d41e05a113bf370cfa404478a65c8351e06270049786cca083dba`; 평가 receipt `55d58c4999bfe481c922d36fc96ed18571a47c1e65a1454d76430b6e9f7d7894`; 별도 검산 receipt `611ee1515293aab26909ed2a857363dd3534e179b9789faaa6120b443d03f536`.
- 완료 보고: `reports/titantpp_backbone_post_attention_ffn_raf6_20261016_v1/manual_completion_v1/RESULT.md`, SHA `8b873d1782231ee6e293b41440591b8711378d4cae906e3c6946322279f0512e`. 전체 결과는 같은 report의 evaluation_v1/THREE_SEED_VALIDATION, THREE_SEED_TEST, EXTERNAL_COMPARISON.json/.md에 보존했다.
- 다음 작업은 결과를 보고 다음 단일 backbone 가설을 별도로 정하는 것이다. 현재 추가 후보는 선정하지 않았고 자동 반복 개선은 종료 상태로 유지한다.
