# 진단 계산 및 최소 실험 계획 QA

## Review Result: PASS

상태: 진단 3종 및 최종 README·MINIMUM_EXPERIMENT_PLAN.md·plan.json 검증 완료. 검토 대상 9개 JSON/Markdown 문서 범위에서 PASS다.

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

## 확인한 근거

- native Validation/Test 결과를 직접 읽어 전체 18쌍의 MSE·MAE 기여, signed error, bias²·오차분산, 양의 총손해/상쇄 이득, 대칭 mix 분해 및 Validation 비중 표준화를 독립 구현으로 재계산했다. 수치 assertion 1,134개 통과.
- 36개 split/대조 pair의 target identity와 truth/history digest가 동일하고, 각 quantity bin count는 population 진단과 일치했다.
- 원본 history 27개 SHA를 검증하고, 최초 strict Validation RMSE 최소 epoch과 `final=selected+40`을 독립 확인했다.
- 3개 원본 parquet에서 series 첫 행을 제외한 적격 target 수, series 수, 수량 평균, tail count, series TV와 split별 series 차집합 수를 재계산했다. 별도 assertion 45개 통과. 모델·checkpoint 로드와 재추론은 수행하지 않았다.
- RAF Test의 Validation 비등장 series 879개, Validation의 Test 비등장 series 1,693개, TV 0.5030192260이 재현됐다. 두 집합은 Train에 있던 series이며 새로운 entity 일반화로 해석하지 않는다.
- frozen `CountAwareTitanCNNGRU.py:53–58,238,243–254`에서 54가 latent width이고 GRU 활성 기준은 observed count>=2이며, cnn_mlp16은 CNN+MLP16임을 확인했다. MLP의 16-lag 가용성을 CNN+GRU의 최소 history 조건으로 바꾸어 설명할 수 없다.

## 해석과 비교 가능성

- 실제 수량 구간에 조건을 건 signed error는 calibration 검증이 아니며, 오차의 모집단 분산은 seed 사이 parameter variance가 아니다.
- 구간 비중 표준화 후 악화가 남는다는 결과는 고정 5구간의 비중만으로 설명되지 않는다는 뜻이다. 구간 안의 수량·series·duration·history 변화는 배제하지 못한다.
- 세 seed가 같은 평가 대상을 사용하므로 population 변화만으로 특정 seed의 반전을 설명하지 않는다.
- RAF52는 Validation부터 양대조보다 불리하므로 세 RAF seed 모두를 새로운 방향 반전으로 표현하지 않는다.
- 현재 Test를 보고 정한 후속 분석·변경은 개발 진단이며, 새 독립 평가로 재명명하지 않는다.

## 최소 후속 구성의 최종 검토

- 기존 checkpoint 27개×Validation/Test = 54 split pass, 신규 fit 0개가 첫 단계다. optional full9×경로 우회2개 = 18 설정/36 split pass의 산술과 문서·JSON 일치를 확인했다.
- RAF의 참된 CNN on/off×GRU on/off×3seed = 12fresh가 조건부이며, 기존 A100/RTX full/제거를 같은 환경의 새 factorial 대조로 재사용하지 않는다. Taxi·Intermittent 확대 24fresh는 자동 실행이 아니다.
- RAF 원본 활성 수 78,851/104,159를 확인했다. CNN 576, GRU 24,732를 더한 계획값 79,427/103,583은 산술상 일치하며 실제 prelaunch 확정이 남아 있음을 명시했다. 용량 동등성 주장은 없다.
- 경로 우회는 학습된 결합 모델의 출력 의존성 진단이고 독립 학습 대조를 대신하지 않는다고 명시했다. norm과 head 변화의 의미를 구분하고, 구성 차이와 단일 seed 실패의 인과관계를 단정하지 않는다.
- 계획 범위·계약/실행 승인·중단 조건·좋은 seed와 불리한 지표 보존·미노출 평가 전 최종 동결 순서가 문서와 JSON에서 일치한다. 원격 자원 유휴·실행 ETA·새 독립 평가 가능성을 확인 완료로 보고하지 않는다.

## Acceptance criteria

- [x] 6개 진단 JSON/Markdown 읽기
- [x] 중요한 수치 원본에서 독립 재계산
- [x] 분모·target identity·stratum 합계 일치 확인
- [x] 학습 선택과 종료 기록 확인
- [x] 인과 주장과 통계적 한계 명시 확인
- [x] 종합 README 및 최소 실험 계획의 최종 문구 확인
- [x] 원격 호출·추론·학습·원본 수정 0

변경 파일: 이 `DIAGNOSIS_QA.md`만 작성했다. 연구 원본과 소스 코드는 변경하지 않았다. 지정된 `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 없었고, `.agents/` 쓰기 금지 지침에 따라 이 reports 경로에 결과를 남겼다. npm audit/lint/type-check는 이 계산·문서 검토 범위에 해당하지 않는다.
