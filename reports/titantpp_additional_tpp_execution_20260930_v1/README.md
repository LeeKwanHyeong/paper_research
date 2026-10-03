# S2P2·AttNHP 추가 공통 head 비교

실행 준비·원문 검토·로컬검증·양 서버 native 검증을 완료했다. 2026-09-30 12:31 KST 실제 관측에서 5080 Taxi S2P2 seed42 9epoch/2700steps,5090 Instacart S2P2 seed42 첫epoch 배치 진입을 확인했다. 소유PID/GPU/진행 증거는 launch_status.json이다. 학습 종료와 결과 감사는 아직 완료되지 않았다.

## 실험 범위 — 확정

S2P2/AttNHP 각각 Taxi·Intermittent·Instacart × seed42/52/62, 총18 fresh fit 및36 selected/last 역할이다. 기존 TitanTPP 및 RMTPP/THP/NHP/SAHP의 결과는 재사용한다. 같은 관측 시간 간격/수량, 동일 head/loss·split·전체train/validation·batch128·최대300/최소40/patience40·strict raw RMSE 선택을 적용한다. MAE와 시간 NLL은 동일 선택 checkpoint에서 보고한다.

비교는 원본 TPP의 encoder를 공통 과제에 연결한 matched-head 비교다. 원본의 intensity/mark decoder와 native NLL 전체 성능 재현이 아니다. H64/2층, 무튜닝의 고정 예산이며 모델별 파라미터 수와 연산량은 다르다. S2P2의 affine doubling 구현은 저자 순차 점화식과 출력/gradient 일치를 검증했지만 N log N work다. AttNHP는 관측 시점 직후의 표현을 사용하고 미래 target 시각을 query에 사용하지 않는다. 이 수정과 한계를 논문 Setup에 명시한다.

## 원본과 검증 — 완료

- 공식 S2P2 소스 c3933240f16a22b43d80d09bda272475526ff24b, AttNHP 소스78f754fde61547b84a0749fd7e7e020e7cd6b4d8. 다운로드SHA·라이선스 보존.
- 로컬: 원본 점화식 출력/gradient, 원본 AttNHP forward, 인과성·마스킹·target 누수·공통head 초기화, optimizer/RNG 복원, 실제 trainer의 합성 학습→checkpoint→selected/last replay 통과. 실제 데이터 학습에 합성update를 더하지 않는다.
- native: 5080/5090 모두 CUDA 동일 검증, 각60개 비용측정 합성update와10개 optimizer/RNG 연속성 검증update, 실제 입력계약, 서버 저장/소유자식 중단, 기존 MLP seed42 validation MAE/RMSE/timeNLL 재현 통과. 해당 GPU별 기존 torch/CUDA/cuDNN/numpy/polars를 유지했다.
- v1 준비의 빈 sample_data 루트 표식 누락과 5090 NVRTC 경로 누락은 scientific fit 전에 발견했다. 최초 로그/claim/검증을 보존했다. v2 사본에서 5090 프로세스 LD_LIBRARY_PATH에 설치된 cu13/lib를 명시하고 양 서버 검증을 완료했다. 이어 실제 학습 전 run 부모 디렉터리 누락으로 두 worker가 종료되어 v3에서 생성 경로를 보완했다. 해당 원본 로그/claim을 회수·보존하고 최초 조건별 절대기한도 그대로 유지했다. 실제 mkdir 코드에 대한 부재 부모/중복 거부 회귀 검증과 학습·checkpoint·replay 로컬 검증을 통과했다. v3의 native 검증과 실제 시작 여부는 해당 receipt와 monitor로 확인한다. 패키지 설치·교체·과학적 설정 변경은 없었다.

## 실행 제한 — 고정

5080은 Taxi/Intermittent12조건,5090은Instacart6조건이다. 서버 간 병렬이며 서버당 한 worker가 순차 처리한다. 조건별36시간, 전체120시간, 고정 마감 **2026-10-05 12:16:19 KST**다. v1 준비시간도 포함하며 연장하지 않는다. 매epoch 서버 자체 checkpoint/SHA와 내부lease/독립timeout을 사용한다. Mac 연결·ACK·스케줄러는 없다. 자동retry/resume은 없고 오류시 해당 큐를 중단한다. 다른 서버의 정상 큐는 독립적이다.

## 증거 위치

프로젝트루트 기준:
- `search_artifacts/titantpp_additional_tpp_20260930_v1/README.md`: 원문과 adaptation 결정.
- `.../verification/`: CPU/pipeline/부정계약검사.
- `.../launch_layout_fix_v3/execution_contract.json`: 현행 계약 canonical SHA `a39790e6c589230e6e94e4ddbf5e421807bb61e702e85a88f5dd61866e230c93`.
- source106개 closure `b5440fa1a29a5abe9adfab2a5c9e1418d1395302746c2bb17f287845e03945e9`.
- `.../launch_layout_fix_v3/native/5080/receipt.json`, `.../native/5090/receipt.json`: 실제 native검증.
- `.../launch_layout_fix_v3/launch/`, `.../monitor/`: 실제 시작·관측 증거. launch요청과 본학습 진입을 구분한다.

## 이후 작업 — 학습 종료 후

선택/last checkpoint·optimizer·노출·배치prefix·원본source/계약·메모리/실측시간·replay를 회수·감사하고 기존 고정표에 추가한다. seed별 결과·평균/표본표준편차와 불리한 결과/실패도 보존한다. 다음 단계는 비교표 통합 → 원고 주장 정리 → 별도 승인된 held-out 평가다. 추가 모델·seed·튜닝·Runtime 변경·commit/Push는 포함하지 않는다.
