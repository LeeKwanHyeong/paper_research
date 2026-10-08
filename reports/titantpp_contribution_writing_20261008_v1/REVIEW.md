# 문서 검토 기록

2026-10-08. 검토 대상은 내부 연구 초안·근거 메모이며 학습 실행이나 새 성능 검증은 아니다.

## 독립 검토 결과 — PASS

별도 QA 역할이 영문 초안·문헌 비교를 전체 읽고 필요한 동결 source/실행 계약을 대조했다. 이어 README와 WRITING_NOTES의 주장 상태를 검토했다. 프로젝트 설정의 QA 역할은 Astra/high이며, 설정 확인과 실제 런타임 모델 식별은 구분한다.

- CRITICAL/HIGH/MEDIUM 지적 없음.
- LOW 지적 1: GRU 수식의 관측 개수와 초기 상태 정의 누락 → 현재 segment의 관측 개수 `n_i`, 경계에서 zero state를 명시했다.
- LOW 지적 2: “학습 중인63조건”의 전체 시작 오해 → Instacart 진행·Q1/Q2 대기로 수정했다.
- LOW 지적 3: 잔여 단계의 상태 분류 → “다음 작업”으로 통일하고 병렬 가능 여부는 설명에 분리했다.
- staging 단계의 아직 없는 링크는 최종 배치에서 검사한다.

## 확인한 항목

1. CNN Q/K/V 보정→첫 block 전체 출력→GRU 갱신 후 residual→두 번째 block의 순서.
2. Full/current-only의 25,308 active correction, current-only의 정보 경계와 depth/cost 차이.
3. Quantity log 회귀/원단위 복원, 기록 정수 duration 질량 및 Instacart top-code.
4. Static prototype64/top4와 fixed/generic512의 실제 수식·temperature1·별도 정규화 한계.
5. Dataset target 수·시간 단위·target를 포함한 sequence cap·native-bin lookback.
6. 63 fresh fits와 mean9 재사용, 최초 strict finite 최소 fullValidation RMSE 선택, 같은 epoch 6지표, 같은2dataset 각3seed 판정, 비용 기준.
7. 완료된 MLP8/parity와 미완료 Q1/Q2·독립 평가·외부 비교의 구분.
8. S2P2 adaptation과 native 모델의 차이; 6편 선행연구의 확인 범위와 최초성 제한.

문헌 원문 대조는 별도의 연구 역할이 수행했고 QA는 그 검토를 전제로 중복 조사하지 않았다. 구현 검증은 정적 코드/계약 대조다. 새 모델 테스트, 재추론, GPU 명령, SSH, Test/held-out 성과 열람은 수행하지 않았다.

## 기계적 검증

- 세 canonical contract SHA와 current pointer 일치.
- 동결 source closure와 파일 raw SHA: Instacart126, history129, retrieval129 모두 일치.
- 세 control-integrity receipt raw SHA를 지정값과 대조. 이번 작업은 도우미 실행 전 운영 검사가 아니며 control의 project_files 전체 검증으로 확대해 보고하지 않는다.
- 사용한 문서/receipt13개의 raw SHA를 evidence manifest에 기록.
- 9+27+27=63, 576+24,732=25,308, fixed/generic512=512 산술 확인.
- 최종 배치된 Markdown의 로컬 링크·줄번호·수식 delimiter·코드 fence 및 문서 raw SHA를 `document_validation.json`에 기록한다. 이는 문서 검증 결과이며 학습 결과 receipt가 아니다.

## 남아 있는 제한

Q1/Q2의 결과, 최종 대표 구조의 외부 비교 정합, 미노출 평가 자료의 독립성, 포괄적 최초성 검토는 미완료다. 현재 PAKDD2027 AI 사용 규정 아래 이 생성된 초안의 투고 적격성을 선언하지 않는다. 기존 논문·실험 계약·자동화를 변경하지 않았다.

## Results·Discussion 추가 검토 — 2026-10-08, PASS

새7·8절 및 RESULTS_UPDATE 문서/파생JSON을 별도 QA가 읽기 전용으로 검토했다. MLP8 54조건의 평균·표본SD·18/18비교·PF/HC판정, Instacart 단일완료조건6지표, qualification 비용의 측정한계와 각 원본 SHA가 일치했다. Intermittent P0 대비 감소율의 이중 반올림(9.02%)을 원본9.01496374620714%에서 직접 반올림한 **9.01%**로 고쳤다. 재확인 후 남은 LOW 및 CRITICAL/HIGH/MEDIUM 지적 없음. 모델 실행·원격 조회·Test 성과 접근 없이 수행한 문서/산술 검증이다.
