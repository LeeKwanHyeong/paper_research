# CNN+GRU54 중심 Results 재배치 — 완료

사용자 승인 범위는 본문의 대표 모델과 성능표를 일치시키고, 과거 MLP4·MLP8 상세를 부록으로 옮기는 것이다. 연구 제목이나 전체 연구 방향은 변경하지 않았다.

## 구성

- 본문 5.1/Table 2: CNN+GRU54 대 외부 6개 공통 출력부 인코더, Taxi·Intermittent·RAF, 3seed 평균·표본 SD.
- 본문 5.2: 대표 구조의 보정 추가 효과. 신규 대조는 미완료로 유지.
- 본문 5.3: 대표 구조의 검색 추가 효과. 신규 예산 일치 대조는 미완료로 유지.
- 부록 A: 완료 CNN9 조건의 selected/final epoch, A100/RTX 실행 경로, 전체/큰 수량 6지표.
- 부록 B: 기존 MLP4 외부 비교표·단순 규칙 및 불리한 결과.
- 부록 C: 기존 MLP8 구조·6개 arm·수치표·대조 결과·시간/큰 수량/비용 한계·시각이 명시된 Instacart 단일 보존 관측.
- 부록 D–F: Train 분포·이력 그림·lag 상관·오차 분해. 기존 본문과 수치 유지.

## 근거와 주장

[evidence.json](evidence.json)에 입력 파일 SHA, CNN9 endpoint 경로/SHA, 재계산한 3seed 통계, 기존 외부18그룹을 보존했다. CNN은 완료9조건=재사용3+후속6이며 새63조건과 별개다. 9개 endpoint의 선택 epoch와 6지표가 원래 완료 집계와 일치함을 검토했다. Intermittent seed42는 원래 집계와 같은 selected endpoint replay 수치를 유지한다.

계약 감사에서는 CNN과 외부 비교군의 Train/Validation target identity·수량SHA, 입력창·loader, Train통계, head·loss·optimizer, budget·selector가 일치했다. CNN 동결source123 rawSHA도 검토했다. CPU binary 재추론은 수행하지 않았다. 비교는 공통 평가 계약의 기술적 Validation 비교다. 같은 초기화나 동일 Runtime을 주장하지 않는다.

본문은 Taxi/Intermittent의 외부 인코더 대비 평균 수량 이득뿐 아니라 RAF의 S2P2 RMSE 우위, 각 데이터의 더 낮은 외부 timing NLL, Intermittent의 직전 수량 반복 MAE/RMSE 우위를 포함한다. 이 완료 성능은 대표 구조의 기여 대조를 대체하지 않는다. Instacart 미완료 대표 행을 MLP 수치로 채우지 않았다.

## 검증

- CNN3그룹×6지표와 기존 외부18그룹×3지표 평균·표본 SD 재계산 일치.
- CNN9 selected endpoint rawSHA 일치.
- Methods/Algorithm, 28개 참고문헌, 기존 MLP4/MLP8 표 수치, 후속 Train 부록 본문 보존.
- 중복 label, 미정의 교차참조·인용, 미인용 참고문헌 없음.
- 본문 표 순서는 Table1 데이터, Table2 CNN+GRU54 외부 비교.
- 초록155단어. 기여 검토에서 미완료 효과를 전제하지 않도록 문구를 조정.
- 열린 main.tex의 내장 편집기 컴파일 성공. 별도PDF 내보내기나 새탭열기 없음. 이번 개정의 독립적인 페이지별 시각 검사는 수행하지 않았다.
- 원격 조회·새학습·추론·Test/혼합 성과 열람·자동화 변경 없음.

[수정 전 원본과 SHA](before_manifest.json), [소스 변경](main_tex.diff), [검증 JSON](validation.json)을 함께 보존했다.

## 남은 순서

**대표 보정 대조의 결과를 반영한다 — 외부 작업 대기**
- 승인된 캠페인의 완료 원본과 3seed·동일 선택epoch·비용을 확보하면 5.2절을 갱신한다.

**검색 대조와 Instacart 비교를 연결한다 — 외부 작업 대기**
- 해당 단계 완료 후 5.3절과 보충 데이터 비교를 채운다. 보정/검색 인과 해석과 외부 성능 위치를 구분한다.

**저자가 기여와 최종 평가 범위를 검토한다 — 다음 작업**
- 미노출 독립 평가, 문헌 차별성, 제출 규정과 분량을 정리한다.
