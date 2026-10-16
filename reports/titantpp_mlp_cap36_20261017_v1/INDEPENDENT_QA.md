# Cap36 독립 검토

## Review Result: PASS — 데이터·운영·worker 정적 검토

최종 worker 소스 SHA를 확인하고 전체 구현과 테스트를 검토했다. 이 문서는 실제 native 실행 완료를 의미하지 않는다.

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

## 완료한 검사

- 데이터 단위검사 2개·하위검사 6개, runtime CPU 검사 15개, 별도 모의 종료·미확인 GPU 검사 2개, manager 검사 14개가 실제 exit0으로 통과했다.
- 임시 격리 경로에서 3개 데이터·9개 입력을 생성하고 SHA, Train 전용 log1p 평균·표준편차와 target RMS를 독립 검산했다. 실제 loader 연결에서 held_out_materialized=false를 확인했다.
- Train/Validation 대상 ID·개수·시간 SHA와 부모 time/model/loader 조건이 유지된다. Test는 대상 ID·개수·고정 Cap 검증에만 사용했다.
- 최종 Polars nearest 분위수는 Intermittent [2,20,20,20], Taxi [7,65,65,65], RAF [2,10,10,10]이다. 중복 경계는 정상이며 strict q>p99는 모든 split에서 비어 있다.
- Python 경로 상속, 배포 중복 차단, 종료 증적과 파일 SHA 확인 후 회수, 안전한 archive 추출, timeout 원문과 미관측 exit=null 보존을 확인했다.

## 허용 범위와 한계

Validation gate는 각 host에 배정된 전체 job에 적용한다. 두 host 36개 전체에 걸친 장벽은 아니다. GPU native 검증·원격·실제 학습·실데이터 예측은 수행하지 않았다. 기존 CNN과 raw 결과는 수정하지 않았다. Cap 결과와 과거 raw-target 성능을 직접 비교하면 안 된다.

변경한 파일은 이 QA JSON/Markdown 두 개뿐이다. 검토 소스 SHA, 정확한 대상 수, 검사 내역은 JSON에 결속했다.

## 최종 worker 검토

최종 runner `a58d3cb2bdc2baa842949e534a31f931632d87b360a202a3aea01b5e43dec9ef`와 tests `121b0563cb29b44eb0939842312cdcc7eead5c061199691210c6a687637bea5d`를 결속했다. 고정 선택 checkpoint·state SHA, 개별 Validation 예측 재현, host별 gate 뒤 Test 읽기, 원본 불변 검사를 확인했다.

폭16의 과거 파일과 현재 파일은 byte 단위로 다르지만, 차이는 기존 폭8/12 지원의 일반화와 폭별 metadata/route 처리다. 폭16 계산 순서와 초기화는 유지된다. 별도 합성 테스트가 초기 state/RNG 및 비영 보정 가중치의 출력·gradient 정확 일치를 검사한다. 작성자는 worker 테스트 18개가 11.026초, actual exit0으로 통과했다고 전달했다. 독립 검토자는 구현을 검토했으며 이 18개를 중복 실행하지 않았다. 최종 전체 통합검사 원문은 부모 작업에서 별도로 보존한다.
