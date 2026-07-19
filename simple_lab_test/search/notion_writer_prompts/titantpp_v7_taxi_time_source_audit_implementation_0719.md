# TitanTPP V7 Taxi Time-Source Audit Implementation

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 날짜 구역: `2026-07-19 | Post-V6 Model Candidate Selection`
- 세부 Step: `Step 2. Taxi P0/P1/P2 Time-Source Isolation Audit`
- 상세 페이지: `TitanTPP V7 Taxi P0/P1/P2 Time-Source Isolation Audit`
- 기존 상세 페이지가 있으면 업데이트하고 중복 페이지를 만들지 않는다.

## 상태

- 상태: `실행 준비 중`
- audit code와 5080 runner 구현 완료
- V7 model path는 Stage-0 통과 전까지 미구현·잠금 유지
- 실행 서버 / tmux: `5080 / titantpp_v7_taxi_time_source_audit_0719`
- 실제 5080 실행 시작 시각: 아직 없음

## 목적

V6의 secondary time signal이 pre-window temporal field 자체에서 나온 것인지,
mark·quantity를 함께 쓴 결과인지 Taxi train split 안에서 분리한다. 이 결과로 V7
time-history adapter 구현을 열지, 모델 구현 전에 종료할지 결정한다.

## Factorial 계약

| Probe | Active-window input | 추가 pre-window source | 역할 |
| --- | --- | --- | --- |
| P0 | 동일 window summary | 없음 | control |
| P1 | P0와 동일 | temporal field만 사용 | V7 primary source test |
| P2 | P0와 동일 | temporal + mark + quantity | attribution only |

P1 대 P0만 Stage-0 통과 여부를 결정한다. P2가 좋아도 P1 실패를 대신 통과시키지
않는다.

## 고정 조건

- dataset: Taxi hourly fixed-split train parquet only
- target: next-event `log1p(delta_t)`
- lookback / max_seq_len: `168 / 256`
- history eligibility: strictly pre-window event `>=8`
- pooling: all strictly pre-window temporal moments; V6 `M/topk` 미사용
- evaluation: 3 expanding rolling-origin folds
- probe: fold-local StandardScaler + Ridge (`alpha=1`)
- uncertainty: series bootstrap `2,000`, seed `42`
- validation/test parquet: 읽지 않음
- artifact: `search_artifacts/model_enhancement_titantpp_v7_taxi_time_source_audit_0719`

## 실행 명령어

```bash
SOURCE_REVISION=<checksum_synced_full_sha> \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
TMUX_SESSION=titantpp_v7_taxi_time_source_audit_0719 \
EXECUTION_SERVER=5080 \
bash simple_lab_test/search/scripts/run_titantpp_v7_taxi_time_source_audit_0719.sh
```

## 결과
