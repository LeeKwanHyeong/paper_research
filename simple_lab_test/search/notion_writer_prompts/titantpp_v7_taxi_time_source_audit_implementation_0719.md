# TitanTPP V7 Taxi Time-Source Audit Implementation

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 날짜 구역: `2026-07-19 | Post-V6 Model Candidate Selection`
- 세부 Step: `Step 2. Taxi P0/P1/P2 Time-Source Isolation Audit`
- 상세 페이지: `TitanTPP V7 Taxi P0/P1/P2 Time-Source Isolation Audit`
- 기존 상세 페이지가 있으면 업데이트하고 중복 페이지를 만들지 않는다.

## 상태

- 상태: `완료 · Stage-0 FAIL · V7 종료`
- audit code와 5080 runner 구현 완료
- V7 model path는 구현하지 않으며 Taxi V3b 유지
- 실행 서버 / tmux: `5080 / titantpp_v7_taxi_time_source_audit_0719`
- 실제 시작 / 종료: `2026-07-19 11:33:58 / 11:34:04 KST`

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

| Probe | Pooled MAE | P0 대비 개선률 | 개선 fold | Series bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| P0 window only | `0.155187` | control | - | - |
| P1 temporal only | `0.157486` | `-1.4811%` | `1/3` | `[-2.5633%,-0.5181%]` |
| P2 full attribution | `0.157746` | `-1.6489%` | `1/3` | `[-2.7612%,-0.6235%]` |

- Data, loader, coverage, causality, finite, rolling-fold gate는 모두 통과했다.
- P1은 pooled `>=1%`, 개선 fold `>=2/3`, bootstrap CI lower `>0`을 모두
  실패했다. 개선 series도 `54/131`에 그쳤다.
- Fold별 P1은 `-2.7363%`, `-4.9420%`, `+2.3561%`로 마지막 fold에서만
  개선돼 시간 구간에 따라 효과 방향이 바뀌었다.
- P2도 악화됐고 attribution-only이므로 P1 실패를 대체하지 않는다.
- 최종 판정: V7을 모델 구현 전에 종료한다. V7a/V7b, CUDA/e1/e50,
  multi-seed와 held-out은 진행하지 않고 Taxi V3b를 유지한다.
- 다음 작업: 별도 Intermittent V5b class-prior correction 계약 재개.

## Local Audit Trail

- source revision: `ea874d28aa01c0cef3bccea5efc6daedc9d61764`
- source sync: local/5080 SHA-256 `8/8 PASS`
- source manifest:
  `search_artifacts/model_enhancement_titantpp_v7_taxi_time_source_audit_0719/source_sync_manifest.json`
- source manifest SHA-256:
  `893ced346bc7c80d8d553090205817c1ce20122acfbfb47461992285f6ba0e41`
- runtime: Python `3.12.13`, matplotlib `3.10.8`, numpy `2.4.4`, polars
  `1.39.3`, sklearn `1.8.0`
- runner: executable, `bash -n PASS`, V6/V7 audit regression `14 passed`
- dataset: `38,524` rows, `131` series, marks `[0,1,2,3]`, quality gate `PASS`
- launch guard: same-name tmux absent, output directory absent
- audit manifest: `completed`, source revision 일치, 종료 시각
  `2026-07-19T11:34:04.224360+09:00`
- remote/local artifact checksum dry-run: `PASS`
- independent recalculation: pooled MAE, fold improvements, 2,000-replicate
  series-bootstrap CI가 summary와 일치
- official Stage-0 decision: `FAIL`, `close_v7_before_model_implementation_and_revisit_v5b`
