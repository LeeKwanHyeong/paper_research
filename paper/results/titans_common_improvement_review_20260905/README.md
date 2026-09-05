# Titans 공통 개선 재검토의 로컬 증적

2026-09-05, `paper_research/master`의 `567196e510e14211e0ee400ea312872bb3ec2703`에서 조사했다. [상세 연구 보고서](../../audits/titans_common_improvement_review_20260905.md)

## 범위와 판정

- `mac_write_visibility_probe.py/.json`: 기존 B1의 합성 CPU forward/backward 감사. H=16에서 write on/off 예측 차이 0 및 writer gradient 0, H=17에서 출력 차이 및 finite/nonzero writer gradient. PASS.
- `train_write_visibility_audit.py/.json`: 실제 train effective target history 전수 집계. H≤16 비율은 Intermittent 20.3063%, Taxi 6.0949%, Instacart 97.0827%. 기존 캐시 169,465개 target의 H와 정확히 일치. PASS.
- 새 성능 후보 구현, optimizer step, GPU 실행, 새로운 validation/test 추론은 없다.
- H≤16은 B1의 segment16/read-before-segment-write 정책에 대한 제한이다. T0의 성능 원인이나 새 후보의 개선을 입증하지 않는다.

Train parquet는 train predicate와 series/seq projection으로만 행을 읽었다. 전체 parquet byte hash는 데이터 identity 확인이다. 기존 train cache는 전체 tensor dict를 로드하되 `target_index/history_length`만 사용했다. 수량·시간·예측 label을 새 분석에 사용하지 않았다. Stable MAC launch metadata는 기존 실행의 context/data/count 정합성만 대조했다.

## 재현

원본 감사는 `/tmp/titans_common_improvement_review/`에서 실행했다. 이 디렉터리에는 실행한 Python과 결과 JSON을 byte-identical하게 복사했다. JSON 내부의 원래 실행 경로와 재현 명령은 provenance를 위해 유지했다. 복사된 경로에서 다시 실행하면 `script_path` 등의 경로 메타데이터가 달라질 수 있지만 같은 source/data/environment의 수치 판정은 같아야 한다.

```sh
/usr/local/bin/python3 -s -B paper/results/titans_common_improvement_review_20260905/mac_write_visibility_probe.py --output /tmp/mac_write_visibility_replay.json
/usr/local/bin/python3 -s -B paper/results/titans_common_improvement_review_20260905/train_write_visibility_audit.py --output /tmp/train_write_visibility_replay.json
```

합성 감사 환경은 Python 3.12.10, PyTorch 2.7.1, CPU float32, 1 thread, eval/dropout-off다. Train 감사에는 NumPy 2.3.1, Polars 1.31.0, PyTorch 2.7.1이 사용됐다. 원본 데이터·기존 캐시는 저장소에 이미 있던 로컬 artifact를 요구한다. 재현에 필요한 source·data·cache checksum은 각 JSON에 기록돼 있다.

`artifact_manifest.json`은 이번에 보존한 문서·스크립트·JSON의 hash와 첨부 PDF/일차 문헌을 기록한다. `validation.json`은 문서 링크, 구문, 저장된 결과 수치와 보존된 script/source hash의 로컬 일치 검증이다. 모델 성능 검증 보고서가 아니다.
