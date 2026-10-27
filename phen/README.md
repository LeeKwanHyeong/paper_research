# PHEN 재현 코드

현재 PHEN과 공통 head를 쓰는 외부 비교군을 기존 실험 코드에서 분리한 패키지다.
원래 모델과 실험 산출물은 삭제하거나 변경하지 않는다. `phen` 브랜치에서 관리한다.

## 구조

```text
phen/
  models/model.py       PHEN의 전체 forward 흐름
  models/components.py  persistent attention, history correction, prototype
  models/heads.py       공통 시간·수량 head
  baselines/           RMTPP, THP, NHP, SAHP, S2P2, AttNHP 각각의 구현
  vendor/              S2P2/AttNHP의 기존 kernel과 라이선스
  configs/             데이터별 고정 설정 4개
  data.py              이벤트 window와 다음 이벤트 target 구성
  training.py          공통 학습·평가·checkpoint 선택
  run.py               단일 모델·seed 실행
  tests/               기존 코드 동등성 및 합성 데이터 통합 테스트
scripts/phen/          모델별 shell 진입점
```

아키텍처와 기존 코드 대응 관계는 [ARCHITECTURE.md](ARCHITECTURE.md)에 정리했다.
학습 경로는 기존 `paper.scripts`, `simple_lab_test`, `models.TPPs`의 실험
factory나 monkey patch에 의존하지 않는다. 동등성 테스트만 원래 코드를 참조한다.

## 재현 범위

- 현재 source의 모델 계산과 데이터 window를 분리한 것이다. 과거 CUDA 학습
  trajectory나 논문 결과표를 다시 재현했다고 주장하지 않는다.
- 비교군 6종은 **matched-head variants**다. 원 논문의 mark/intensity likelihood를
  그대로 재현하는 모델이 아니며, PHEN과 같은 입력·시간 head·수량 head·loss를 쓴다.
- 원래 8개 history gate, 고정 divisor 8, persistent token 16개/block,
  prototype 64개와 top-4 평균을 유지한다. branch 축소는 별도 모델 변경이다.
- 기본 실행은 train/validation만 materialize한다. test는 별도 평가 명령에서
  `--allow-held-out`을 명시해야 한다. shell 예시는 원격 실행 승인이 아니다.

## 환경

Python 3.12에서 CPU 검증했다. `requirements.txt`는 로컬에서 확인한 버전이며,
과거 GPU 환경의 lock 파일은 아니다. 기존 환경에서는 자동 설치하지 말고 버전을
먼저 확인한다. 새 환경에서 설치할 때는 다음 명령을 사용한다.

```bash
python -m pip install -r phen/requirements.txt
bash scripts/phen/test.sh
```

CUDA용 PyTorch wheel, driver, CUDA/cuDNN 버전은 실행 서버에서 별도 확인해야 한다.
각 실행은 실제 `pip freeze`, Python/PyTorch/CUDA 버전, source snapshot을 기록한다.
다른 Python을 쓸 때는 `PYTHON_BIN=/absolute/path/to/python`을 지정한다.
source snapshot만 풀어서 실행할 수도 있다. Git metadata가 없으면 manifest에
`null`로 기록하고 source hash를 보존한다. snapshot에는 기존 실험 source가 없으므로
원래 구현과 비교하는 테스트는 skip되지만, 독립 runtime 테스트는 실행할 수 있다.

## 데이터와 설정

입력 parquet에는 다음 열이 있어야 한다.

```text
oper_part_no, seq, delta_t, demand_qty, chronological_split
```

`chronological_split`은 기존 train/validation/test 분할을 사용한다. 이 패키지는
분할을 다시 만들거나 수량을 cap하지 않는다. `seq`는 각 데이터의 기존 정수
시간축이고, `delta_t`는 기존 loader와 동일하게 정수 변환 후 최소 1로 처리한다.

| config / dataset | lookback (`seq` 단위) | 최대 길이 (target 포함) | 시간 단위 | top code |
| --- | ---: | ---: | --- | ---: |
| `yellow_trip_hourly` | 168 | 256 | hour | 없음 |
| `intermittent_frozen_5000` | 520 | 256 | week | 없음 |
| `raf_spare_parts` | 84 | 84 | month | 없음 |
| `insta_market_basket` | 52 | 64 | day | 30 |

공통 학습 설정은 batch 128, AdamW(lr=0.001, weight_decay=0.01), gradient clip 1,
최대 300 epoch, 최소 40 epoch, patience 40, seed 42/52/62다.
checkpoint는 validation의 **원 수량 RMSE가 엄격하게 낮아진 최초 epoch**로 선택한다.
time NLL과 log1p-quantity MSE의 가중치는 각각 1이다.
수량 구간별 지표도 config의 기존 고정 경계값을 사용한다. 분위수를 다시 계산해서
반올림 경계나 구간 소속이 달라지게 만들지 않는다.

기본 데이터 위치는 `sample_data/phen/<dataset>/train_validation.parquet`다.
이 경로는 기존 `.gitignore`의 연구 데이터 제외 규칙을 따른다. 데이터는 배포하지
않는다. `--data`로 다른 위치를 지정할 수 있지만, 허용된 파일 SHA256과 canonical
target 목록·수량 hash, train 초기화 통계가 모두 맞아야 한다.
재저장한 parquet는 내용이 같아도 파일 hash가 달라질 수 있으므로 자동 허용하지 않는다.

Taxi/Intermittent/RAF는 raw9 실행 계약에서, Instacart는 기존 observed-time
계약에서 설정을 가져왔다. 각 JSON의 `provenance`와
[추출 기준 기록](reference_manifest.json)을 확인한다. 설정 파일을 바꿔 새로운
데이터를 쓰는 경우에는 기존 실험의 재현이 아니라 별도 실험으로 기록해야 한다.

## 실행 전 확인

저장소 밖에서 shell 파일을 호출해도 저장소 root로 이동한다. GPU는 자동 선택하지
않으며 기본 device는 CPU다. `--dry-run`은 config만 읽고 데이터·출력 파일은 열지 않는다.

```bash
SEEDS=42 bash scripts/phen/run_all.sh yellow_trip_hourly --dry-run

SEEDS=42 bash scripts/phen/run_phen.sh yellow_trip_hourly \
  --data /absolute/path/to/events.parquet --mode preflight
```

`preflight`는 파일·분할·초기화 통계를 확인하지만 모델을 생성하거나 학습하지 않는다.

## 모델별 학습

```bash
bash scripts/phen/run_phen.sh yellow_trip_hourly \
  --data /absolute/path/to/events.parquet --device cuda:0 \
  --output-root search_artifacts/phen/repro_v1
```

기본은 42/52/62 seed를 순차 실행한다. 한 seed만 실행하려면 앞에 `SEEDS=42`를 붙인다.
아래 shell 파일은 동일한 옵션을 받는다.

```text
run_phen.sh
run_rmtpp.sh
run_thp.sh
run_nhp.sh
run_sahp.sh
run_s2p2_matched_head.sh
run_attnhp_matched_head.sh
```

`run_all.sh`는 같은 데이터에서 7개 모델을 순차 실행한다. 추가 병렬 실행이나
원격 접속, tmux 생성, GPU process 제어는 하지 않는다. 짧은 별도 실행을 할 때는
`--epochs 1 --min-epochs 1`처럼 함께 지정하고, 본실험과 다른 output root를 쓴다.

## 선택된 checkpoint 평가

validation replay:

```bash
SEEDS=42 bash scripts/phen/run_phen.sh yellow_trip_hourly \
  --data /absolute/path/to/events.parquet --device cpu \
  --output-root search_artifacts/phen/repro_v1 \
  --mode evaluate \
  --checkpoint search_artifacts/phen/repro_v1/yellow_trip_hourly/phen/seed_42/best.pt
```

별도 승인 후 test를 평가할 때만 위 명령에 `--split test --allow-held-out`을 추가한다.
test가 포함된 원본 파일도 config의 허용 hash에 등록되어 있어야 한다. 현재 RAF
config는 train/validation snapshot만 묶여 있으므로 그대로 test를 평가할 수 없다.
추가 test 입력은 별도 검토한 설정이 필요하다. test로 checkpoint를 선택하지 않는다.

## 산출물

```text
<output-root>/<dataset>/<model>/seed_<seed>/
  manifest.json                   설정, 데이터 hash·대상 목록, source/runtime 정보
  source.tar.gz                   실행 시 사용한 패키지와 shell source
  pip_freeze.txt                  실제 실행 환경
  history.jsonl                   epoch별 train/validation 지표
  best.pt                         validation raw-RMSE 선택 모델과 optimizer 상태
  last.pt                         마지막 epoch 상태
  validation_predictions.parquet  선택된 모델의 validation 예측과 target 식별자
  summary.json                    선택 epoch, 전체 및 수량 구간별 지표
  evaluation_validation/          명시적으로 요청한 replay 결과
```

기존 출력 디렉터리는 덮어쓰지 않는다. 실패한 실행도 보존하며 새 output root로
다시 시작한다. 자동 resume은 지원하지 않는다. checkpoint의 optimizer/RNG 정보는
기록용이며, 완전한 학습 재개 계약을 의미하지 않는다.

기존 Shared15 가중치는 `phen.legacy.translate_legacy_state`로 이름을 변환한 뒤
`strict=True`로 로드할 수 있다. 이는 weight 변환이며, 과거 checkpoint의 전체
실험 metadata 검증이나 optimizer resume이 아니다.
