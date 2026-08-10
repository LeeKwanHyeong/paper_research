# Paper workspace

이 디렉토리는 논문 본문, 표, 그림, 실험 계약과 생성 증적의 기준 위치다.
앞으로 논문에 직접 사용하거나 논문 산출물을 생성하는 작업은 이 디렉토리 아래에서 관리한다.

## Directory contract

- `contracts/`: 데이터셋과 공정 비교 조건의 사람이 검토하는 원천 계약
- `scripts/`: 표와 그림을 원천 데이터에서 다시 생성하는 코드
- `tables/`: 논문 본문과 appendix에 사용할 Markdown/CSV 표
- `figures/`: 논문용 그림과 그림별 원천 데이터
- `data/`: hash, 데이터 감사 결과와 표 생성 provenance
- `drafts/`: section outline과 원고 초안
- `references/`: section별 핵심 문헌, 사용 근거와 주장 범위
- `manifests/`: experiment artifact의 최종표 사용 자격과 재실행 필요 여부

## Editing rule

`tables/`와 `data/`의 generated file은 직접 수정하지 않는다. 데이터 정의는
`contracts/`에서, 계산과 출력 형식은 `scripts/`에서 수정한 뒤 생성 명령을 다시 실행한다.

```bash
python paper/scripts/build_t1_t2.py
```

생성 스크립트는 fixed split manifest와 실제 parquet를 대조하며, row 수, sequence 수,
split 순서, mark 분포와 SHA-256이 유효할 때만 T1과 T2를 출력한다.

## Current artifacts

- Manuscript v0.6: professor feedback revision with Taxi and Instacart e300 validation results
- v0.6 revision notes: applied feedback, evidence boundaries and submission blockers
- Qualified e300 tables: seed-level and mean/standard-deviation results for Taxi and Instacart
- T1: fixed-split dataset statistics, task construction and dataset identity
- T2: matched model configuration and frozen training/evaluation protocol
- F1: event-based quantity-aware problem formulation
- F2: TitanTPP architecture and controlled RMTPP backbone comparison
- F3: frozen dataset sequence-length and quantity survival distributions
- Introduction: paragraph outline and claim-evidence map
- Related Work: recurrent/attention TPP, continuous quantity mark, intermittent-demand 문헌 지도
- Draft history: section drafts and manuscript versions v0.1 through v0.6
- Artifact manifest: 39-run frozen contract qualification by dataset, model and seed

```bash
python paper/scripts/build_f1_f3.py
```
