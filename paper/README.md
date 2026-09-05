# Paper workspace

이 디렉토리는 논문 본문, 표, 그림, 실험 계약과 생성 증적의 기준 위치다.
현재 제출 경로는 **TitanTPP v0.7 validation freeze**이며 held-out test는 잠겨 있다.

## Directory contract

- `contracts/`: 최종 모델 identity, 데이터셋과 공정 비교 조건
- `scripts/`: 표와 그림을 원천 데이터에서 재생성하는 코드
- `tables/`: 본문과 appendix용 Markdown/CSV 표
- `figures/`: 논문용 그림과 그림별 source data
- `data/`: 데이터 감사 결과와 생성 provenance
- `drafts/`: revision note, section outline과 이전 원고
- `references/`: 문헌, 사용 근거와 주장 범위
- `manifests/`: 원천 artifact와 생성 산출물의 hash 및 사용 자격
- `results/`: validation 결과, 진단과 독립 검산 증적

`tables/`, `figures/source_data/`와 `manifests/`의 generated file은 직접 수정하지
않는다. 데이터 정의 또는 출력 형식은 생성 스크립트에서 수정한 뒤 다시 빌드한다.

## 현재 기준선 — v0.7

- [최종 모델·주장 계약](contracts/titantpp_v0_7_final_model_claim_contract_v1.md):
  original mark-free Hard-LMM / TitanTPP-T0와 허용 주장 동결
- [v0.7 manuscript](titantpp_short_paper_draft_v0_7_manuscript.md): RMTPP·THP·TitanTPP
  공정 비교, 세 데이터셋의 validation 결과와 한계
- [Revision notes](drafts/v0_7_revision_notes.md): 감사 결정, 데이터셋 역할과 남은 작업
- [Validation freeze](results/titantpp_v0_7_validation_freeze_20260905/README.md):
  Taxi·Instacart seed·구간 결과, quantity-interface 감사와 독립 검산
- [Time-head revision audit](results/titantpp_v0_7_validation_freeze_20260905/time_head_revision_audit.json):
  validation 실행 revision의 cap 300·10과 현재 기본값 차이 검증
- [T1](tables/T1_v0_7_dataset_statistics.md): 평가 artifact와 identity가 일치하는
  Intermittent-5000·Taxi·Instacart identity와 train-only 분포 통계
- [T2](tables/T2_v0_7_model_training_contract.md): 공통 모델·학습 계약
- [T3](tables/T3_v0_7_backbone_validation.md): 3 datasets × 3 models × 3 seeds
  validation 집계
- [T4](tables/T4_v0_7_paired_titan_deltas.md): 같은 seed의 TitanTPP paired delta
- [T5](tables/T5_v0_7_quantity_strata_validation.md): train에서 경계를 고정한 세
  데이터셋의 validation quantity-stratum MAE·RMSE
- [Figure register](figures/F1_F3_figure_register.md): 현재 F1·F2와 v0.6 archive 구분
- [Artifact manifest](manifests/titantpp_v0_7_validation_freeze_manifest.md): source,
  generated artifact와 publication document SHA-256, held-out 잠금 상태

기존 quantity-interface 결과는 v0.7 공정 비교 계약을 충족하지 못하므로 진단
증적으로만 유지한다. Direct log regression은 공통 인터페이스이며 categorical
interface 대비 우위는 현재 기여에서 제외했다.

## Rebuild and verify

```bash
python paper/results/titantpp_v0_7_validation_freeze_20260905/build_freeze.py
python paper/results/titantpp_v0_7_validation_freeze_20260905/independent_verify.py
python paper/scripts/audit_v0_7_time_head_revisions.py
python paper/scripts/generate_v0_7_mark_free_figures.py
python paper/scripts/build_v0_7_paper_artifacts.py
python paper/scripts/verify_v0_7_paper_artifacts.py
```

첫 두 명령은 Taxi·Instacart validation 증적을 재구성하고 329개 행을 독립 검산한다.
세 번째 명령은 세 validation 실행 revision의 time-head 수식을 확인한다. 다음 두
명령은 mark-free Figure 1과 frozen source에 연결된 v0.7 표·Figure 2·manifest를
생성하고 루트 README를 v0.7 원고와 동기화한다. 마지막 검산은 세 데이터셋의
27개 seed 행, T1–T5, 두 그림, 원고·README와 manifest hash를 원천 파일에서 다시 확인하며 결과를
[`titantpp_v0_7_validation_freeze_verification.json`](manifests/titantpp_v0_7_validation_freeze_verification.json)에
기록한다. 어느 명령도 held-out test를 실행하지 않는다.

## 이전 revision

`titantpp_short_paper_draft_v0_6_manuscript.md`, 과거 T1/T2/F1/F2/F3와
`final_fair_artifact_manifest.*`는 v0.6 provenance를 보존한다. 현재 v0.7 표나
주장의 원천으로 사용하지 않는다.
