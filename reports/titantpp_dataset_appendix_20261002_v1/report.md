# TitanTPP 데이터 특성·구간별 validation 오차 및 Appendix 보강

2026-10-02. 대상은 현재 LNCS 원고와 이미 완료된 네 데이터의 validation 결과다.

## 반영 결과

- **본문 5.4:** Taxi·Intermittent의 큰 수량 구간 개선, RAF의 구간별 상쇄, Instacart의 MAE/RMSE 순위 역전을 연결했다. train의 이력 길이·상관은 설명 가설로 구분했다.
- **Appendix A:** 2026-10-02 07:14 KST의 완료 부분집합과 미완료/감사 대기 설명을 그대로 보존했다. 이 시각 이후의 학습 상태로 갱신했다고 주장하지 않는다.
- **Appendix B:** 네 데이터의 사건·수량·시간 단위, train 표본 수·수량 분포·이력 길이·인접 상관 및 이력 분포 그림을 추가했다.
- **Appendix C:** 완료된 84조건의 저장된 selected validation 기록에서 수량 20행·이력 15행의 비교표와 Instacart 오차 기여 그림을 구성했다. 빈 RAF 이력 구간도 유지했다.

현재 원고는 표 9개·그림 4개·Appendix A/B/C를 포함한다.

## 분석 범위와 재사용

Taxi·Intermittent·Instacart train 기술통계는 `reports/titantpp_data_characteristics_20260928_v1/analysis.json`의 `datasets`만 재사용했다. 이 파일에 함께 있는 과거 미완료 validation 표는 사용하지 않았다. 세 train 원본 SHA와 동결 계약의 target 수를 다시 확인했다.

RAF는 같은 train 분석 함수를 사용하되 `chronological_split == train` 필터를 먼저 적용했다. 원래 RAF 배포 묶음은 train/validation만 들어 있는 파일을 사용했으므로, 보존된 parent 데이터의 train target identity·quantity SHA 및 초기화 통계가 일치하는지 확인했다. 분석 함수의 선택적 장기 이력 하위집단이 RAF에서 비어 있는 경우는 상관 미정의로 기록했다. 기존 함수나 과학소스는 변경하지 않았다.

모델별 완료된 checkpoint와 source의 기존 CPU 감사는 재사용했다. 이번 작업은 저장된 selected endpoint JSON, 소형 파일 SHA, 표본 수 및 오차 합계를 확인한 분석이다. 학습·model forward·replay·새 binary 감사·원격 관측·held-out 성능 열람을 실행하지 않았다.

## 공통 정의

| 항목 | 정의 |
|---|---|
| 사건/수량 | Taxi: 활성 공간 셀의 시간별 승차 수; Intermittent: site–part 양수 주문량; RAF: 월별 양수 부품 수요; Instacart: 같은 사용자·기록 활동일의 상품 행 합계로 만든 basket-size proxy |
| 수량 train 통계 | 각 train 사건 행을 한 번씩 사용; 첫 사건 포함; nearest 분위수 |
| gap/history 통계 | 각 계열의 첫 사건을 제외한 canonical train target; gap은 해당 데이터의 기록 단위 |
| usable history | target·padding 제외, 실제 lookback 및 sequence cap 안에 남는 관측 사건 수 |
| 인접 상관 | 인접한 `log1p(quantity)` 쌍에서 predecessor와 successor의 계열별 평균을 각각 제거한 뒤 residual 쌍을 모은 Pearson 상관; 계열별 상관의 단순 평균이 아님 |
| validation checkpoint | 최초 strict 최소 raw quantity RMSE epoch; 모든 지표·구간에 동일 checkpoint |
| seed 집계 | 각 seed 안에서는 사건 단위 집계, seed42/52/62의 산술평균·표본표준편차; 같은 target을 독립 표본 3배로 계산하지 않음 |
| 비교 모델 | 데이터 전체 mean RMSE로 고정한 외부 모델: Taxi/Intermittent=RMTPP, RAF/Instacart=S2P2. 구간별 최상위 모델로 바꾸지 않음 |

원고의 구간표는 간결하게 세 seed 평균을 표시한다. `strata_three_seed.csv`에는 모든 7모델의 MAE·RMSE·시간 NLL 평균과 표본표준편차가 있고, `strata_by_seed.csv`에는 각 seed 원값과 원본 연결이 있다.

## 핵심 관측

| Train 특성 | Taxi | Intermittent | RAF | Instacart |
|---|---:|---:|---:|---:|
| usable history 중앙값 | 129 | 40 | 3 | 5 |
| usable history 95분위 | 168 | 144 | 8 | 14 |
| 계열 평균 제거 log 인접 상관 | 0.749 | 0.928 | -0.105 | 0.037 |

RAF의 음수 상관은 짧은 계열에 평균 제거를 적용한 기술통계다. 안정적인 음의 의존관계나 구조 성능 차이의 인과적 증거로 해석하지 않는다. train 기술통계와 validation 오차는 다른 모집단이며, 이 둘을 동일 행의 상관 분석처럼 설명하지 않는다.

구간별 MSE 기여는 `mean_seed((SSE_MLP_bin − SSE_external_bin) / N_total)`이다. 음수는 MLP의 개선이다. 전체 RMSE 평균을 제곱하거나 구간 RMSE를 가중평균한 값이 아니다.

| 데이터·고정 비교군 | 작은 수량 쪽 MSE 기여 | 큰 수량 쪽 MSE 기여 | 전체 MSE 차이 |
|---|---:|---:|---:|
| Taxi vs RMTPP | q≤686: +200.886 | q>686: −2,822.868 | −2,621.982 |
| Intermittent vs RMTPP | q≤31: +0.05969 | q>31: −3.77546 | −3.71577 |
| RAF vs S2P2 | q≤60: −31.463 | q>60: +28.463 | −3.000 |
| Instacart vs S2P2 | q≤20: −1.58816 | q>20: +1.94581 | +0.35765 |

수량 단위가 다르므로 데이터 사이 MSE 절대 크기를 성능 순위로 비교하지 않는다.

- Taxi·Intermittent의 RMSE 이득은 큰 수량 구간에서 나온다. 두 데이터 모두 하위 두 수량 구간의 RMSE는 RMTPP가 낮다.
- RAF는 수량 60 이하의 이득이 60 초과 손실에 대부분 상쇄된다. 이 비교는 S2P2 기준이며, 전체 MAE 최상위는 기존 표와 같이 RMTPP다.
- Instacart는 q≤20이 validation의 89.37%다. 전체 MAE는 MLP가 0.00712 낮지만, 큰 수량의 손실로 전체 MSE는 0.35765 높다. 모든 비어 있지 않은 저장 이력 구간에서도 S2P2의 평균 RMSE가 낮다.
- 이력 길이와 이득이 단조롭게 대응하지 않는다. Taxi는 >128 구간에서 이득이 집중되지만, Intermittent는 ≤64와 (64,128]에서 개선되고 >128에서 악화된다.

## 구간 경계와 해상도

수량은 동결된 train 유래 경계를 사용한다. Taxi `(7,686,1562,3449)`, Intermittent `(2,31,46,187)`, RAF `(2,30,60,200)`, Instacart `(8,20,25,35)`다. 경계값은 아래 구간에 포함한다. 첫 구간 `q≤a`, 중간 `(a,b]`, 마지막 `q>d`다.

이력은 저장된 추가 구간을 재사용한다. Taxi·Intermittent·RAF는 `(64,128)`, Instacart는 `(1,3,7,15,31)`이다. 정의는 같지만 경계가 같지는 않으므로 데이터 간 같은 길이 집단 비교로 제시하지 않는다. **RAF validation 6,690건은 모두 ≤64에 들어가므로 기존 기록만으로 RAF의 세밀한 이력 길이별 차이는 확인할 수 없다.** 두 빈 구간은 N=0, 오차 미정의로 보존했다. 이를 메우기 위한 새 추론은 수행하지 않았다.

## 원본 연결과 산출물

- `sources.json`: 분석이 읽은 101개 원본의 경로·SHA.
- `selected_conditions.csv`: 완료 84조건의 dataset/model/seed, selected epoch, validation 지표, endpoint 경로·SHA와 model tensor SHA.
- `train_profiles.json` / `.csv`: 재사용/보충한 train 특성.
- `strata_by_seed.csv` / `strata_three_seed.csv`: 전 모델의 seed별/3seed 구간 지표와 표본 수.
- `anchor_strata.csv`: 원고의 35개 구간 비교와 오차 기여.
- `error_decomposition.json`: partition별 전체 차이 합계.
- `figures/`: 이력 분포와 Instacart 오차 기여의 독립 PDF·PNG. 원고에는 같은 수치를 embedded TikZ로 넣어 외부 그림 의존성을 추가하지 않았다.
- `appendices_B_C.tex`: 통합된 Appendix의 생성 원문.
- `before/`: 편집 전 원고·README·검증기·검증 결과·source manifest·ZIP.

기존 원본 위치는 Core의 `final_conditions.csv`, 추가 TPP의 5080/5090 `retrieved/.../endpoint_replays.json`, RAF의 `comparison.json` 및 해당 endpoint 원본을 따른다. Instacart는 초기 미완료 비교표 대신 `titantpp_instacart_final_analysis_20261001_v1`의 완료 조건을 사용했다. 정확한 경로는 `sources.json`과 `selected_conditions.csv`에 있다.

## 검증

- 84조건의 selected 지표·epoch가 확정 표와 일치하며, 모든 partition의 count·SSE·가중 MAE·시간 NLL이 전체 값과 일치했다.
- 원고의 기존 수치·구조 검사 41개와 새 Appendix 검사 69개를 통과했다. train 표 56개 숫자, 수량/이력 35행 및 source SHA를 확인했다.
- 새 그림 PDF 두 개를 Poppler로 렌더링하고 시각적으로 확인했다. MAE 축의 촘촘한 tick 간격을 수정한 뒤 최종 확인했다.
- 내장 LaTeX 컴파일러는 첫 시도에 기존 `\Description` 미정의 오류를 보고했다. 구형 class용 `\providecommand`를 추가한 뒤 두 번째 시도에 성공했다.
- 내장 컴파일은 LNCS 2.21을 사용했다. 배포 ZIP의 공식 LNCS 2.25 파일은 그대로이며 해당 파일을 사용한 별도 컴파일은 수행하지 않았다. 전체 원고 PDF의 export·페이지 수 확인·육안 조판 검토는 남아 있다.

## 남은 순서

1. **현재 LNCS 원고의 전체 조판을 확인한다 — 다음 작업.** 표·그림 위치, 가독성, 본문/Appendix 분량을 PDF에서 확인한다.
2. **진행 중 36조건의 완료 묶음을 확정한다 — 진행 중 실험 이후.** 기존 관측/감사 범위에 따라 결과가 확정되면 Appendix A와 본문 구조 대조 해석을 갱신한다.
3. **최종 평가의 독립성·범위를 확정해 제출본에 연결한다 — 다음 작업.** 현재 내용은 development validation 분석이며 독립 최종 평가를 대신하지 않는다.
