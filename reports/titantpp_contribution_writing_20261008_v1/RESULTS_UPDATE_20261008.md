# 완료 결과를 지금 본문에 반영한 기록

2026-10-08. 기다려야 하는 것은 새 Q1/Q2의 최종 판정이다. 완료된 근거의 Results·Discussion 작성은 지금 수행할 수 있어 [영문 초안](/Users/igwanhyeong/PycharmProjects/paper_research/paper/drafts/titantpp_contribution_methods_20261008_v1.md)의 **7절 결과·8절 논의**를 추가했다.

## 반영한 내용 — 완료

| 근거 | 본문 반영 | 해석 경계 |
|---|---|---|
| 완료된 MLP8 구조54 | 6구조×3데이터의 RMSE 평균±표본SD, H1/H0 18/18 개선 | CNN+GRU54 성과로 이전하지 않음 |
| 동일 예산 대조 | Taxi·RAF에서 PF/HC 대비 각3seed 기준 충족; Intermittent 미충족 | 지정한 구조 package 대조이며 통계적 유의성 아님 |
| prototype 반례 | H1에 P1 추가 시 Taxi0.46%·Intermittent2.92% 평균 RMSE 악화; RAF0.12% 개선 | prototype 보편적 필요성·시너지 주장 불가 |
| 시간·큰 수량의 손해 | Taxi 전체 TimeNLL +53.17%/+69.64%, Taxi PF대비 큰수량RMSE 악화, RAF 꼬리 지표 악화 | 수량 최적 epoch를 유지; 별도 시간최적값과 섞지 않음 |
| MLP8 구조54의 비용 | 같은 GPU 내 합성 qualification-step wall time과 peak allocation | 순수 GPU throughput·추론 latency·전체 fit 시간 아님 |
| Instacart 완료1조건 | MLP8 seed52의 같은 선택 epoch 6지표 | 중간 기술적 결과. 구조 비교·3seed평균·전체9조건 완료 아님 |

## Instacart 중간 행의 증거 범위

보존된 실제 관측은 **2026-10-08 14:38:07 KST**, fit 완료 기록은 **13:40:21 KST**다. 관측시각을 새 시각으로 바꾸지 않았다. 조건은 `insta_market_basket / titantpp_history_mlp_width8 / seed52 / RTX5090`, 마지막 저장E55·선택E15다.

| Validation 범위 | 대상 수 | RMSE | MAE | TimeNLL |
|---|---:|---:|---:|---:|
| 전체 | 503,733 | 5.884102 | 3.994055 | 2.811975 |
| 큰 수량 >35 | 6,036 | 24.665215 | 22.087655 | 2.899737 |

[관측 snapshot](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5090_1791437887192863000.json)의 로컬 raw SHA는 `8db0339244646b0e230e8099cb11de373952490475c60ce8349a00e0af4dd9f3`이며 저장 pointer와 일치한다. 원시 readback도 해당 completed job과 일치한다. `rawSHA_verified=true`인 관측·endpoint 근거다.

전체 stage completion receipt와 selected/last binary의 최종 로컬 회수는 이 기록에서 아직 없다. 그러므로 단일 fit의 완료 기록을 전체 stage의 원본 감사 완료로 쓰지 않는다. 진행 중 fit의 수치를 이 행과 비교해 최종 우열을 내리지 않았다.

## 계산비용의 불리한 결과도 보존

Dataset 이름은 합성 길이 시나리오를 가리키며 실제 dataset batch의 throughput 측정이 아니다.

| GPU / 합성 길이 | P1_H1 wall time | PF_H1 wall time | PF대비 시간 증가 | PF대비 peak 증가 |
|---|---:|---:|---:|---:|
| 5080 / T256 | 36.459 ms | 25.340 ms | 43.88% | 67.37% |
| 5080 / T84 | 14.148 ms | 11.486 ms | 23.17% | 117.84% |
| 5090 / T256 | 23.108 ms | 17.740 ms | 30.26% | 67.37% |

원본은 구조54 [5080 qualification](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/qualification/5080/receipt.json)과 [5090 qualification](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/qualification/5090/receipt.json)이다. [측정 코드](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/operation/qualify.py:48)는 seed42·합성 batch128의 **3 optimizer step 중앙값**을 사용하며 별도 warmup을 버리지 않는다. 시간에는 CPU gradient 복사·SHA·finite 검사가 포함되고 peak는 이후 eval 검사도 포함한다.

따라서 동일 예산에도 측정 비용 증가가 있었다는 기술만 가능하다. 다른 GPU 시간을 섞거나 새 검색 실험의 warm5/measure20 비용 gate를 이 과거 측정에 소급하지 않는다. 전체54 fit의 epoch timing을 이번 작업에서 새로 집계하지 않았으며 실제 end-to-end 학습시간 결론은 별도다.

## 검증과 아직 남은 작업

- **완료:** 원본 Validation 요약54행의 고유성을 확인하고, 18그룹×6지표의 평균108개·표본SD108개를 표준 라이브러리로 재계산해 일치 확인. H1/H0 18/18 승리와 본문 비율 재계산.
- **완료:** 원래 완료 receipt·Validation summary의 SHA를 보존 provenance와 대조. Instacart snapshot/readback SHA 확인. 비용 비율은 같은host/길이 원본값에서 재계산. [파생 근거](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_writing_20261008_v1/results_update_evidence.json)에 저장.
- **다음 작업:** Instacart9조건의 완결된 구조/seed 비교를 갱신하고, 새 보정27 및 검색27의 사전 판정에 따라 해당 결과 절을 추가한다.
- **학습과 병렬 가능:** 외부 비교 provenance와 미노출 평가 자료의 접근 계보, 가장 가까운 선행연구를 추가 정리한다.

학습·재평가·SSH·자동화 변경은0회다. Test 및 혼합 성과를 열지 않았다. 기존 원본·selector·계약은 유지했다. 문서는 계속 AI-assisted 내부 연구 검토용이며, 이전에 명시한 PAKDD2027 투고 정책 경계도 유지한다.
