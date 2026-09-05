# 첫 Taxi B1 e1 데이터 접근 범위 감사

- 대상 소스: `add02baae27c25da02008873c578f11926f8d125`. 아래 줄 번호는 이 revision 기준이다.
- 방법: 실행 소스와 기존 분할 설정·생성 코드의 읽기 전용 추적. 실제 test parquet/행을 다시 조회하지 않았고, 새 학습·평가·데이터 통계 계산도 하지 않았다.
- 실행 감사에서 보고된 **test 8,327행의 메모리 적재는 계약 위반**이다. 이 결과는 정상 e1 완료 증적·대조군·성능 채택에 사용하지 않는다.

## 확인한 범위

| 경로 | 코드 근거와 판단 |
|---|---|
| 원본 프레임 적재 | [runner:393](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/run_count_aware_tpp_backbone_control.py:393)의 필터 분기에 B1·prior-prefix가 없어서, B1의 `experimental` 실행은 400행의 `pl.read_parquet(args.data)`로 전체 프레임을 적재한다. |
| test 행의 추가 전처리 | [prepare_count_frame:29](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/count_aware_tpp_backbone/core.py:29)는 전체 프레임에 수량 음수 검사·상수 mark 생성·수량 형 변환을 적용한다. [Dataset:276](/Users/igwanhyeong/PycharmProjects/paper_research/data_loader/event_seq_data_module.py:276)은 전체 series의 seq·시간·수량·split을 Python lists로 보관한다. 따라서 “test는 전혀 처리하지 않았다”라고 표현할 수 없다. |
| 수량 통계·interface | [train_quantile_contract:212](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/run_taxi_quantity_interface_ablation.py:212)의 quantile과 [runner:474](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/run_count_aware_tpp_backbone_control.py:474)의 log 수량 평균·표준편차는 명시적인 `chronological_split == train` 필터 뒤에 계산한다. test 수량을 이 통계에 합산하는 경로는 없다. |
| 시간 통계 | [derive_train_time_contract:106](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/run_count_aware_tpp_backbone_control.py:106)은 `target_splits={train}`인 Dataset의 target 시간만 모아 통계를 계산한다. test target을 사용하는 경로는 없다. |
| 학습·평가 대상 | [train_one:288](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/count_aware_tpp_backbone/training.py:288)은 train loader와 validation loader만 만든다. [make_loader:299](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/run_taxi_quantity_interface_ablation.py:299)가 전달한 target split을 [Dataset:305](/Users/igwanhyeong/PycharmProjects/paper_research/data_loader/event_seq_data_module.py:305)가 필터한다. 이 실행 경로에는 test target loader가 없다. |
| checkpoint 선택 | [training:401](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/count_aware_tpp_backbone/training.py:401)의 epoch 평가와 [training:489](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/count_aware_tpp_backbone/training.py:489)의 복원 후 평가는 모두 validation loader를 사용한다. 447행의 선택 기준도 validation joint objective다. |

## 입력 이력에 대한 판단과 한계

[Dataset:340](/Users/igwanhyeong/PycharmProjects/paper_research/data_loader/event_seq_data_module.py:340)은 선택된 target 직전까지의 과거 index만 이력으로 사용한다. **이력 자체에는 별도의 split 필터가 없다.** 따라서 test 행이 history에 들어오지 않는다는 판단은 고정 chronological split의 순서 계약에 의존한다.

[분할 생성 코드:120](/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/notebooks/preprocessing/tpp_split_utils.py:120)는 series별 seq 순서를 매긴 뒤 153–158행에서 train → validation → test 순서로 배치한다. 기존 Taxi manifest의 설정은 entity=`oper_part_no`, order=`seq`, 비율 0.7/0.15/0.15다. **이 고정 순서가 유지되는 전제에서는 test 행이 train·validation target의 과거 이력으로 선택되지 않는다.** 이번 감사는 실제 test 행을 재조회하여 그 전제를 다시 검증한 것은 아니다.

[target_outputs:69](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/count_aware_tpp_backbone/core.py:69)는 선택 target의 수량과 memory write를 차단하고 직전 history state로 예측한다. [CountAwareTPP:867](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTPP.py:867)의 일반 학습 호출은 별도 online state를 전달하지 않아 입력 window마다 memory가 초기화되므로, 프레임에 보관된 다른 행의 memory를 넘겨받는 경로도 없다.

정확한 결론은 **“test 행의 적재·전처리는 발생했다. 코드와 고정 chronological split 계약상 test target을 학습·평가하거나 checkpoint 선택에 사용하는 경로는 확인되지 않았다”**이다. `held_out_test_evaluated=false`는 test 프레임 미적재를 증명하는 표지가 아니며, 이번 접근 계약 위반을 해소하지 않는다.
