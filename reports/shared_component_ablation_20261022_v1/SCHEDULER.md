# A–D 공유 이력·prototype 내부 비교 실행 기록

## 현재 기준선 — 완료
- 사용자가 A 재사용 감사→계약 고정→구현·검증→5080/5090 공통 3seed 학습·Validation/Test·속도/메모리 비교를 승인했다. MAC은 계속 보류한다.
- 프로젝트 `/Users/igwanhyeong/PycharmProjects/paper_research`, 브랜치 `codex/shared-component-ablation-20261022`.
- 과학 계약 `SCIENCE_SPEC.json` SHA256 `3c3aaf9ee9fd8c20b0705bc4d45a4604dd6425810728656ca8105f498f7ad2ce`.
- A 재사용 감사 `A_REUSE_AUDIT.json` SHA256 `276834024cdded7128d5be7418bac7218efff432b01bd925c4d243004333b6e2`: source989/A983·input9·원본222파일 SHA, 선택9 binary/tensor/history, Train9+Validation9 실제 exit0, 개별 Validation 예측 일치 확인.
- seed42는5080, seed52/62는5090. 같은 데이터·seed의4arm은같은GPU. A6학습재사용+Afresh3+BCD27=신규30fit, Validation36/Test36을새경로에서수행한다.
- A는원래모델factory/identity를보존한다. B/C/D만후단prototype4096개와중간history9600개를실제제거한다. persistent16/encoder2block/causal/공통head·loss·목표mask는동일하다.
- 큰수량 cutoff는원래 `quantity_boundaries_all_train_rows[-1]`이며임의로바꾸지않는다. 결과유불리로seed/구조/기준을변경하지않는다.

## 구현과 독립 검증 — 완료
- 원본 과학 소스와 모델 파일은 수정하지 않는다. 신규 중립 model/runner/profile/manager 파일만 사용한다.
- 원래A52/62훈련은5090/cuDNN92000·3병렬, 원래Test는5080/cuDNN91900이다. 새A–D비교는fit별동일GPU에선택본평가와cost를맞춘다. 과거A총훈련시간을신규단독훈련시간과직접비교하지않는다.
- 독립 검증은97 tests+4 subtests, 최종 통합 검사는102 tests+4 subtests 모두 actual exit0이다. 실행 소스4개의 검사 전후 SHA가 같으며 `INDEPENDENT_QA.json`과 `LOCAL_TESTS_v2.json`에 연결했다.
- 실제 GPU 학습/평가/qualification은 아직 시작하지 않았다. CPU 합성 검증을 실제 GPU 용량 검증이나 과학 실험 완료로 해석하지 않는다.

## 실제 GPU 자격 검증 — 다음 작업
- 커밋된 과학소스·입력9·기존A사본6종의전수SHA를묶어별도root에배포한다. 기존root나checkpoint를덮어쓰지않는다.
- 각GPU에서dataset/arm synthetic12개와B128/길이256 capacity4개가실제exit0이어야학습을시작한다.
- 5080의원래정지프로세스351034/351048/351049/353905는T/start_ticks/argv/cwd/pgid를보존한다. 해당GPU예약메모리를측정자신allocator메모리와구분한다. 다른active소유자는차단한다.

## 학습과 평가·회수 — 다음 작업
- GPU당과학worker1개. 최대300/min40/patience40, AdamW .001/wd.01/clip1, batch128, 전체Validation raw수량RMSE 최초엄격최소선택을고정한다.
- 부모가각scientific child와dispatcher의실제OSexit를관측해보존한다. 자체supervisor의OSexit를직접관측했다고표현하지않는다.
- 전체36개선택본의개별Validation재현과실제exit를묶은두hostglobalgate이후에만고정Test36개를평가한다. 네사전지정구조를모두보고하며Test를보고모델·seed를변경하지않는다.
- 선택본RAM사본으로Train전용37batch(5warmup/32측정), train/eval각3반복을별도process에서측정한다. checkpoint저장0, 원본SHA전후불변을확인한다.
- 장시간SSH연결이나별도background finisher를만들지않는다. 확인은짧은read-onlySSH, 회수는완료/실제exit0뒤한번만수행한다. scientific retry/claim삭제/partial덮어쓰기0.
- 새보고는Validation/Test 전체·큰수량RMSE/MAE·recordedTimeNLL 평균±표본SD(ddof1), 모든seed절대/상대차이·악화, GPU/seed별시간·학습/평가allocated/reserved peak·파라미터를포함한다.

## 보존과 범위 — 유지
- 기존A/외부5seed·MAC결과/checkpoint/실패기록을보존한다. 기존외부5seed는A결과이므로다른변형으로이름만바꾸지않는다.
- 원고자동작성·새MAC학습·외부baseline재학습·별도단순MLP추가실험으로확대하지않는다.
