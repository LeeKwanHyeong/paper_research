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
- 위 결과는 CPU 합성 검증이다. 실제 GPU 용량 검증과 학습 시작은 아래 별도 실제 증적에 연결한다.

## 실제 GPU 자격 검증 — 완료
- 커밋된 과학소스·입력9·기존A사본6종의전수SHA를묶어별도root에배포했다. 기존root나checkpoint는덮어쓰지않았다.
- 각GPU에서dataset/arm synthetic12개와B128/길이256 capacity4개가실제exit0이어야학습을시작한다.
- 5080의원래정지프로세스351034/351048/351049/353905는T/start_ticks/argv/cwd/pgid를보존한다. 해당GPU예약메모리를측정자신allocator메모리와구분한다. 다른active소유자는차단한다.
- `paper_research`의 `codex/shared-component-ablation-20261022` 브랜치 commit `b1455baec1c084e422ce044dec7cd454acd95631`을 두 서버의 새 전용 root에 배포했다. 실행 계약 raw SHA256 `462996103c87523785965f733404d9c56f6e03e73418aa38f593838c4a6c696a`, canonical `e81652c0b644fd00bf6e8a3499294aca6b111a2d744226dab0a224feea416db6`이다.
- 실제 배포 SHA는 source987/input9/legacy36개 모두 일치했다. 양GPU 각각 synthetic12+capacity4와 dispatcher 실제 exit0을 확인했다. 원문은 `deployment/`와 `observations/qualification_check_2/`, 연결 요약은 `NATIVE_QUALIFICATION.json`이다.

## 학습과 A 재사용 검증 — 진행 중
- 실제 최초 실행은2026-10-22 09:41 KST다. 5080 dispatcher848518/start_ticks102629746, parent848511; 5090 dispatcher1832601/start_ticks7186554, parent1832598을 실제 argv/cwd와 함께 확인했다. 원문 `launch/`와 `continuation_v1/LAUNCHED.json`을 보존한다.
- 초기 관측에서5080 worker848523의A학습,5090 A재사용 admission 실제exit0 뒤 worker1832803의B학습을 확인했다. 이 과거 초기 관측은 전체학습·Validation/Test 완료를 뜻하지 않는다.

## 전체 Validation gate 뒤 Test와 비용 비교 — 다음 작업
- GPU당과학worker1개. 최대300/min40/patience40, AdamW .001/wd.01/clip1, batch128, 전체Validation raw수량RMSE 최초엄격최소선택을고정한다.
- 부모가각scientific child와dispatcher의실제OSexit를관측해보존한다. 자체supervisor의OSexit를직접관측했다고표현하지않는다.
- 전체36개선택본의개별Validation재현과실제exit를묶은두hostglobalgate이후에만고정Test36개를평가한다. 네사전지정구조를모두보고하며Test를보고모델·seed를변경하지않는다.
- 선택본RAM사본으로Train전용37batch(5warmup/32측정), train/eval각3반복을별도process에서측정한다. checkpoint저장0, 원본SHA전후불변을확인한다.
- 장시간SSH연결이나별도background finisher를만들지않는다. 확인은짧은read-onlySSH, 회수는완료/실제exit0뒤한번만수행한다. scientific retry/claim삭제/partial덮어쓰기0.
- 새보고는Validation/Test 전체·큰수량RMSE/MAE·recordedTimeNLL 평균±표본SD(ddof1), 모든seed절대/상대차이·악화, GPU/seed별시간·학습/평가allocated/reserved peak·파라미터를포함한다.

## 보존과 범위 — 유지
- 기존A/외부5seed·MAC결과/checkpoint/실패기록을보존한다. 기존외부5seed는A결과이므로다른변형으로이름만바꾸지않는다.
- 원고자동작성·새MAC학습·외부baseline재학습·별도단순MLP추가실험으로확대하지않는다.

## 후속 확인과 비교 제어 — 완료, 학습 종료 대기
- `paper_research`의 `codex/shared-component-ablation-20261022` 로컬 제어 commit `8300e2348f49d3b03f0b7d13c924ea2b8ecbea7f`에 controller/comparator와 독립 QA를 별도로 기록했다. 실제 원격 실행 소스는 앞의 `b1455bae`에서 변경하지 않았다.
- controller 독립67 tests, comparator 실제 합성21확인 및CLI3process, 최종 전체170 tests+4 subtests가 actual exit0이다. 현재 코드와 QA·원본 source/input/legacy·실제 시작 신원을 `continuation_v1/CONTROL_BINDINGS.json`에 묶었다. binding SHA `446dc5af7e1558e5003e9fe970c8ffb097b93e6f467eabff51b5b7ac282e7e90`, authorization SHA `3905719993b7179cbf7f892a9ddaa24395e98d900459c3a97b6a4434fd72dd42`다.
- 09:56:34 KST 최초 actual advance exit0에서 두 host의 읽기 전용 SSH actual exit0와 waiting·고정 dispatcher 신원 일치를 확인했다. RAF 후속 history와 5080 Taxi A epoch27이 관측됐으나 해당 tail 값은 선택본 결과표나 전체완료 근거가 아니다. 전체 Validation/Test/회수/비교는 아직 미완료다.
- 이 채팅의 새 heartbeat `a-d-3seed`를 ACTIVE/15분으로 생성했다. 실제 원문은 `continuation_v1/FIRST_ADVANCE.*`, 각 round, `AUTOMATION.json`과 `AUTOMATION_PROMPT.txt`에 보존한다. 기존2seed/mac9/다른자동화는 변경하지 않았다. 정상대기는조용히유지하고주요전환/실패/완료때만알린다.
- 다음 순서는 진행 중인 양host fit/Validation 회수 → 전역36개 개별 Validation gate → 고정 Test36 및 비용216 → 전수회수·별도비교·독립실제결과검토다. 기존 partial/실패/claim을 삭제하거나 재시작하지 않는다.

## 새 논문 중심 문서 — 완료
- 사용자가 지정한 논문 작성 상위 페이지 아래 `논문 중심 개선안`과 7개 주제+과거기록 보관 페이지를 생성하고 모든 제목·하위 경로·내용·목차8개 연결을 다시 확인했다. 외부5seed 표·악화seed·A–D 미완료 상태를 구분했다.
- 새URL `https://app.notion.com/p/401bbe405613815eafbff5c04fec8744`. 생성·readback 증적은 `NOTION_IMPROVEMENT_COMPLETE.json`, 실제 export 본문은 `notion_improvement_drafts/`다. 기존 `논문 중심`은 수정하지 않았다. 서버PID·내부경로·commit·인증정보는 새문서에 전송하지 않았다.
- 앞선 Notion 상태갱신의 자동승인거부는 원문으로 보존했다. 이후 사용자가 새 목적지와 문서 재구성을 명시승인해 새 페이지를 생성한 별도 작업이며, 거부된 내부정보 payload를 재전송하지 않았다.
