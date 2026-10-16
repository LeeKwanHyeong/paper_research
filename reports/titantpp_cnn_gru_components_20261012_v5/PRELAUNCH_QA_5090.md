# 5090 독립 시작 사전 QA

상태: `5090_NATIVE_PASS_THREE_INITIAL_FITS_VERIFIED`. 기존 v5 봉인과 최신 사용자 승인만 검증했다. QA SSH/GPU 실행0, source/code/bundle/main.tex 변경0. 코드 변경이 없어 기존 통과 tests를 반복하지 않았다.

## Review Result: PASS — 로컬 봉인과 승인

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

사용자 승인 원문 “5090도 진행하자”가 `hosts=["5090"]`와 기존 canonical에 결합됐다. 승인 대상은 Intermittent seed42/52/62 각각 NONE/CNN-only/GRU-only 동시3개 후 FULL1개, 총12개 fresh fit이다. Train/Validation만 사용하고 Test 평가, warm start, retry/resume는 금지한다.

- canonical: `e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074`
- source131: `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53`
- operation4: `12c836dc5fe1d9146b83a352ba77abc297f4d69c6f68d83ad00935a2bc442903`
- local tool40 seal raw: `16e317c3c16dddede95c6aa6174026dca0ab772879bf438c3d1e6b752eff7074`
- 5090 승인 raw: `c97cedf36b312a46409d1f1669fe617f88e0bbf1d7f15980986f5ea658862004`
- 5090 release raw: `9a8edd6bf4f562e9984b6300a7f866188af52099f0934851ce12f56b89f249d6`
- 원본 hold raw: `8933173570f0fec56174091bf139f2057a6fdbc976bb37396b975241514b93fa`

`check_bundle`과 `authorization(...,'5090')`를 독립 실행해 source131/input7/operation4/local tool40/archive143 member SHA, pointer, receipt, 승인/release/hold 결합 PASS를 확인했다. 5090 고정 GPU는 `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`, 전용 resume runtime_env Python3.12.13/Torch2.11.0+cu130/CUDA13.0/cuDNN92000을 요구한다. 새 host 고유 qualification과 actual exit0가 있어야 fit을 시작할 수 있다.

5080 기존 파일을 다시 SHA 확인했다. 기존 승인 `8029057d…`, release `28706aa2…`, deployment `f90e1136…`, training permit `f72eda31…`가 모두 불변이다. 5080의 lease나 실행을 변경하거나 새 원격 조회하지 않았다. 기존 5080 native 통과를 5090 native 통과로 대신하지 않는다.

## 완료 조건

- [x] 기존 봉인 전체 SHA 및 과학 조건 불변
- [x] 새5090 singleton 승인·release 결합
- [x] 5080 기존 승인/배포/permit 불변
- [x] 5090 health와 실제 native/capacity/qualifier exit0·종료 후 GPU idle
- [x] 5090 최초 scientific fit PID/argv/GPU 및 저장 상태 확인

같은 자료·seed의 네 구조는 동일 GPU/Runtime에서 비교한다. 자료 간5080/5090 Runtime·cuDNN 차이와3+1 시간 차이를 명시하며, 이 대조를 참신성·보편적 필요성·독립 일반화 증명으로 확대하지 않는다.

실제 native 및18:41:08 KST 최초3fit 실행/첫저장전 상태는 NATIVE_QA_5090.md에 별도 기록했다. 전체12개 완료 판정은 아니다.
