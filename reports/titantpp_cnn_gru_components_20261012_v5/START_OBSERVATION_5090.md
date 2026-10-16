# 5090 Intermittent12 학습 시작 관측

**승인·기존 Runtime·원본과 실제GPU검증을 확인한다 — 완료**

- 최신사용자 “5090도 진행하자” 원문을 기존v5 계약에host[5090]만 결합했다. 전체source131/input7/op4/local40/archive143 SHA와고정GPU UUID·전용Runtime/cuDNN92000·GPU유휴를확인했다. 기존5080의승인·배포·학습permit은원문SHA가동일하다.
- 5090 native 네구조검증·세합성worker·supervised qualification campaign 실제exit0 및종료후GPUidle을확인했다. 합성작업을과학fit이나시간효율근거로집계하지않는다. q rawSHA4e0c035f12f8dd3cc75d8b8ddde349f46375bd8e5709a275a4fdc38986914fdb를유지한다.
- 새5090 최초host168시간permit이며5080기한은연장하지않는다. 기존GPUdriver·인증·공용Runtime·원고·science source변경0이다.

**5090 Intermittent12개 fresh학습을 진행한다 — 진행 중**

- trainphase 시작 2026-10-12 18:39:56 KST, 최초실행관측 2026-10-12 18:41:08 KST.
- Intermittent42 NONE/CNN/GRU 세GPU worker1059784/1059786/1059788, guard1059783/1059785/1059787, dispatcher1059740의정확argv·PID/PPID/pgrp·lease·claim을확인했다. 초기관측은모두첫checkpoint저장전이며GPU/PID존재를완료epoch로표현하지않는다.
- 같은seed 세작업종료→FULL1개→Intermittent52→62의순서다. 총12개중현재완료0/진행3, FULL과뒤seed는대기다. Train/Validation·원래loss/optimizer/B128/min40/max300/patience40/최초strictfinite전체Validation수량RMSE 선택을유지한다. Test·warmstart·resume·자동fitretry0이다.
- 전체큐ETA미확정이다. 초기확인완료후지속polling과새자동화는하지않는다.

**5080과5090 모든종료·원본을확인하고구성요소근거를판정한다 — 다음 작업**

- 5080 Taxi/RAF24와5090 Intermittent12를중복없이36ID로확인한다. 실제worker/dispatcher exit0·마지막phasecomplete·GPU/PID부재·선택/최종binary·manifest/archive/localSHA 이후여섯Validation지표·seed별대조·평균/표본SD를집계한다.
- Runtime/GPU차이와활성파라미터수·3+1부하차이를보존한다.통계유의성·인과시너지·새독립일반화증명으로확대하지않는다.이후CNN+GRU54 Results/Methods/주장정합과필요한최소개선을진행한다.

**초기실행을마지막으로확인하고자연진행한다 — 시작확인 완료 / 학습 진행 중**

- 마지막초기관측 2026-10-12 18:44:38 KST, 실제세GPU/PID·argv·lease는일치한다. 세조건모두첫저장전이며완료epoch/best를추정하지않는다. dispatcher와학습worker는실행중이고12개완료는0이다. 이후지속원격조회는수행하지않는다.
