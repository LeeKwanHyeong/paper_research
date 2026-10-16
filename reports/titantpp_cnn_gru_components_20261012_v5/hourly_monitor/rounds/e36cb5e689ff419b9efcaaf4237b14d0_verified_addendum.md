# CNN·GRU36 시간별 관측 보완 보고

**저장 checkpoint와 실제 종료를 확인한다 — 진행 중**

실제 관측: 2026-10-12 22:13:01 KST. 전체36 중 종료4·진행6·대기26.

| GPU | 완료 / 전체 | 진행 | 대기 |
|---|---:|---:|---:|
| 5080 | 4 / 24 | 3 | 17 |
| 5090 | 0 / 12 | 3 | 9 |

| GPU | dataset·seed·arm | 상태 | 저장 / 이력 best | best 미갱신·속도 유지 | 300epoch·속도 유지 |
|---|---|---|---:|---:|---:|
| 5080 | Taxi·42·보정 없음 | 종료 | 87 / 47 | 종료 | — |
| 5080 | Taxi·42·CNN만 | 종료 | 101 / 61 | 종료 | — |
| 5080 | Taxi·42·GRU만 | 종료 | 85 / 45 | 종료 | — |
| 5080 | Taxi·42·CNN+GRU | 종료 | 86 / 46 | 종료 | — |
| 5080 | Taxi·52·보정 없음 | 진행 중 | 72 / 68 | 25분 | 2시간37분 |
| 5080 | Taxi·52·CNN만 | 진행 중 | 71 / 37 | 4분 | 2시간39분 |
| 5080 | Taxi·52·GRU만 | 진행 중 | 35 / 29 | 47분 | 6시간7분 |
| 5090 | Intermittent·42·보정 없음 | 진행 중 | 42 / 30 | 2시간19분 | 21시간17분 |
| 5090 | Intermittent·42·CNN만 | 진행 중 | 42 / 28 | 2시간10분 | 21시간32분 |
| 5090 | Intermittent·42·GRU만 | 진행 중 | 20 / 14 | 5시간57분 | 48시간57분 † |

- 5080Taxi42네구조는실제exit0·종료manifest각13파일원격SHA(선택/최종binary·endpoint replay포함)로검증됐다. FULL원관측종료21:23:18KST/저장86/best46이며 현재Taxi52세대조가진행중이다. 기존세대조원관측종료시각도새시각으로바꾸지않는다.
- 진행6개ETA는각fit자신의최근10완료epoch finite양수elapsed_seconds 중앙값으로계산했다. best는최초strictfinite 최소 fullValidation전체raw수량RMSE epoch다. 진행fit의최종selected binary는아직생성전이다. 조기종료목표min(300,max(40,best+40)), best미갱신·속도유지조건이며 부하/속도/best갱신시변한다. 전체36큐ETA미확정이다.
- †5090 GRU만의300epoch대안48시간57분는남은조건기한32시간27분을넘는다. 현기한내300epoch를보장하지않으며 실제timeout나기한연장승인으로해석하지않는다.
- 기존round의완료0·종료fit저장전은보고집계오류가남은표시다. 같은raw snapshot의종료증적으로별도보완했으며원본round/봉인/코드/학습/자동화설정은변경하지않았다.
- helper1회·livehost별읽기전용SSH1회·추가SSH0·Test0·원본회수0. endpoint성과수치열람0. pipeline wrapper자체OSexit는직접관측하지않았다. 원격완료와localimmutable원본회수·최종Validation집계는별도이며자동화유지다.

**종료보고를보완하고원본과논문근거를확정한다 — 다음 작업**

- terminal증적을반영하는보고기수정·검증이필요하다. 시간별관측→36개자연종료확인→원본회수범위/계약확인→archive/manifest·선택/최종binary 로컬SHA확정→여섯Validation지표·모든seed/대조평균/표본SD→Results·Methods·주장정합순서다.

원본회차: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor/rounds/e36cb5e689ff419b9efcaaf4237b14d0.json
rawSHA: 9a38ac5d2b6e3c438a42e23c36ba4e8eef6621b8349bd974e3a99ed6db73b4f4
