# 수정 전 원문 기준

**완료판정 오류를 저장된 증적으로 재현한다 — 완료**

- 현재 저장소 브랜치: `codex/contextual-retrieval-pilot-20261007`. 기존 사용자 변경과 staged draft.tex는 보존한다.
- 원래 v5 `check_bundle --mode check --host 5080`은 source131·input7·operation4·local40·payload·current·계약 대조에 통과했다. 실제 remote_calls=0이다.
- 원 v5 launcher focused tests18개는 통과했다. 존재하지 않는 hourly observer 테스트 모듈을 함께 지정했던 첫 테스트 명령은 import 오류로 종료했으며, 해당 모듈이 없음을 확인한 뒤 실제 launcher 테스트18개만 실행해 통과했다. 제품 오류나 새학습 실패로 해석하지 않는다.
- 원래 `training_status`를 원문 5080 snapshot에 적용하면 `Terminal completion or GPU idle unconfirmed` 오류가 재현된다. 이 snapshot은 실제24 terminal manifests의 모든 파일 SHA가 검증돼 있다. 원격 호출0으로 재현했다.

| 대상 | rawSHA256 |
|---|---|
| 기존 과학 계약 canonical | e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074 |
| source131 closure | 15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53 |
| operation4 closure | 12c836dc5fe1d9146b83a352ba77abc297f4d69c6f68d83ad00935a2bc442903 |
| 기존 local40 integrity | 16e317c3c16dddede95c6aa6174026dca0ab772879bf438c3d1e6b752eff7074 |
| 기존 monitor integrity | a2b1095b7cbdd9fb0a65442378b5e7817b4dbfe448c3667d3506fca0efee35ec |
| 08:27 원본 round36bd73… | 88a650f7f474bead0c02809328ca3e78a83a7c6623770fb0ea544ccf7d15820a |
| 원본 hourly latest | b16917bd3fd681d575f859e89cc8eab2bdbfbd4492d19bacadffcacc20d23a30 |
| 5080 snapshot1ecf… | 352b65c7f5c55a11fae7720b67c26c437b6999f662cfa0b8297fe0b2c33556f2 |
| 5090 snapshotf3d… | 71dbc280f6581c8b193a6799cac04b8420905d34987312885cde677377ea828a |

- 전체고유36 ID는 완료27·진행1·대기8이다. 이 시각의 원격 조회를 새 조회처럼 기록하지 않는다.
- 5080 actual supervised dispatcher exit0 원시각은 2026-10-13 06:50:48.901582 KST, 마지막 phase complete는 .925819 KST다. pipeline wrapper 자체 OS exit는 직접 관측하지 않았다.
- local immutable 원본 회수·최종Validation 집계·원고변경·Test·검색은 이번 baseline 검증에서 수행하지 않았다.
