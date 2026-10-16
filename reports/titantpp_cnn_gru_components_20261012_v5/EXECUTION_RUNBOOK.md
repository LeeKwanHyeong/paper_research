# 5080 단독 실행 및 5090 대기

**현재 승인과 봉인을 확인한다 — 진행 중**

- 프로젝트 `/Users/igwanhyeong/PycharmProjects/paper_research`, 브랜치 `codex/contextual-retrieval-pilot-20261007`에서 source131/input7·운영파일·로컬도구·배포tar의 SHA와 singleton5080 authorization/release를 확인한다.
- 이전v2/v3 root의 claim/failure/실제exit 원본을 유지한다. v5는 새 격리root이며 원래5080 absolute lease를 계승한다.

**5080의 native·동시 수용 검증 뒤 학습을 시작한다 — 다음 작업**

```sh
/usr/local/bin/python3 -m paper.scripts.launch_titantpp_cnn_gru_components_v5 --bundle search_artifacts/titantpp_cnn_gru_components_20261012_v5 --mode deploy --host 5080
/usr/local/bin/python3 -m paper.scripts.launch_titantpp_cnn_gru_components_v5 --bundle search_artifacts/titantpp_cnn_gru_components_20261012_v5 --mode qualify --host 5080
/usr/local/bin/python3 -m paper.scripts.launch_titantpp_cnn_gru_components_v5 --bundle search_artifacts/titantpp_cnn_gru_components_20261012_v5 --mode status --host 5080
/usr/local/bin/python3 -m paper.scripts.launch_titantpp_cnn_gru_components_v5 --bundle search_artifacts/titantpp_cnn_gru_components_20261012_v5 --mode train --host 5080
```

- 명령은 단계별 의존관계가 있다. health busy이면 후속0; native 실제 자식·세 합성worker·campaign exit0 및 종료후GPU idle 검증 cache가 있어야 train gate가 열린다. 단순 tmux 시작이나 transport exit0는 실제native PASS가 아니다.
- 실제초기PID·argv·GPU·checkpoint를 확인하고 지속polling은 하지 않는다. fitfailure/claim을 지우거나 resume·자동retry하지 않는다. 전체큐ETA는 미확정이다.

**5090도 승인된12개큐를 진행한다 — 진행 중**

- 별도 명시적 사용자 지시 전 authorization/release/SSH·deploy·qualification·train 모두0이다. 지시가 오면 그host만 승인 원문과 봉인에 결합하고 건강·유휴·기존Runtime·ownnative gate를 거친다. 현재5080 permit을5090 승인으로 사용하지 않는다.

**끝난 실험의 원본과 Validation을 판정한다 — 다음 작업**

- host별24/12 ID·실제worker/dispatcher exit0·마지막phasecomplete·ownerGPU/PID부재 및 모든binary원본/archive/manifest SHA를 확인한 뒤 전체36개를 집계한다. 별도Test·검색·새loss/head개선은 이 실행에 포함되지 않는다.

2026-10-12 최신5090시작승인과실제native/초기3fit확인은 [5090 관측](START_OBSERVATION_5090.md)을참조한다. 기존조건부승인문구는미승인host의일반gate이며현재5090은별도승인됨이다.
