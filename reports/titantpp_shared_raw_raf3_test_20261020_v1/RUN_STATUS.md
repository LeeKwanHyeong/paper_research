# 원본 RAF 공유형의 Test 평가 후속 상태

**기존 Test 비교군과 평가 도구 검증 — 완료**

- 원본RAF의 MLP16 Test3개와 공통head 외부6종 Test18개를 저장 결과로 검증했다. 동일한 Test5226개 대상이며 기존결과 재평가는없다.
- 새 worker·운영·집계 도구의 최종 합성57개/하위2개 검사가 통과했다. 독립 검토에도 열린 발견이 없다. 실제native평가를 완료했다는 뜻은 아니다.
- 신규코드는 paper_research / codex/contextual-retrieval-pilot-20261007 브랜치의 a6cc0404에 커밋했다. 기존모델·checkpoint·source·결과·실패 기록과공용Runtime은보존했다.

**원본9개 실험의 종료·검증·회수를 기다린다 — 외부 작업 대기**

- 2026-10-20 11:12:49 KST 유한 후속 프로세스PID87103를등록했다. 실제로컬PID실행·활성영수증·실패기록없음을확인했다.
- 5090 기존원본완료프로세스PID71592와5080 Cap완료프로세스PID16494도실행중임을확인했다. 현재학습이나그프로세스는변경하지않았다.
- 원본9개 fit/Validation9 재현/회수/최종비교완료영수증과모든SHA,기존finisher종료전에는추가원격평가를시작하지않는다.
- 현재신규RAF Test는0개이고실제native평가도미시작이다. 반복개선자동화를추가한것이아니라한번종료하는후속대기다.

**고정 checkpoint를 같은5090에서 평가한다 — 다음 작업**

1. 실제유휴·기존GPU/Runtime을확인하고, 별도격리경로에서합성qualification3개를통과한다.
2. Validation선택epoch9·5·19의binary/tensor/metadata를고정해Validation3개의여섯지표·개별예측·실제workerexit0를모두재현한다.
3. 전체gate통과뒤Test3개를최대3병렬로평가한다. Test로checkpoint/구조를다시선택하지않는다.
4. 실제종료·소유PID/GPU부재·원본archive/manifestSHA를검증하고기존MLP16/외부6종저장21개와3seed RMSE·MAE/큰수량/시간지표및모든seed차이를보고한다.

실패·timeout·기존attempt·SHA불일치에는자동재시도하지않으며,원문을새경로에보존한다. 기존절대기한은연장하지않는다.
