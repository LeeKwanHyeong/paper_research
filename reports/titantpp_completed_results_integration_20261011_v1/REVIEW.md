# Instacart·이력 보정 완료 결과의 논문 반영

2026-10-11. 대상 저장소는 paper_research, 현재 브랜치는 `codex/contextual-retrieval-pilot-20261007`이다. 사용자 요청은 “Instacart랑 이력 보정 결과를 논문 초안에 반영하고, CNN+GRU54 중심으로 Results·Methods·주장 정합 작업부터 진행하자.”였다. 현재 초안은 [main.tex](../../paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1/main.tex)이며 원고 문서만 편집했다. 기존 변경은 보존했고 커밋·Push·원격 실행은 수행하지 않았다.

**완료 근거와 기존 원고를 구분한다 — 완료**

- Instacart9와 보정27의 완료 receipt·Validation 선택 기록·최종 감사 SHA를 확인했다. 기존 CNN+GRU54/외부 비교와 신규 full9는 서로 다른 캠페인이다. 기존 표의 숫자를 신규 값으로 교체하지 않았다.
- 편집 전 `main.tex`, `main.pdf`, README, Writing Notes와 작업 순서를 `before/`에 보존했다. [before_manifest.json](before_manifest.json)에 초기 SHA·브랜치·요청 범위가 있다. LNCS class와 안내 파일은 그대로 유지했다.
- 각 fit의 최초 strict finite 최소 전체 raw 수량 RMSE epoch를 따른다. 같은 epoch의 전체/큰 수량 MAE·RMSE·TimeNLL과 3seed 평균·표본 SD를 사용했다. Test/held-out 성과·예측, Validation/Test 혼합 파일은 열지 않았다.

**CNN+GRU54의 Results·Methods·주장을 일치시킨다 — 완료**

- 본문 순서는 기존 대표 구조와 외부 인코더 → 보정27 → Instacart9 → 대기 중 검색이다. 부록 G/H에 같은 여섯 지표와 선택/최종 epoch를 추가했다. 단일 Instacart 모니터링 관측은 완료9 비교로 대체했다.
- 보정 연구는 Taxi·Intermittent에서 모든 3seed가 두 대조보다 strict 전체 RMSE 개선을 보여 사전 Q1 개발 일관성 기준을 충족한다. RAF52는 두 대조보다 나빠 실패한다. 18 paired 비교 중 전체 TimeNLL11개, 큰 수량 RMSE·MAE 각각10개의 손해를 보존했다.
- Instacart의 작은 평균 수량 이득과 seed52 MLP12 우위, 평균 전체 TimeNLL 악화를 함께 적었다. 예산이 다른 구조 비교이며 보정 경로의 인과 효과를 증명하지 않는다.
- Method 수식·Algorithm1은 유지했다. 보정 우회 모듈은 등록되어 있지만 동결·실행0·활성 학습 파라미터0임을 설명했다. full/current-only의 활성25308은 같지만 계산·깊이가 다르고 current hidden에도 prefix 이력이 있다. CNN/GRU별 효과나 특정 lag의 원인을 분리하지 않는다.
- 실제 A10015+RTX12 배정, RAF42 full/current-only의 물리 A100 이관과 seed 간 GPU·Runtime 차이를 공개했다. 병렬 학습 wall time을 구조 효율로 해석하지 않는다.
- Abstract·Introduction·Discussion·Conclusion도 같은 범위로 고쳤다. 검색은 공통 GPU mean·자원 계약 전 대기이고 A100 full을 RTX 동일 GPU mean으로 간주하지 않는다. CPU binary 재추론·Test·새 독립 평가는 미완료다.

**독립 수치 검토와 컴파일·페이지 배치를 확인한다 — 완료**

- [통합 ledger](integration_evidence.json)의 새 mean±sample SD 108개 LaTeX 셀을 원본에 연결했다. 별도 QA가 history54개·Instacart18개 지표 요약을 독립 재계산하고 기존9개 table 블록과 참고문헌28개 보존, split·예외·완료 상태를 대조했다. [QA 결과](manuscript_claims_qa.json)는 PASS이며 미해결 CRITICAL/HIGH/MEDIUM/LOW는0이다.
- 내장 LaTeX compiler와 로컬 TeX Live2026 컴파일 모두 성공했다. 로컬 output-directory는 이 보고서의 `build/`다. 최초 시도는 build 디렉터리 부재로 중단됐고, 디렉터리 생성 뒤 정상 컴파일했다. 최종 정적 PDF는 초안 디렉터리에 내보냈다.
- 최종 PDF33쪽을 PNG로 렌더링하고 모든 페이지의 접촉 시트와 새 핵심 표 페이지를 직접 확인했다. 표·수식·그림의 겹침·잘림, 미정의 인용·참조, overfull box가 없다. 기존 `amsmath`의 사용하지 않는 `vec` 재정의 경고와 그림·Algorithm 경계 페이지5의 비치명적 underfull vbox가 남는다. 해당 페이지의 내용은 정상 표시된다.
- 부록 C의 긴 removal/addition 표현은 같은 의미의 짧은 문장으로 고쳐 가로 넘침을 해결했다. 문법 `a RTX`는 `an RTX`로 수정했다. 숫자와 보호된 Method·표·참고문헌·부록 D–F는 유지했다.
- [검증 기록](validation.json)에 최종 소스·PDF SHA, 페이지 수·보존·참조·렌더 확인을 남겼다. 수치 검산·원본 SHA 검증은 CPU binary 재추론 감사나 새 독립 평가가 아니다. 실제 투고 규정·분량 준수는 판정하지 않았다.

**남은 작업을 의존관계로 재정렬한다 — 완료 / 다음 작업은 검색 포함 여부 결정**

- [PAPER_WORK_ORDER.md](../titantpp_history_rtx_handoff_20261010_v1/PAPER_WORK_ORDER.md)를 현재 완료 상태에 맞춰 갱신했다. 완료 학습과 원본·Validation 판정·이번 원고 반영은 현재 기준선이다.
- 다음은 검색을 논문에 포함할지 결정하는 일이다. 유지한다면 공통 GPU mean과 자원·비교 계약을 준비하고 새 실행 범위의 승인을 받는다. 27/36조건을 자동 실행하지 않는다.
- 이어서 미노출 평가 확보 가능성과 CPU 감사 계약을 각각 고정한다. 실행은 별도 승인 대상이다. 문헌·출처 점검은 독립 문서에서 병렬 준비할 수 있고, 공통 원고 편집은 확정된 범위에 따라 직렬로 진행한다.
- 최종 평가·포함 범위가 정해진 뒤 문헌·부록·재현성·제출 분량을 정리하고 재컴파일한다. 현재33쪽은 저자 검토용 확장 초안이다.

새 학습·추론·SSH·Pod·Runtime·인증·자동화 변경은 모두0이다. 기존 연구 원본을 변경하지 않았다.
