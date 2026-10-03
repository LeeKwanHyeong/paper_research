# Deep Renewal 원고 비교 범위 변경

사용자 확인: “원고의 비교에서만 제외하고 학습은 유지”.

- 현재 LNCS main.tex의 Deep Renewal 비교 설정, Appendix A 표 행과 결과 해석을 제외했다.
- Appendix A는 기존 13조건 감사 중 구조 대조 10조건(Taxi 6, Intermittent seed42 2, Instacart seed42 2)으로 범위를 맞췄다. 표는 2026-10-02 10:51:58 KST 감사 기준이며 최신 학습 상태를 뜻하지 않는다.
- 관련 연구 및 참고문헌 인용은 유지했다. 주 비교의 주장 범위는 공통 head를 붙인 여섯 TPP encoder이며, native 수요 예측 모델 전체에 대한 우월성을 뜻하지 않는다.
- 제외 이유는 사용자의 원고 범위 결정이다. Deep Renewal 실험의 무효성이나 TitanTPP의 우월성이 새로 입증된 것으로 해석하지 않는다.
- 원래 13조건 감사·모든 실험 원본·학습·실행 계약·스케줄러는 변경하지 않았다.
- 같은 main.tex에 대해 Codex 내장 컴파일 성공을 확인했다. 변경 전 파일과 changes.diff, SHA receipt를 이 폴더에 보존한다.

다음 작업: 새로 완료된 후속 조건의 원본 회수·검증 후, 원고 범위에 포함되는 구조 대조 결과를 갱신한다.
