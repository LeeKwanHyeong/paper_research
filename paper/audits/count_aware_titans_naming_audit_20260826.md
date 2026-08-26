# Count-aware Titans 명칭 감사

- 감사일: 2026-08-26
- 대상: `models/Titan/common/memory.py`, Count-aware T0 계약, 향후 B1/B2

## 판정

기존 `LMM` class는 원본 Titans Long-term Memory Module이 아니라 static hard
prototype retrieval이다. 이름 중복으로 인해 현재 TitanTPP-T0가 원본 Titans의
test-time neural memory를 구현한 것처럼 읽힐 위험이 있었다.

## 조치

- 정식 코드 이름을 `HardLocalMemoryMatcher`로 분리했다.
- `LMM` alias와 `static_hard_lmm` metadata는 기존 checkpoint·artifact 호환용으로만
  유지한다.
- B0를 `Current Hard-LMM`, B1을 `Faithful Titans-MAC`, B2를
  `TPP-specific Gated Memory`로 고정했다.
- 원본 `Long-term Memory Module` 명칭은 surprise gradient, momentum, adaptive
  forgetting 및 test-time online update를 모두 구현한 B1에만 사용한다.

과거 결과 파일은 실험 당시 증적이므로 일괄 수정하지 않는다. 향후 표와 본문에서는
B0를 `Hard Local Memory Matcher`로 풀어 쓰고 원본 Titans LTM과 명시적으로 구분한다.
