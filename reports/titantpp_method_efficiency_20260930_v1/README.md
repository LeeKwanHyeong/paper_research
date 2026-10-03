# TitanTPP 방법 절과 Titans-MAC 효율 근거

2026-09-30 · paper_research · `titantpp_history_mlp` · validation-only 기존 결과

**방법 절·기호 정의·구조 그림·코드 대응표 및 기존 비용 감사 작성을 완료했다.** 동결 구현의 입력·인과성·8분기 MLP·정적 검색·두 head·손실·선택 규칙을 수식으로 고정했다. 작은 CPU 텐서 검증에서 명시적 MLP 수식과 실제 구현의 최대 차이는 `5.204170427930421e-18`이었다. 방법 관련8개 source SHA와 원본97개 manifest closure를 확인했다. 새로운 GPU 작업은 수행하지 않았다.

Notion: [방법 절·설계 근거·MAC 효율 감사](https://app.notion.com/p/3ebbbe40561381e2be20fbeb63fd7329). 기존 설계안에도 완료 상태와 후속 측정 범위를 반영하고 재조회로 확인했다.

## 산출물

- [영문 방법 절·기호 정의표](method_draft.md)
- [구조 그림 SVG](architecture.svg) · [PNG](architecture.png) · [벡터 PDF](architecture.pdf) · [편집용 Mermaid](architecture.mmd)
- [코드와 수식의 대응표](code_equation_map.md)
- [설계 목적과 실험 근거·claim–evidence map](design_evidence.md)
- [MAC 효율 근거와 최소 후속 측정 판단](efficiency_audit.md)
- [현재 MLP 비용9조건](current_mlp_costs.csv) · [과거 MAC 비용](historical_costs.csv) · [B/Full 대비 효과](design_effects.csv)
- [검증 증적](verification.json) · [기계 판독 비용 판단](efficiency_audit.json) · [출처 SHA](sources.json)

## 핵심 판단

방법의 표현은 **온라인 신경 메모리 갱신을 생략하고, causal encoder 사이에서 인접 이력의 저차원 잔차를 계산하는 TitanTPP**로 정리했다. Persistent bank와 정적 검색은 유지된다. 전체 Memory를 제거했다는 표현은 사용하지 않는다.

과거 MAC/B0 시간비는 Taxi6.86배·Intermittent5.45배다. 모델·head·장치·timer 조건이 현재 MLP와 달라 이를 현재 TitanTPP의 가속 배수로 쓰지 않는다. 현재 MLP 기록은 Taxi13.015±0.011초, Intermittent123.096±1.186초/완료 epoch이나 train-only 직접 계측이 아닌 실행 구간 상각값이다. 과거 MAC의 peak allocated가 더 낮았던 기록도 함께 보존했다.

## 남은 순서

**방법과 증거를 원고에 고정한다 — 완료**
- 대상: 위 Method 초안, 그림, 설계 근거 및 Notion 설계안.
- 최종 MLP와 비교 구성의 이름을 분리하고, causal masking·동일 초기화·수량 MAE/RMSE 및 시간 NLL 해석을 명시했다.

**최소 효율 측정 범위를 선택한다 — 다음 작업 / 실제 GPU 실행은 승인 필요**
- 대상: 개인5080, MLP와 MAC 두 모델, Taxi·Intermittent train 입력.
- 권장안은 총 GPU 점유30분 상한의 짧은 동일 부하 profile이다. 범위·반복·실패 처리·측정 경계는 [비용 감사 §5](efficiency_audit.md)에 구체화했다. 현재 승인된 실행 계약은 아니다.
- 실제 초/epoch가 필요하면 profile 이후 전량 epoch 계측의 시간을 산정하고 별도 승인한다. 장기 MAC9조건을 속도 측정의 기본값으로 두지 않는다.

**최종 증거와 독립 평가를 닫는다 — 이후 작업**
- 대상: 5090 최종 binary/source 회수 감사 잔여분, 논문 결과표, 관련 연구와 독립 평가 설계.
- 이번 문서 작업으로 최종 binary 감사나 held-out 평가가 완료된 것은 아니다. 신규 학습·held-out은 각각 승인 범위를 확정한 뒤 진행한다.

공통 계산식·계약·증거에 의존하므로 이번 작업은 현재 세션에서 직렬로 진행했다. 기존 과학적 소스·결과를 수정하지 않았으며 스케줄러·학습·재평가·커밋·Push는 시작하지 않았다.
