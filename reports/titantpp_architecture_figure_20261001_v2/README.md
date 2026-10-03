# TitanTPP 논문용 구조 그림 v2

**그림 작성 — 완료**

- (a) 전체 인과 encoder·보완·정적 검색·두 head 경로, (b) History correction의 입력 쌍·독립8분기·128→4→64·GELU·가용성·고정/8·잔차를 확대했다.
- 본문 Figure 1과 캡션에 반영했다. 기존 방법 보고서의 그림과 원고 수정 전 사본은 보존한다.
- [PDF](architecture.pdf), [SVG](architecture.svg), [PNG](architecture.png), [작성 코드](draw.py), [PDF 렌더 미리보기](pdf_preview.png)를 제공한다.
- 흰 배경, 검은 문자·중립색 encoder·녹색 보완·연한 주황 head를 사용하며 그라데이션·원근 큐브는 쓰지 않는다. 그림의 모든 상태·연산은 기존 계산식에 대응한다.
- 도형 본체는 벡터이며 PDF는1페이지, raster image object0개다. SVG는 편집 가능한 텍스트를 유지하고 PDF는 폰트를 포함한다. PNG는400dpi다.
- 제안 배치 폭122mm에서 기본 라벨은 약7pt이며 첨자와수식은 더 작다. 최종 템플릿 적용 시 본문에서 가독성을 다시 확인한다. 현재의 대조 가능한 PNG와PDF 미리보기를 육안 점검했다.

**정확성 확인 — 완료**

- [검증](verification.json): 원본 그림 파일 보존, 벡터 PDF, 텍스트 페이지 경계, 소스·산출물 SHA를 확인한다. 모델 동작은 기존 수식–동결 코드 검증을 재사용한다.
- 모든 분기는 동일 직전 관측을 읽는다. c_i는 withheld valid row 이후 관측 수이며 padding과마스크의 상세 정의는본문식3에 있다.
- 출력 V_b 영 초기화와 predecessor 없는 위치의영 보완, target 차단, 추론 중파라미터 고정은캡션에 유지했다.
- 신규 학습·GPU 계측·checkpoint 실행·원격조회는 없다.

**남은 작업 — 다음 작업**

- 참고문헌은별도 [범위 검토](../titantpp_manuscript_integration_20261001_v1/reference_coverage_review.md)에 따라관련연구 본문에 연결한다.
- 기존5090 실험 종결 감사 이후 최종 비교표와 초록·결론을 확정한다. 본그림제작으로 실험상태를 새관측한 것은 아니다.
