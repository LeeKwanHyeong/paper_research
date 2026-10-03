# TitanTPP 참고문헌·관련 연구 보강 — 2026-10-01

**원고의 연구 위치와 인용 근거를 연결한다 — 완료**

- 사용자 요청 “그러면 그것도 좀 업데이트를 해볼까? 지금 하는게 낫겠지?”에 따라 현재 원고의 참고문헌을 9편에서 32편으로 확장했다. 학회 인용 개수 규정을 가정하지 않았으며, 본문에 실제 역할이 있는 문헌만 포함했다.
- [원고](../../paper/titantpp_history_mlp_manuscript_20261001_v1.md), [References JSON](references.json), [BibTeX](references.bib)를 같은 목록으로 정리했다. 참고문헌은 본문 최초 인용 순서이며 원고의 인용은 기존 Markdown 저자–연도 링크 형식을 유지한다.
- [수정 전 원고·메타데이터](before_reference_expansion/), [문헌별 확인 범위와 본문 위치](reference_additions.json), [편집 재현 코드](expand_references.py), [수정 기록](reference_expansion_changes.json)을 보존한다.

## 추가 23편의 역할

| 영역 | 추가 수 | 본문에 연결한 내용 |
|---|---:|---|
| 사건·시간 모델링 | 5 | Hawkes, neural TPP review, fully neural integrated intensity, intensity-free TPP, HiPPO |
| 수요·시계열 예측 | 6 | Croston, Syntetos–Boylan, TSB, DeepAR, PatchTST, iTransformer |
| 잔차·메모리 구조 | 5 | ResNet, bottleneck adapters, ReZero, end-to-end memory networks, Transformer attention |
| 실제 구현의 출처 | 4 | Layer normalization, GELU, AdamW, PyTorch |
| 평가·재현성 | 3 | EasyTPP, scale-dependent forecast errors, proper scoring rules |

## 본문 변경과 해석

| 위치 | 개정 내용 | 근거와 해석의 구분 |
|---|---|---|
| 서론 | 사건의 크기와 발생을 구분하는 동기에 Croston을 연결 | 수요 표현의 선행 연구. 재고 비용 개선을 실험했다는 주장은 추가하지 않음 |
| 2.1 | history encoder와 시간 분포를 구분하고 recurrent·attention·state-space·direct-distribution 연구를 연결 | 앞선 연구의 분류와 자체 모델의 위치를 설명. 다른 논문의 성능을 로컬 실험 수치로 사용하지 않음 |
| 2.2 | 수요 예측 계보 및 일반 forecasting과 현재 next-event 목표의 관계를 설명 | 수치형 mark의 최초 도입 주장 없음. 관련 연구 인용은 새 실험 완료를 뜻하지 않음 |
| 2.3 | 병목 잔차·영 초기화·메모리 읽기의 선행 연구와 TitanTPP의 특수화를 설명 | 인접 contextual state·encoder 사이 보완·가용성 규칙·공동 학습이 설계 초점. 기존 연구의 수렴 보장을 차용하지 않음 |
| 3.2·3.3·3.6 | attention·LayerNorm·GELU·AdamW 출처를 실제 연산 설명에 부착 | 동결 수식과 optimizer 설정은 그대로 유지 |
| 4.2·4.3 | 비교 인터페이스, raw quantity 오차의 척도 의존성과 시간 log score의 의미 | EasyTPP가 로컬 계약을 정의했다는 주장 없음. RMSE 선택은 본 실험의 고정 기준이며 선행 논문의 보편적 권고로 쓰지 않음 |
| 6.1 | 실제 사용한 PyTorch의 소프트웨어 인용 | 버전·시간·메모리는 기존 로컬 계측이 근거이며 2019 논문에서 가져온 수치가 아님 |

관련 연구 본문은 기존 약 403단어에서 약 700단어로 확장했다(Markdown 공백 기준, 인용 링크 포함). 새 성능 주장·수식·실험 조건·결과 행은 추가하지 않았다. 각 단락은 선행 연구의 기능을 설명한 뒤 현재 설계와 연결한다. 반복적인 면책 문장을 늘리는 대신 기존 7절의 한계를 유지했다.

## 서지 확인의 범위

추가 항목은 공식 학회 proceedings, 출판사 원문 페이지, 저자 arXiv 원고 또는 저자 소속기관의 원문으로 확인했다. [문헌 기록](reference_additions.json)의 `verification_extent`는 초록·메타데이터 확인과 본문 특정 절 확인을 구분한다. 모든 논문을 처음부터 끝까지 정독한 포괄적 신규성 조사로 보고하지 않는다.

- Houlsby의 2.1절에서 bottleneck/skip/초기화 구조를 확인했다. ReZero는 UAI 2021, intensity-free TPP는 ICLR 2020, EasyTPP는 ICLR 2024, AdamW는 ICLR 2019로 출판연도를 맞췄다.
- LayerNorm과 GELU는 arXiv 문헌으로 표시했다. 확인되지 않은 학회·DOI·쪽수는 채우지 않았다.
- Hyndman–Koehler의 척도 의존 오차 설명과 Gneiting–Raftery의 logarithmic score 설명을 확인했다. 전자는 RMSE만을 선택 기준으로 삼으라는 근거가 아니며, 후자는 수량 점 회귀를 확률밀도로 바꾸는 근거가 아니다.
- PatchTST OpenReview의 browser challenge와 ResNet CVF 직접 조회의 403은 우회하지 않고 공개된 저자 원고와 확인 가능한 공식 서지 기록을 이용했다.
- 새 데이터셋 원출처는 추정하여 붙이지 않았다. 기존 Intermittent 원출처·재배포 확인 과제는 유지한다.

## 남은 작업 순서

**추가 비교군 결과를 확정한다 — 외부 작업 대기 / 다음 확인**
- 저장된 08:59 KST 관측 이후의 5090 완료 여부를 별도 요청 범위에서 확인하고, 원본 회수·CPU 감사 후 두 Pending 그룹을 확정한다. 이번 편집에서는 서버 상태를 새로 관측하지 않았다.

**데이터 출처와 최종 논지·분량을 정리한다 — 다음 작업**
- Intermittent를 포함한 데이터 원출처와 재배포 근거를 확인한다. 완성된 비교표에 맞춰 초록·결론과 문헌별 필요성을 재검토한다.
- 최종 LNCS 조판 때 저자–연도 링크를 문서의 정식 인용 명령과 번호로 변환하고 참고문헌·그림의 페이지 배치를 점검한다. 본 단계는 Markdown/BibTeX 편집으로, 조판 컴파일을 완료한 것이 아니다.

**독립 최종 평가의 범위를 확정한다 — 승인 필요**
- validation 개발 결과와 구분할 평가 split·모델·selector·통계 절차를 합의한 뒤 실행 여부를 정한다. 이번 인용 추가는 held-out 접근이나 GPU 실행을 포함하지 않는다.
