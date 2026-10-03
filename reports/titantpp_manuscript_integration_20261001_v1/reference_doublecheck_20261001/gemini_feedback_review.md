# Gemini 참고문헌 피드백 재검토

검토일: 2026-10-01 KST. 대상은 현재 `references.json`과 원고의 32개 참고문헌 중 Gemini가 지적한 출판본 링크·학회명 서식이다. 원고와 서지 원본은 이번 검토에서 수정하지 않았다.

## 판단

| 제안 | 판단 | 현재 적용할 내용 |
|---|---|---|
| 정식 출판물의 링크를 우선한다 | 타당한 편집 개선 | 아래 ICLR 5편의 공식 링크를 제출용 대표 링크로 사용할 수 있다. arXiv 링크 자체가 오인용이라는 뜻은 아니다. |
| ResNet을 CVF/DOI로 바꾼다 | 이미 반영 | 현재 registry는 CVF 공식 페이지와 DOI 10.1109/CVPR.2016.90을 사용한다. |
| S2P2 공식 proceedings가 아직 없을 수 있으니 arXiv v2를 유지한다 | 현재 사실과 불일치 | NeurIPS 2025 정식 페이지가 공개되어 있고 현재 원고에도 이미 반영했다. |
| ICML의 회차와 Proceedings of를 없애야 서식이 일관된다 | 의무로 해석하면 부정확 | 해당 문구는 PMLR의 공식 booktitle이다. 출판명은 보존하고 제출 템플릿의 출력 규칙을 적용한다. |
| 32편이 모든 면에서 100% 완벽하다 | 검증 범위를 넘는 단정 | 존재·서지·링크 대조와 문헌의 포괄성, 본문 모든 주장의 타당성, 제출 양식 완성을 구분한다. |

## 공식 링크 후보와 확인 범위

| 번호 | 문헌 | 공식 OpenReview 링크 | 확인 근거 |
|---|---|---|---|
| 10 | AttNHP | https://openreview.net/forum?id=Rty5g9imm7H | [ICLR 2022 공식 페이지](https://iclr.cc/virtual/2022/poster/6600)의 OpenReview 연결 확인 |
| 14 | Intensity-Free | https://openreview.net/forum?id=HygOjhEYDH | [저자 공식 구현](https://github.com/shchur/ifl-tpp)의 링크 및 [ICLR 2020](https://iclr.cc/virtual/2020/poster/1564) 발표 확인 |
| 19 | PatchTST | https://openreview.net/forum?id=Jbdc0vTOcol | [ICLR 2023 공식 페이지](https://iclr.cc/virtual/2023/poster/10876) 및 저자 arXiv 출판 정보 확인 |
| 28 | AdamW | https://openreview.net/forum?id=Bkg6RiCqY7 | [저자 소속 연구실의 출판 목록](https://ml.informatik.uni-freiburg.de/profile/loshchilov/)과 [ICLR 2019](https://iclr.cc/virtual/2019/poster/935) 확인 |
| 29 | EasyTPP | https://openreview.net/forum?id=PJwAkg0z7h | OpenReview 공식 PDF의 검색 색인 및 [ICLR 2024 공식 페이지](https://iclr.cc/virtual/2024/poster/18712) 확인 |

OpenReview 직접 열람은 브라우저 확인/403으로 제한됐다. 이를 링크 오류나 미채택으로 판정하지 않았다. 직접 PDF 전체를 읽은 것으로 표시하지 않으며, 학회 페이지·저자 자료·검색 색인으로 확인한 범위를 구분한다. 링크 교체는 아직 수행하지 않았다.

S2P2 현재 공식 출판본: https://proceedings.neurips.cc/paper_files/paper/2025/hash/ee39348acc798915d2d15a8bbbd417b8-Abstract-Conference.html

## AttNHP 저자 순서의 플랫폼 차이

ICLR 행사 페이지에는 Hongyuan Mei, Chenghao Yang, Jason Eisner 순서로 표시된다. 반면 [저자가 제공한 최종 원고](https://www.cs.jhu.edu/~jason/papers/yang%2Bal.iclr22.pdf) 첫 페이지와 [저자 출판 목록](https://www.cs.jhu.edu/~jason/papers/)에는 Chenghao Yang, Hongyuan Mei, Jason Eisner 순서가 명시되어 있다. 현재 원고는 저자 원고와 일치한다. 행사 메타데이터만 보고 저자 순서를 바꾸지 않는다. OpenReview의 최종 PDF 자체는 이번 요청에서 직접 읽지 못했다.

이는 이전 검토의 '확인한 저자 원고와 일치' 판단을 뒤집지 않지만, 모든 플랫폼의 모든 필드가 완전히 동일하다는 Gemini의 표현은 정확하지 않음을 보여준다.

## LNCS 서식의 실제 기준

[Springer 공식 저자 안내](https://link.springer.com/series/558/information-for-authors-and-editors)에서 연결된 [현재 Instructions for Authors PDF](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/27852130/data/Instructions%20for%20Authors%20PDF)의 §4.9는 번호 인용, 본문과 목록의 상호 일치, DOI 포함을 설명한다. 참고문헌 예시에는 `10th IEEE International Symposium ...`처럼 회차와 풀네임을 쓴 항목과 `Euro-Par 2006`처럼 약칭을 쓴 항목이 함께 있다. 따라서 학회명을 전부 약어로 바꾸거나 회차를 일괄 삭제하는 것을 필수 규칙으로 해석하지 않는다.

[THP의 공식 PMLR BibTeX](https://proceedings.mlr.press/v119/zuo20a.html)는 `Proceedings of the 37th International Conference on Machine Learning`을 booktitle로 제공한다. 현재 표기는 잘못된 학회명이 아니다.

실제 통일 대상은 저자 성·이니셜 출력, 구두점, 제목 대소문자 보호, 권·호·페이지, DOI/URL 출력 방식과 인용 번호다. 최종 제출 시 대상 연도 PAKDD가 배포한 템플릿과 지침을 우선한다. 이번 검토에서는 특정 PAKDD 연도의 제출 규정을 별도로 확정하지 않았다.

## 남은 작업

**출판본 대표 링크를 정리한다 — 다음 작업**

- ICLR 5편의 URL을 공식 링크로 정리하고 저자 순서·학회 연도는 보존한다. 원고·BibTeX·registry의 동일성을 확인한다.

**최종 제출 양식을 적용한다 — 이후 작업**

- 해당 연도 PAKDD 템플릿을 기준으로 참고문헌을 출력한다. 공식 출판명을 일괄 축약하거나 서지정보를 임의 삭제하지 않는다.

이번 작업은 문헌·서식 검토이며 실험, GPU 조회, 일정 변경, 외부 게시를 수행하지 않았다.
