"""Apply the manually reviewed primary-source bibliography audit (no network/model work)."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
OUT = AUDIT.parent
MAIN = ROOT / 'paper/titantpp_history_mlp_manuscript_20261001_v1.md'
stamp = datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before_sha = json.loads((AUDIT / 'before_sha256.json').read_text())
for relative in ['paper/' + MAIN.name, str((OUT / 'references.json').relative_to(ROOT)),
                 str((OUT / 'references.bib').relative_to(ROOT))]:
    assert sha(ROOT / relative) == before_sha[relative], 'Already applied or intervening edit: ' + relative

registry = json.loads((OUT / 'references.json').read_text())
records = registry['records']
by_key = {r['key']: r for r in records}
old_text = MAIN.read_text()
old_records = json.loads(json.dumps(records))

s2_url = 'https://proceedings.neurips.cc/paper_files/paper/2025/hash/ee39348acc798915d2d15a8bbbd417b8-Abstract-Conference.html'
cvf_url = 'https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html'
crossref_url = lambda doi: 'https://api.crossref.org/works/' + doi

patches = {
    'croston1972forecasting': {'number': '3'},
    'behrouz2025titans': {'doi': '10.52202/085713-3786', 'pages': '125925--125962'},
    'du2016rmtpp': {
        'venue': 'Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining',
        'pages': '1555--1564'},
    'chang2025s2p2': {'venue': 'Advances in Neural Information Processing Systems 38',
        'url': s2_url, 'doi': '10.52202/085713-5431', 'pages': '180555--180596'},
    'draxler2025flextpp': {'pages': '127441--127473'},
    'he2016residual': {'url': cvf_url, 'pages': '770--778', 'doi': '10.1109/CVPR.2016.90'},
    'houlsby2019adapters': {'bibtex_author':
        'Houlsby, Neil and Giurgiu, Andrei and Jastrzebski, Stanislaw and Morrone, Bruna and De Laroussilhe, Quentin and Gesmundo, Andrea and Attariyan, Mona and Gelly, Sylvain'},
    'hyndman2006accuracy': {'url': 'https://doi.org/10.1016/j.ijforecast.2006.03.001'},
}
notes = {
    1: ('complete_issue', '출판사와 DOI 등록명인 JORS는 일치한다. 발행 당시 명칭 Operational Research Quarterly와 구별하고, 기존 표기를 오인용으로 판정하지 않는다. 누락된 3호만 보완한다.'),
    2: ('match', '저자 4명·순서·2021년·16(11)·e0259764·DOI가 일치한다. 연속시간 수요 도착과 신경 renewal 모형의 선행연구로 적절하다.'),
    3: ('complete_doi_pages', 'NeurIPS 2025 / 38권과 저자 3명이 맞다. 공식 proceedings DOI와 등록 페이지를 보완한다. 논문은 온라인 메모리 문맥의 근거이고 현지 속도 배수의 근거는 실측이다.'),
    4: ('match', 'OUP의 1971년 58(1):83–90 및 DOI가 일치한다. 1970년 접수일을 출판 연도로 바꾸지 않는다.'),
    5: ('match', '공식 IJCAI 2021의 저자 4명·4585–4593·DOI가 모두 일치한다.'),
    6: ('complete_venue_pages', '공식 PDF의 제목·부제·저자와 ACM DOI 등록이 일치한다. KDD 22회 및 정식 페이지 1555–1564를 보완한다. rpp1081이라는 배포 파일명을 페이지로 해석하지 않는다.'),
    7: ('match_name_variant', '2017년 30권·논문 제목·저자 순서가 일치한다. 공식 웹 메타데이터의 Jason M Eisner와 논문/기존 목록의 Jason Eisner는 동일 저자 표기 변형이다.'),
    8: ('match', 'PMLR 공식 BibTeX의 저자 5명·ICML 2020·119권·11692–11702와 일치한다.'),
    9: ('match_name_variant', 'PMLR 공식 BibTeX의 저자 4명·ICML 2020·119권·11183–11193과 일치한다. Omer Kirnap은 공식 등록의 ASCII 표기다.'),
    10: ('match_year_basis', '저자 3명·제목이 일치하며 저자 최종 원고에 ICLR 2022 Final이 명시되어 있다. 2021년 최초 arXiv 공개와 구분한다.'),
    11: ('match', '공식 NeurIPS의 저자 5명·2020년 33권과 일치한다. 다항식 투영으로 이력을 압축한다는 본문 설명이 초록과 맞다.'),
    12: ('complete_publication_record', 'NeurIPS 2025 정식 proceedings로 연결하고 38권·DOI·페이지를 보완한다. Cao Xiao/Cao (Danica) Xiao 및 Andrew Warrington/andrew warrington은 웹·저자 원고 간 표기 변형이다. 저자 원고 이름을 유지한다.'),
    13: ('match', '공식 NeurIPS 2019 / 32권과 일치한다. 적분 강도를 신경망으로 표현하고 미분한다는 설명이 초록과 일치한다.'),
    14: ('match_year_basis', '저자 3명 및 ICLR 2020이 맞다. 최초 arXiv 공개는 2019년이다. inter-event-time 분포를 직접 모형화한다는 설명에 적절하다.'),
    15: ('match', '출판사 검색 색인의 저자·2005년 21(2):303–314·DOI를 확인했다. 이 논문은 간헐수요 추정법 비교 근거이며 모든 상황에서 RMSE 선택을 권고하는 문헌으로 쓰지 않는다.'),
    16: ('match', '출판사 본문 색인과 DOI 등록에서 세 저자·2011년 214(3):606–615를 확인했다. 무수요 기간의 발생확률 갱신과 재고 노후화 맥락이 본문 설명과 맞다.'),
    17: ('complete_pages', '공식 NeurIPS 2025 / 38권·7명 저자·DOI가 일치한다. 등록 페이지를 보완한다. mixed-type/continuous attributes 선행연구이며 discrete mark만 가능하다는 주장을 뒷받침하지 않는다.'),
    18: ('match_year_basis', '최종 저널 논문의 저자 4명과 2020년 36(3):1181–1191·DOI가 일치한다. 초기 preprint의 연도·저자 목록을 최종 저널 정보와 섞지 않는다.'),
    19: ('match_year_basis', '저자 4명과 ICLR 2023이 맞다. 2022년은 최초 arXiv 공개 연도다. patching/channel-independence 설명이 초록과 일치한다.'),
    20: ('match', 'ICLR 2024 공식 proceedings의 저자 7명·순서·제목과 일치한다. variate token에 대한 attention 설명이 맞다.'),
    21: ('complete_publication_record', '2015년 arXiv 원고 링크를 CVPR 2016 출판 페이지로 바꾸고 770–778 및 DOI를 보완한다. 기존 2016년은 맞으며 저자·제목도 맞다.'),
    22: ('fix_bibtex_name_parsing', '표시된 저자 목록은 PMLR 공식 기록과 맞다. BibTeX에서 복합 성씨 De Laroussilhe가 Laroussilhe로만 분리되지 않게 공식 family, given 순서를 사용한다. 병목과 잔차 구조는 원문 §2.1/Fig.2로 확인했다.'),
    23: ('match_year_basis', 'ReZero의 정식 발표는 UAI 2021, PMLR 161:1352–1361이다. 현재 표기가 맞으며 ICML 논문으로 바꾸면 안 된다. 영 초기화 스칼라 gate와 TitanTPP의 영 출력 행렬을 구별한다.'),
    24: ('match', '공식 NeurIPS 2015 / 28권·저자 4명과 일치한다. 외부 메모리에 attention으로 접근한다는 설명에 사용한다.'),
    25: ('match_capitalization', '제목의 대소문자 외에 공식 NeurIPS 2017 / 30권·저자 8명과 일치한다. Transformer 원문의 post-LN을 현재 pre-LN 구현과 동일하다고 서술하지 않는다.'),
    26: ('match_preprint', 'Ba–Kiros–Hinton, 2016, arXiv:1607.06450이 일치한다. 확인하지 않은 정식 학회명을 덧붙이지 않는다.'),
    27: ('match_preprint', 'Hendrycks–Gimpel, 2016, arXiv:1606.08415가 일치한다. 2023년은 최신 수정일이다. GELU의 xΦ(x) 정의와 원고의 exact formulation이 맞다.'),
    28: ('match_year_basis', 'AdamW 논문은 ICLR 2019이다. 최초 공개 2017년과 구분한 현재 표기가 맞다.'),
    29: ('match_year_basis', '12명 저자와 ICLR 2024 camera-ready가 일치한다. 2023년 arXiv 최초 공개와 구분한다. 본문은 benchmarking 배경으로만 인용하고 EasyTPP 구현 사용을 주장하지 않는다.'),
    30: ('clarify_published_version_link', '2006년 22(4):679–688·DOI는 맞다. 기존 링크의 저자 원고 표지는 2005년이므로 정식 출판 DOI로 연결한다. §2.1은 scale dependence 및 RMSE의 outlier 민감성을 설명하며 MASE를 제안한다.'),
    31: ('match_year_basis', 'JASA 2007년 102(477):359–378·저자·DOI가 일치한다. 현재 출판사 페이지의 2012년 온라인 등록일로 연도를 바꾸지 않는다. 로그 점수의 근거이며 혼합 학습 목적 전체의 properness 증명은 아니다.'),
    32: ('match_name_variant', '공식 NeurIPS 2019 / 32권의 21명 저자 순서와 일치한다. Andreas Kopf는 공식 메타데이터 표기다. PyTorch 소개 인용이지 현지 설치 버전이나 속도 수치의 출처가 아니다.'),
}
extra_sources = {
    1: [crossref_url('10.1057/jors.1972.50'), 'https://www.tandfonline.com/doi/abs/10.1057/jors.1972.50'],
    3: [crossref_url('10.52202/085713-3786')],
    4: ['https://academic.oup.com/biomet/issue/58/1'],
    6: [crossref_url('10.1145/2939672.2939875')],
    12: [s2_url, crossref_url('10.52202/085713-5431')],
    16: [crossref_url('10.1016/j.ejor.2011.05.018')],
    17: [crossref_url('10.52202/085713-3833')],
    18: [crossref_url('10.1016/j.ijforecast.2019.07.001')],
    21: [cvf_url, crossref_url('10.1109/CVPR.2016.90')],
    22: ['https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf'],
    30: [crossref_url('10.1016/j.ijforecast.2006.03.001')],
    31: ['https://www.tandfonline.com/doi/abs/10.1198/016214506000001437',
         'https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf',
         crossref_url('10.1198/016214506000001437')],
}
rows, changes, url_replacements = [], [], {}
for i, r in enumerate(records, 1):
    original = old_records[i - 1]
    patch = patches.get(r['key'], {})
    field_changes = {k: {'before': r.get(k), 'after': v} for k, v in patch.items() if r.get(k) != v}
    if 'url' in field_changes:
        url_replacements[r['url']] = patch['url']
    r.update(patch)
    if field_changes:
        changes.append({'number': i, 'key': r['key'], 'fields': field_changes,
                        'classification': notes[i][0]})
    sources = list(dict.fromkeys([original['url']] + extra_sources.get(i, [])))
    rows.append({'number': i, 'key': r['key'], 'title': r['title'],
                 'status': notes[i][0], 'identity_verified': True,
                 'authors_and_order_verified': True, 'publication_year_verified': True,
                 'venue_or_preprint_status_verified': True,
                 'existing_doi_verified': True if original.get('doi') else None,
                 'existing_volume_issue_pages_verified': True,
                 'citation_claim_check': 'consistent_with_current_passage',
                 'note_ko': notes[i][1], 'evidence_urls': sources,
                 'access_extent': ('Publisher-indexed metadata/abstract plus Crossref or author manuscript where noted; not a cover-to-cover review'
                     if i in [4, 15, 16, 18, 21, 31] else
                     'Official proceedings/publisher or author arXiv metadata and abstract; targeted method passage for refs 22 and 30'),
                 'checked_utc': stamp})

text = old_text
for old_url, new_url in url_replacements.items():
    text = text.replace('](' + old_url + ')', '](' + new_url + ')')
body, tail = text.split('## References', 1)
internal = '## Internal' + tail.split('## Internal', 1)[1]
lines = []
for i, r in enumerate(records, 1):
    authors = ', '.join(r['authors'][:-1]) + ', and ' + r['authors'][-1] if len(r['authors']) > 2 else ' and '.join(r['authors'])
    details = r['venue']
    if r.get('volume'): details += ' ' + r['volume']
    if r.get('number'): details += '(' + r['number'] + ')'
    if r.get('pages'): details += ':' + r['pages'].replace('--', '–')
    lines.append(f"{i}. {authors} ({r['year']}). *{r['title']}*. {details}. [Source]({r['url']}).")
MAIN.write_text(body + '## References\n\n' + '\n'.join(lines) + '\n\n' + internal)
registry['updated_utc'] = stamp
registry['reference_doublecheck'] = {'checked_utc': stamp, 'record_count': 32,
    'report': str((AUDIT / 'report.md').relative_to(ROOT)),
    'result': 'No incorrect work identity, author order, or publication year found; bibliographic completion and BibTeX surname parsing revised'}
dump(OUT / 'references.json', registry)

entries = []
for r in records:
    fields = {'title': '{' + r['title'] + '}', 'author': r.get('bibtex_author', ' and '.join(r['authors'])), 'year': str(r['year'])}
    fields['journal' if r['entry_type'] == 'article' else ('howpublished' if r['entry_type'] == 'misc' else 'booktitle')] = r['venue']
    for field in ['volume', 'number', 'pages', 'series', 'doi', 'url']:
        if r.get(field): fields[field] = str(r[field])
    entries.append('@' + r['entry_type'] + '{' + r['key'] + ',\n' + ',\n'.join('  ' + k + ' = {' + v + '}' for k, v in fields.items()) + '\n}')
(OUT / 'references.bib').write_text('\n\n'.join(entries) + '\n')

# The addition list and expansion receipt are historical records and stay intact.
# The current registry and this review are authoritative for subsequent export.
dump(AUDIT / 'audit.json', {'checked_utc': stamp, 'status': 'primary_source_review_complete',
    'scope': 'User attachment and all 32 current manuscript references',
    'reference_count': 32, 'incorrect_work_identity_found': 0,
    'incorrect_author_order_found': 0, 'incorrect_publication_year_found': 0,
    'incorrect_existing_doi_found': 0, 'modified_reference_count': len(changes),
    'full_texts_read_cover_to_cover': False,
    'systematic_literature_coverage_or_novelty_certified': False,
    'current_claims_reviewed_against_abstract_or_targeted_passage': True,
    'records': rows})
dump(AUDIT / 'changes.json', {'created_utc': stamp, 'reference_changes': changes,
    'body_url_replacements': url_replacements,
    'manuscript_before_sha256': before_sha[str(MAIN.relative_to(ROOT))],
    'manuscript_after_sha256': sha(MAIN), 'body_prose_or_results_changed': False,
    'bibliography_count_before': 32, 'bibliography_count_after': 32})

table = '\n'.join('| ' + ' | '.join([str(r['number']), r['key'], r['note_ko'],
                    '[1차 출처](' + (s2_url if r['number'] == 12 else r['evidence_urls'][0]) + ')']) + ' |' for r in rows)
report = f'''# 참고문헌 32편 재대조

**결론 — 완료:** 첨부 목록 32편에서 존재하지 않는 논문, 다른 논문으로 연결되는 링크, 잘못된 저자 순서·출판 연도·기존 DOI는 발견하지 않았다. **8개 항목의 누락 정보·출판본 링크·BibTeX 성씨 처리를 보완**했다. 제목이나 저자 순서를 바꾼 항목은 없다.

대조 시각(UTC): {stamp}. 첨부 파일의 32개 제목이 현재 원고 목록과 일치함을 확인했다. 출판사·학회 공식 기록, 저자 arXiv 원고, DOI 등록기관 Crossref를 사용했다. 자동 문자열 검사만으로 외부 검증을 대신하지 않았다.

## 반영한 변경

| 번호 | 문헌 | 반영 내용 | 성격 |
|---|---|---|---|
| 1 | Croston | 23(3)의 호수 추가 | 누락 보완 |
| 3 | Titans | 공식 DOI 10.52202/085713-3786 및 125925–125962 추가 | 누락 보완 |
| 6 | RMTPP | KDD 22회, 1555–1564 추가 | 누락 보완 |
| 12 | S2P2 | NeurIPS 38, 정식 proceedings 링크, DOI 10.52202/085713-5431, 180555–180596 추가 | 출판 정보 보완 |
| 17 | FlexTPP | 127441–127473 추가 | 누락 보완 |
| 21 | ResNet | CVPR 2016 정식 링크, 770–778, DOI 10.1109/CVPR.2016.90 추가 | 출판 정보 보완 |
| 22 | Houlsby adapters | BibTeX의 `De Laroussilhe, Quentin` 성·이름 경계 명시 | 형식상 오류 예방 |
| 30 | Hyndman–Koehler | 2005년 날짜가 찍힌 저자 원고 대신 2006년 출판 DOI에 연결 | 판본 혼동 방지 |

NeurIPS 2025의 긴 페이지 번호는 추정값이 아니라 공식 DOI 등록자료에 있는 전체 proceedings 페이지다. Croston의 JORS 표기는 현재 출판사와 Crossref가 사용하는 명칭이므로 틀린 저널 인용으로 판정하지 않았다. 역사적 명칭 Operational Research Quarterly와의 관계는 편집 주의사항으로 남긴다.

## 연도·저자 표기에서 확인한 사항

- AttNHP는 ICLR 2022, Intensity-Free는 ICLR 2020, PatchTST는 ICLR 2023, AdamW는 ICLR 2019, EasyTPP는 ICLR 2024다. 최초 arXiv 연도를 출판 연도로 옮겨 쓰지 않는다.
- ReZero는 UAI 2021이다. HiPPO는 NeurIPS 2020, S2P2·Titans·FlexTPP는 NeurIPS 2025다.
- Layer Normalization과 GELU는 이 목록에서 arXiv 논문으로 인용한다. GELU의 2023년 수정일을 최초 공개 2016년과 혼동하지 않는다.
- DeepAR는 2020년 최종 저널의 네 저자를 기준으로 한다. 초기 preprint의 저자·연도로 교체하지 않는다.
- Jason Eisner/Jason M Eisner, Cao Xiao/Cao (Danica) Xiao, ASCII 이름과 악센트가 있는 이름은 출판 플랫폼별 표기 차이다. 저자 누락이나 다른 사람으로 판정하지 않았다. 본문에 표시하는 이름은 기존 공식 기록·저자 원고 표기를 유지했다.

## 본문 인용의 의미도 대조한 결과

현재 본문의 관련 연구 설명은 확인한 초록 또는 필요한 방법 문단과 일치한다. 특히 다음 경계를 유지한다.

- Türkmen·FlexTPP는 수량/연속 속성 모형화의 **선행연구**다. TitanTPP가 continuous mark를 처음 도입했다는 근거가 아니다.
- Hyndman–Koehler §2.1은 MAE·RMSE의 척도 의존성과 RMSE의 이상치 민감성을 설명한다. 이 논문의 주 제안은 MASE이며, 현재 RMSE checkpoint 선택을 보편적으로 정당화하는 문헌으로 쓰지 않는다.
- Gneiting–Raftery의 로그 점수 설명은 duration NLL의 해석에 사용한다. 시간 NLL과 수량 MSE를 합한 학습 목적 전체가 joint likelihood 또는 strictly proper score라는 증명으로 확대하지 않는다.
- Houlsby의 병목 adapter와 ReZero의 영 초기화는 관련 구조다. 현재 TitanTPP의 공동 학습·인접 상태 입력·영 출력 행렬과 같은 구현이라고 서술하지 않는다.
- DeepAR·PatchTST·iTransformer는 관련 연구이며 현재 수행한 외부 TPP 성능 비교군에 포함된 것으로 표현하지 않는다. EasyTPP는 benchmark 배경, PyTorch 논문은 소프트웨어 소개 근거다.

## 항목별 대조

| 번호 | 식별자 | 판정과 확인 사항 | 출처 |
|---|---|---|---|
{table}

## 검증 범위와 산출물

- [항목별 기록](audit.json)에는 각 문헌의 1차 출처와 검토 범위를 남겼다. [변경 기록](changes.json), [첨부 동일성](attachment_identity.json), [수정 전 해시](before_sha256.json)를 보존했다.
- [Crossref 조회 결과](crossref_metadata_verified_tls.json)와 [S2P2 등록정보](s2p2_crossref.json)를 저장했다. 일부 DOI 요청은 전송 오류였으며 해당 문헌은 공식 proceedings·출판사 색인으로 확인했다. 최초 Python TLS 인증서 조회 실패도 [별도 기록](crossref_metadata.json)으로 남겼고, 검증을 끄지 않고 시스템 curl의 TLS 검증으로 조회했다.
- OUP·Elsevier·T&F·CVF 일부 직접 열람에는 접근 제한이 있었다. 출판사 검색 색인, DOI 등록자료 또는 저자 원고로 보완했다. 접근 실패를 잘못된 DOI나 없는 논문으로 판정하지 않았다.
- 32편을 모두 처음부터 끝까지 정독했다거나 문헌 포괄성·독창성·투고 준비가 확정됐다는 뜻은 아니다. 이번 작업은 서지 정확성과 현재 인용 문장의 대응 검토다.
- 현재 원고·BibTeX·references.json을 갱신했다. 과거 `reference_additions.json`과 `reference_expansion_changes.json`은 당시 기록으로 보존한다. 과거 `expand_references.py`를 현재 원고에 재실행하면 이번 보완이 사라지므로, 이후 편집·내보내기는 현재 registry를 기준으로 한다.

**인용 정확성 점검 — 완료**

- 32편 대조와 8편 보완을 마쳤다. 본문 문장·수식·표·그림·실험 결과는 보존하고 세 출판본 링크만 바꾸었다.

**최종 원고의 주장과 문헌 범위를 점검한다 — 다음 작업**

- 최종 실험 결과가 반영된 뒤 초록·기여·관련 연구의 문장별 인용 범위를 다시 맞춘다. 단순한 참고문헌 개수 목표로 문헌을 추가하지 않는다.

**제출 양식으로 참고문헌을 편집한다 — 이후 작업**

- 제출본의 인용 스타일·지면에 맞춰 표시를 통일한다. 이번 서지 점검에서는 새 학습, 원격 GPU 조회, 일정 변경, 외부 게시를 수행하지 않았다.
'''
(AUDIT / 'report.md').write_text(report)
print(json.dumps({'references_checked': 32, 'references_modified': len(changes),
                  'body_url_changes': len(url_replacements), 'manuscript_sha256': sha(MAIN)}, ensure_ascii=False))
