import json,csv,re,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
j=read(D/'comparison.json');v=read(D/'verification.json');p=R/'paper/titantpp_pakdd_2027_draft/main.tex';s=p.read_text();old=(D/'main.before.tex').read_text();t=read(D/'generated_tables.json')
checks={}
checks['generated_full_and_partial_table_rows_present']=all(x in s for x in t['full_rows']+t['partial_rows'])
checks['table_source_SHA']=t['comparison_sha256']==sha(D/'comparison.json')
for env in ['figure','table']:
 def blocks(text):
  return re.findall(r'\\begin\{'+env+r'\}[\s\S]*?\\end\{'+env+r'\}',text)
 before=blocks(old);after=blocks(s)
 if env=='table':
  before=[x for x in before if 'tab:extension' not in x];after=[x for x in after if 'tab:extension' not in x]
 checks['all_'+env+'_blocks_outside_appendix_A_unchanged']=before==after
checks['authors_abstract_introduction_method_unchanged']=old[old.index(r'\documentclass'):old.index('Two parameter-matched structural controls')]==s[s.index(r'\documentclass'):s.index('Two parameter-matched structural controls')]
checks['bibliography_unchanged']=old[old.index(r'\begin{thebibliography}'):] == s[s.index(r'\begin{thebibliography}'):]
checks['appendix_C_unchanged']=old[old.index(r'\section{Validation Errors'):old.index(r'\begin{thebibliography}')] == s[s.index(r'\section{Validation Errors'):s.index(r'\begin{thebibliography}')]
checks['Deep_Renewal_comparison_excluded_but_citation_retained']='Deep Renewal &' not in s and r'\bibitem{turkmen2021renewal}' in s
checks['scope_counts']=j['counts']=={'extension_complete_audited':31,'a100_complete_audited':4,'newly_audited':22,'reused_audited':13,'manuscript_structural_complete':21,'extension_incomplete_at_cutoff':5,'a100_incomplete_at_cutoff':2}
labels=re.findall(r'\\label\{([^}]+)\}',s);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s)
checks['unique_labels_and_resolved_references']=len(set(labels))==len(labels) and set(refs)<=set(labels)
checks['no_stale_scope_language']='10 completed structural' not in s and 'their seed-42 results' not in s
checks['mixed_GPU_and_partial_groups_explicit']='RTX PRO 4500' in s and '21 completed structural conditions' in s and 'no partial-group' in s
assert all(checks.values()),checks
(D/'manuscript_verification.json').write_text(json.dumps({'status':'passed','checked_unix':time.time(),'checks':checks,'main_sha256':sha(p),'comparison_sha256':sha(D/'comparison.json'),'before_sha256':sha(D/'main.before.tex'),'compilation':'Codex compile_latex_document success','new_training_or_replay':False,'heldout_read':False},ensure_ascii=False,indent=2)+'\n')
(D/'compilation_receipt.json').write_text(json.dumps({'path':str(p),'source_sha256':sha(p),'tool':'compile_latex_document','kind':'success','message':"The current source compiled successfully with the desktop editor's compiler.",'separate_pdf_exported':False},indent=2)+'\n')
DS={'yellow_trip_hourly':'Taxi','intermittent_frozen_5000':'Intermittent','raf_spare_parts':'RAF','insta_market_basket':'Instacart'}
MS={'titantpp_history_mlp':'기존 MLP','titantpp_current_only_param_matched':'현재 상태 전용','titantpp_all_available_history_mlp':'전체 분기·고정 /8','deep_renewal_event_native_nb':'Deep Renewal','titantpp_history_cross_product':'Cross-product','titantpp_history_recent4_mean':'Recent-four-mean'}
new=[x for x in v['retrievals'] if not x['reused']]
lines=['# 추가 완료 22조건 원본 검증과 원고 갱신','', '**요청 범위의 후속 신규 18조건과 A100 4조건이 모두 회수·CPU 검증을 통과했다.** 기존 13조건 감사는 재실행하지 않고 원본 SHA와 감사 결과를 재확인해 재사용했다. 원고에는 Deep Renewal을 제외한 구조 대조 21조건을 반영했다.','',
'관측 범위는 **2026-10-02 21:57 KST**에 고정했다. 5080은 19:53 종료 관측을 재사용했다. 회수 이후 추가 완료되었을 수 있는 조건의 현재 상태를 뜻하지 않는다. 실제 회수 시각과 CPU 감사 시각은 서버별 영수증에 있다.','',
'## 원본 검증 결과','', '| 서버 | 신규 조건 | checkpoint | 파일 | source 사본 | byte |','|---|---:|---:|---:|---:|---:|']
for x in new:lines.append(f"| {x['host']} | {x['conditions']} | {x['checkpoints']} | {x['files']} | {x['source_files']} | {x['bytes']:,} |")
lines.extend(['',f"합계: 신규 {sum(x['conditions'] for x in new)}조건, checkpoint {sum(x['checkpoints'] for x in new)}개, {sum(x['files'] for x in new)}파일. source 개수는 서버별 사본 수이며 고유 소스 파일 수의 합계가 아니다.",'',
'- 종료 명세와 파일 SHA, 과학 계약·동결 source closure·실행 승인·native qualification 연결을 확인했다.',
'- selected/last checkpoint의 CPU strict loading, 유한 tensor, 가중치 SHA, 선택 epoch와 embedded best를 확인했다.',
'- last checkpoint의 optimizer 상태·step, CPU RNG·shuffle 복원, CUDA RNG 저장 byte, 전체 epoch 노출과 기존 MLP의 공통 batch prefix를 확인했다.',
'- 첫 최소 raw 수량 RMSE 선택과 첫 종료 조건, 저장 지표, selected/last validation replay 기록의 population·지표·구간 합계를 대조했다. 마지막 epoch exposure receipt와 종료 후 추가 validation exposure는 구분했다.',
'- **모델 forward·재학습·새 replay·GPU 측정은 수행하지 않았다.** 원본 데이터는 다시 읽지 않고 native input identity와 노출 영수증을 사용했다. CPU runtime 검증을 원래 CUDA 실행 재현으로 표현하지 않는다.',
'- 5080의 첫 읽기 전용 회수 시도는 RAF 완료 기록이 별도 이관 관리자에 있어 종료되었다. 원래 관리자와 RAF 관리자 기록을 각각 보존하고 완료 SHA를 연결한 뒤 회수했다. 첫 시도 파일도 보존했으며 학습 오류나 학습 재시도가 아니다.',
'- PRO4500은 이관 reservation SHA와 원래 계약을 확인했다. 원래 109소스 중 실행기 한 파일의 차이를 구분했다. A100은 별도 110소스·과학 계약과 3개 병렬 qualification을 검증했으며 효율 측정에 합치지 않았다.','',
'## 완료와 원본 검증을 구분한 전체 목록','',
'| 캠페인 | 학습 완료·CPU 감사 완료 | 진행 | 미시작 대기 | 원고에 반영한 구조 대조 |','|---|---:|---:|---:|---:|',
'| 후속 36조건 | 31 (기존 13 + 신규 18) | 2 | 3 | 21 |','| A100 후보 6조건 | 4 | 2 | 0 | 별도 탐색 보고 |','',
'전체 42조건의 데이터·모델·seed·GPU·학습 상태·감사 상태는 [condition_registry.csv](condition_registry.csv)에 빠짐없이 기록했다. 원고에서 제외한 Deep Renewal 완료 10조건도 해당 목록과 아래 연구 결과에 보존한다. 진행 중 4조건과 대기 3조건은 이 감사의 완료 범위 밖이다.','',
'## 원고에 반영한 구조 대조','',
'- 완전한 3seed 그룹: Taxi 현재 상태 전용·전체 분기, Intermittent 현재 상태 전용, RAF 현재 상태 전용·전체 분기(5그룹, 15조건).',
'- 부분 그룹: Intermittent 전체 분기 seed42/52, Instacart 두 변형 seed42/52(6조건). 세 seed 평균·표본 표준편차는 만들지 않았다.',
'- 원래 MLP 대비 현재 상태 전용 비교에서 MLP의 MAE/RMSE 감소율은 Taxi 11.07%/11.48%, Intermittent 4.78%/4.11%, RAF 0.58%/0.75%다. 분모는 현재 상태 전용 평균이다.',
'- Intermittent는 seed42/52에서 MLP가 수량 두 지표 모두 낮고 seed62에서 반대다. seed62 현재 상태 전용은 PRO4500이며 기존 MLP와 나머지 두 seed는 5080이다. 현재 상태 전용의 평균 시간 NLL은 세 데이터 모두 낮다.',
'- 전체 분기 사용은 Taxi 수량 평균에서 앞서고 RAF는 차이가 작다. Intermittent 완료 두 seed는 순위가 반대이며 Instacart 완료 두 seed는 변형의 RMSE가 약0.05–0.14% 낮다. 단계적 분기 활성화의 보편적 우월성으로 해석하지 않는다.',
'- 현재 LNCS Appendix A, 실험 설정·구성 비교·한계와 Appendix B 연결 문단을 갱신했다. Appendix B/C의 기존 통계·오차 표와 모든 그림·히트맵, 주 비교84조건·효율·초록·Contribution·참고문헌은 유지했다.',
'- A100 네 조건은 seed42·이종GPU 탐색으로 아래 별도 기록에 둔다. 원고의 주모델을 교체하지 않았다.','',
'## 확정된 연구 결과: 후속 실험 전체 완료분','',
'다음은 원고 채택 여부와 무관한 원본 검증 결과다. 모든 값은 동일 RMSE 선택 epoch의 validation 지표이며, Deep Renewal은 고유 NB 손실을 사용하는 adapter다.','',
'| 데이터 | 모델 | seed | GPU | 완료/선택 epoch | MAE | RMSE | 시간 NLL |','|---|---|---:|---|---:|---:|---:|---:|'])
for x in sorted(j['completed'],key=lambda x:(x['campaign'],x['dataset'],x['model'],x['seed'])):
 if x['campaign']!='extension36':continue
 lines.append(f"| {DS[x['dataset']]} | {MS[x['model']]} | {x['seed']} | {x['host']} | {x['completed_epochs']}/{x['selected_epoch']} | {x['qty_mae']:.6f} | {x['qty_rmse']:.6f} | {x['time_nll']:.6f} |")
lines.extend(['','## A100 완료 4조건: 별도 탐색 결과','', '| 데이터 | 후보 | 완료/선택 epoch | MAE | RMSE | 시간 NLL | MLP 대비 MAE/RMSE/NLL 변화 |','|---|---|---:|---:|---:|---:|---|'])
for x in j['completed']:
 if x['campaign']!='a100_candidates':continue
 g=next(g for g in j['groups'] if g['dataset']==x['dataset'] and g['model']==x['model']);d=g['paired'][0]['candidate_vs_MLP_percent']
 lines.append(f"| {DS[x['dataset']]} | {MS[x['model']]} | {x['completed_epochs']}/{x['selected_epoch']} | {x['qty_mae']:.6f} | {x['qty_rmse']:.6f} | {x['time_nll']:.6f} | "+' / '.join(f'{d[k]:+.2f}%' for k in ['qty_mae','qty_rmse','time_nll'])+' |')
lines.extend(['','음수는 오차 감소다. Taxi cross-product의 RMSE 감소는0.06% 수준이며 MAE·시간 NLL은 증가했다. RAF recent-four-mean은 MAE·시간 NLL 감소와 RMSE 증가가 상충한다. 완료 네 조건에서 기존 MLP를 일관되게 대체할 근거는 없다. Intermittent 두 후보의 진행 중 best는 이 표에 넣지 않았다.','',
'## 증거와 재현 경로','',
'- [scope.json](scope.json): 고정된 원래 관측과 조건별 terminal SHA.',
'- [verification.json](verification.json): 신규/재사용 감사, source·checkpoint 수, 원본 SHA 재확인.',
'- [comparison.json](comparison.json): seed별 지표·완전한 3seed 평균과 표본 SD(ddof=1), 짝지은 변화율.',
'- [condition_registry.csv](condition_registry.csv): 42조건 전체 상태 및 원본·감사 경로.',
'- [manuscript.diff](manuscript.diff), [manuscript_verification.json](manuscript_verification.json), [compilation_receipt.json](compilation_receipt.json): 원고 변경과 검증·컴파일.',
'- 기존 13조건 감사는 reports/titantpp_extension_completed13_audit_20261002_v1 및 원래 retrieved/completed13_20261002_v1 아래에서 재사용했다.',
'- 새 회수본은 search_artifacts/titantpp_completed22_audit_20261002_v1/retrieved/{5080,5090,pro4500,a100}/original 에 있다. 최초 retrieval_receipt의 pending은 회수 시점 이력이며 각 terminal_audit.json이 후속 CPU 감사 결과다.','',
'## 남은 작업 순서','',
'**남은 승인 학습을 관측하고 종료분을 검증한다 — 진행 중**',
'- 5090·PRO4500·A100의 나머지 조건을 기존 계약과 스케줄러로 관측한다. 새로운 완료 조건만 같은 방식으로 회수·검증한다. Deep Renewal의 학습과 연구 기록은 유지한다.',
'', '**완전한 구조 대조 그룹과 A100 탐색 결론을 정리한다 — 다음 작업 / 종료분 검증 이후**',
'- 남은 세 구조 대조 조건이 검증되면 원고의 미완료 그룹을 갱신한다. A100은 여섯 조건을 단일seed 탐색으로 종합하고 추가 실험이나 주모델 교체를 자동 실행하지 않는다.',
'', '**RunPod 비용과 원본 회수를 마무리한다 — 진행 중 / 각 Pod 종료 시**',
'- 기존 관리자에 따른 전체 원본 회수·비용보고·Pod 삭제 확인은 이번 부분 회수와 별도다. 이번 결과만으로 캠페인 종료·전체 회수 완료라고 하지 않는다.',
'', '**최종 평가 계약을 확정한다 — 다음 작업 / 모델 선택 고정 이후**',
'- 기존 평가 코드와 checkpoint를 재사용하고 모집단·대상 모델·불확실성 계산·실행 상한을 고정한다. 실제 held-out 평가와 성능 열람은 승인된 범위에서 수행한다.'])
(D/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'checks':checks,'report':str(D/'report.md'),'registry_conditions':42,'newly_audited_conditions':22,'manuscript_structural_conditions':21},ensure_ascii=False))
