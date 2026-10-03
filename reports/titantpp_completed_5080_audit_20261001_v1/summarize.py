"""Reconcile terminal CPU audits with the previously published local table.

Only local audit/plan documents change. Original experiment and comparison
artifacts remain immutable; no network, training or inference is performed.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDITS = {
    'additional': ROOT / 'search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5080_20261001_v1/terminal_audit.json',
    'normalization': ROOT / 'search_artifacts/titantpp_instacart_norm_5080_20260930_v1/retrieved/terminal_20261001_v1/terminal_audit.json',
}
COMPARISON = ROOT / 'reports/titantpp_completed_external_comparison_20261001_v1'
BASELINE = ROOT / 'reports/titantpp_baseline_reset_20261001_v1'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rel = lambda p: str(Path(p).relative_to(ROOT))


def write(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    assert not (OUT / 'verification.json').exists(), 'Preserve completed summary'
    now = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    audits = {key: read(p) for key, p in AUDITS.items()}
    assert all(a['status'] == 'passed' and a['originals_unchanged'] for a in audits.values())
    csv_path = COMPARISON / 'selected_conditions.csv'
    table = list(csv.DictReader(csv_path.open()))
    rows, matches = [], []
    for kind, audit in audits.items():
        script_digest = audit['audit_script_sha256']
        version = OUT / 'audit_versions' / (script_digest + '.py')
        if not version.exists():
            assert sha(OUT / 'audit.py') == script_digest
            version.write_bytes((OUT / 'audit.py').read_bytes())
        assert sha(version) == script_digest
        for row in audit['conditions']:
            job = row['job']
            candidates = [x for x in table if (x['dataset'], x['model'], int(x['seed'])) ==
                          (job['dataset'], job['arm'], job['seed'])]
            assert len(candidates) == 1
            x = candidates[0]
            assert int(x['epoch']) == row['selected_epoch'] and int(x['count']) == row['validation_targets']
            assert all(float(x[k]) == v for k, v in row['selected_validation'].items())
            matches.append({'job': job['id'], 'selected_metrics_exact_match': True,
                            'selected_epoch_exact_match': True, 'validation_count_exact_match': True})
            rows.append({'campaign': kind, **row})
    assert len(rows) == 13
    reuse = {
        'core': ROOT / 'reports/titantpp_core_ablation_execution_20260928_v1/verification.json',
        'efficiency': ROOT / 'reports/titantpp_efficiency_5080_20260930_v1/verification.json',
    }
    assert read(reuse['core'])['final_core_binary_source_audit_complete']
    assert read(reuse['efficiency'])['execution_status'] == 'complete'
    total_files = sum(a['verified_retrieved_files'] for a in audits.values())
    total_checkpoints = sum(a['verified_terminal_checkpoints'] for a in audits.values())
    assert (total_files, total_checkpoints) == (466, 26)
    summary = {
        'status': 'passed', 'completed_kst': now, 'scope': 'completed_5080_additional12_and_instacart_norm1',
        'completed_conditions': 13, 'terminal_checkpoint_files_cpu_audited': 26,
        'preserved_recovery_checkpoint_files_cpu_audited': 1, 'selected_last_replay_records_audited': 26,
        'retrieved_files_sha_verified': 466, 'source_files_per_campaign': {'additional': 106, 'normalization': 104},
        'campaign_audits': {key: {'path': rel(p), 'sha256': sha(p)} for key, p in AUDITS.items()},
        'existing_selected_conditions_csv': {'path': rel(csv_path), 'sha256': sha(csv_path)},
        'existing_table_reconciliation': matches, 'conditions': rows,
        'reused_completed_evidence': {key: {'path': rel(p), 'sha256': sha(p)} for key, p in reuse.items()},
        'no_new_model_forward_training_replay': True, 'held_out_read': False,
        'scheduler_changes_this_audit': False, 'originals_unchanged': True,
        'ongoing_raf_and_5090_training_untouched': True,
        'remaining': ['5080 RAF approved queue and terminal artifact audit',
                      '5090 Instacart additional TPP approved queue and terminal artifact audit',
                      'Paper method/related-work integration can proceed in parallel',
                      'Final four-dataset table after remaining campaigns; held-out execution requires separate approval'],
    }
    write(OUT / 'audit_summary.json', summary)
    metric_lines = []
    labels = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent', 'insta_market_basket': 'Instacart'}
    arms = {'s2p2_matched_head': 'S2P2', 'attnhp_matched_head': 'AttNHP', 'titantpp_history_mlp_active_norm': '활성 분기 정규화'}
    for r in rows:
        j = r['job']; m = r['selected_validation']
        metric_lines.append(f"| {labels[j['dataset']]} | {arms[j['arm']]} | {j['seed']} | {r['selected_epoch']} / {r['completed_epochs']} | {m['qty_mae']:.9f} | {m['qty_rmse']:.9f} | {m['time_nll']:.9f} | 통과 |")
    norm = audits['normalization']['conditions'][0]
    report = f"""# 완료된 5080 결과의 원본 감사

감사 완료: **{now}**. 대상은 추가 TPP 12조건(Taxi·Intermittent × S2P2·AttNHP × seed42·52·62)과 Instacart 활성 분기 정규화 seed42 1조건이다.

**13조건 모두 통과했다.** 서버에서 원본466파일을 회수해 SHA를 확인했고, terminal checkpoint26개를 CPU에서 strict loading·tensor SHA로 검증했다. Instacart의 기존4090 epoch2 복구 checkpoint1개도 별도로 검증했다. 원본 source는 캠페인별106개·104개이며, 두 묶음에 공통 파일이 있으므로210개 고유 source라는 의미는 아니다.

**기존 전체 비교표의 결과는 바뀌지 않았다.** 13조건의 선택 epoch·validation 표본 수·MAE·RMSE·시간 NLL이 기존 CSV와 정확히 일치한다. 이번에 해소한 것은 원본 checkpoint/source 최종 감사 미완료 상태다. 감사 통과가 성능 우월성이나 독립 평가 완료를 의미하지는 않는다.

**원본 회수와 무결성 — 완료**

- 추가 TPP: 325파일·106source·12조건·24terminal checkpoint. [회수 영수증]({AUDITS['additional'].parent / 'retrieval_receipt.json'}) · [CPU 감사]({AUDITS['additional']}).
- Instacart 정규화: 141파일·104source·1조건·2terminal checkpoint 및 보존된복구 checkpoint1개. [회수 영수증]({AUDITS['normalization'].parent / 'retrieval_receipt.json'}) · [CPU 감사]({AUDITS['normalization']}).
- 계약·승인·native qualification·supervisor 완료 상태·terminal manifest·전체 파일 SHA를 연결했다. 추가 TPP 계약은`a39790e6…`, Instacart5080 운영 계약은`f93a456e…`다. 정규화 checkpoint 안의 과학 계약`ae7a1bff…`는 최초4090 계약으로 유지되는 것이 정상이며,5080 운영 계약과 구분해 확인했다. 전체SHA는 audit_summary.json과 각 terminal_audit.json에 있다.
- 종료된 캠페인 root의 소유 프로세스가 없음을 회수시 확인했다. 같은5080의 RAF는 별도 실행으로 유지했고5090에 새 조회·실행을 하지 않았다. 원격에서는 저우선순위 읽기·전송만 수행했다.

**checkpoint·학습 이력·재평가 기록 — 완료**

- 동결 source의 CPU 사본으로 모델을 구성해 selected/last와 last 안의best tensor를 strict loading했다. 현재 작업 트리의 변경된 과학 코드를 사용하지 않았다.
- optimizer 상태의 shape·유한값·학습률·누적 step을 검사하고 CPU 복원을 확인했다. Python/NumPy/Torch RNG와 shuffle 상태는 CPU에서 복원하고,CUDA RNG는 바이트 보존만 검사했다.
- 최초 strict 최소 validation raw 수량 RMSE 선택과 최초 patience40 종료, batch128/min40/max300, 전체train/validation 노출·표본 수·원래배치 prefix를 검증했다. 초기화 동일성은 당시 실제Runtime의 통과한 native qualification과 checkpoint 초기SHA 체인을 재사용했다.
- selected/last26역할의 저장된 validation 재평가를 tensorSHA·history와 대조했다. 수량·이력길이·추가이력 구간의 표본 수·SSE·MAE·시간NLL 합계, body/tail, 비어 있는 구간을 확인했다. 새로운 forward나 재평가는 실행하지 않았다.
- epoch checkpoint 영수증은 마지막epoch 직후 노출을, terminal manifest는 종료후 추가validation이 포함된 최종 노출을 가리킨다. validation 마지막기록을 제외한 prefixSHA까지 확인했으며 원본파일은 수정하지 않았다.
- S2P2의 첫층 `delta_net.weight`는 원본연산에서 입력이None인 경로라 optimizer 상태가 생기지 않는다. 동결vendor/adapter 코드로 확인한 이 항목만 허용했으며, 그 외 누락상태는 허용하지 않았다. summary의 history_rows는epoch수가 아니라 이력길이 구간표이며,빈구간 생략을 endpoint의명시적0개구간과 대응했다.

**Instacart 복구 이력 — 완료**

- 4090의epoch1~2 history·exposure·timing·train.log prefix와epoch2 checkpointSHA가5080완료 원본에 보존돼 있다. epoch2의optimizer31,114steps/RNG/shuffle/모델/내부best 상태를 CPU검증했고,5080은epoch3부터 이어졌다.
- 종료70epoch·선택30epoch, 전체optimizer **1,088,990steps**. 시간·수량 head와과학계약·source·초기SHA·resume identity를 유지했다. GPU이전 후 중단 없는4090실행과 같은 수치궤적이라는 보장은 하지 않는다.
- summary elapsed **{norm['fit_elapsed_seconds']:.6f}초는5080복구 이후**다. 전체70epoch의저장된 timing합은 **{norm['epoch_timing_seconds_sum']:.6f}초**이며,첫2epoch는4090기록이다. 원래RunPod 예산검사 중단과복구대기·장비차이를 삭제하거나 추정 보정하지 않았다. 원래마감 안에서 종료했다.

**원본까지 연결한 동일 선택 checkpoint 결과 — 완료**

모든 값은 validation이며 낮을수록 좋다. 선택/종료는epoch번호다. 추가TPP는공통head비교이고,정규화Instacart는seed42탐색으로 구분한다.

| 데이터 | 모델 | seed | 선택 / 종료 | MAE | RMSE | 시간 NLL | CPU 감사 |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(metric_lines)}

기존 [전체 비교보고서]({COMPARISON / 'report.md'})의07:07KST 수치와114고유조건 집계는 그대로 보존했다. 그보고서의verification에있는pending은생성당시 기록이며, 최신감사상태는[후속 감사상태]({COMPARISON / 'audit_completion.json'})가 보완한다. 추가TPP **5080의12조건만 최종감사완료**이며,5090전체를감사완료로표시하지 않는다.

**측정과 감사의 한계**

- 로컬CPU Runtime은Python{audits['normalization']['local_cpu_audit_runtime']['python']}/Torch{audits['normalization']['local_cpu_audit_runtime']['torch']}다. 학습당시CUDA Runtime을재현한실행이아니며,실제환경·공통초기화·인과성·마스킹검증은SHA로확인된native기록을재사용했다. CPU에서 같은초기tensor를새로생성해CUDA와동일하다고주장하지않는다.
- 원본데이터를다시로드하지않았다. 계약데이터SHA·native입력통계와population identity·실제노출/배치순서기록을대조했다. held-out/test성능은열람하지않았다.
- 초기합성qualification update는scientificfit step에포함하지않는다. 추가TPP native60updates와5080정규화복구검증3updates는기존준비증거로만재사용했다.
- Core54조건의최종감사와Titans-MAC효율측정은완료증거를재사용했다. 새학습·평가·Pod·스케줄러·소스변경·commit/push는없다.

**남은 작업 순서**

**1. 방법과 관련 연구를 원고로 연결한다 — 다음 작업 / 지금 착수 가능**
- 대상은paper_research의단일논문원고다. 완료한수식·구조그림·코드대응·효율자료와이번감사결과를재사용해기여·공통head비교범위·RMSE선택이유를연결한다. 기존GPU큐와독립적으로병행가능하다.
- 완료조건은방법·관련연구·실험설정초안에각주장의근거와미검증범위가표시되는것이다. 최종결과수치는다음단계를따른다.

**2. 승인된5080 RAF와5090 Instacart 추가TPP를 마친다 — 진행 중 / 서버 작업 대기**
- 이미실행중인서버별큐는독립병렬로유지한다. 이번감사를위해중단·재시작·추가실험을하지않았다. 이문서는새현재진척을조회한보고가아니다.
- 종료후완료·실패·미시작조건을빠짐없이회수하고동일한원본감사를수행한다. 기존3데이터정규화seed52·62는보류한다. 자동스케줄러는삭제상태를유지한다.

**3. 네 데이터의 최종 비교와 주장–근거표를 확정한다 — 다음 작업 /2번 이후**
- 고정대표TitanTPP MLP와외부6모델을같은평가기준으로비교한다. 정규화·Gate는별도탐색,Core는구성비교로두고시간NLL악화·구간별반례·실측비용을포함한다.
- 완료조건은기존3데이터와RAF의전체결과가원본까지추적되고,같은seed및3seed집계의주장범위가명확한것이다.

**4. 평가 규칙을 고정하고 독립 평가·제출본을 준비한다 — 이후 작업 / 실행·열람 및 제출 승인 필요**
- 모델·checkpoint·지표·비교범위를먼저고정한실행계약을작성한다. held-out결과를현재validation분석에섞지않으며,실제독립평가와외부제출은별도승인범위를따른다.

[기계판독 감사요약](audit_summary.json) · [검증결과](verification.json) · [현재 기준선과 전체 작업순서]({BASELINE / 'README.md'})
"""
    (OUT / 'report.md').write_text(report)
    # Keep the original analysis and its generation-time verification immutable.
    completion = {'recorded_kst': now, 'status': 'passed', 'scope': summary['scope'],
                  'additional_binary_cpu_audit': {'5080': 'complete_12_conditions_24_terminal_checkpoints',
                                                 '5090': 'pending_not_audited_in_this_request'},
                  'instacart_norm_local_binary_cpu_audit': 'complete_1_condition_2_terminal_and_1_recovery_checkpoint',
                  'selected_table_values_unchanged': True, 'original_comparison_cutoff_kst': '2026-10-01T07:07:00+09:00',
                  'original_verification_preserved_as_generation_time_record': True,
                  'audit_report': rel(OUT / 'report.md'), 'audit_summary': rel(OUT / 'audit_summary.json'),
                  'campaign_audits': summary['campaign_audits']}
    write(COMPARISON / 'audit_completion.json', completion)
    (COMPARISON / 'AUDIT_UPDATE.md').write_text(f"# 原本 감사 후속 상태\n\n{now}에5080추가TPP12조건과Instacart정규화1조건의원본회수·CPU감사를완료했다. 선택epoch·표본수·MAE·RMSE·시간NLL은기존표와정확히일치하며수치는변경하지않았다.\n\n기존verification.json의pending은07:07KST분석생성당시상태로보존한다. 최신상태는[audit_completion.json](audit_completion.json)과[감사보고서]({OUT / 'report.md'})를따른다. 5090추가TPP의최종binary감사는이번완료범위에포함되지않는다.\n")
    backup = BASELINE / 'before_completed_5080_audit'
    backup.mkdir(exist_ok=False)
    for name in ('README.md', 'baseline.json', 'verification.json'):
        (backup / name).write_bytes((BASELINE / name).read_bytes())
    baseline = read(BASELINE / 'baseline.json')
    baseline['completed_artifact_audit_updated_kst'] = now
    baseline['additional_tpp']['host5080']['binary_cpu_source_final_audit'] = 'complete_12_conditions_24_checkpoints'
    baseline['additional_tpp']['host5080']['audit_evidence'] = summary['campaign_audits']['additional']
    baseline['additional_tpp']['binary_cpu_source_final_audit'] = '5080_complete_5090_pending'
    baseline['active_norm_existing_three_datasets']['instacart']['local_binary_source_final_audit'] = 'complete'
    baseline['active_norm_existing_three_datasets']['instacart']['audit_evidence'] = summary['campaign_audits']['normalization']
    baseline['next_action'] = 'Draft method/related work using completed evidence while existing RAF/5090 queues continue; audit them after terminal completion'
    baseline['artifact_audit_report'] = rel(OUT / 'report.md')
    baseline['artifact_audit_summary_sha256'] = sha(OUT / 'audit_summary.json')
    write(BASELINE / 'baseline.json', baseline)
    doc = (BASELINE / 'README.md').read_text()
    doc = doc.replace('이번 요청: 불필요한 스케줄러를 삭제하고, 현재 증거를 기준으로 남은 작업을 재정렬한다.',
        f'**후속 완료 반영({now})**: 추가TPP5080의12조건과Instacart정규화1조건의원본회수·CPU감사를마쳤다. [감사보고서]({OUT / "report.md"}). 서버진척은아래07:35관측이력으로유지하며이번감사시각을새진척관측으로표시하지않는다.\n\n최초 요청: 불필요한 스케줄러를 삭제하고, 현재 증거를 기준으로 남은 작업을 재정렬한다.')
    doc = doc.replace('아래 순서는 이후 작업 계획이며, 이번에 감사와 원고 작업까지 완료했다는 뜻이 아니다.',
                      '원본 감사 후속완료는위에반영했다. 원고작업과진행중두캠페인의최종감사는아직남아있다.')
    doc = doc.replace('| 로컬 checkpoint binary/source 최종 회수·CPU 감사 |', '| 후속13조건감사에서회수·CPU감사완료;재사용 |')
    doc = doc.replace('Instacart 로컬 binary/source 최종 감사.', 'Instacart 로컬 binary/source 최종감사완료;재사용.')
    start = doc.index('**1. 이미 완료된 새 결과의 원본 감사를 마무리한다')
    end = doc.index('**2. 승인된 RAF', start)
    doc = doc[:start] + f'''**새 완료 결과의 원본 감사를 확정한다 — 완료 / 현재 기준선**

- 대상: 추가TPP5080의12조건과Instacart정규화1조건. 원본466파일SHA·캠페인별106/104source·terminal checkpoint26개와복구 checkpoint1개의CPU감사를통과했다.
- 전체노출·공통배치prefix·optimizer/RNG·최초RMSE선택·기존selected/last26역할을검증했다. 기존비교표의13조건수치는변경없다. [최종감사]({OUT / 'report.md'}).
- Core와효율측정은완료증거를재사용한다.5080 RAF와5090추가TPP의남은실행·최종감사는별도다.

''' + doc[end:]
    for old, new in [('**2. 승인된 RAF', '**1. 승인된 RAF'),
                     ('**3. 네 데이터', '**2. 네 데이터'), ('다음 작업 / 1·2번 이후', '다음 작업 / 1번 최종감사 이후'),
                     ('**4. 방법과 관련 연구', '**3. 방법과 관련 연구'), ('다음 작업 / 1·2번과 병행 가능', '다음 작업 / 1번과 병행 가능'),
                     ('최종 수치·전체 결과표·결론은3번을 따른다', '최종 수치·전체 결과표·결론은2번을 따른다'),
                     ('**5. 최종 평가 규칙', '**4. 최종 평가 규칙'), ('**6. PAKDD 제출본', '**5. PAKDD 제출본'),
                     ('대상: 원고·표·그림·재현 부록.5번 결과', '대상: 원고·표·그림·재현 부록.4번 결과')]:
        doc = doc.replace(old, new)
    doc = doc.replace('종료 후 실패·미완료까지 보존해1번과 같은 원본 감사를 수행한다.', '종료후실패·미완료까지보존해이번완료감사와같은원본감사를수행한다.')
    old = '**바로 다음 작업**은1번의완료된5080'  # Actual old text includes spaces; replace the full paragraph below.
    paragraph_start = doc.index('**바로 다음 작업**')
    paragraph_end = doc.index('\n\n', paragraph_start)
    doc = doc[:paragraph_start] + '**바로 진행할 수 있는 작업**은3번의방법·관련연구·실험설정원고통합이다. 1번의기존5080 RAF와5090추가TPP는계속실행되고종료후회수·감사를한다. 최종네데이터비교표는1번이끝난뒤같은세션에서직렬로확정한다. 새스케줄러·학습·sub-agent는생성하지않았다.' + doc[paragraph_end:]
    (BASELINE / 'README.md').write_text(doc)
    verification = read(BASELINE / 'verification.json')
    verification['original_plan_verification_preserved_at'] = rel(backup / 'verification.json')
    verification['audit_update_kst'] = now
    verification['new_binary_audits_claimed'] = True
    verification['binary_audit_scope'] = summary['scope']
    verification['completed_artifact_audit_passed'] = True
    verification['plan_sha256'] = sha(BASELINE / 'README.md')
    verification['audit_summary_sha256'] = sha(OUT / 'audit_summary.json')
    write(BASELINE / 'verification.json', verification)
    result = {'status': 'passed', 'completed_kst': now, 'conditions': 13,
              'terminal_checkpoints_cpu_audited': 26, 'extra_recovery_checkpoints_cpu_audited': 1,
              'sha_verified_files': 466, 'replay_roles_verified': 26,
              'existing_selected_table_exact_matches': 13, 'scientific_integrity_failures': 0,
              'scope_validation_only': True, 'held_out_read': False, 'new_gpu_work': False,
              'new_forward_or_replay': False, 'original_comparison_preserved': True,
              'audit_summary_sha256': sha(OUT / 'audit_summary.json'),
              'report_sha256': sha(OUT / 'report.md'),
              'campaign_audits': summary['campaign_audits'],
              'remaining_5090_and_raf_not_claimed_complete': True}
    write(OUT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
