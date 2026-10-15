"""Current selected state tensors only: fixed V-difference filter arithmetic.

No model/project-module construction/import, forward, dataset/prediction read,
remote call, prior CNN checkpoint read, threshold search, or training decision.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BUNDLE=ROOT/'search_artifacts/titantpp_backbone_value_difference_rows_20261016_v1'
CNN_SUMMARY=ROOT/'reports/titantpp_backbone_location_raf6_20261015_v1/diagnostics_dc_v1/RESULT.json'
CONTRACT_SHA='8f5a4ed206ff36297b29f7fdc1d3191ba536301213d72764bd8777e26b8793dc'
BOOTSTRAP_SHA='e009b432b29e07649289e558feff7cc806c00b0d1f2cd0f9a6cb1fbeb21851a5'
CNN_SUMMARY_SHA='c14dcb2786729a932d4d3fbcb333cb8fb47485733145bb6ebba48315be69b114'
SOURCE_CLOSURE='4fce860e41ee716592abf168af8c3985859b74ea6d080357690124b19a88d361'
STATE_KEY='encoder.layers.0.attn.value_difference_kernel'
EPOCHS={42:9,52:5,62:22}
BINARY={42:'d1d0c36e4e8eda3cba947bece753fbbe43310d41027b15bb7567e01041778839',52:'f4deb6c52d1560c6671c6eb7a78d325f43aad5f86e217f03807141dff895a8c0',62:'0995fb284437779d7b575d9fcfe30d9accfac4baae2c1d2b4dda5f090d855c8a'}
STATES={42:'d752a98efd462463dde61956a3d5a3116ceedaee50a38b1879e5453b299eb0d0',52:'d86c2a6deead4c8fa612d09dcf9eb31dc5d8427f47a2261347206b7e49509f1c',62:'4e0adfca29d5f2041dd833958a3e2f051492b7310bd340a26f8254c49f048078'}
METHOD={
 'scope':'Only current selected ValueDifference RAF42/52/62 parameter tensors; old CNN summary reused without checkpoint read.',
 'full_history':{'h_current_lag1_lag2':'[1+a,b-a,-b]','dc':'1','nyquist_signed':'1+2*a-2*b'},
 'two_observations':{'h_current_lag1':'[1+a,-a]','dc':'1','nyquist_signed':'1+2*a'},
 'one_observation':{'h_current':'[1]','dc':'1','nyquist_signed':'1'},
 'summary':'64 latent channels, unweighted mean/sample SD/min/q10/median/q90/max; quantiles use linear interpolation.',
 'frequency_domain':'Observed-event index inside one segment, not physical elapsed time. Short-history values are finite-boundary coefficient-polynomial evaluations.',
 'selection':'Existing first strict finite minimum of fullValidation raw RMSE, unchanged; no outcome-based threshold or model selection.',
 'fixed_assertion_tolerance':1e-12,
 'limitations':['Gain is a latent linear-filter coefficient property, not raw quantity error or an activation-weighted effect.',
 'Q/K, projections, attention, memory and head are not included in this scalar response.',
 'DC=1 preserves a constant projected V segment, not necessarily a constant raw quantity or the final prediction.',
 'No causal attribution or next-training admission criterion is derived.',
 'Channels are unweighted and latent coordinate axes are not aligned between independently trained models/seeds.']}
INPUTS={}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def pin(path,expected):
    need(sha(path)==expected,'Frozen input SHA differs: '+str(path));INPUTS[str(path.relative_to(ROOT))]=expected;return path


def write(path,v):
    with Path(path).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def quantile(values,q):
    values=sorted(values);index=(len(values)-1)*q;lo=math.floor(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def summary(values):
    values=list(map(float,values));need(len(values)>0 and all(map(math.isfinite,values)),'Nonfinite or empty audit vector')
    return {'count':len(values),'min':min(values),'q10':quantile(values,.1),'median':quantile(values,.5),'q90':quantile(values,.9),
            'max':max(values),'mean':statistics.fmean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None}


def response(a,b,length):
    need(length in (1,2,3),'Boundary length must be 1,2,3')
    h=[1.] if length==1 else [1+a,-a] if length==2 else [1+a,b-a,-b]
    nyquist=1. if length==1 else 1+2*a if length==2 else 1+2*a-2*b
    dc=math.fsum(h);actual=math.fsum(v*(-1.)**i for i,v in enumerate(h))
    need(math.isclose(dc,1.,rel_tol=0.,abs_tol=1e-12),'DC identity failed')
    need(math.isclose(actual,nyquist,rel_tol=1e-12,abs_tol=1e-12),'Nyquist identity failed')
    return {'coefficients':h,'dc':dc,'nyquist_signed':nyquist,'nyquist_amplitude':abs(nyquist)}


def synthetic_checks():
    # Fixed rational coefficients/inputs chosen without inspecting trained state.
    cases=[(0.,0.),(.25,-.5),(-.75,.125),(2.,-3.),(-2.,4.)]
    vectors=[(7.,7.,7.),(1.,-1.,1.),(2.5,-4.,9.),(-6.,.5,2.)]
    checks=0
    for a,b in cases:
        for n in (1,2,3):
            h=response(a,b,n)['coefficients']
            for v in vectors:
                direct=v[0]+(a*(v[0]-v[1]) if n>=2 else 0)+(b*(v[1]-v[2]) if n>=3 else 0)
                expanded=math.fsum(x*y for x,y in zip(h,v))
                need(math.isclose(direct,expanded,rel_tol=1e-12,abs_tol=1e-12),'Expanded difference formula differs');checks+=1
            need(math.isclose(math.fsum(x*7 for x in h),7.,rel_tol=0.,abs_tol=1e-12),'Constant latent is not preserved');checks+=1
    need(response(.25,-.5,3)['nyquist_signed']==2.5,'Known full Nyquist value differs');checks+=1
    need(response(.25,-.5,2)['nyquist_signed']==1.5,'Known 2-observation Nyquist differs');checks+=1
    need(response(2.,-3.,1)=={'coefficients':[1.],'dc':1.,'nyquist_signed':1.,'nyquist_amplitude':1.},'Single-observation identity differs');checks+=1
    need(summary([1,2,3])['mean']==2 and summary([1,2,3])['sample_sd']==1,'Descriptive summary differs');checks+=1
    return {'status':'passed','arithmetic_cases':checks,'fixed_cases':cases,'model_forward_calls':0}


def state_hash(state,torch):
    digest=hashlib.sha256()
    for name in sorted(state):
        t=state[name];need(isinstance(t,torch.Tensor),'State contains non-tensor');t=t.detach().cpu().contiguous()
        meta=json.dumps({'name':name,'dtype':str(t.dtype),'shape':list(t.shape)},sort_keys=True,separators=(',',':')).encode()
        raw=t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b''
        digest.update(len(meta).to_bytes(8,'big'));digest.update(meta);digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    return digest.hexdigest()


def strict_epoch(history):
    rows=history.get('history') if isinstance(history,dict) else history
    need(isinstance(rows,list),'Selection history layout differs')
    best=None
    for row in rows:
        value=row.get('val_qty_rmse');epoch=row.get('epoch')
        if type(value) in (float,int) and math.isfinite(value) and (best is None or value<best[1]):best=(epoch,value)
    return best


def main():
    need(not any((OUT/n).exists() for n in ('RESULT.json','RESULT.md','RECEIPT.json')),'Existing diagnostic result; no overwrite')
    checks=synthetic_checks()
    c=read(pin(BUNDLE/'contract.json',CONTRACT_SHA));boot=read(pin(BUNDLE/'bootstrap_integrity.json',BOOTSTRAP_SHA))
    need(c['contract_sha256']==canon({k:v for k,v in c.items() if k!='contract_sha256'}) and boot['contract_raw_sha256']==CONTRACT_SHA,'Contract binding differs')
    need(c['source']['closure']==SOURCE_CLOSURE==canon(c['source']['files']) and len(c['source']['files'])==141,'Source closure differs')
    need(c['allowed_splits']==['train','validation'] and c['held_out_test_evaluated'] is False and c['historical_CNN_inference'] is False,'Diagnostic scope differs')
    for rel in ('models/TPPs/CountAwareTitanValueDifferenceCNN.py','simple_lab_test/search/common/runner.py'):
        pin(BUNDLE/c['source']['root']/rel,c['source']['files'][rel])
    need(len(c['jobs'])==3 and {j['seed'] for j in c['jobs']}==set(EPOCHS),'Candidate cohort differs')
    bound=[]
    for job in c['jobs']:
        s=job['seed'];need(job['arm']=='titantpp_cnn_value_difference_v1' and job['selected_epoch']==EPOCHS[s] and job['checkpoint']['sha256']==BINARY[s]
                         and job['checkpoint_state_sha256']==STATES[s],'Frozen selected candidate changed')
        bp=pin(BUNDLE/job['checkpoint']['relative_path'],BINARY[s]);meta=read(pin(BUNDLE/job['checkpoint_metadata_path'],job['checkpoint_metadata_sha256']))
        hist=read(pin(BUNDLE/job['selection_history_path'],job['selection_history_sha256']))
        need(strict_epoch(hist)==(EPOCHS[s],job['selected_metric_value']),'Strict finite Validation selection differs')
        need(meta['best_epoch']==EPOCHS[s] and meta['model_state_sha256']==STATES[s] and meta['seed']==s and meta['held_out_test_evaluated'] is False,'Metadata state/selection differs')
        bound.append((job,bp))
    old=read(pin(CNN_SUMMARY,CNN_SUMMARY_SHA));need(old['model_forward_calls']==old['remote_calls']==0 and old['held_out_test_read'] is False,'Stored CNN summary scope differs')
    input_before=dict(INPUTS);reader_file_sha=sha(__file__)
    import torch  # Tensor deserialization only. No project model/runner import.
    rows=[]
    for job,path in bound:
        payload=torch.load(path,map_location='cpu',weights_only=True);state=payload['model_state_dict']
        need(state_hash(state,torch)==job['checkpoint_state_sha256'],'Canonical selected tensor SHA differs')
        need(STATE_KEY in state and not any('causal_v_kernel' in k for k in state),'Wrong ValueDifference state')
        tensor=state[STATE_KEY];need(tuple(tensor.shape)==(2,64) and tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()),'Wrong coefficient tensor')
        coefficients=tensor.detach().cpu().to(torch.float64).tolist();channels=[]
        for i,(a,b) in enumerate(zip(*coefficients)):
            channels.append({'channel':i,'a':a,'b':b,'full_history':response(a,b,3),'history_two':response(a,b,2),'history_one':response(a,b,1)})
        summaries={'a':summary(coefficients[0]),'b':summary(coefficients[1])}
        for label in ('full_history','history_two','history_one'):
            summaries[label]={field:summary(x[label][field] for x in channels) for field in ('dc','nyquist_signed','nyquist_amplitude')}
        rows.append({'seed':job['seed'],'selected_epoch':job['selected_epoch'],'binary_path':str(path.relative_to(ROOT)),'binary_sha256':job['checkpoint']['sha256'],
                     'state_sha256':job['checkpoint_state_sha256'],'tensor_key':STATE_KEY,'tensor_shape':[2,64],'tensor_dtype':str(tensor.dtype),
                     'tensor_raw_sha256':hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                     'channels':channels,'summary':summaries})
        need(state_hash(state,torch)==job['checkpoint_state_sha256'],'In-memory selected state changed');del payload,state,tensor
    old_v=[{'seed':x['seed'],'selected_epoch':x['selected_epoch'],'full_history_dc_summary':x['summary']['total_constant_gain_1_plus_sum'],
            'one_observation_residual_sum_summary':x['prefix_sum_for_one_available_observation'],'two_observation_residual_sum_summary':x['prefix_sum_for_two_available_observations'],
            'state_sha256_reported_by_prior_audit':x['state_sha256'],'tensor_raw_sha256_reported_by_prior_audit':x['tensor_raw_sha256']} for x in old['per_seed_branch'] if x['branch']=='V']
    need(len(old_v)==3 and {x['seed'] for x in old_v}==set(EPOCHS),'Stored CNN V summary cohort differs')
    for rel,h in input_before.items():need(sha(ROOT/rel)==h,'Input bytes changed during audit')
    need(sha(__file__)==reader_file_sha,'Audit code changed during execution')
    result={'schema':'current_value_difference_filter_response_v1','created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'method':METHOD,'method_sha256':canon(METHOD),
            'per_seed':rows,'stored_CNN_V_reference':old_v,'stored_CNN_summary_sha256':CNN_SUMMARY_SHA,
            'comparison_limits':{'different_structure':'Old CNN independent 3-tap residual h=[1+k0,k1,k2], partial tap sums at short boundaries; current difference requires available pairs and is identity with one observation.',
             'hash_domains':'Checkpoint raw bytes, serialization-independent named-state tensor SHA, and standalone coefficient raw bytes are distinct domains. Current kernel shape2x64 vs old3x64; no equality or channel alignment is implied.',
             'native_training_runtime':'Current5090/cuDNN92000; storedCNN5080/cuDNN91900. Local tensor reading does not recreate either Runtime.',
             'old_CNN_parameters_reread':False},
            'source_closure':SOURCE_CLOSURE,'training_source_revision':c['source']['revision'],'local_tensor_reader_torch':torch.__version__,
            'mathematical_synthetic_checks':checks,'all_selected_states_unchanged':True,'before_after_input_sha_equal':True,
            'model_class_constructions':0,'model_forward_calls':0,'remote_calls':0,'old_CNN_checkpoint_reads':0,'held_out_test_read':False,
            'activation_weighted_effect_measured':False,'gain_threshold_selected':False,'next_training_admission_selected':False,
            'next_evidence_needed':'Separate branch-mechanism and activation/paired-error diagnostic; this appendix alone cannot select a new training candidate.'}
    write(OUT/'RESULT.json',result)
    md=['# 현재 V 차분 CNN 선택계수의 응답 감사','', '현재 선택 epoch는 seed42=9, seed52=5, seed62=22로 고정했다. 모델을 만들거나 예측을 계산하지 않고 state tensor만 읽었다.','',
        '| seed | a 평균 | b 평균 | 관측3 이상 Nyquist 평균 [최소, 최대] | 관측2 Nyquist 평균 [최소, 최대] | DC(관측1/2/3+) |', '|---|---:|---:|---|---|---|']
    for x in rows:
        z=x['summary'];f=z['full_history']['nyquist_signed'];t=z['history_two']['nyquist_signed']
        md.append(f"| {x['seed']} | {z['a']['mean']:.6f} | {z['b']['mean']:.6f} | {f['mean']:.6f} [{f['min']:.6f}, {f['max']:.6f}] | {t['mean']:.6f} [{t['min']:.6f}, {t['max']:.6f}] | 1 / 1 / 1 |")
    md += ['', '여기서 Nyquist는 관측 순번에 따른 교대 입력 [1, −1, 1]의 부호를 포함한 응답이다. 실제 시간 간격이나 실수량의 변화를 뜻하지 않는다. 관측2의 값은 짧은 이력 경계에서 계수 다항식을 계산한 값이며, 전체 모델의 정상상태 주파수 응답이 아니다.','',
       '`h3=[1+a,b−a,−b]`, `h2=[1+a,−a]`, `h1=[1]`이므로 상수 잠재 V의 DC는 모두1이다. 기존 CNN의 저장된 V DC 평균은 seed42/52/62에서 각각 '+', '.join(f"{x['full_history_dc_summary']['mean']:.6f}" for x in old_v)+'였다. 기존 CNN checkpoint는 다시 읽지 않았다.','',
       '각 채널의 a·b와 세 경계의 전체 계수·DC·Nyquist, 부호 없는 응답 크기 및 기술통계는 RESULT.json에 기록했다. 채널별 표현의 좌표가 서로 다른 학습 모델 사이에서 맞춰져 있지 않으므로 계수 변화만으로 출력 효과를 직접 비교할 수 없다.','',
       '필터 계수의 gain은 raw 수량 오류, 실제 활성값으로 가중한 효과, 또는 실패 원인이 아니다. 상수 잠재 V 보존도 최종 수량 편향 제거를 보장하지 않는다. 임의 gain 문턱이나 새 학습 진입 기준을 정하지 않았다. 다음 후보 선정에는 별도의 분기 기전 진단이 필요하다.','',
       '원본 binary·선택 metadata/history·계약 및 기존 CNN 요약의 SHA를 처리 전후 확인했고 모두 같았다. 학습·모델 생성·forward·원격·Test 접근은0이다. 로컬 torch는 '+str(torch.__version__)+', 학습 Runtime은 후보5090/cuDNN92000과 기존CNN5080/cuDNN91900으로 다르다.','',
       f"고정 수학 합성 검사 {checks['arithmetic_cases']}개 통과. 전체 이름·dtype·shape·bytes를 포함한 state SHA와 개별 계수 raw SHA, checkpoint 파일 SHA는 서로 다른 hash domain으로 기록했다."]
    with (OUT/'RESULT.md').open('x') as f:f.write('\n'.join(md)+'\n')
    outputs={n:sha(OUT/n) for n in ('analyze.py','RESULT.json','RESULT.md')}
    receipt={'status':'complete','inputs_before':input_before,'inputs_after':{rel:sha(ROOT/rel) for rel in input_before},'output_files':outputs,
             'new_checkpoint_tensor_reads':3,'old_CNN_checkpoint_reads':0,'model_class_constructions':0,'model_forward_calls':0,'remote_calls':0,'held_out_test_read':False,
             'synthetic_arithmetic_cases_passed':checks['arithmetic_cases'],'state_sha_verified_count':3,'before_after_identical':True,'method_sha256':canon(METHOD)}
    write(OUT/'RECEIPT.json',receipt)
    print(json.dumps({'status':'complete','output_sha256':{**outputs,'RECEIPT.json':sha(OUT/'RECEIPT.json')},'summaries':[{'seed':x['seed'],'a_mean':x['summary']['a']['mean'],'b_mean':x['summary']['b']['mean'],'nyquist_full':x['summary']['full_history']['nyquist_signed'],'nyquist_two':x['summary']['history_two']['nyquist_signed']} for x in rows]},ensure_ascii=False))


if __name__=='__main__':main()
