"""Immutable current QueryOnly/KeyOnly tensors and stored Validation summaries.
No project model import/construction, forward, optimizer, remote or held-out read.
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
PLAN_SHA='d554c02a5d55ff848ca2c1b077d93117ca602a01559cf0d4511de3c592933ceb'
BUNDLE=ROOT/'search_artifacts/titantpp_backbone_qk_location_raf6_20261016_v1'
OLD=ROOT/'reports/titantpp_backbone_location_raf6_20261015_v1/diagnostics_dc_v1/RESULT.json'
VALIDATION=OUT.parent/'THREE_SEED_VALIDATION.json'
ARMS={'titantpp_cnn_query_only_v1':'QueryOnly','titantpp_cnn_key_only_v1':'KeyOnly'}
METRICS={'RMSE':(None,'qty_rmse'),'MAE':(None,'qty_mae'),'Large_RMSE':('tail','qty_rmse'),
 'Large_MAE':('tail','qty_mae'),'TimeNLL':(None,'time_nll'),'Large_TimeNLL':('tail','time_nll')}

def need(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,x):
    with Path(path).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
def quantile(v,q):
    s=sorted(v);i=(len(s)-1)*q;lo=math.floor(i);hi=math.ceil(i)
    return s[lo]+(s[hi]-s[lo])*(i-lo)
def summary(v):
    v=list(map(float,v));need(v and all(map(math.isfinite,v)),'Empty/nonfinite descriptive values')
    return {'count':len(v),'min':min(v),'q10':quantile(v,.1),'median':quantile(v,.5),'q90':quantile(v,.9),'max':max(v),
            'mean':statistics.fmean(v),'sample_sd':statistics.stdev(v) if len(v)>1 else None}
def stats(v):return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)}
def domain_counts(v):
    return {'negative':sum(x<0 for x in v),'zero':sum(x==0 for x in v),'between_zero_and_one':sum(0<x<1 for x in v),
            'one':sum(x==1 for x in v),'above_one':sum(x>1 for x in v)}
def response(w,n):
    need(n in (1,2,3) and len(w)==3,'Invalid tap availability')
    r=list(w[:n]);h=r.copy();h[0]+=1.
    freq=[]
    for omega in (0.,math.pi/2,math.pi):
        real=math.fsum(x*math.cos(omega*i) for i,x in enumerate(h))
        imag=math.fsum(-x*math.sin(omega*i) for i,x in enumerate(h))
        freq.append({'omega_radians':omega,'real':real,'imaginary':imag,'amplitude':math.hypot(real,imag)})
    dc=math.fsum(h);nyquist=math.fsum(x*((-1.)**i) for i,x in enumerate(h))
    need(close(freq[0]['real'],dc) and close(freq[2]['real'],nyquist),'Frequency endpoint arithmetic differs')
    return {'total_coefficients':h,'dc_gain':dc,'nyquist_signed':nyquist,'nyquist_amplitude':abs(nyquist),
      'residual_l1':math.fsum(abs(x) for x in r),'residual_l2':math.sqrt(math.fsum(x*x for x in r)),
      'total_l1':math.fsum(abs(x) for x in h),'total_l2':math.sqrt(math.fsum(x*x for x in h)),'frequency_samples':freq}
def decomposition(w):
    level=statistics.fmean(w);detail=[x-level for x in w]
    total=math.fsum(x*x for x in w);level_energy=3*level*level;detail_energy=math.fsum(x*x for x in detail)
    need(close(total,level_energy+detail_energy) and close(math.fsum(detail),0.),'Orthogonal tap decomposition failed')
    return {'level_component_coefficients':[level]*3,'zero_sum_component_coefficients':detail,
            'residual_energy':total,'level_component_energy':level_energy,'zero_sum_component_energy':detail_energy,
            'level_energy_fraction':level_energy/total if total else None}
def synthetic():
    weights=[(0.,0.,0.),(.25,-.5,.125),(-.75,.25,.5),(2.,-3.,4.),(-2.,4.,-1.)]
    vectors=[(7.,7.,7.),(1.,-1.,1.),(2.5,-4.,9.),(-6.,.5,2.)];count=0
    for w in weights:
        decomposition(w);count+=1
        for n in (1,2,3):
            r=response(w,n);h=r['total_coefficients']
            for x in vectors:
                need(close(x[0]+math.fsum(w[i]*x[i] for i in range(n)),math.fsum(h[i]*x[i] for i in range(n))),'Direct versus expanded residual differs');count+=1
            need(close(math.fsum(x*7 for x in h),7*r['dc_gain']),'Constant response failed');count+=1
            need(close(math.fsum(x*(-1.)**i for i,x in enumerate(h)),r['nyquist_signed']),'Alternating response failed');count+=1
    need(response((.25,-.5,.125),1)['dc_gain']==1.25,'One-observation response incorrectly forced to identity');count+=1
    need(response((.25,-.5,.125),3)['nyquist_signed']==1.875,'Known alternating response differs');count+=1
    need(summary([1,2,3])['sample_sd']==1. and summary([1,2,3])['mean']==2.,'Summary arithmetic failed');count+=1
    return {'status':'PASS','fixed_arithmetic_checks':count,'model_constructions':0,'model_forward_calls':0}
def state_hash(state,torch):
    h=hashlib.sha256()
    for name in sorted(state):
        t=state[name];need(isinstance(t,torch.Tensor),'Non-tensor selected state')
        t=t.detach().cpu().contiguous();m=json.dumps({'name':name,'dtype':str(t.dtype),'shape':list(t.shape)},sort_keys=True,separators=(',',':')).encode()
        raw=t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b''
        h.update(len(m).to_bytes(8,'big'));h.update(m);h.update(len(raw).to_bytes(8,'big'));h.update(raw)
    return h.hexdigest()
def first_best(x):
    h=x['history'];need([v['epoch'] for v in h]==list(range(1,len(h)+1)),'History epoch order differs')
    finite=[v for v in h if type(v.get('val_qty_rmse')) in (int,float) and math.isfinite(v['val_qty_rmse'])]
    need(finite,'No finite selection metric');v=min(finite,key=lambda r:(r['val_qty_rmse'],r['epoch']))
    return v['epoch'],v['val_qty_rmse']
def verify_validation(v):
    need(v['split']=='validation' and v['held_out_test_evaluated'] is False and v['new_fit_count']==6 and v['baseline_reference_count']==3,'Stored Validation scope differs')
    indexed={(x['arm'],x['seed']):x for x in v['rows']};expected={(arm,s) for arm in ('CNN',*ARMS.values()) for s in (42,52,62)}
    need(len(v['rows'])==len(indexed)==9 and set(indexed)==expected,'Stored Validation cohort differs')
    count=0
    for arm in ('CNN',*ARMS.values()):
        for metric in METRICS:
            z=stats([indexed[(arm,s)][metric] for s in (42,52,62)])
            need(all(close(z[k],v['aggregate'][arm][metric][k]) for k in z),'Validation aggregate differs');count+=1
    contrasts={x['contrast'] for x in v['seed_differences']};need(contrasts=={'QueryOnly−CNN','KeyOnly−CNN','KeyOnly−QueryOnly'} and len(v['seed_differences'])==9,'Validation contrasts differ')
    for row in v['seed_differences']:
        left,right=row['contrast'].split('−');s=row['seed']
        for metric in METRICS:
            l,r=indexed[(left,s)][metric],indexed[(right,s)][metric]
            need(close(l-r,row[metric]['absolute']) and close(100*(l-r)/r,row[metric]['relative_percent']),'Validation seed difference differs');count+=1
    return indexed,count

def main():
    need(not any((OUT/n).exists() for n in ('RESULT.json','RESULT.md','RECEIPT.json')),'Existing result; no overwrite')
    need(sha(OUT/'PLAN.json')==PLAN_SHA,'Frozen analysis plan changed');plan=read(OUT/'PLAN.json')
    inputs=dict(plan['input_files']);inputs[str((OUT/'PLAN.json').relative_to(ROOT))]=PLAN_SHA
    c=read(BUNDLE/'execution_contract.json');r=read(BUNDLE/'retrieval/final_receipt.json')
    need(r['status']=='complete' and r['new_fit_count']==6 and r['held_out_test_evaluated'] is False and r['all_training_terminal_verified'] is True,'Training receipt incomplete')
    need(c['source']['files_sha256']==canon(c['source']['files'])==plan['source_closure'] and len(c['source']['files'])==145,'Scientific source closure differs')
    for name,digest in c['source']['files'].items():inputs[str((BUNDLE/'source'/name).relative_to(ROOT))]=digest
    for name,digest in inputs.items():need(sha(ROOT/name)==digest,'Input SHA differs: '+name)
    script_before=sha(__file__);checks=synthetic();v=read(VALIDATION);indexed,nchecks=verify_validation(v)
    old=read(OLD);need(old['model_forward_calls']==old['remote_calls']==0 and old['held_out_test_read'] is False,'Old summary scope differs')
    old_rows=[x for x in old['per_seed_branch'] if x['branch'] in ('Q','K')]
    need(len(old_rows)==6,'Stored CNN Q/K summaries incomplete')
    import torch # Tensor deserialization only; project models are never imported.
    rows=[];actual_checks=0
    for j in plan['jobs']:
        p=ROOT/j['checkpoint']['path'];payload=torch.load(p,map_location='cpu',weights_only=True);state=payload['model_state_dict']
        endpoint=read(ROOT/j['endpoint']['path'])['selected'];h=read(ROOT/j['history']['path'])
        need(first_best(h)==(j['selected_epoch'],payload['selected_metric_value']),'Validation selection differs')
        need(payload['best_epoch']==j['selected_epoch'] and payload['seed']==j['seed'] and payload['backbone']==j['arm'] and payload['held_out_test_evaluated'] is False,'Selected payload identity differs')
        digest=state_hash(state,torch);need(digest==payload['model_state_sha256']==j['selected_state_sha256']==endpoint['state_sha256'],'Selected tensor SHA differs')
        meta=payload['encoder_config'];loc=j['retained_projection'];key='encoder.layers.0.attn.causal_'+loc+'_kernel'
        need(meta['backbone_contract_id']=='titantpp_qk_location_cnn_20261016_v1' and meta['qk_location_arm']==j['arm'] and meta['retained_projection']==loc
             and meta['active_local_cnn_parameters']==192 and meta['active_value_cnn_parameters']==0,'Selected filter metadata differs')
        need({k for k in state if k.startswith('encoder.layers.0.attn.causal_')}=={key},'Unexpected additional local filter')
        tensor=state[key];need(tuple(tensor.shape)==(3,64) and tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()),'Coefficient tensor differs')
        coeff=tensor.detach().cpu().to(torch.float64).tolist();channels=[]
        for ch,w in enumerate(zip(*coeff)):
            item={'channel':ch,'w0':w[0],'w1':w[1],'w2':w[2],'tap_decomposition':decomposition(w)}
            for n in (1,2,3):item['history_'+str(n)]=response(w,n)
            channels.append(item);actual_checks+=4
        summaries={name:summary(coeff[i]) for i,name in enumerate(('w0','w1','w2'))}
        for n in (1,2,3):
            name='history_'+str(n);summaries[name]={k:summary(x[name][k] for x in channels) for k in ('dc_gain','nyquist_signed','nyquist_amplitude','residual_l1','residual_l2','total_l1','total_l2')}
            summaries[name]['dc_domain_counts']=domain_counts([x[name]['dc_gain'] for x in channels])
            summaries[name]['nyquist_domain_counts']=domain_counts([x[name]['nyquist_signed'] for x in channels])
        decomp={k:summary(x['tap_decomposition'][k] for x in channels) for k in ('residual_energy','level_component_energy','zero_sum_component_energy')}
        fractions=[x['tap_decomposition']['level_energy_fraction'] for x in channels if x['tap_decomposition']['level_energy_fraction'] is not None]
        decomp['nonzero_channel_level_energy_fraction']=summary(fractions) if fractions else None
        decomp['exact_zero_residual_channels']=64-len(fractions)
        validation=indexed[(ARMS[j['arm']],j['seed'])];need(validation['best_epoch']==j['selected_epoch'],'Stored Validation selected epoch differs')
        for metric,(group,k) in METRICS.items():need(close(validation[metric],(endpoint if group is None else endpoint[group])[k]),'Stored Validation endpoint differs')
        old_row=next(x for x in old_rows if x['seed']==j['seed'] and x['branch']==loc.upper())
        rows.append({'arm':j['arm'],'label':ARMS[j['arm']],'seed':j['seed'],'selected_epoch':j['selected_epoch'],'retained_projection':loc,
            'checkpoint_path':j['checkpoint']['path'],'checkpoint_sha256':j['checkpoint']['sha256'],'state_sha256':digest,
            'tensor_key':key,'tensor_shape':[3,64],'tensor_dtype':str(tensor.dtype),
            'tensor_raw_sha256':hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            'channels':channels,'summary':summaries,'tap_decomposition_summary':decomp,'stored_validation':validation,
            'validation_minus_CNN':next(x for x in v['seed_differences'] if x['seed']==j['seed'] and x['contrast']==ARMS[j['arm']]+'−CNN'),
            'stored_CNN_same_branch':{'seed':old_row['seed'],'branch':old_row['branch'],'selected_epoch':old_row['selected_epoch'],
                'summary':old_row['summary'],'prefix_sum_for_one_available_observation':old_row['prefix_sum_for_one_available_observation'],
                'prefix_sum_for_two_available_observations':old_row['prefix_sum_for_two_available_observations'],
                'prior_tensor_sha256':old_row['tensor_raw_sha256'],'prior_state_sha256':old_row['state_sha256']}})
        need(state_hash(state,torch)==digest,'Selected state modified in memory');del payload,state,tensor
    for name,digest in inputs.items():need(sha(ROOT/name)==digest,'Input SHA changed during diagnostic')
    need(sha(__file__)==script_before,'Analysis script changed during diagnostic')
    arm_summary={}
    for arm,label in ARMS.items():
        selected=[x for x in rows if x['arm']==arm]
        arm_summary[label]={'mean_of_seed_dc_means':stats([x['summary']['history_3']['dc_gain']['mean'] for x in selected]),
            'mean_of_seed_nyquist_means':stats([x['summary']['history_3']['nyquist_signed']['mean'] for x in selected]),
            'validation':v['aggregate'][label],
            'RMSE_better_than_CNN_seeds':[x['seed'] for x in selected if x['validation_minus_CNN']['RMSE']['absolute']<0],
            'MAE_better_than_CNN_seeds':[x['seed'] for x in selected if x['validation_minus_CNN']['MAE']['absolute']<0]}
    result={'schema':'qk_location_selected_filter_diagnostic_result_v1','status':'complete','created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
      'plan_sha256':PLAN_SHA,'source_closure':plan['source_closure'],'source_revision':plan['source_revision'],'per_selected_model':rows,
      'arm_summaries':arm_summary,'stored_validation':{'split':'validation','aggregate':v['aggregate'],'seed_differences':v['seed_differences']},
      'fixed_arithmetic_validation':checks,'actual_channel_boundary_and_decomposition_checks':actual_checks,'stored_validation_recalculation_checks':nchecks,
      'interpretation':{'confirmed':'선택된 잠재 Q/K 계수의 경계별 선형 응답과 별도의 저장 Validation 수치다.',
        'unmeasured':['실제 Q/K 활성값 크기','event/memory attention 점수·질량','수량 예측 방향·개별 행 오류','DC 성분의 인과효과'],
        'next_mechanism_diagnostic':'현재 KeyOnly 선택모델에서 K 필터 전후 활성값·event 점수·memory 질량을 같은 대상의 Train/Validation 오류와 결속하는 진단이 우선이다. 필요하면 총 잔차를 평균tap 성분과 zero-sum 성분으로 나눈 개입을 별도 계약으로 검증한다. 본 감사는 새학습/구조 채택을 결정하지 않는다.',
        'no_new_model_selected':True,'no_gain_threshold_optimized':True,'no_causal_claim':True,'limitations':plan['limits']},
      'hash_domains':'checkpoint 파일SHA, 이름/dtype/shape/bytes로 만든 전체state SHA, 개별kernel rawbytesSHA는 서로 다른 영역이다.',
      'boundary_semantics':'관측 순서 tap; padding을 건너뛰고 valid withheld 경계에서 이력을 재설정한다. 짧은 이력에서도 현재tap w0는 남아 h1=[1+w0]이다. memory token은 localfilter를 우회한다.',
      'runtime_limit':'후보5090/cuDNN92000 대 기존CNN5080/cuDNN91900. 로컬 텐서 읽기는 native평가 재현이 아니다.',
      'local_tensor_reader_torch':str(torch.__version__),'model_class_constructions':0,'model_forward_calls':0,'remote_calls':0,'old_CNN_checkpoint_reads':0,
      'held_out_test_read':False,'new_training':False,'all_selected_states_unchanged':True,'before_after_input_sha_equal':True}
    write(OUT/'RESULT.json',result)
    md=['# Q-only/K-only 선택 필터 계수와 저장 Validation 감사','',
        '선택 checkpoint6개의 tensor만 읽었다. 모델 생성·forward·새 추론·학습·원격·Test 열람은 하지 않았다. 기존 CNN은 저장된 Q/K 계수 요약만 재사용했다.','',
        '| 구조 | seed | epoch | DC 평균 [최소, 최대] | 교대 응답 평균 [최소, 최대] | 잔차 L2 평균 | ΔRMSE(CNN 대비) | ΔMAE(CNN 대비) |',
        '|---|---:|---:|---|---|---:|---:|---:|']
    for x in rows:
        s=x['summary']['history_3'];d=s['dc_gain'];n=s['nyquist_signed'];v=x['validation_minus_CNN']
        md.append(f"| {x['label']} | {x['seed']} | {x['selected_epoch']} | {d['mean']:.6f} [{d['min']:.6f}, {d['max']:.6f}] | {n['mean']:.6f} [{n['min']:.6f}, {n['max']:.6f}] | {s['residual_l2']['mean']:.6f} | {v['RMSE']['absolute']:+.6f} | {v['MAE']['absolute']:+.6f} |")
    md+=['','DC는 상수 잠재값에 대한 배율, 교대 응답은 관측 순번 [1,−1,1]에 대한 부호 포함 응답이다. 둘 다 원수량 예측의 배율이나 실제 시간 주파수를 뜻하지 않는다.',
       '`h3=[1+w0,w1,w2]`, `h2=[1+w0,w1]`, `h1=[1+w0]`이다. 짧은 이력에서는 없는 tap만 빠지며 현재tap은 남는다. V차분 후보처럼 관측1개에서 반드시 항등인 구조가 아니다.','',
       '| 구조 | seed | DC<0 | 0≤DC<1 | DC=1 | DC>1 | 기존CNN 동일분기 DC 평균 |', '|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        d=x['summary']['history_3']['dc_domain_counts'];old=x['stored_CNN_same_branch']['summary']['total_constant_gain_1_plus_sum']['mean']
        md.append(f"| {x['label']} | {x['seed']} | {d['negative']} | {d['zero']+d['between_zero_and_one']} | {d['one']} | {d['above_one']} | {old:.6f} |")
    md+=['','각64채널의 원계수, 관측1/2/3 경계 응답, 고정0·π/2·π 응답, L1/L2, 평균tap 성분과 zero-sum 성분의 직교 에너지는 RESULT.json에 보존했다. 0과1은 선형 응답의 정의상 기준이며 성능 문턱으로 정하지 않았다.',
       'KeyOnly는 Validation RMSE가3seed 모두 기존CNN보다 낮지만 MAE 개선은seed42에만 있다. QueryOnly는 RMSE가3seed 모두 높고 MAE 개선은seed42에만 있다. 이 사실과 계수 통계를 병기했지만 여섯 모델 간 상관이나 인과관계를 추정하지 않았다.',
       '서로 독립 학습한 모델의 잠재 채널좌표는 정렬되지 않았다. 기존CNN은 Q/K/V를 모두 학습했고 후보는 한 분기만 학습했다. Runtime과 용량도 다르므로 같은 채널번호나 계수 평균만으로 효과를 직접 대조할 수 없다.',
       '다음 기전 진단은 현재 KeyOnly 선택모델의 필터 전후 K 활성값과 event 점수·memory 질량을 같은 대상의 Train/Validation 오류에 연결하는 범위로 좁힐 수 있다. 평균tap 성분과 zero-sum 성분의 별도 개입은 그 뒤 별도 계약으로 검토할 사항이다. 이 감사만으로 DC 제한이나 새학습 후보를 채택하지 않는다.',
       'Query는 memory key 점수에도 작용한다. Key는 이벤트 key만 바꾸지만 정규화 분모를 통해 memory의 할당 질량에도 영향을 줄 수 있다. 실제 attention·head 효과는 이번 작업에서 측정하지 않았다.','',
       f"고정 합성 수학 검사 {checks['fixed_arithmetic_checks']}개, 실제 채널 경계/직교분해 {actual_checks}개, 저장 Validation 집계·차이 {nchecks}개를 검산했다. 입력 {len(inputs)}개와 분석 코드는 처리 전후 SHA가 같고 선택state6개도 불변이다."]
    with (OUT/'RESULT.md').open('x') as f:f.write('\n'.join(md)+'\n')
    outputs={n:sha(OUT/n) for n in ('PLAN.json','analyze.py','RESULT.json','RESULT.md')}
    receipt={'status':'complete','inputs_before':inputs,'inputs_after':{n:sha(ROOT/n) for n in inputs},'output_files':outputs,
       'state_sha_verified_count':6,'current_checkpoint_tensor_reads':6,'old_CNN_checkpoint_reads':0,
       'synthetic_arithmetic_checks':checks['fixed_arithmetic_checks'],'actual_channel_checks':actual_checks,'validation_recalculation_checks':nchecks,
       'before_after_identical':True,'model_constructions':0,'model_forward_calls':0,'remote_calls':0,'held_out_test_read':False}
    write(OUT/'RECEIPT.json',receipt)
    print(json.dumps({'status':'complete','output_sha256':{**outputs,'RECEIPT.json':sha(OUT/'RECEIPT.json')},
         'summary':[{'arm':x['label'],'seed':x['seed'],'DC':x['summary']['history_3']['dc_gain'],'Nyquist':x['summary']['history_3']['nyquist_signed']} for x in rows]},ensure_ascii=False))
if __name__=='__main__':main()
