import hashlib,json,pathlib,shutil
P=pathlib.Path(__file__).resolve().parent;old=P/'execution_v6';E=P/'execution_v8'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def remap(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [remap(v,a,b) for v in x]
 if isinstance(x,dict):return {k:remap(v,a,b) for k,v in x.items()}
 return x
approval=P/'USER_LOG_MSE_GATE_APPROVAL.json';normalized=P/'LOG_MSE_GATE_AUTHORIZATION_V8.json'
assert json.loads(approval.read_text())['approved']
assert (P/'collector_v5.py').is_file()
E.mkdir();shutil.copytree(old/'bundle',E/'bundle');shutil.copy2(P/'collector_v5.py',E/'collector_v5.py')
for path in [approval,normalized,old/'USER_NUMERIC_GATE_APPROVAL.json',old/'NUMERIC_GATE_AUTHORIZATION_V6.json']:shutil.copy2(path,E/path.name)
for name in ['remote_control.py','control_transport.py','test_remote_control.py','test_transport.py']:
 text=(old/name).read_text().replace('_SPEC_v6.json','_SPEC_v8.json').replace('collector_v4.py','collector_v5.py')
 if name=='control_transport.py':
  text=text.replace("files += [P/'USER_NUMERIC_GATE_APPROVAL.json'", "files += [P/'USER_LOG_MSE_GATE_APPROVAL.json',P/'LOG_MSE_GATE_AUTHORIZATION_V8.json',P/'USER_NUMERIC_GATE_APPROVAL.json'")
 with (E/name).open('x') as f:f.write(text)
host='5090';g=json.loads((old/(host+'_GUARD.json')).read_text());previous=g['remote_root'];current=previous.replace('_execution_v6','_execution_v8');g=remap(g,previous,current);write(E/(host+'_GUARD.json'),g)
for mode in ['QUALIFICATION','FULL']:
 op=old/(host+'_'+mode+'_SPEC_v6.json');s=remap(json.loads(op.read_text()),previous,current)
 s['comparison_tolerances']['log_qty_mse']['atol']=1e-6
 s.update(runner_sha256=sha(E/'collector_v5.py'),runner_filename='collector_v5.py',log_mse_gate_revision=dict(field='log_qty_mse.atol',previous=1e-7,approved=1e-6,rtol=1e-5,authorization=dict(path=current+'/'+normalized.name,bytes=normalized.stat().st_size,sha256=sha(normalized))),operational_revision=dict(previous_spec_sha256=sha(op),reason='Explicit additional user-approved log-MSE-only numerical gate revision; retain approved cosine gate and all other gates; seed52 missing cohorts only',previous_failures_preserved=True))
 write(E/(host+'_'+mode+'_SPEC_v8.json'),s)
write(E/'REVISION.json',dict(collector_sha256=sha(E/'collector_v5.py'),approval_sha256=sha(approval),remaining_rows=873933,reuse_verified_rows=1046503,previous_attempts_overwritten=False))
print(str(E))
