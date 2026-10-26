import hashlib,json,pathlib,shutil
P=pathlib.Path(__file__).resolve().parent;old=P/'execution_v4';E=P/'execution_v6'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def remap(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [remap(v,a,b) for v in x]
 if isinstance(x,dict):return {k:remap(v,a,b) for k,v in x.items()}
 return x
approval=P/'USER_NUMERIC_GATE_APPROVAL.json';original=json.loads(approval.read_text());assert original['approved']
normalized=P/'NUMERIC_GATE_AUTHORIZATION_V6.json'
write(normalized,dict(approved=True,field='topk_similarity.atol',previous=1e-7,approved_atol=1e-6,rtol=1e-5,user_instruction=original['answer'],user_question=original['question'],original_approval=dict(path=str(approval),bytes=approval.stat().st_size,sha256=sha(approval))))
E.mkdir();shutil.copytree(old/'bundle',E/'bundle');shutil.copy2(P/'collector_v4.py',E/'collector_v4.py')
for path in [approval,normalized]:shutil.copy2(path,E/path.name)
for name in ['remote_control.py','control_transport.py','test_remote_control.py','test_transport.py']:
 text=(old/name).read_text().replace('_SPEC_v4.json','_SPEC_v6.json').replace('collector_v3.py','collector_v4.py')
 if name=='control_transport.py':
  text=text.replace("files += [P/'collector_v4.py'", "files += [P/'USER_NUMERIC_GATE_APPROVAL.json',P/'NUMERIC_GATE_AUTHORIZATION_V6.json',P/'collector_v4.py'")
 with (E/name).open('x') as f:f.write(text)
for host in ['5080','5090']:
 g=json.loads((old/(host+'_GUARD.json')).read_text());previous=g['remote_root'];current=previous.replace('_execution_v4','_execution_v6');g=remap(g,previous,current);write(E/(host+'_GUARD.json'),g)
 for mode in ['QUALIFICATION','FULL']:
  op=old/(host+'_'+mode+'_SPEC_v4.json');s=remap(json.loads(op.read_text()),previous,current)
  if host=='5080':s['jobs']=[j for j in s['jobs'] if j['checkpoint_kind']=='last']
  s['comparison_tolerances']['topk_similarity']['atol']=1e-6
  s.update(runner_sha256=sha(E/'collector_v4.py'),runner_filename='collector_v4.py',qualification_batches=16,numeric_gate_revision=dict(field='topk_similarity.atol',previous=1e-7,approved=1e-6,rtol=1e-5,authorization=dict(path=current+'/'+normalized.name,bytes=normalized.stat().st_size,sha256=sha(normalized))),operational_revision=dict(previous_spec_sha256=sha(op),reason='Explicit user-approved cosine-only numerical gate revision; all other gates unchanged; collect only five missing cohorts',previous_failures_preserved=True))
  write(E/(host+'_'+mode+'_SPEC_v6.json'),s)
write(E/'REVISION.json',dict(collector_sha256=sha(E/'collector_v4.py'),approval_sha256=sha(approval),remaining_rows=1354042,reuse_verified_rows=566394,previous_attempts_overwritten=False))
print(str(E))
