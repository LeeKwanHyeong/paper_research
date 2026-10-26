import hashlib,json,pathlib,shutil
P=pathlib.Path(__file__).resolve().parent;old=P/'execution_v4';E=P/'execution_v5_seed52_5080'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def remap(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [remap(v,a,b) for v in x]
 if isinstance(x,dict):return {k:remap(v,a,b) for k,v in x.items()}
 return x
E.mkdir();shutil.copytree(old/'bundle',E/'bundle');shutil.copy2(old/'collector_v3.py',E/'collector_v3.py')
for name in ['remote_control.py','control_transport.py','test_remote_control.py','test_transport.py']:
 with (E/name).open('x') as f:f.write((old/name).read_text().replace('_SPEC_v4.json','_SPEC_v5.json'))
g=json.loads((old/'5080_GUARD.json').read_text());newroot=g['remote_root'].replace('_execution_v4','_seed52_execution_v5');g=remap(g,g['remote_root'],newroot);g['process_environment']['SOURCE_REVISION']='5d8ee1d64601a67d5ae0d0b4b97dd5547b63ebdd';write(E/'5080_GUARD.json',g)
previousroot=json.loads((old/'5090_GUARD.json').read_text())['remote_root']
for mode in ['QUALIFICATION','FULL']:
 op=old/('5090_'+mode+'_SPEC_v4.json');s=remap(json.loads(op.read_text()),previousroot,newroot)
 s.update(expected_device_name='NVIDIA GeForce RTX 5080',expected_device_uuid=g['gpu_uuid'],process_environment=g['process_environment'],qualification_batches=16,operational_revision=dict(previous_spec_sha256=sha(op),reason='Observe identical seed52 checkpoints on 5080 after 5090 cosine-only parity gate failed. No tolerance/model/input changes; no mixed-device new observation cohorts. Qualification uses all 2048 existing reference rows per split.',failed_attempt_preserved='failed_gpu_qualification_v4/5090'))
 write(E/('5080_'+mode+'_SPEC_v5.json'),s)
write(E/'REVISION.json',dict(actual_observation_host='5080',seeds=[52],collector_sha256=sha(E/'collector_v3.py'),no_retraining=True,scientific_tolerances_unchanged=True,do_not_overlap_seed42=True))
print(str(E))
