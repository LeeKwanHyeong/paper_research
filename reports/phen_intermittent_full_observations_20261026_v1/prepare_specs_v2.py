import hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
runner=P/'collector_v2.py';assert runner.is_file()
records={}
for old in [P/'CPU_QUALIFICATION_SPEC.json',*[P/(h+'_'+m+'_SPEC.json') for h in ['5080','5090'] for m in ['QUALIFICATION','FULL']]]:
    s=json.loads(old.read_text());s['runner_sha256']=sha(runner);s['runner_filename']='collector_v2.py'
    root=pathlib.Path(s['working_dir'])
    s['split_manifest_record']=dict(path=str(root/'inputs/data/intermittent_frozen_5000/split_manifest.json'),sha256='393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04')
    s['operational_revision']=dict(previous_spec_sha256=sha(old),reason='Fix isolated imports and restore identical missing input sidecar before model evaluation; no scientific setting or tolerance changes',failed_attempt_preserved='cpu_qualification_v1')
    dest=old.with_name(old.stem+'_v2.json')
    with dest.open('x') as f:json.dump(s,f,indent=2)
    records[dest.name]=sha(dest)
print(json.dumps(records))
