"""Local-only preparation of a new immutable remote inference sandbox."""
from pathlib import Path
import importlib.util,json,hashlib,shutil,tarfile
HERE=Path(__file__).resolve().parent
P=HERE.parent/'phen_intermittent_full_observations_20261026_v1'
def rec(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    spec=importlib.util.spec_from_file_location('prepare_hidden_spec',HERE/'extraction/prepare_spec.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    refs=HERE/'extraction/references_v1/REFERENCES.json'
    manifest=json.loads(refs.read_text())
    for host,seed,v in [('5090',52,8),('5080',42,6)]:
        stage=HERE/'execution_v1'/host/'stage';stage.mkdir(parents=True,exist_ok=False)
        remote=f'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/phen_intermittent_time_head_control_20261026_v1_{host}_hidden_v1'
        result,context=m.prepare(seed,remote,HERE/'SCIENCE_SPEC.json',remote+'/SCIENCE_SPEC.json',refs,'qualification')
        for name,src in [('remote_control.py',HERE/'remote_control.py'),('collector_hidden.py',HERE/'extraction/collector_hidden.py'),('SCIENCE_SPEC.json',HERE/'SCIENCE_SPEC.json')]:shutil.copyfile(src,stage/name)
        (stage/'references').mkdir()
        for r in manifest['references']:
            if r['seed']==seed:
                src=Path(r['reference']['path']);assert rec(src)=={k:r['reference'][k] for k in ['bytes','sha256']}
                shutil.copyfile(src,stage/'references'/src.name)
        (stage/(host+'_QUALIFICATION_SPEC.json')).write_text(json.dumps(result,indent=2)+'\n')
        old=json.loads((P/f'execution_v{v}/{host}_GUARD.json').read_text())
        fresh=json.loads((HERE/(host+'_PREFLIGHT_01.parsed.json')).read_text())
        assert fresh['gpu']['actual_exit']==fresh['gpu_apps']['actual_exit']==0
        assert old['protected']==fresh['protected'],'Protected task changed from prior verified collection'
        guard={**old,'remote_root':remote,'boot_id':fresh['boot_id'],'protected':fresh['protected'],
               'gpu_uuid':fresh['gpu']['stdout'].split(',')[1].strip(),'process_environment':context['process_environment']}
        (stage/(host+'_GUARD.json')).write_text(json.dumps(guard,indent=2)+'\n')
        for d in ['cache/matplotlib','cache/torchinductor']:(stage/d).mkdir(parents=True,exist_ok=True)
        files={str(p.relative_to(stage)):rec(p) for p in stage.rglob('*') if p.is_file()}
        (stage/'DEPLOY_MANIFEST.json').write_text(json.dumps(dict(files=files,empty_directories=['cache/matplotlib','cache/torchinductor']),indent=2)+'\n')
        tarpath=stage.parent/'deploy.tar.gz'
        with tarfile.open(tarpath,'x:gz') as t:
            for p in sorted(stage.rglob('*')):t.add(p,arcname=str(p.relative_to(stage)),recursive=False)
        (stage.parent/'PREPARED.json').write_text(json.dumps(dict(remote_root=remote,host=host,seed=seed,python=guard['python'],archive=rec(tarpath),manifest=rec(stage/'DEPLOY_MANIFEST.json'),qualification_spec=rec(stage/(host+'_QUALIFICATION_SPEC.json'))),indent=2)+'\n')
        print(host,json.dumps(rec(tarpath)),len(files),flush=True)
if __name__=='__main__':main()
