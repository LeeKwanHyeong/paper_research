"""One bounded approved activation/observation/finalization round; no retry."""
from __future__ import annotations
import concurrent.futures
import json
from pathlib import Path
import sys
import time
_ROOT=Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:sys.path.insert(0,str(_ROOT))
from paper.scripts import titantpp_contribution_runtime as op
from paper.scripts.deploy_titantpp_contribution_campaigns import local_contract,activate_host
from paper.scripts.observe_titantpp_contribution_campaigns import observe
from paper.scripts.finalize_titantpp_contribution_campaigns import finalize
from paper.scripts.prepare_titantpp_contribution_campaigns import REPORT,write

def advance():
    stages={s:local_contract(s) for s in op.STAGES}
    # Revalidate completed immutable originals locally, without remote calls.
    active=None;completed={}
    for stage,(b,c) in stages.items():
        if (b/'analysis/completion_receipt.json').exists():completed[stage]=finalize(stage)
        else:
            active=stage;break
    if active is None:return {'status':'all63_complete_originals_and_reports_verified','stages':completed,'remote_calls':0}
    activation={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={h:pool.submit(activate_host,active,h) for h in ('5080','5090')}
        for h,f in futures.items():
            try:activation[h]=f.result()
            except Exception as exc:activation[h]={'host':h,'status':'failed_preserved','type':type(exc).__name__,'message':str(exc),'user_action_required':True}
    observations=observe(active)
    try:finalization=finalize(active)
    except Exception as exc:finalization={'status':'failed_preserved','type':type(exc).__name__,'message':str(exc),'user_action_required':True}
    result={'active_stage':active,'activation':activation,'observations':observations,'finalization':finalization,
        'completed_stages':completed,'following_stages_wait_for_previous_finalization':True,'automatic_retry':False,'created_unix':time.time()}
    write(REPORT/'rounds'/f'{time.time_ns()}.json',result)
    return result

if __name__=='__main__':print(json.dumps(advance(),ensure_ascii=False))
