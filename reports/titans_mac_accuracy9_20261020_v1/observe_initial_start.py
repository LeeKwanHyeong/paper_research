"""One bounded read-only SSH wait for qualification and the first saved epochs."""
import json
import shlex
from pathlib import Path
from paper.scripts import manage_titans_mac_accuracy9 as manager

INTEGRITY = "1dd974eccccce4c9b338bfdcd6b3f3755ee96a1172ee46de6b63c6f19db0cd04"
c = manager.verify_local(INTEGRITY)
h = c["hosts"]["5090"]
first = [j["id"] for j in c["jobs"][:3]]
program = f'''import json,pathlib,time
from paper.scripts import titans_mac_accuracy9_runtime as r
root=pathlib.Path({h['root']!r});contract=r.read(root/'execution_contract.json')
first={first!r};deadline=min(time.time()+1800.,contract['hosts']['5090']['absolute_host_deadline_unix'])
reason='initial_epoch_wait_timeout'
while time.time()<deadline:
 if (root/'dispatcher_exit.json').exists():reason='dispatcher_terminated';break
 ready=(root/'start.json').exists()
 for job in first:
  histories=list((root/'run'/job).glob('runs/*/*/*/history.json'))
  try:ready=ready and len(histories)==1 and len(r.read(histories[0])['history'])>=1
  except (FileNotFoundError,json.JSONDecodeError,KeyError):ready=False
 if ready:reason='three_workers_saved_initial_epoch';break
 time.sleep(5.)
value=r.status(contract,'5090');value['initial_wait_reason']=reason
print(json.dumps(value,ensure_ascii=False,allow_nan=False))
'''
command = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join([
    "env", *[f"{k}={v}" for k,v in h["environment"].items()],h["python"],"-c",program])
raw = manager._transport.save_transport("5090","initial_start_wait",
    manager._transport.ssh("5090",command,timeout=1830))
value=json.loads(raw)
manager.write(manager.REPORT/"INITIAL_START_OBSERVATION.json",value)
print(json.dumps({"status":"observed","reason":value["initial_wait_reason"],
                  "observed_at":value["observed_at"],"remote_status_ssh_calls":1},ensure_ascii=False))
