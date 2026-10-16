"""Guard only the recovery Pod created under the original combined USD50 cap."""
import fcntl,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
CLI='/private/tmp/paper-runpodctl-v2.14.0'
def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def main():
 with (R/'provision_guard.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   a=json.loads((R/'provision_allocation.json').read_text())
   if (R/'provision_guard_handoff.json').exists() or (R/'provision_stop_receipt.json').exists():return
   now=time.time();write('provision_guard_heartbeat.json',{'time':now,'pid':os.getpid(),'pod_id':a['pod_id'],'deadline':a['setup_deadline'],'estimated_combined_cost_usd':a['parent_billed']+max(0,now-a['created_unix'])/3600*a['conservative_rate']})
   if now>=a['setup_deadline']:
    r=subprocess.run([CLI,'pod','stop',a['pod_id']],capture_output=True,text=True,timeout=45)
    write('provision_stop_receipt.json',{'pod_id':a['pod_id'],'time':now,'request_exit':r.returncode,'requested':True,'confirmed':False,'reason':'Recovery setup deadline; preserve owned volume'})
    return
   time.sleep(15)
if __name__=='__main__':main()
