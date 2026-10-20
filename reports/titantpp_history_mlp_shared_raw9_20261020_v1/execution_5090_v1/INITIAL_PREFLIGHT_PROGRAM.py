import os,sys,json,subprocess,datetime,pathlib
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
def run(args):
 p=subprocess.run(args,capture_output=True,text=True)
 return {"actual_exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
x={"host":"5090","observed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"gpu":run(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.used","--format=csv,noheader,nounits"]),"gpu_processes":run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"]),"runtime":{"python":sys.version.split()[0],"python_executable":sys.executable,"real_python_executable":os.path.realpath(sys.executable),"torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads()},"old_paused_process_paths_present":{},"new_root_exists":pathlib.Path("/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1_execution_5090_v1").exists(),"real_data_training_executed":False}
for pid in (3510525,3530812,3533434,3535733):
 path=pathlib.Path('/proc')/str(pid);x['old_paused_process_paths_present'][str(pid)]=path.exists()
print(json.dumps(x,ensure_ascii=False))
