import importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
root=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
sys.path.insert(0,'/private/tmp/ffn_model_frozen147_v1')
from paper.scripts import run_titantpp_post_attention_ffn as runner
spec=importlib.util.spec_from_file_location('new_ffn_runtime',root/'paper/scripts/titantpp_post_attention_ffn_raf6_runtime.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.runner=runner
runtime.start_ticks=lambda pid:1  # macOS-only synthetic fixture, no native identity claim
contract=json.loads((root/'search_artifacts/titantpp_backbone_qk_location_raf6_20261016_v1/execution_contract.json').read_text())
contract['arms']=list(runtime.ARMS);contract['schema']=runtime.SCHEMA
original_fork=torch.random.fork_rng
torch.set_num_threads(4)
results={}
with patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self),patch.object(torch.nn.Module,'cuda',lambda self,*a,**k:self),patch.object(torch.random,'fork_rng',lambda **k:original_fork(devices=[])),patch.multiple(torch.cuda,set_per_process_memory_fraction=lambda *a,**k:None,empty_cache=lambda:None,reset_peak_memory_stats=lambda:None,synchronize=lambda:None,max_memory_reserved=lambda:0,max_memory_allocated=lambda:0):
 for arm in runtime.ARMS:
  results[arm]=runtime.synthetic_arm(contract,Path('/private/tmp/ffn_ops'),arm)
print(json.dumps({'status':'PASS_CPU_synthetic_scientific_route_only','actual_native_gpu_validation':False,'cuda_allocation_device_moves_and_start_ticks_mocked':True,'runtime_path':str(Path(runtime.__file__).resolve()),'runner_path':str(Path(runner.__file__).resolve()),'candidate_results':results},indent=2))
