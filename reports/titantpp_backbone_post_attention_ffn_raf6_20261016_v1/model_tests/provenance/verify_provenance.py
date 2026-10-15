import json,sys,platform,torch,hashlib
from pathlib import Path
from paper.scripts import run_titantpp_post_attention_ffn as route
from models.TPPs import CountAwareTitanPostAttentionFFN as model
data=route.fabricated_data();data["loader"]["max_seq_len"]=84
counts={}
for arm in model.ARMS:
 m,meta=route.build_model(data,arm)
 counts[arm]={"actual_active":sum(p.numel() for p in m.parameters() if p.requires_grad),"actual_registered":sum(p.numel() for p in m.parameters()),"metadata_active":meta["active_trainable_parameter_count"],"metadata_registered":meta["registered_parameter_count"]}
modules={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if n.startswith(("models.","paper.","simple_lab_test.","data_loader.","utils.")) and getattr(m,"__file__",None)}
assert all(p.startswith("/private/tmp/ffn_model_frozen147_v1/") for p in modules.values())
print(json.dumps({"status":"passed","python":platform.python_version(),"torch":torch.__version__,"device":"cpu","actual_counts_context84":counts,"module_paths":modules,"native_validation_performed":False,"real_data_loaded":False,"held_out_test_evaluated":False},sort_keys=True))