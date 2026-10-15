"""Fresh Taxi history-CNN execution, six fits, 5080 max3, Validation only.
Operational guards are copied from the existing v5 component campaign;
the scientific fit loop is loaded unchanged from frozen_runtime.py.
"""
from __future__ import annotations
from contextlib import contextmanager
import argparse, csv, fcntl, hashlib, importlib, importlib.util, json, math, os
from pathlib import Path
import shutil, signal, subprocess, sys, time, traceback
ARMS=("titantpp_cnn_history_adaptive_v1","titantpp_cnn_history_capacity_v1")
SEEDS=(42,52,62)
SCHEMA="titantpp_cnn_history_taxi6_20261015_v1"
PREPARATION_SHA="c5ae7ffed7e89c798dcb7bf7264e15a781adf10c6162a632501c3b8019bcdb3b"
TRAINING={"batch_size":128,"maximum_epochs":300,"minimum_epochs":40,"patience":40,
"monitor":"validation_raw_quantity_rmse","tie":"strict_earliest_finite_minimum","warm_start":False}
PARALLELISM={"maximum_workers":3,"grouping":"each_candidate_three_seeds_then_other_candidate",
"per_process_cuda_fraction":.30,"aggregate_gpu_fraction":.8}
ROOT=None
common=base=runner=frozen=None
_GPU_IDENTITY_WITNESSES={}
class OwnedTermination(RuntimeError):
    """A supervisor signal requests cleanup before this controller exits."""

@contextmanager
def owned_signal_guard():
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        # A repeated supervisor signal must not interrupt owned-child cleanup.
        signal.signal(signum, signal.SIG_IGN)
        raise OwnedTermination("SIGTERM received; stopping owned descendants")
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()

def safe_relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe relative path")
    resolved = (Path(root) / path).resolve()
    require(resolved.is_relative_to(Path(root).resolve()), "Path escaped owned root")
    return resolved

def write_json(path, value, *, exclusive=False):
    path = Path(path)
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode()
    require(len(encoded) < 64 * 1024 ** 2, "JSON exceeds artifact limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as handle:
            handle.write(encoded)
    else:
        temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
        with temporary.open("xb") as handle:
            handle.write(encoded)
        os.replace(temporary, path)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load sealed operation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def authorization(c, root, *, live=True):
    validate(c)
    require(not any((root / name).exists() for name in ("failure.json", "qualification/failure.json")),
            "Preserved failure; no automatic retry")
    approval, start = read(root / "approval.json"), read(root / "start_permit.json")
    host = next(h for h, spec in c["hosts"].items() if Path(spec["root"]) == root)
    require(approval["approved"] is True and approval["contract_sha256"] == sha_json(c)
            and approval["hosts"] == [host] and approval["user_instruction"], "Host-specific user launch approval missing")
    require(start["host"] == host and start["contract_sha256"] == sha_json(c)
            and start["approval_sha256"] == sha_json(approval), "Start permit binding drift")
    require(start["deadline_unix"] - start["started_at_unix"] == c["limits"]["total_wall_seconds"], "Extended host lease")
    if live:
        require(start["started_at_unix"] <= time.time() < start["deadline_unix"], "Expired host lease")
    return start

def job_lease_path(root, name):
    require(name and "/" not in name and ".." not in name, "Unsafe job lease id")
    return root / "leases" / (name + ".json")

def owned_root_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total

class PulseBudget:
    def __init__(self, root, digest, job, deadline, lease=False):
        self.root, self.digest, self.job, self.deadline, self.lease = root, digest, job, deadline, lease
        self.last = self.storage_check = 0.
        self.monotonic_deadline = time.monotonic() + max(0, deadline - time.time())

    def __call__(self):
        now = time.time()
        require(now < self.deadline and time.monotonic() < self.monotonic_deadline, "Absolute budget exhausted")
        if now - self.last < 2:
            return
        if self.lease:
            lease = read(job_lease_path(self.root, self.job))
            require(lease["contract_sha256"] == self.digest and lease["job"] == self.job
                    and now < lease["expires_unix"] <= self.deadline
                    and lease["worker_group_pid"] == os.getpgrp()
                    and lease["supervisor_pid"] != os.getpid(), "Per-job lease expired or foreign")
        require(shutil.disk_usage(self.root).free > 5 * 1024 ** 3, "Free disk below5GiB")
        rss = int(next(row.split()[1] for row in Path("/proc/self/status").read_text().splitlines()
                       if row.startswith("VmRSS:"))) * 1024
        require(rss < 16 * 1024 ** 3, "Worker CPU RSS over16GiB")
        if now - self.storage_check > 30:
            require(owned_root_bytes(self.root) < 8 * 1024 ** 3, "Owned output over8GiB")
            self.storage_check = now
        import torch
        if torch.cuda.is_initialized():
            fraction = .30 if self.lease else .8
            require(torch.cuda.max_memory_allocated() <= fraction * torch.cuda.get_device_properties(0).total_memory,
                    "CUDA allocation exceeds admitted fraction")
        write_json(self.root / "progress" / (self.job + ".json"), {"job": self.job, "pid": os.getpid(),
                   "updated_unix": now, "deadline_unix": self.deadline,
                   "source": "genuine_phase_or_batch_progress", "mac_required": False})
        self.last = now

def timeout_command(script, c, host, tail, deadline):
    remaining = int(deadline - time.time())
    require(remaining > 0, "No remaining native timeout budget")
    return ["timeout", "--signal=TERM", "--kill-after=15s", str(remaining),
            c["hosts"][host]["python"], str(script), "--contract",
            str(Path(c["hosts"][host]["root"]) / "execution_contract.json"), "--host", host, *tail]

def stop_owned(child):
    if child.poll() is not None:
        return
    require(os.getpgid(child.pid) == child.pid, "Owned child process group changed")
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)

def observe_exit(child, command, digest, **fields):
    require(child.poll() is not None, "Child is still live")
    return {**fields, "returncode": child.returncode, "worker_group_pid": child.pid,
            "supervisor_pid": os.getpid(), "argv": command, "observed_unix": time.time(),
            "contract_sha256": digest, "actual_child_exit_observed": True}

def verify_gpu_membership(gpu_pids, processes, active, *, script, used_bytes, total_bytes):
    require(total_bytes > 0 and 0 <= used_bytes <= .8 * total_bytes, "Aggregate GPU memory over80 percent")
    for pid in gpu_pids:
        proc = processes.get(pid)
        require(proc is not None, "GPU PID absent from process snapshot")
        owners = [item for item in active.values() if proc["ppid"] == item["child"].pid]
        require(len(owners) == 1 and proc["pgrp"] == owners[0]["child"].pid, "Foreign GPU process group")
        argv, command = proc["argv"], owners[0]["command"]
        require(argv == command[4:] and str(script) in argv, "Foreign GPU worker command")
    return True

def _gpu_process_snapshot(pid):
    """Read one PID with stable kernel identity around its exact command line."""
    proc = Path("/proc") / str(pid)
    before_text = (proc / "stat").read_text()
    before = before_text.rsplit(")", 1)[1].split()
    argv = (proc / "cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
    after_text = (proc / "stat").read_text()
    after = after_text.rsplit(")", 1)[1].split()
    require(int(before_text.split(" ", 1)[0]) == int(after_text.split(" ", 1)[0]) == pid,
            "GPU stat PID mismatch")
    initial = {"state": before[0], "ppid": int(before[1]), "pgrp": int(before[2]), "start_ticks": int(before[19])}
    final = {"state": after[0], "ppid": int(after[1]), "pgrp": int(after[2]), "start_ticks": int(after[19])}
    return {**final, "argv": argv, "before": initial,
            "consistent": all(initial[name] == final[name] for name in ("ppid", "pgrp", "start_ticks"))}

def _fresh_gpu_inventory(host_spec, timeout_seconds):
    """One read-only NVML refresh bounded by the remaining settling budget."""
    require(math.isfinite(timeout_seconds) and 0 < timeout_seconds <= 5., "Invalid GPU settling query budget")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=timeout_seconds)
    pids = set()
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        require(len(row) == 2, "Unknown GPU process inventory")
        if row[0].strip() == host_spec["gpu_uuid"]:
            pids.add(int(row[1]))
    return pids

def gpu_observation(c, host, active, script):
    """Keep exact live ownership; reconcile only known PID exit boundaries.

    NVML enumeration and /proc reads are not atomic. A witnessed owned PID may
    become a zombie (empty cmdline), disappear, or be reparented during exit.
    At most25 fresh inventories, spaced0.2s inside a hard5s monotonic budget,
    can reconcile those transitions. A stable
    foreign command, PID reuse, unreadable /proc, or still-listed unknown PID
    always fails. No scientific fit is retried or resumed by this observation.
    """
    output = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=15).stdout
    rows = [row.strip().split(",") for row in output.splitlines() if row.strip()]
    match = [row for row in rows if row[0].strip() == c["hosts"][host]["gpu_uuid"]]
    require(len(match) == 1, "Pinned GPU missing")
    total, used = [float(match[0][i].strip()) * 1024 ** 2 for i in (1, 2)]
    require(total > 0 and 0 <= used <= .8 * total, "Aggregate GPU memory over80 percent")
    cache = _GPU_IDENTITY_WITNESSES.setdefault((sha_json(c), host, str(script)), {})
    initial_pids = set(common.gpu_pids(c["hosts"][host]))
    pids, processes, pending, reconciled = initial_pids, {}, {}, []
    inventories = [sorted(initial_pids)]

    def capture(pid):
        try:
            proc = _gpu_process_snapshot(pid)
        except FileNotFoundError:
            return None
        before = proc["before"]
        require(before["start_ticks"] == proc["start_ticks"], "GPU PID reused during identity capture")
        witness = cache.get(pid)
        # Changed identity cannot borrow a prior worker's vanished-PID proof.
        if witness is not None:
            require(witness["start_ticks"] == proc["start_ticks"], "GPU PID reused after exact ownership witness")
        if not proc["consistent"] or proc["state"] in ("Z", "X") or not any(proc["argv"]):
            return proc
        # This original positive membership check is intentionally unchanged.
        # Stable nonempty foreign argv fails here, without any reconciliation.
        verify_gpu_membership({pid}, {pid: proc}, active, script=script, used_bytes=used, total_bytes=total)
        owner = next(item for item in active.values() if item["child"].pid == proc["ppid"])
        cache[pid] = {"ppid": proc["ppid"], "pgrp": proc["pgrp"], "start_ticks": proc["start_ticks"],
                      "argv": list(proc["argv"]), "guard_pid": owner["child"].pid,
                      "guard_command": list(owner["command"]), "observed_unix": time.time()}
        return proc

    def assign(current):
        result, unresolved = {}, {}
        for pid in current:
            proc = capture(pid)
            if proc is None or not proc["consistent"] or proc["state"] in ("Z", "X") or not any(proc["argv"]):
                unresolved[pid] = proc
            else:
                result[pid] = proc
        return result, unresolved

    processes, pending = assign(pids)
    settling_started = time.monotonic()
    settling_deadline = settling_started + 5.
    for _ in range(25):
        if not pending:
            break
        remaining = settling_deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(.2, remaining))
        remaining = settling_deadline - time.monotonic()
        if remaining <= 0:
            break
        fresh = set(_fresh_gpu_inventory(c["hosts"][host], remaining))
        require(time.monotonic() < settling_deadline, "GPU settling hard5s monotonic deadline reached")
        inventories.append(sorted(fresh))
        for pid in set(pending) - fresh:
            require(pid in cache, "Unwitnessed GPU PID disappeared; ownership remains unconfirmed")
            witness = cache[pid]
            owners = [item for item in active.values() if item["child"].pid == witness.get("guard_pid")
                      and item["command"] == witness.get("guard_command")]
            age = time.time() - witness["observed_unix"]
            require(len(owners) == 1 and witness["ppid"] == witness["pgrp"] == witness["guard_pid"]
                    and witness["argv"] == owners[0]["command"][4:]
                    and math.isfinite(age) and 0 <= age <= 60,
                    "Disappeared GPU PID lacks a recent exact current active-owner witness")
            proc = pending[pid]
            if proc is not None:
                require(proc["start_ticks"] == cache[pid]["start_ticks"], "Disappeared GPU PID identity drift")
            reconciled.append({"pid": pid, "reason": "owned_exit_boundary_confirmed_absent_in_fresh_NVML",
                               "prior_exact_owned_witness": cache[pid], "last_process_snapshot": proc})
        pids = fresh
        processes, pending = assign(pids)
    require(not pending, "GPU PID identity unresolved after bounded5s settling: " + json.dumps(pending, sort_keys=True))
    verify_gpu_membership(set(processes), processes, active, script=script, used_bytes=used, total_bytes=total)
    return {"gpu_uuid": c["hosts"][host]["gpu_uuid"], "gpu_pids": sorted(processes),
            "processes": processes, "used_bytes": used, "total_bytes": total,
            "observed_unix": time.time(), "initial_gpu_pids": sorted(initial_pids),
            "gpu_inventory_samples": inventories, "exit_boundary_reconciliation": reconciled,
            "maximum_fresh_inventory_rechecks": 25, "settling_poll_interval_seconds": .2,
            "settling_hard_limit_seconds": 5., "settling_elapsed_seconds": time.monotonic() - settling_started}

def validate_terminal(c, root, job):
    folder = root / "run" / job["id"]
    manifest = read(folder / "terminal_manifest.json")
    require(manifest["scientific_success"] is True and manifest["status"] == "complete"
            and manifest["job"] == job and manifest["contract_sha256"] == sha_json(c)
            and manifest["held_out_test_evaluated"] is False
            and manifest["evaluation_scope"] == "validation_only", "Missing/foreign terminal")
    for name, digest in manifest["files"].items():
        require(sha_file(safe_relative(folder, name)) == digest, "Terminal binary/file integrity mismatch")
    require(any(name.endswith("best_val_qty_rmse_model.pt") for name in manifest["files"])
            and any(name.endswith("last_epoch_state.pt") for name in manifest["files"]), "Selected/final binary missing")
    return {"terminal_manifest_sha256": sha_file(folder / "terminal_manifest.json")}

def write_parallel_pulse(c, host, root, start, active, results):
    now = time.time()
    require(now < start["deadline_unix"], "Campaign absolute deadline reached")
    require(len(active) <= 3, "Parallel worker ceiling exceeded")
    state = {}
    for name, item in active.items():
        require(now < item["deadline"], "Condition absolute deadline reached")
        write_json(job_lease_path(root, name), {"job": name, "contract_sha256": sha_json(c),
                   "issued_unix": now, "expires_unix": min(now + 90, item["deadline"]),
                   "supervisor_pid": os.getpid(), "worker_group_pid": item["child"].pid})
        state[name] = {"job": item["job"], "worker_group_pid": item["child"].pid,
                       "worker_guard_argv": item["command"], "condition_deadline_unix": item["deadline"]}
    write_json(root / "status.json", {"status": "running", "active_jobs": state,
               "completed": results, "supervisor_pid": os.getpid(), "supervisor_argv": [sys.executable, *sys.argv],
               "updated_unix": now, "deadline_unix": start["deadline_unix"], "maximum_concurrency": 3,
               "grouping": PARALLELISM["grouping"], "mac_required": False})

def run_fit(c, host, root, job, deadline):
    require(job in c["jobs"] and job["host"] == host, "Foreign fit")
    lease_path = job_lease_path(root, job["id"])
    until = min(time.time() + 10, deadline)
    while time.time() < until:
        try:
            lease = read(lease_path)
            if lease["worker_group_pid"] == os.getppid():
                break
        except FileNotFoundError:
            pass
        time.sleep(.05)
    else:
        raise ValueError("Matching per-job timeout lease not published")
    parent_stat = Path(f"/proc/{os.getppid()}/stat").read_text().rsplit(")", 1)[1].split()
    tokens = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
    require(os.getppid() == os.getpgrp() == lease["worker_group_pid"]
            and int(parent_stat[1]) == lease["supervisor_pid"], "Timeout PID/PPID ownership mismatch")
    require(len(tokens) == 16 and Path(tokens[0]).name == "timeout"
            and tokens[1:3] == ["--signal=TERM", "--kill-after=15s"]
            and float(tokens[3]) > 0 and tokens[4:] == [c["hosts"][host]["python"],
                str(Path(__file__).resolve()), "--contract", str(root / "execution_contract.json"),
                "--host", host, "--mode", "fit", "--job", job["id"], "--deadline", str(deadline)],
            "Timeout exact command/deadline mismatch")
    require(lease["job"] == job["id"] and lease["contract_sha256"] == sha_json(c), "Foreign fit lease")
    write_json(root / "worker_identities" / (job["id"] + ".json"), {"job": job, "pid": os.getpid(),
               "ppid": os.getppid(), "pgrp": os.getpgrp(), "supervisor_pid": lease["supervisor_pid"],
               "argv": [sys.executable, *sys.argv], "contract_sha256": sha_json(c),
               "started_unix": time.time(), "deadline_unix": deadline}, exclusive=True)
    # Only the legacy operational lease lookup is redirected; trainer, selector,
    # loss, optimizer, loader and endpoint replay remain the sealed function.
    def read_bridge(path):
        return read(lease_path if Path(path) == root / "server_lease.json" else path)
    frozen.read = read_bridge
    import torch
    torch.cuda.set_per_process_memory_fraction(.30, 0)
    common.ENVIRONMENT = {**common.ENVIRONMENT, "PYTHONHASHSEED": str(job["seed"])}
    frozen.run_fit(c, host, root, job, deadline)

def expected_jobs():
    return [{'id':f'yellow_trip_hourly__{s}__{a}','host':'5080','dataset':'yellow_trip_hourly','seed':s,'arm':a}
            for a in ARMS for s in SEEDS]


def validate(c, *, verify_files=True):
    require(c['schema']==SCHEMA and c['jobs']==expected_jobs() and c['arms']==list(ARMS), 'Scope drift')
    require(c['training']==TRAINING and c['parallelism']==PARALLELISM, 'Training/concurrency drift')
    require(c['evaluation_scope']=='validation_only' and c['held_out_test_evaluated'] is False
            and c['test_evaluation'] is False, 'Test is not authorized')
    require(c['limits']['total_wall_seconds']==168*3600 and c['limits']['per_condition_seconds']==36*3600
            and c['limits']['automatic_retry'] is False, 'Deadline/retry drift')
    require(set(c['hosts'])=={'5080'}, 'Only 5080 approved')
    h=c['hosts']['5080']; root=Path(h['root'])
    require(str(root)=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+SCHEMA+'_5080'
            and h['source_root']==str(root/'source') and h['operation_root']==str(root/'operation'), 'Root drift')
    require(h['gpu_uuid']=='GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6'
            and h['python']=='/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12', 'Host drift')
    require(sha_json(c['source']['files'])==c['source']['files_sha256'], 'Source closure drift')
    if verify_files:
        location=Path(__file__).resolve().parent.parent
        for name,digest in {**c['source']['files'],**c['input_files']}.items():
            require(sha_file(safe_relative(ROOT,name))==digest,'Source/input SHA mismatch: '+name)
        for name,digest in c['operation']['files'].items():
            require(sha_file(safe_relative(location,name))==digest,'Operation SHA mismatch: '+name)
        p=location/'preparation/contract.json'
        require(sha_file(p)==PREPARATION_SHA,'Preparation changed')
        prepared=read(p)
        require(c['datasets']==[prepared['dataset']] and c['baseline_jobs']==prepared['baseline_jobs'], 'Science/baseline drift')
        require(all(c['source']['files'].get(n)==v for n,v in prepared['source_files'].items()),'Prepared source changed')
    return c


def bootstrap(c,host):
    global ROOT,common,base,runner,frozen
    ROOT=Path(c['hosts'][host]['source_root']).resolve()
    require(Path.cwd().resolve()==ROOT and Path(__file__).resolve().parent==Path(c['hosts'][host]['operation_root']), 'Wrong cwd/operation')
    validate(c)
    sys.path.insert(0,str(ROOT))
    common=importlib.import_module('paper.scripts.observed_slot_parallel_common')
    base=importlib.import_module('paper.scripts.run_multilag_detail_execution')
    prior=importlib.import_module('paper.scripts.run_local_detail_benchmark')
    shared=importlib.import_module('paper.scripts.run_observed_slot_partition')
    runner=importlib.import_module('paper.scripts.run_titantpp_adaptive_history_cnn');runner.install_hooks()
    for m in (common,base,prior,shared,runner):
        require(Path(m.__file__).resolve().is_relative_to(ROOT),'Import escaped frozen source')
    frozen=load_module('history_scientific_fit',Path(__file__).with_name('frozen_runtime.py'))
    frozen.ROOT,frozen.common,frozen.base=ROOT,common,base
    frozen.prior,frozen.shared,frozen.engines=prior,shared,{a:runner for a in ARMS}
    frozen.validate,frozen.authorization=validate,authorization
    frozen.initial_states=runner.initial_states
    frozen.verify_training_permit,frozen.PulseBudget=verify_training_permit,PulseBudget


def verify_training_permit(c,root):
    p=read(root/'training_permit.json');q=read(root/'qualification/receipt.json')
    require(p['contract_sha256']==sha_json(c) and p['qualification']==q
            and p['qualification_sha256']==sha_file(root/'qualification/receipt.json')
            and p['start_permit_sha256']==sha_json(read(root/'start_permit.json')),'Training permit changed')
    require(q['status']=='passed' and q['contract_sha256']==sha_json(c)
            and read(root/'qualification/actual_exit.json')['returncode']==0,'Qualification did not succeed')
    require(q['capacity_check']['maximum_workers']==3 and q['capacity_check']['passed'],'GPU capacity missing')
    return p


def native_qualification(c,host,root):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    authorization(c,root)
    require(not common.gpu_pids(c['hosts'][host]),'GPU occupied before qualification')
    runtime=common.runtime_check(c,host)
    require(runtime['python']=='3.12.13' and runtime['threads']==4 and runtime['interop_threads']==1,'Native training runtime drift')
    q={'status':'checking','host':host,'contract_sha256':sha_json(c),'runtime':runtime,
       'inputs':{},'initialization':{},'synthetic':{},'held_out_test_evaluated':False,'scientific_optimizer_updates':0}
    data=c['datasets'][0];name=data['dataset_id']
    frame,receipt=base.prepare_admitted_data(data,data_root=ROOT)
    q['inputs'][name]={'input_receipt':receipt,'input_receipt_sha256':sha_json(receipt),
        'interface_sha256':{a:sha_json(runner.time_interface(data,frame,c,arm=a)) for a in ARMS}}
    q['initialization'][name]={str(s):runner.initial_states(data,s) for s in SEEDS}
    for job in c['baseline_jobs']:
        torch.manual_seed(job['seed']);m,_=runner.build_model(data,runner.CNN)
        require(canonical_state_dict_sha256(m.state_dict())==job['original']['initial_state_sha256'],'CNN initialization mismatch')
        del m
    # Fabricated full-shape updates only, never an abbreviated scientific fit.
    for arm in ARMS:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(42);model,_=runner.build_model(data,arm);model.cuda()
            generator=torch.Generator().manual_seed(831)
            shape=(128,data['loader']['max_seq_len'])
            dt=torch.randint(1,8,shape,generator=generator).float().cuda();dt[:,0]=0
            qty=torch.randint(0,40,shape,generator=generator).float().cuda();mask=torch.ones_like(dt,dtype=torch.bool)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            names={n for n,p in model.named_parameters() if n.startswith(runner.EXTRA_PREFIX)};active=set()
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)['joint_loss'].mean()
                require(bool(torch.isfinite(loss)),'Nonfinite synthetic loss');loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Nonfinite gradient')
                active.update(n for n,p in model.named_parameters() if n in names and p.grad is not None and bool(p.grad.ne(0).any()))
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            require(active==names,'Inactive candidate network parameters')
            model.eval()
            with torch.no_grad():
                d=dt[:1,:8].clone();v=qty[:1,:8].clone();valid=mask[:1,:8];write=valid.clone();write[:,-1]=False
                before=canonical_state_dict_sha256(model.state_dict())
                a=model.encode_task_states(d,v,valid,memory_write_mask=write)[0]
                d[:,-1]=float('nan');v[:,-1]=float('inf')
                b=model.encode_task_states(d,v,valid,memory_write_mask=write)[0]
                require(torch.equal(a,b),'Target leak in native candidate')
                require(before==canonical_state_dict_sha256(model.state_dict()),'Evaluation mutated model')
            torch.cuda.synchronize()
            q['synthetic'][arm]={'status':'passed','shape':list(shape),'updates':4,'scientific_updates':0,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'active_added_parameters':243,'target_blocked':True,'eval_state_unchanged':True}
            del model,optimizer,loss,dt,qty,mask,a,b
    peak=max(x['peak_reserved_bytes'] for x in q['synthetic'].values())
    total=torch.cuda.get_device_properties(0).total_memory
    bound=3*(1.25*peak+512*1024**2)
    require(peak<.30*total and bound<.8*total,'Three-worker conservative memory gate failed')
    q['capacity_check']={'passed':True,'maximum_workers':3,'method':'3*(1.25*max_reserved+512MiB_context_reserve)',
        'bound_bytes':bound,'total_bytes':total,'simultaneous_synthetic_workers_executed':False}
    q.update(status='passed',completed_unix=time.time())
    write_json(root/'qualification/receipt.json',q,exclusive=True)


def dispatch(c,host,root):
    start=authorization(c,root);verify_training_permit(c,root)
    require(not common.gpu_pids(c['hosts'][host]),'GPU occupied before fresh fits')
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(root/'supervisor.claim',{'pid':os.getpid(),'argv':[sys.executable,*sys.argv],
            'contract_sha256':sha_json(c),'started_unix':time.time()},exclusive=True)
        active={};results={};script=Path(__file__).resolve()
        try:
            # One complete three-seed wave per candidate. Both waves use the
            # same fixed max3 scheduling and are independent fresh fits.
            for arm in ARMS:
                gpu_observation(c,host,active,script)
                for job in [j for j in c['jobs'] if j['arm']==arm]:
                    require(not (root/'run'/job['id']).exists(),'Existing fit; no retry')
                    deadline=min(start['deadline_unix'],time.time()+c['limits']['per_condition_seconds'])
                    write_json(root/'claims'/(job['id']+'.json'),{'job':job,'contract_sha256':sha_json(c),'deadline_unix':deadline},exclusive=True)
                    command=timeout_command(script,c,host,['--mode','fit','--job',job['id'],'--deadline',str(deadline)],deadline)
                    log=root/'logs'/(job['id']+'.log');log.parent.mkdir(exist_ok=True)
                    with log.open('xb') as stream:
                        child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONHASHSEED':str(job['seed'])},
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    active[job['id']]={'job':job,'child':child,'deadline':deadline,'command':command}
                    write_parallel_pulse(c,host,root,start,active,results)
                while active:
                    write_parallel_pulse(c,host,root,start,active,results)
                    gpu_observation(c,host,active,script)
                    finished=[]
                    for name,item in active.items():
                        if item['child'].poll() is None: continue
                        receipt=observe_exit(item['child'],item['command'],sha_json(c),job=item['job'])
                        write_json(root/'worker_exits'/(name+'.json'),receipt,exclusive=True)
                        require(receipt['returncode']==0,'Worker failed; no retry: '+name)
                        results[name]=validate_terminal(c,root,item['job']);finished.append(name)
                    for name in finished:del active[name]
                    if active:time.sleep(5)
                gpu_observation(c,host,active,script)
            for job in c['jobs']:
                p=root/'run'/job['id']/'runs'/job['arm']/frozen.VARIANT/f"seed_{job['seed']}"/'exposure.json'
                frozen.audit_host_batch_prefix(c,host,root,job,read(p))
            write_json(root/'status.json',{'status':'complete','completed':results,'completed_conditions':6,
                'updated_unix':time.time(),'maximum_concurrency':3,'held_out_test_evaluated':False})
        except BaseException as exc:
            stopped=[]
            for item in active.values():
                stop_owned(item['child']);stopped.append(observe_exit(item['child'],item['command'],sha_json(c),job=item['job']))
            write_json(root/'failure.json',{'message':str(exc),'traceback':traceback.format_exc(),
                'completed':results,'owned_stopped':stopped,'automatic_retry':False},exclusive=True)
            raise


def pipeline(c,host,root):
    validate(c);approval=read(root/'approval.json')
    require(approval['approved'] and approval['contract_sha256']==sha_json(c),'Approval mismatch')
    write_json(root/'pipeline.claim',{'pid':os.getpid(),'started_unix':time.time()},exclusive=True)
    now=time.time();write_json(root/'start_permit.json',{'host':host,'started_at_unix':now,
        'deadline_unix':now+c['limits']['total_wall_seconds'],'contract_sha256':sha_json(c),
        'approval_sha256':sha_json(approval)},exclusive=True)
    start=authorization(c,root);child=None
    try:
        for mode in ('native','dispatch'):
            deadline=min(start['deadline_unix'],time.time()+c['limits']['qualification_seconds']) if mode=='native' else start['deadline_unix']
            command=timeout_command(Path(__file__).resolve(),c,host,['--mode',mode],deadline)
            path=root/('qualification/native.log' if mode=='native' else 'logs/dispatcher.log');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                child.wait()
            receipt=observe_exit(child,command,sha_json(c),phase=mode)
            write_json(root/('qualification/actual_exit.json' if mode=='native' else 'dispatcher_exit.json'),receipt,exclusive=True)
            require(child.returncode==0,mode+' actual exit '+str(child.returncode))
            require(not common.gpu_pids(c['hosts'][host]),'Child GPU process remains')
            if mode=='native':
                q=read(root/'qualification/receipt.json')
                write_json(root/'training_permit.json',{'contract_sha256':sha_json(c),'qualification':q,
                    'qualification_sha256':sha_file(root/'qualification/receipt.json'),
                    'start_permit_sha256':sha_json(start)},exclusive=True)
        write_json(root/'train_complete.json',{'status':'complete','scientific_worker_count':6,
            'dispatcher_actual_exit':0,'contract_sha256':sha_json(c),'completed_unix':time.time()},exclusive=True)
    except BaseException as exc:
        if child is not None:stop_owned(child)
        write_json(root/'pipeline_failure.json',{'message':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False},exclusive=True)
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--host',choices=['5080'],required=True)
    p.add_argument('--mode',choices=['pipeline','native','dispatch','fit'],required=True);p.add_argument('--job');p.add_argument('--deadline',type=float)
    a=p.parse_args();c=read(a.contract);root=Path(c['hosts'][a.host]['root'])
    require(Path(a.contract).resolve()==root/'execution_contract.json','Contract location mismatch')
    env=dict(c['hosts'][a.host]['environment'])
    if a.mode=='fit':env['PYTHONHASHSEED']=str(next(j for j in c['jobs'] if j['id']==a.job)['seed'])
    require(all(os.environ.get(k)==v for k,v in env.items()),'Process environment mismatch')
    bootstrap(c,a.host)
    common.ENVIRONMENT={k:env[k] for k in common.ENVIRONMENT}
    with owned_signal_guard():
        if a.mode=='pipeline':pipeline(c,a.host,root)
        elif a.mode=='native':native_qualification(c,a.host,root)
        elif a.mode=='dispatch':dispatch(c,a.host,root)
        else:run_fit(c,a.host,root,next(j for j in c['jobs'] if j['id']==a.job),a.deadline)

if __name__=='__main__':main()
