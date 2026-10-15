"""Fresh RAF V-difference CNN execution, three fits, 5090 max3, Validation only.
Operational guards are copied from the existing v5 component campaign;
the scientific fit loop is loaded unchanged from frozen_runtime.py.
"""
from __future__ import annotations
from contextlib import contextmanager
import argparse, csv, fcntl, hashlib, importlib, importlib.util, json, math, os
from pathlib import Path
import shutil, signal, subprocess, sys, time, traceback
ARMS=("titantpp_cnn_value_difference_v1",)
SEEDS=(42,52,62)
ADMISSION_JOBS=tuple(("difference"+str(s),ARMS[0],s) for s in SEEDS)
SCHEMA="titantpp_backbone_value_difference_raf3_20261016_v1"
TRAINING={"batch_size":128,"maximum_epochs":300,"minimum_epochs":40,"patience":40,
"monitor":"validation_raw_quantity_rmse","tie":"strict_earliest_finite_minimum","warm_start":False}
PARALLELISM={"maximum_workers":3,"grouping":"single_candidate_three_seeds",
"per_process_cuda_fraction":.30,"aggregate_gpu_fraction":.8}
SCIENCE_SHA="f74fa52053d5ee4a8771ad0b25d3740c58a811bd7c0a256efd50a0da23b82b6f"
INHERITED_DEADLINE=1792679647.915156
INHERITED_START_SHA="b3faca44824ce90972a62a740bb338b2cf5f54b0dfeb0d3fb291e06bed2a7f5a"
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

def valid_start_deadline(c, start):
    """The new root receives only the remainder of the original host lease."""
    begin, end = start.get("started_at_unix"), start.get("deadline_unix")
    return (type(begin) in (int, float) and type(end) in (int, float)
            and math.isfinite(begin) and math.isfinite(end)
            and end == c["limits"]["inherited_absolute_host_deadline_unix"]
            and 0 < end - begin <= c["limits"]["total_wall_seconds"])


def new_start_permit(c, host, approval, now):
    start = {"host": host, "started_at_unix": now,
             "deadline_unix": c["limits"]["inherited_absolute_host_deadline_unix"],
             "contract_sha256": sha_json(c), "approval_sha256": sha_json(approval)}
    require(valid_start_deadline(c, start), "Expired or extended inherited host lease")
    return start


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
    require(valid_start_deadline(c, start), "Extended inherited host lease")
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

def start_ticks(pid):
    return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])

def launch_identity(child,command,digest,**fields):
    return {**fields,'pid':child.pid,'start_ticks':start_ticks(child.pid),
        'observer_pid':os.getpid(),'observer_start_ticks':start_ticks(os.getpid()),
        'argv':command,'contract_sha256':digest,'started_unix':time.time()}

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
                   "supervisor_pid": os.getpid(), "worker_group_pid": item["child"].pid,
                   "supervisor_start_ticks":start_ticks(os.getpid()),
                   "worker_group_start_ticks":item['launch']['start_ticks']})
        state[name] = {"job": item["job"], "worker_group_pid": item["child"].pid,
                       "worker_guard_argv": item["command"], "condition_deadline_unix": item["deadline"],
                       "worker_guard_start_ticks":item['launch']['start_ticks']}
    write_json(root / "status.json", {"status": "running", "active_jobs": state,
               "completed": results, "supervisor_pid": os.getpid(), "supervisor_argv": [sys.executable, *sys.argv],
               "updated_unix": now, "deadline_unix": start["deadline_unix"], "maximum_concurrency": 3,
               "grouping": PARALLELISM["grouping"], "mac_required": False,
               "contract_sha256":sha_json(c),"supervisor_start_ticks":start_ticks(os.getpid())})

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
    launch=read(root/'worker_launches'/(job['id']+'.json'))
    require(launch['job']==job and launch['pid']==os.getppid()
        and launch['start_ticks']==start_ticks(os.getppid())==lease['worker_group_start_ticks']
        and launch['observer_pid']==lease['supervisor_pid']
        and launch['observer_start_ticks']==start_ticks(lease['supervisor_pid'])==lease['supervisor_start_ticks'],
        'Worker launch process identity drift')
    write_json(root / "worker_identities" / (job["id"] + ".json"), {"job": job, "pid": os.getpid(),
               "ppid": os.getppid(), "pgrp": os.getpgrp(), "supervisor_pid": lease["supervisor_pid"],
               "argv": [sys.executable, *sys.argv], "contract_sha256": sha_json(c),
               "started_unix": time.time(), "deadline_unix": deadline,"start_ticks":start_ticks(os.getpid()),
               "guard_start_ticks":start_ticks(os.getppid()),"supervisor_start_ticks":lease['supervisor_start_ticks'],
               "launch_sha256":sha_file(root/'worker_launches'/(job['id']+'.json'))}, exclusive=True)
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
    return [{'id':f'raf_spare_parts__{s}__{a}','host':'5090','dataset':'raf_spare_parts','seed':s,'arm':a}
            for a in ARMS for s in SEEDS]


def validate(c, *, verify_files=True):
    require(c['schema']==SCHEMA and c['jobs']==expected_jobs() and c['arms']==list(ARMS), 'Scope drift')
    require(c['training']==TRAINING and c['parallelism']==PARALLELISM, 'Training/concurrency drift')
    require(c['evaluation_scope']=='validation_only' and c['held_out_test_evaluated'] is False
            and c['test_evaluation'] is False, 'Test is not authorized')
    require(c['limits']['total_wall_seconds']==168*3600 and c['limits']['per_condition_seconds']==36*3600
            and c['limits']['automatic_retry'] is False, 'Deadline/retry drift')
    require(c['limits'].get('inherited_absolute_host_deadline_unix')==INHERITED_DEADLINE
            and c['limits'].get('inherited_start_permit_sha256')==INHERITED_START_SHA, 'Inherited absolute deadline drift')
    require(set(c['hosts'])=={'5090'}, 'Only 5090 approved')
    h=c['hosts']['5090']; root=Path(h['root'])
    require(str(root)=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+SCHEMA+'_5090'
            and h['source_root']==str(root/'source') and h['operation_root']==str(root/'operation'), 'Root drift')
    require(h['gpu_uuid']=='GPU-c9adc246-ef66-7906-bc07-6e152fa9952f'
            and h['python']=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12', 'Host drift')
    require(sha_json(c['source']['files'])==c['source']['files_sha256'], 'Source closure drift')
    if verify_files:
        location=Path(__file__).resolve().parent.parent
        for name,digest in {**c['source']['files'],**c['input_files']}.items():
            require(sha_file(safe_relative(ROOT,name))==digest,'Source/input SHA mismatch: '+name)
        for name,digest in c['operation']['files'].items():
            require(sha_file(safe_relative(location,name))==digest,'Operation SHA mismatch: '+name)
        p=location/'preparation/contract.json'
        require(sha_file(p)==c["preparation_contract_sha256"],'Preparation changed')
        prepared=read(p)
        require(c['datasets']==[prepared['dataset']] and c['baseline_jobs']==prepared['baseline_jobs'], 'Science/baseline drift')
        require('stored_channel_reference_jobs' not in c and len(c['baseline_jobs'])==3
            and {j['seed'] for j in c['baseline_jobs']}==set(SEEDS),'Only historical CNN3 references allowed')
        science=location/'preparation/science_spec.json'
        inherited=location/'preparation/inherited_5090_start_permit.json'
        require(sha_file(science)==c['science_spec_sha256']==prepared['science_spec_sha256']==SCIENCE_SHA,
            'Scientific specification changed')
        require(sha_file(inherited)==prepared['inherited_start_permit_sha256']==INHERITED_START_SHA,
            'Inherited start evidence changed')
        prior=read(inherited)
        require(prior['host']=='5090' and prior['deadline_unix']==INHERITED_DEADLINE
            and prior['deadline_unix']-prior['started_at_unix']==168*3600
            and prepared['inherited_absolute_host_deadline_unix']==INHERITED_DEADLINE,
            'Inherited host deadline changed')
        require(c['architecture']==prepared['design'] and sha_json(c['architecture'])==c['design_sha256'],'Architecture drift')
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
    runner=importlib.import_module('paper.scripts.run_titantpp_value_difference_cnn');runner.install_hooks()
    for m in (common,base,prior,shared,runner):
        require(Path(m.__file__).resolve().is_relative_to(ROOT),'Import escaped frozen source')
    frozen=load_module('history_scientific_fit',Path(__file__).with_name('frozen_runtime.py'))
    frozen.ROOT,frozen.common,frozen.base=ROOT,common,base
    frozen.prior,frozen.shared,frozen.engines=prior,shared,{a:runner for a in ARMS}
    frozen.validate,frozen.authorization=validate,authorization
    frozen.initial_states=lambda data,seed:runner.initial_states(data,seed,arms=ARMS)
    frozen.verify_training_permit,frozen.PulseBudget=verify_training_permit,PulseBudget


def verify_training_permit(c,root):
    digest=sha_json(c);p=read(root/'training_permit.json');q=read(root/'qualification/receipt.json')
    start=read(root/'start_permit.json');approval=read(root/'approval.json')
    require(approval.get('approved') is True and approval.get('contract_sha256')==digest
        and approval.get('fresh_fit_count')==3 and approval.get('maximum_concurrent_workers')==3
        and start.get('approval_sha256')==sha_json(approval) and start.get('contract_sha256')==digest
        and start.get('host')=='5090' and valid_start_deadline(c,start),
        'Start approval binding changed')
    require(p['contract_sha256']==digest and p['qualification']==q
        and p['qualification_sha256']==sha_file(root/'qualification/receipt.json')
        and p['start_permit_sha256']==sha_json(start),'Training permit changed')
    launch=read(root/'qualification/launch.json');exit0=read(root/'qualification/actual_exit.json')
    pipeline=read(root/'pipeline.claim');pipeline_launch=read(root/'pipeline_launch.json')
    base=[c['hosts']['5090']['python'],str(root/'operation/runtime.py'),'--contract',str(root/'execution_contract.json'),'--host','5090']
    require(q.get('status')=='passed' and q.get('contract_sha256')==digest and q.get('host')=='5090'
        and q.get('source_closure')==c['source']['files_sha256'] and q.get('held_out_test_evaluated') is False
        and q.get('scientific_optimizer_updates')==0 and q.get('argv')==base+['--mode','native']
        and type(q.get('start_ticks')) is int and q['start_ticks']>0
        and q.get('ppid')==launch.get('pid') and q.get('guard_start_ticks')==launch.get('start_ticks'),
        'Qualification identity changed')
    require(pipeline.get('contract_sha256')==pipeline_launch.get('contract_sha256')==digest
        and pipeline.get('pid')==pipeline_launch.get('pid') and pipeline.get('start_ticks')==pipeline_launch.get('start_ticks')
        and pipeline.get('ppid')==pipeline_launch.get('observer_pid') and pipeline.get('wrapper_start_ticks')==pipeline_launch.get('observer_start_ticks')
        and pipeline.get('argv')==pipeline_launch.get('argv')==base+['--mode','pipeline']
        and launch.get('contract_sha256')==digest and launch.get('observer_pid')==pipeline.get('pid')
        and launch.get('observer_start_ticks')==pipeline.get('start_ticks') and launch.get('phase')=='native'
        and launch.get('argv',[])[:3]==['timeout','--signal=TERM','--kill-after=15s']
        and launch.get('argv',[])[4:]==base+['--mode','native'], 'Qualification launch changed')
    require(exit0.get('actual_child_exit_observed') is True and type(exit0.get('returncode')) is int
        and exit0['returncode']==0 and exit0.get('contract_sha256')==digest and exit0.get('phase')=='native'
        and exit0.get('worker_group_pid')==launch['pid'] and exit0.get('supervisor_pid')==pipeline['pid']
        and exit0.get('argv')==launch['argv'] and exit0.get('launch_sha256')==sha_file(root/'qualification/launch.json'),
        'Qualification actual exit binding missing')
    require(set(q.get('synthetic',{}))==set(ARMS) and all(q['synthetic'][a].get('status')=='passed'
        and q['synthetic'][a].get('contract_sha256')==digest and q['synthetic'][a].get('arm')==a
        and q['synthetic'][a].get('active_trainable_parameter_count')==79363
        and q['synthetic'][a].get('registered_parameter_count')==104095
        and q['synthetic'][a].get('active_local_cnn_parameters')==512
        and q['synthetic'][a].get('active_value_difference_parameters')==128
        and q['synthetic'][a].get('original_value_kernel_absent') is True
        and q['synthetic'][a].get('value_operation')=='adjacent_observed_differences' for a in ARMS),
        'Candidate synthetic qualification missing')
    cap=q['capacity_check']
    require(cap.get('maximum_workers')==3 and cap.get('passed') is True
        and cap.get('simultaneous_synthetic_workers_executed') is True
        and cap.get('worker_actual_exit0_count')==3 and bool(cap.get('evidence_files')),'GPU capacity missing')
    for n,d in cap['evidence_files'].items():require(sha_file(safe_relative(root,n))==d,'Admission evidence drift')
    for phase in ('ready','trained'):
        v=read(root/'qualification/admission'/(phase+'_concurrency.json'));m=v['memory']
        require(v['contract_sha256']==digest and len(v['workers'])==3 and m['gpu_uuid']==c['hosts']['5090']['gpu_uuid']
            and m['total_bytes']>0 and 0<=m['used_bytes']<=.8*m['total_bytes'] and m['limit_fraction']==.8,
            'Admission aggregate memory/identity missing')
    for i,a,seed in ADMISSION_JOBS:
        directory=root/'qualification/admission';l=read(directory/(i+'.launch.json'));e=read(directory/(i+'.actual_exit.json'));r=read(directory/(i+'.ready.json'))
        require(l.get('contract_sha256')==digest and e.get('contract_sha256')==digest and r.get('contract_sha256')==digest
            and e.get('admission_id')==i and e.get('arm')==a and e.get('seed')==seed
            and e.get('actual_child_exit_observed') is True and type(e.get('returncode')) is int and e['returncode']==0
            and e.get('worker_group_pid')==l.get('pid')==r.get('pid') and l.get('start_ticks')==r.get('start_ticks')
            and e.get('supervisor_pid')==q['pid']==l.get('observer_pid') and e.get('argv')==l.get('argv')
            and e.get('launch_sha256')==sha_file(directory/(i+'.launch.json')),'Admission actual exit binding missing')
    return p



def synthetic_arm(c, root, arm, *, admission_id=None, seed=42):
    """Exercise native gradients and target isolation using fabricated data only."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    data=c['datasets'][0]
    torch.cuda.set_per_process_memory_fraction(.30,0)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed);model,metadata=runner.build_model(data,arm);model.cuda()
        require(sum(p.numel() for p in model.parameters() if p.requires_grad)==metadata['active_trainable_parameter_count']==79363
            and sum(p.numel() for p in model.parameters())==metadata['registered_parameter_count']==104095
            and metadata['cnn_parameter_count']==512 and metadata['value_difference_parameter_delta']==-64,
            'RAF model parameter identity changed')
        attn=model.encoder.layers[0].attn
        require('causal_v_kernel' not in dict(attn.named_parameters())
            and tuple(attn.value_difference_kernel.shape)==(2,64)
            and tuple(attn.causal_q_kernel.shape)==tuple(attn.causal_k_kernel.shape)==(3,64),
            'V difference / preserved QK parameter shape drift')
        generator=torch.Generator().manual_seed(831);shape=(128,data['loader']['max_seq_len'])
        dt=torch.randint(1,8,shape,generator=generator).float().cuda();dt[:,0]=0
        qty=torch.randint(0,40,shape,generator=generator).float().cuda()
        mask=torch.ones_like(dt,dtype=torch.bool)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        prefix='encoder.layers.0.attn.'
        difference_name=prefix+'value_difference_kernel'
        names={prefix+'causal_q_kernel',prefix+'causal_k_kernel',difference_name};active=set()
        parameters=dict(model.named_parameters())
        require(names<=set(parameters) and sum(parameters[n].numel() for n in names)==512,
            'Local CNN parameter budget changed')
        ready=time.time()
        if admission_id is not None:
            write_json(root/'qualification/admission'/(admission_id+'.ready.json'),
                {'admission_id':admission_id,'arm':arm,'seed':seed,'start_ticks':start_ticks(os.getpid()),'pid':os.getpid(),'argv':[sys.executable,*sys.argv],
                 'ready_unix':ready,'contract_sha256':sha_json(c)},exclusive=True)
            until=time.monotonic()+180
            while not (root/'qualification/admission/go.json').exists():
                require(time.monotonic()<until,'Admission barrier timed out')
                time.sleep(.1)
            go=read(root/'qualification/admission/go.json')
            require(go['contract_sha256']==sha_json(c),'Foreign admission barrier')
        start=time.time()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)['joint_loss'].mean()
            require(bool(torch.isfinite(loss)),'Nonfinite synthetic loss');loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Nonfinite gradient')
            active.update(n for n,p in model.named_parameters() if n in names and p.grad is not None and bool(p.grad.ne(0).any()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        require(active==names,'Inactive local CNN parameters')
        model.eval()
        with torch.no_grad():
            d=dt[:1,:8].clone();v=qty[:1,:8].clone();valid=mask[:1,:8]
            write=valid.clone();write[:,-1]=False
            before=canonical_state_dict_sha256(model.state_dict())
            a=model.encode_task_states(d,v,valid,memory_write_mask=write)[0]
            d[:,-1]=float('nan');v[:,-1]=float('inf')
            b=model.encode_task_states(d,v,valid,memory_write_mask=write)[0]
            require(torch.equal(a,b),'Target leak in native model')
            require(before==canonical_state_dict_sha256(model.state_dict()),'Evaluation mutated model')
        torch.cuda.synchronize()
        result={'status':'passed','arm':arm,'admission_id':admission_id,'seed':seed,'start_ticks':start_ticks(os.getpid()),'pid':os.getpid(),'shape':list(shape),
            'updates':4,'scientific_updates':0,'started_unix':start,'completed_unix':time.time(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'active_trainable_parameter_count':metadata['active_trainable_parameter_count'],
            'registered_parameter_count':metadata['registered_parameter_count'],
            'active_local_cnn_parameters':sum(parameters[n].numel() for n in active),
            'active_value_difference_parameters':parameters[difference_name].numel(),
            'original_value_kernel_absent':True,'value_operation':'adjacent_observed_differences',
            'target_blocked':True,'eval_state_unchanged':True,'contract_sha256':sha_json(c)}
        require(result['active_local_cnn_parameters']==512 and result['active_value_difference_parameters']==128,'Parameter budget changed')
        if admission_id is not None:
            write_json(root/'qualification/admission'/(admission_id+'.trained.json'),result,exclusive=True)
            until=time.monotonic()+180
            while not (root/'qualification/admission/release.json').exists():
                require(time.monotonic()<until,'Admission release timed out')
                time.sleep(.1)
        return result


def admission_worker(c,host,root,admission_id):
    authorization(c,root)
    spec=next((x for x in ADMISSION_JOBS if x[0]==admission_id),None)
    require(spec is not None,'Foreign admission ID')
    _,arm,seed=spec
    common.runtime_check(c,host)
    result=synthetic_arm(c,root,arm,admission_id=admission_id,seed=seed)
    write_json(root/'qualification/admission'/(admission_id+'.result.json'),result,exclusive=True)


def admission_memory(c,host):
    output=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout
    rows=[r for r in csv.reader(output.splitlines()) if r and r[0].strip()==c['hosts'][host]['gpu_uuid']]
    require(len(rows)==1,'Pinned GPU missing at admission')
    total,used=[float(rows[0][i].strip())*1024**2 for i in (1,2)]
    require(math.isfinite(total) and math.isfinite(used) and total>0 and 0<=used<=PARALLELISM['aggregate_gpu_fraction']*total,
        'Aggregate GPU memory over80 percent before scientific training')
    return {'gpu_uuid':c['hosts'][host]['gpu_uuid'],'total_bytes':total,'used_bytes':used,
        'aggregate_fraction':used/total,'limit_fraction':PARALLELISM['aggregate_gpu_fraction']}


def three_worker_admission(c,host,root):
    """Three distinct fabricated candidate workers; no CNN optimizer steps."""
    directory=root/'qualification/admission';directory.mkdir(exist_ok=False)
    children={};script=Path(__file__).resolve();start=time.monotonic()
    try:
        for admission_id,arm,seed in ADMISSION_JOBS:
            command=[c['hosts'][host]['python'],str(script),'--contract',str(root/'execution_contract.json'),
                     '--host',host,'--mode','admission','--admission-id',admission_id]
            with (directory/(admission_id+'.log')).open('xb') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True)
            launch=launch_identity(child,command,sha_json(c),admission_id=admission_id,arm=arm,seed=seed)
            write_json(directory/(admission_id+'.launch.json'),launch,exclusive=True)
            children[admission_id]=(child,command,launch)
        for phase in ('ready','trained'):
            while not all((directory/(i+'.'+phase+'.json')).exists() for i,_,_ in ADMISSION_JOBS):
                require(time.monotonic()-start<300,'Three-worker admission timed out')
                require(all(child.poll() is None for child,_,_ in children.values()),'Admission worker exited before barrier')
                time.sleep(.1)
            inventory=common.gpu_pids(c['hosts'][host]);pids={child.pid for child,_,_ in children.values()}
            require(pids.issubset(inventory),'Not all three admission workers are GPU-live')
            require(not (set(inventory)-pids-{os.getpid()}),'Foreign GPU process during admission')
            memory=admission_memory(c,host);proofs=[]
            for admission_id,arm,seed in ADMISSION_JOBS:
                child,command,launch=children[admission_id]
                value=read(directory/(admission_id+'.'+phase+'.json'))
                require(value['arm']==arm and value['seed']==seed and value['admission_id']==admission_id
                    and value['pid']==child.pid and value['start_ticks']==launch['start_ticks']
                    and value['contract_sha256']==sha_json(c),'Admission identity mismatch')
                live=_gpu_process_snapshot(child.pid)
                require(live['consistent'] and live['state'] not in ('Z','X') and live['start_ticks']==launch['start_ticks']
                    and live['argv']==command and live['pgrp']==child.pid,'Admission process identity mismatch')
                proofs.append({'pid':child.pid,**live})
            write_json(directory/(phase+'_concurrency.json'),{'workers':proofs,'gpu_pids':sorted(inventory),
                'memory':memory,'observed_unix':time.time(),'contract_sha256':sha_json(c)},exclusive=True)
            target='go.json' if phase=='ready' else 'release.json'
            write_json(directory/target,{'contract_sha256':sha_json(c),'issued_unix':time.time()},exclusive=True)
        for admission_id,arm,seed in ADMISSION_JOBS:
            child,command,launch=children[admission_id];child.wait(timeout=30)
            receipt=observe_exit(child,command,sha_json(c),arm=arm,seed=seed,admission_id=admission_id,
                launch_sha256=sha_file(directory/(admission_id+'.launch.json')))
            write_json(directory/(admission_id+'.actual_exit.json'),receipt,exclusive=True)
            require(receipt['returncode']==0,'Native admission failed: '+admission_id)
            require(not Path('/proc',str(child.pid)).exists(),'Admission child PID remains')
        require(not ({child.pid for child,_,_ in children.values()}&set(common.gpu_pids(c['hosts'][host]))),'Admission GPU worker remains')
        return {'passed':True,'maximum_workers':3,'simultaneous_synthetic_workers_executed':True,
            'method':'three fabricated candidate workers at full batch and sequence length; live PID/GPU and memory barriers',
            'admission_jobs':[{'id':i,'arm':a,'seed':n} for i,a,n in ADMISSION_JOBS],
            'worker_actual_exit0_count':3,'scientific_optimizer_updates':0,
            'evidence_files':{str(p.relative_to(root)):sha_file(p) for p in directory.iterdir() if p.is_file()}}
    except BaseException:
        for child,_,_ in children.values():stop_owned(child)
        raise


def native_qualification(c,host,root):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    authorization(c,root)
    require(not common.gpu_pids(c['hosts'][host]),'GPU occupied before qualification')
    runtime=common.runtime_check(c,host)
    require(runtime['python']=='3.12.13' and runtime['threads']==4 and runtime['interop_threads']==1,'Native runtime drift')
    data=c['datasets'][0];name=data['dataset_id'];frame,receipt=base.prepare_admitted_data(data,data_root=ROOT)
    q={'status':'checking','host':host,'contract_sha256':sha_json(c),'runtime':runtime,
       'source_closure':c['source']['files_sha256'],'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),
       'ppid':os.getppid(),'guard_start_ticks':start_ticks(os.getppid()),'argv':[sys.executable,*sys.argv],
       'inputs':{name:{'input_receipt':receipt,'input_receipt_sha256':sha_json(receipt),
          'interface_sha256':{a:sha_json(runner.time_interface(data,frame,c,arm=a)) for a in ARMS}}},
       'initialization':{name:{str(s):runner.initial_states(data,s,arms=ARMS) for s in SEEDS}},
       'synthetic':{},'held_out_test_evaluated':False,'scientific_optimizer_updates':0}
    for job in c['baseline_jobs']:
        initial_sha=runner.baseline_reference_initial_state_sha(data,job['seed'])
        require(initial_sha==job['original']['initial_state_sha256'],'Common initialization mismatch against stored CNN hash')
    for arm in ARMS:q['synthetic'][arm]=synthetic_arm(c,root,arm)
    torch.cuda.empty_cache()
    q['capacity_check']=three_worker_admission(c,host,root)
    q.update(status='passed',completed_unix=time.time())
    write_json(root/'qualification/receipt.json',q,exclusive=True)


def dispatch(c,host,root):
    start=authorization(c,root);verify_training_permit(c,root)
    require(not common.gpu_pids(c['hosts'][host]),'GPU occupied before fresh fits')
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(root/'supervisor.claim',{'pid':os.getpid(),'argv':[sys.executable,*sys.argv],
            'start_ticks':start_ticks(os.getpid()),'ppid':os.getppid(),'guard_start_ticks':start_ticks(os.getppid()),
            'contract_sha256':sha_json(c),'started_unix':time.time()},exclusive=True)
        active={};results={};script=Path(__file__).resolve()
        try:
            # One candidate, three independent fresh seeds, maximum three workers.
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
                    launch=launch_identity(child,command,sha_json(c),job=job,deadline_unix=deadline)
                    write_json(root/'worker_launches'/(job['id']+'.json'),launch,exclusive=True)
                    active[job['id']]={'job':job,'child':child,'deadline':deadline,'command':command,'launch':launch}
                    write_parallel_pulse(c,host,root,start,active,results)
                while active:
                    write_parallel_pulse(c,host,root,start,active,results)
                    gpu_observation(c,host,active,script)
                    finished=[]
                    for name,item in active.items():
                        if item['child'].poll() is None: continue
                        receipt=observe_exit(item['child'],item['command'],sha_json(c),job=item['job'],
                            launch_sha256=sha_file(root/'worker_launches'/(name+'.json')),
                            worker_identity_sha256=sha_file(root/'worker_identities'/(name+'.json')),
                            worker_claim_sha256=sha_file(root/'claims'/(name+'.json')))
                        write_json(root/'worker_exits'/(name+'.json'),receipt,exclusive=True)
                        require(receipt['returncode']==0,'Worker failed; no retry: '+name)
                        results[name]=validate_terminal(c,root,item['job']);finished.append(name)
                    for name in finished:del active[name]
                    if active:time.sleep(5)
                gpu_observation(c,host,active,script)
            for job in c['jobs']:
                p=root/'run'/job['id']/'runs'/job['arm']/frozen.VARIANT/f"seed_{job['seed']}"/'exposure.json'
                frozen.audit_host_batch_prefix(c,host,root,job,read(p))
            write_json(root/'status.json',{'status':'complete','completed':results,'completed_conditions':3,
                'updated_unix':time.time(),'maximum_concurrency':3,'held_out_test_evaluated':False,
                'contract_sha256':sha_json(c)})
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
    write_json(root/'pipeline.claim',{'pid':os.getpid(),'started_unix':time.time(),'start_ticks':start_ticks(os.getpid()),
        'ppid':os.getppid(),'wrapper_start_ticks':start_ticks(os.getppid()),'argv':[sys.executable,*sys.argv],
        'contract_sha256':sha_json(c)},exclusive=True)
    write_json(root/'start_permit.json',new_start_permit(c,host,approval,time.time()),exclusive=True)
    start=authorization(c,root);child=None
    try:
        for mode in ('native','dispatch'):
            deadline=min(start['deadline_unix'],time.time()+c['limits']['qualification_seconds']) if mode=='native' else start['deadline_unix']
            command=timeout_command(Path(__file__).resolve(),c,host,['--mode',mode],deadline)
            path=root/('qualification/native.log' if mode=='native' else 'logs/dispatcher.log');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                launch_path=root/('qualification/launch.json' if mode=='native' else 'dispatcher_launch.json')
                write_json(launch_path,launch_identity(child,command,sha_json(c),phase=mode),exclusive=True)
                child.wait()
            receipt=observe_exit(child,command,sha_json(c),phase=mode,launch_sha256=sha_file(launch_path))
            write_json(root/('qualification/actual_exit.json' if mode=='native' else 'dispatcher_exit.json'),receipt,exclusive=True)
            require(child.returncode==0,mode+' actual exit '+str(child.returncode))
            require(not common.gpu_pids(c['hosts'][host]),'Child GPU process remains')
            if mode=='native':
                q=read(root/'qualification/receipt.json')
                write_json(root/'training_permit.json',{'contract_sha256':sha_json(c),'qualification':q,
                    'qualification_sha256':sha_file(root/'qualification/receipt.json'),
                    'start_permit_sha256':sha_json(start)},exclusive=True)
        write_json(root/'train_complete.json',{'status':'complete','scientific_worker_count':3,
            'dispatcher_actual_exit':0,'contract_sha256':sha_json(c),'completed_unix':time.time()},exclusive=True)
    except BaseException as exc:
        if child is not None:stop_owned(child)
        write_json(root/'pipeline_failure.json',{'message':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False},exclusive=True)
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--host',choices=['5090'],required=True)
    p.add_argument('--mode',choices=['pipeline','native','admission','dispatch','fit'],required=True);p.add_argument('--job');p.add_argument('--deadline',type=float);p.add_argument('--admission-id',choices=[x[0] for x in ADMISSION_JOBS])
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
        elif a.mode=='admission':admission_worker(c,a.host,root,a.admission_id)
        elif a.mode=='dispatch':dispatch(c,a.host,root)
        else:run_fit(c,a.host,root,next(j for j in c['jobs'] if j['id']==a.job),a.deadline)

if __name__=='__main__':main()
