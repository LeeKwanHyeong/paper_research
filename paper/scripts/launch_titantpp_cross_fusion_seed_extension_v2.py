"""Read file-based admission receipts before the first seed-extension launch."""
import argparse
import hashlib
import json
import shlex

from paper.scripts import manage_titantpp_cross_fusion_seed_extension as m


def check_install_admit(host):
    """Only the known successful pre-launch stage can be continued."""
    paths = sorted((m.BUNDLE / "transport" / host).glob("*.json"))
    if len(paths) != 2 or [p.stem.rsplit("_", 1)[-1] for p in paths] != ["install", "admit"]:
        raise ValueError("Expected exactly one successful install and admission, and no launch")
    for p in paths:
        r = m.read(p)
        if r["actual_transport_exit"] != 0:
            raise ValueError("Failed transport cannot be resumed")
        for suffix in ("stdout", "stderr"):
            data = p.with_suffix("." + suffix).read_bytes()
            if len(data) != r[suffix + "_bytes"] or hashlib.sha256(data).hexdigest() != r[suffix + "_sha256"]:
                raise ValueError("Transport evidence changed")
    attempt = m.read(m.BUNDLE / "attempts" / f"{host}.json")
    if attempt["host"] != host or attempt["contract_sha256"] != m.canonical(m.verify_local()):
        raise ValueError("Attempt identity mismatch")


def check_snapshot(snapshot, c, host):
    q = snapshot["qualification"]
    if (snapshot["existing"] or snapshot["tmux_exit"] != 1 or snapshot["processes"]
            or snapshot["contract_raw_sha256"] != m.read(m.REPORT / "PREPARED.json")["contract_sha256"]
            or q["status"] != "passed" or q["host"] != host
            or q["contract_sha256"] != m.canonical(c)
            or q["inherited_native_qualification"] != c["parent"]["qualification_sha256"][host]):
        raise ValueError("First-start qualification or explicit absence check failed")


def launch(host):
    c = m.verify_local()
    h = c["hosts"][host]
    intent = m.BUNDLE / "launch_v2" / f"{host}.json"
    if intent.exists():
        raise FileExistsError("First launch already attempted; no automatic retry")
    if not (m.BUNDLE / "attempts" / f"{host}.json").exists():
        if m.sha(m.BUNDLE / "deployment.tar.gz") != m.read(m.REPORT / "PREPARED.json")["archive_sha256"]:
            raise ValueError("Archive changed")
        m.write(m.BUNDLE / "attempts" / f"{host}.json", {"created_utc": m.now(), "host": host,
            "contract_sha256": m.canonical(c), "automatic_retry": False})
        install = "mkdir " + shlex.quote(h["root"]) + " && tar -xzf - -C " + shlex.quote(h["root"])
        m.save_transport(host, "install", m.ssh(host, install, stdin=(m.BUNDLE / "deployment.tar.gz").read_bytes()))
        m.save_transport(host, "admit", m.ssh(host, m.invocation(c, host, "admit"), timeout=180))
    check_install_admit(host)
    code = '''import pathlib,json,subprocess,hashlib,os
r=pathlib.Path(__ROOT__)
names=('start.json','status.json','final_receipt.json','dispatcher_exit.json','dispatch_failure.json','run','logs')
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().split(b'\\0')
  if str(r/'execution_contract.json').encode() in argv:processes.append(int(p.name))
 except (FileNotFoundError,ProcessLookupError):pass
tmux=subprocess.run([__TMUX__,'has-session','-t',__SESSION__],capture_output=True)
print(json.dumps({'qualification':json.loads((r/'qualification.json').read_text()),
 'contract_raw_sha256':hashlib.sha256((r/'execution_contract.json').read_bytes()).hexdigest(),
 'existing':[n for n in names if (r/n).exists()],'processes':processes,'tmux_exit':tmux.returncode}))
'''.replace("__ROOT__", repr(h["root"])).replace("__TMUX__", repr(h["tmux_binary"])).replace("__SESSION__", repr(h["tmux"]))
    snapshot = json.loads(m.save_transport(host, "prelaunch_v2", m.ssh(host, shlex.join([h["python"], "-c", code]))))
    check_snapshot(snapshot, c, host)
    m.write(intent, {"created_utc": m.now(), "host": host, "contract_sha256": m.canonical(c),
        "qualification_reexecuted": False, "snapshot": snapshot, "launcher_sha256": m.sha(__file__)})
    command = [h["python"], "-u", "-m", m.RUNTIME, "--contract", h["root"] + "/execution_contract.json", "--host", host, "--mode", "dispatch"]
    supervisor = (
        "import os,subprocess,json,pathlib,sys,datetime; "
        f"p=subprocess.Popen({command!r},cwd={h['source_root']!r},env={{**os.environ,**{h['environment']!r}}}); "
        "code=p.wait(); "
        f"path=pathlib.Path({h['root']!r})/'dispatcher_exit.json'; "
        "path.write_text(json.dumps({'pid':p.pid,'actual_os_returncode':code,"
        "'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),"
        "'supervisor_own_exit_directly_observed':False})); sys.exit(code)")
    shell = shlex.join([h["python"], "-c", supervisor]) + " > " + shlex.quote(h["root"] + "/dispatcher.log") + " 2>&1"
    m.save_transport(host, "launch_v2", m.ssh(host, shlex.join([h["tmux_binary"], "new-session", "-d", "-s", h["tmux"], shell])))
    result = {"status": "queue_dispatched_not_yet_verified", "host": host, "utc": m.now(),
        "root": h["root"], "maximum_combined_workers": 3, "qualification": snapshot["qualification"]}
    m.write(m.REPORT / f"DISPATCHED_{host}.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    print(json.dumps(launch(parser.parse_args().host), indent=2))
