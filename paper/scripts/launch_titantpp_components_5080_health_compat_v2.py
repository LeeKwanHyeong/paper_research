"""Identify the observed shutdown waiter; retain active-update blocking.

Sealed science and v2 operations are unchanged. The exact helper is accepted
only with pinned source, sleeping/root/systemd identity and logind shutdown=false.
"""
import argparse
import hashlib
import json
from pathlib import Path

from paper.scripts import launch_titantpp_components_5080_health_compat as previous

sealed = previous.sealed
PREVIOUS_SHA = "fac25ea166692e1500169cce6ad7a1c8bef566a4ee9fabe3b4274c6ead6e31be"
UPDATES_OLD = '''   if comm in ('apt','apt-get','dpkg','unattended-upgr','packagekitd'): updates.append({'pid':int(p.name),'comm':comm})'''
UPDATES_NEW = '''   if comm in ('apt','apt-get','dpkg','unattended-upgr','packagekitd'):
    observer=False
    if comm=='unattended-upgr':
     try:
      before=(p/'stat').read_text().rsplit(')',1)[1].split()
      argv=(p/'cmdline').read_bytes().decode().rstrip('\\x00').split('\\x00')
      status=(p/'status').read_text()
      after=(p/'stat').read_text().rsplit(')',1)[1].split()
      expected=['/usr/bin/python3','/usr/share/unattended-upgrades/unattended-upgrade-shutdown','--wait-for-signal']
      if argv==expected and before[19]==after[19] and after[0]=='S' and int(after[1])==1:
       uid=next(line.split()[1:] for line in status.splitlines() if line.startswith('Uid:'))
       helper_sha=sha(expected[1])
       shutdown=capture(['busctl','get-property','org.freedesktop.login1','/org/freedesktop/login1','org.freedesktop.login1.Manager','PreparingForShutdown']).strip()
       observer=uid==['0','0','0','0'] and helper_sha=='8fbf4e278c4ad457e82cc05586631b1a0550dd9d10892a70487ef74c623066e0' and shutdown=='b false'
       if observer: idle_shutdown_observers.append({'pid':int(p.name),'argv':argv,'state':after[0],'ppid':1,'start_ticks':after[19],'helper_source_sha256':helper_sha,'PreparingForShutdown':False})
     except (OSError,ValueError,StopIteration,subprocess.SubprocessError): observer=False
    if not observer: updates.append({'pid':int(p.name),'comm':comm})'''


def health_script():
    sealed.require(sealed.sha(previous.__file__) == PREVIOUS_SHA, "Previous administrative bridge changed")
    script = previous.health_script()
    sealed.require(script.count(UPDATES_OLD) == 1, "Observer health block changed")
    script = script.replace(" updates=[]", " updates=[]; idle_shutdown_observers=[]", 1)
    script = script.replace(UPDATES_OLD, UPDATES_NEW, 1)
    script = script.replace("'update_processes':updates,", "'update_processes':updates,'idle_shutdown_observers':idle_shutdown_observers,", 1)
    compile(script, "health_observer_compat", "exec")
    return script


class CompatibleRemote(previous.CompatibleRemote):
    def call(self, host, action, payload):
        sealed.require(host == payload["host"] == "5080", "Cannot contact5090")
        sealed.require(payload["contract_sha256"] == previous.CONTRACT_SHA
            and sealed.canonical(payload["contract"]) == previous.CONTRACT_SHA, "Foreign contract")
        original = sealed.REMOTE_SCRIPT
        if action == "health": sealed.REMOTE_SCRIPT = health_script()
        try: return sealed.SSHRemote.call(self, host, action, payload)
        finally: sealed.REMOTE_SCRIPT = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "deploy", "qualify", "train", "status"), required=True)
    a = parser.parse_args()
    c, proof = sealed.check_bundle(a.bundle)
    sealed.require(proof["contract_sha256"] == previous.CONTRACT_SHA, "Foreign contract")
    script = health_script()
    if a.mode != "check":
        sealed.authorization(a.bundle, c, "5080")
        record = {"contract_sha256": previous.CONTRACT_SHA, "host": "5080",
            "sealed_launcher_raw_sha256": previous.LAUNCHER_SHA,
            "previous_bridge_raw_sha256": PREVIOUS_SHA, "compatibility_tool_raw_sha256": sealed.sha(__file__),
            "health_script_raw_sha256": hashlib.sha256(script.encode()).hexdigest(),
            "scope": "Read-only package locks and pinned shutdown-waiter classification; active/unknown updates remain blocking"}
        path = a.bundle / "administrative_health_compatibility_v2.json"
        if path.exists(): sealed.require(sealed.read(path) == record, "Administrative provenance changed")
        else: sealed.write_exclusive(path, record)
    print(json.dumps(sealed.run_mode(a.bundle, a.mode, "5080", remote=CompatibleRemote()),
        indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__": main()
