"""Read-only health permission bridge; preserve sealed v2 execution tools.

Only package-lock observation changes: stat + /proc/locks instead of opening
root-owned lock files. All deployment, qualification and training use sealed v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper.scripts import launch_titantpp_cnn_gru_components_v2 as sealed

CONTRACT_SHA = "5d5abc55beb4c7543504f52ccf1fe43e8576bef9554226e40afe0397a5bbad0d"
LAUNCHER_SHA = "2d1a6a11511f7f9df41b60402b9f6c124428d127c0ec67d21065b9fe95cbb78c"
OLD = ''' locks=[]
 for name in ('/var/lib/dpkg/lock','/var/lib/dpkg/lock-frontend','/var/lib/apt/lists/lock','/var/cache/apt/archives/lock'):
  if not pathlib.Path(name).exists(): continue
  fd=os.open(name,os.O_RDONLY)
  try:
   try: fcntl.lockf(fd,fcntl.LOCK_SH|fcntl.LOCK_NB); fcntl.lockf(fd,fcntl.LOCK_UN)
   except BlockingIOError: locks.append(name)
  finally: os.close(fd)
'''
NEW = ''' locks=[]
 import re
 lock_text=pathlib.Path('/proc/locks').read_text()
 locked_objects=set()
 for row in lock_text.splitlines():
  for token in row.split():
   match=re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9]+)',token)
   if match:
    locked_objects.add((int(match.group(1),16),int(match.group(2),16),int(match.group(3))))
 for name in ('/var/lib/dpkg/lock','/var/lib/dpkg/lock-frontend','/var/lib/apt/lists/lock','/var/cache/apt/archives/lock'):
  path=pathlib.Path(name)
  if not path.exists(): continue
  stat=path.stat()
  if (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino) in locked_objects: locks.append(name)
'''


def health_script():
    sealed.require(sealed.sha(sealed.__file__) == LAUNCHER_SHA, "Sealed launcher changed")
    sealed.require(sealed.REMOTE_SCRIPT.count(OLD) == 1, "Health bridge match changed")
    result = sealed.REMOTE_SCRIPT.replace(OLD, NEW, 1)
    compile(result, "health_compat_remote", "exec")
    return result


class CompatibleRemote(sealed.SSHRemote):
    def call(self, host, action, payload):
        sealed.require(host == payload["host"] == "5080", "Compatibility bridge cannot contact5090")
        sealed.require(payload["contract_sha256"] == CONTRACT_SHA
            and sealed.canonical(payload["contract"]) == CONTRACT_SHA, "Compatibility contract changed")
        original = sealed.REMOTE_SCRIPT
        if action == "health":
            sealed.REMOTE_SCRIPT = health_script()
        try:
            return super().call(host, action, payload)
        finally:
            sealed.REMOTE_SCRIPT = original

    def upload(self, host, local, remote_path):
        sealed.require(host == "5080", "Compatibility bridge cannot upload to5090")
        return super().upload(host, local, remote_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--mode", choices=("check", "deploy", "qualify", "train", "status"), required=True)
    a = p.parse_args()
    c, proof = sealed.check_bundle(a.bundle)
    sealed.require(proof["contract_sha256"] == CONTRACT_SHA, "Compatibility contract changed")
    health_script()
    if a.mode != "check":
        sealed.authorization(a.bundle, c, "5080")
        r = Path(a.bundle) / "administrative_health_compatibility.json"
        content = {"contract_sha256": CONTRACT_SHA, "host": "5080", "sealed_launcher_raw_sha256": LAUNCHER_SHA,
            "compatibility_tool_raw_sha256": sealed.sha(__file__),
            "health_script_raw_sha256": hashlib.sha256(health_script().encode()).hexdigest(),
            "scope": "package locks: read-only stat/proc locks; remote permissions, runtime and science unchanged"}
        if r.exists():
            sealed.require(sealed.read(r) == content, "Administrative bridge provenance changed")
        else:
            sealed.write_exclusive(r, content)
    result = sealed.run_mode(a.bundle, a.mode, "5080", remote=CompatibleRemote())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
