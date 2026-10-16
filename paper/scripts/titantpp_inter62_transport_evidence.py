"""Preserve bounded SSH transport output; a timeout never proves remote exit."""
import hashlib,json,subprocess,time
from pathlib import Path


def _bytes(value):
 return value.encode() if isinstance(value,str) else (value or b'')


def recorded_run(bundle,label,argv,*,input=None,timeout=120,stdout=None):
 folder=Path(bundle)/'control';folder.mkdir(exist_ok=True)
 started=time.time();timed_out=False;code=None;out=b'';err=b''
 try:
  result=subprocess.run(argv,input=input,stdout=stdout or subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
  code=result.returncode;out=_bytes(result.stdout);err=_bytes(result.stderr)
 except subprocess.TimeoutExpired as error:
  timed_out=True;out=_bytes(error.stdout);err=_bytes(error.stderr)
  result=None
 finally:
  if stdout is None:
   op=folder/(label+'_stdout.bin')
   with op.open('xb') as f:f.write(out)
   out_sha=hashlib.sha256(out).hexdigest();out_bytes=len(out)
  else:
   stdout.flush();op=Path(stdout.name);raw=op.read_bytes();out_sha=hashlib.sha256(raw).hexdigest();out_bytes=len(raw)
  with (folder/(label+'_stderr.bin')).open('xb') as f:f.write(err)
  receipt={'actual_transport_exit':code,'actual_transport_exit_observed':type(code) is int,'timeout':timed_out,'remote_exit_inferred_from_timeout':False,'stdout_sha256':out_sha,'stderr_sha256':hashlib.sha256(err).hexdigest(),'stdout_bytes':out_bytes,'stderr_bytes':len(err),'started_unix':started,'completed_unix':time.time(),'automatic_retry':False}
  if label=='native_audit':receipt['model_predictions']=0
  with (folder/(label+'_exit.json')).open('x') as f:json.dump(receipt,f,indent=2)
 if timed_out:raise subprocess.TimeoutExpired(argv,timeout,output=out,stderr=err)
 return result
