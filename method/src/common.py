import json,select
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))

def worker_read(worker,timeout=120):
 ready,_,_=select.select([worker.stdout],[],[],timeout)
 if not ready:raise TimeoutError('Camera model worker did not respond')
 text=worker.stdout.readline()
 if not text:raise RuntimeError('Camera model worker ended')
 return json.loads(text)

def send(worker,obj):
 worker.stdin.write(json.dumps(obj)+'\n');worker.stdin.flush()
 x=worker_read(worker)
 if 'error' in x:raise RuntimeError(x['error'])
 return x
