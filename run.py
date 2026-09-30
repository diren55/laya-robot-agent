"""Portable launcher for the frozen four-family no-generative-LLM agent."""
import argparse,datetime,json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',default=None);p.add_argument('--seeds',type=int,nargs='+',default=[1061201])
 p.add_argument('--tasks',nargs='+',choices=['CubeLift','CubeStack','CubeRestack','SpillWipe'],default=['CubeLift','CubeStack','CubeRestack','SpillWipe'])
 p.add_argument('--sim-python',default=str(ROOT/'.venvs/sim/bin/python'))
 p.add_argument('--vision-python',default=str(ROOT/'.venvs/vision/bin/python'))
 p.add_argument('--motion-python',default=str(ROOT/'.venvs/motion/bin/python'))
 p.add_argument('--ik-port',type=int,default=18116)
 args=p.parse_args()
 for exe in [args.sim_python,args.vision_python,args.motion_python]:
  if not Path(exe).is_file():p.error('Missing runtime: '+exe+'; first run bash setup.sh')
 out=Path(args.out).expanduser().resolve() if args.out else ROOT/'runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
 if out.exists():p.error('Output already exists; choose a new directory')
 out.parent.mkdir(parents=True,exist_ok=True);(ROOT/'runtime_state').mkdir(exist_ok=True)
 env=os.environ.copy()
 env.update(LAYA_AGENT_CAPX_REPO=str(ROOT/'vendor/capx'),LAYA_AGENT_MODEL_ROOT=str(ROOT/'runtime'),
  LAYA_AGENT_CHECKPOINT=str(ROOT/'weights/T4'),LAYA_AGENT_DINO=str(ROOT/'runtime/models/grounding_dino_tiny'),
  LAYA_AGENT_WORKER_PYTHON=args.vision_python,LAYA_AGENT_GPU_LOCK=str(ROOT/'runtime_state/gpu.lock'),
  LAYA_AGENT_IK_URL='http://127.0.0.1:'+str(args.ik_port),MUJOCO_GL='egl',OMP_NUM_THREADS='2',
  TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
  PYTHONPATH=os.pathsep.join([str(ROOT/'vendor/robosuite'),str(ROOT/'vendor/capx')]))
 lib='/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
 if Path(lib).exists():env.setdefault('LD_PRELOAD',lib)
 with socket.socket() as sock:
  if sock.connect_ex(('127.0.0.1',args.ik_port))==0:p.error('IK port occupied. Select another --ik-port.')
 log=out.parent/(out.name+'_motion.log')
 with log.open('w') as stream:
  service=subprocess.Popen([args.motion_python,'-u',str(ROOT/'start_motion.py'),'--port',str(args.ik_port)],
      stdout=stream,stderr=subprocess.STDOUT,env={**env,'JAX_PLATFORMS':'cpu'})
  try:
   deadline=time.monotonic()+180
   while True:
    if service.poll() is not None:raise RuntimeError('Motion service failed; inspect '+str(log))
    try:
     with urllib.request.urlopen(env['LAYA_AGENT_IK_URL']+'/openapi.json',timeout=2):break
    except (OSError,ValueError):
     if time.monotonic()>deadline:raise TimeoutError('Motion startup timed out; inspect '+str(log))
     time.sleep(.25)
   cmd=[args.sim_python,'-u',str(ROOT/'method/src/run_native.py'),'--out',str(out),'--seeds',*map(str,args.seeds),'--tasks',*args.tasks]
   result=subprocess.run(cmd,env=env)
   if result.returncode:raise SystemExit(result.returncode)
   complete=out/'COMPLETE.json'
   if not complete.exists():raise RuntimeError('Runner returned without COMPLETE.json')
   print(json.dumps({'output':str(out),'summary':json.loads(complete.read_text())['summary']},ensure_ascii=False))
  finally:
   if service.poll() is None:
    service.terminate()
    try:service.wait(timeout=10)
    except subprocess.TimeoutExpired:service.kill();service.wait()
if __name__=='__main__':main()
