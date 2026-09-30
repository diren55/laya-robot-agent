import bootstrap
import ast,signal,traceback,io,contextlib,copy
import numpy as np
from capx.envs.tasks.base import CodeExecutionEnvBase,CodeExecEnvConfig
from native_spill import SpillMotion
def checked_code(code,api_names):
 tree=ast.parse(code)
 safe_np={'array','asarray','zeros','ones','copy','clip','isfinite','all','any','float64','float32','sqrt','sum','abs','max','min','pi'}
 banned={'env','APIS','INPUTS','obs','eval','exec','compile','open','getattr','setattr','globals','locals','vars','type','object','input','breakpoint','help','exit','quit'}
 for n in ast.walk(tree):
  if isinstance(n,ast.Name) and (n.id in banned or n.id.startswith('__')):raise ValueError('Execution boundary: forbidden name '+n.id)
  if isinstance(n,(ast.Import,ast.ImportFrom)):
   if not isinstance(n,ast.Import) or any(a.name!='numpy' or a.asname not in (None,'np') for a in n.names):
    raise ValueError('Only numpy import is supported in this bounded adapter')
  if isinstance(n,ast.Attribute):
   if n.attr.startswith('_'):raise ValueError('Private attribute access forbidden')
   safe_methods={'copy','reshape','tolist','shape','dtype','ndim','size','get','items','keys','values','append','extend','pop'}
   if n.attr not in safe_np|safe_methods:raise ValueError('Attribute outside bounded numerical/data API: '+n.attr)
   if isinstance(n.value,ast.Name) and n.value.id in ('np','numpy') and n.attr not in safe_np:raise ValueError('Numpy operation outside numerical API: '+n.attr)
  if isinstance(n,(ast.ClassDef,ast.Global,ast.Nonlocal)):raise ValueError('Unsupported execution construct')
 return compile(tree,'<native_agent_code>','exec')

def restricted_import(name,*args,**kwargs):
 if name!='numpy':raise ImportError('Only numpy allowed')
 return np

class SkillExecutionEnv(CodeExecutionEnvBase):
 def __init__(self,low,ctrl,prompt):
  self.ctrl=ctrl
  extra="""\nCamera adapter: no privileged simulator state is available. The attached end-effector is a wiping sponge, not a grasping gripper. Use only the documented wipe-compatible APIs and numerical Python/numpy; no files, networking, introspection or other imports. observe_scene() returns current measured relations. After visual feedback verifies the requested goal, set RESULT='done'. If another observation/action is needed, leave RESULT='continue'. If required information is unavailable, set RESULT='clarify'. Do not declare completion merely because code executed. No simulator reset is allowed within an episode. The reusable wipe_region skill is available to both compared agents and returns current visual evidence; it is not a completion oracle. Prefer it when appropriate instead of rewriting the same low-level trajectory."""
  if not isinstance(ctrl,SpillMotion):
   extra="""\nCamera adapter: no privileged simulator state is available. Use only documented motion/perception APIs and numerical Python/numpy; no files, networking, introspection or other imports. observe_scene() returns current measured relations. After visual feedback verifies the requested goal, set RESULT='done'. If another observation/action is needed, leave RESULT='continue'. If required information is unavailable, set RESULT='clarify'. Do not declare completion merely because code executed. No simulator reset is allowed within an episode. The reusable lift_cube and stack_cube skills are available to both agents in every cube task; they are not completion oracles. Prefer them where appropriate instead of rewriting low-level trajectories. Failure can be retried within the same ten-turn and native physics budget."""
  super().__init__(CodeExecEnvConfig(low_level=low,apis=[],prompt=prompt+extra,privileged=True))
  self._apis={'CameraMotion':ctrl}
  self._full_prompt=[dict(role='system',content=self._system_prompt),dict(role='user',content=self._get_complete_prompt())]
  self._init_exec_globals()
 def reset(self,*,seed=None,options=None):
  # Original base reset merges raw low-level fields with the camera view.
  # Do not serialize, send or expose those fields to either decision system.
  _,info=super().reset(seed=seed,options=options)
  camera=self.ctrl.observe_scene()
  self._exec_globals['INPUTS']=copy.deepcopy(camera)
  return camera,{'task_prompt':self._task_prompt}
 def _get_observation(self):
  return self.ctrl.observe_scene()
 def _init_exec_globals(self):
  self._exec_globals={'__name__':'__main__','RESULT':None,'np':np,
   '__builtins__':dict(__import__=restricted_import,print=print,range=range,len=len,min=min,max=max,abs=abs,float=float,int=int,bool=bool,str=str,list=list,dict=dict,tuple=tuple,enumerate=enumerate,zip=zip,all=all,any=any,next=next,iter=iter,sorted=sorted,sum=sum,round=round,set=set,ValueError=ValueError,RuntimeError=RuntimeError,Exception=Exception)}
  self._exec_globals.update(self.ctrl.functions())
 def _exec_user_code(self,code):
  out=io.StringIO();err=io.StringIO();ok=True
  try:
   obj=checked_code(code,self.ctrl.functions())
   self._exec_globals['RESULT']=None
   def limit(*args):raise TimeoutError('Bounded code execution exceeded 60 seconds')
   old=signal.signal(signal.SIGALRM,limit);signal.alarm(60)
   try:
    with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):exec(obj,self._exec_globals,self._exec_globals)
   finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
  except BaseException:
   ok=False;traceback.print_exc(file=err)
  return dict(ok=ok,stdout=out.getvalue(),stderr=err.getvalue(),result=self._exec_globals.get('RESULT'))
