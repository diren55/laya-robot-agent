import copy,time
import numpy as np
from motion import OfficialMotion
from common import send,dump
class CameraMotion(OfficialMotion):
 def __init__(self,env,worker):
  super().__init__(env);self.worker=worker;self.cache_key=None;self.cache=None
  self.out=None;self.frame=0;self.initial={};self.capture_log=[];self.motion_log=[];self.last_grip='open';self.clearing_view=False
 def reset_episode(self,out):
  self.out=out;self.frame=0;self.initial={};self.cache_key=None;self.cache=None;self.capture_log=[];self.motion_log=[];self.calls=[];self.cfg=None;self.last_grip='open';self.clearing_view=False
 def observe_scene(self):
  """Return current camera object geometry, relation evidence and missing/ambiguous detections, not simulator truth. Cached only if no physical step occurred. After release, if a previously seen object is missing, the shared observer once clears the arm to its original home pose and observes again; no simulator truth is used."""
  key=int(self._env._sim_step_count)
  if self.cache_key==key:return copy.deepcopy(self.cache)
  t=time.perf_counter();obs=self._env.get_observation();cam=obs['robot0_robotview']
  # Upstream control intentionally skips image updates outside video capture.
  # Explicit sensor rendering refreshes RGBD without advancing physical time.
  from robosuite.utils.camera_utils import get_real_depth_map
  fresh_rgb,fresh_depth=self._env.robosuite_env.sim.render(camera_name='robot0_robotview',
    width=self._env._render_width,height=self._env._render_height,depth=True)
  cam['images']={'rgb':fresh_rgb[::-1].copy(),
    'depth':get_real_depth_map(self._env.robosuite_env.sim,fresh_depth[::-1]).copy()}
  packet=self.out/('camera_'+str(self.frame)+'.npz')
  np.savez_compressed(packet,rgb=cam['images']['rgb'],depth=cam['images']['depth'],intrinsics=cam['intrinsics'],pose_mat=cam['pose_mat'])
  capture=time.perf_counter()-t;t=time.perf_counter()
  p=send(self.worker,dict(packet=str(packet)));vision=time.perf_counter()-t
  if self.frame==0:self.initial={o['name']:copy.deepcopy(o) for o in p['objects']}
  objects=p['objects'];rels=[]
  for o in objects:
   initial=self.initial.get(o['name'])
   lift='unknown' if not initial else ('true' if o['lower_base'][2]>initial['top_z_base']+.04 and self.last_grip=='closed' else 'false')
   rels.append(dict(object=o['name'],relation='raised',destination='initial support',truth=lift,source='current image and initial camera reference; gripper command completed'))
   for other in objects:
    if other['name']==o['name']:continue
    dx=float(np.linalg.norm(np.array(o['center_base'][:2])-other['center_base'][:2]))
    bottom=o['top_z_base']-o['estimated_side']
    on=dx<.65*other['estimated_side'] and abs(bottom-other['top_z_base'])<.018 and self.last_grip=='open'
    rels.append(dict(object=o['name'],relation='on',destination=other['name'],truth='true' if on else 'false',source='current RGBD cuboid-fit bounds and completed release command'))
  state=dict(objects=objects,relation_evidence=rels,source_frame=str(packet),
    ambiguous_names=p.get('ambiguous_names',[]),missing_names=[k for k in self.initial if k not in {o['name'] for o in objects}],
    observed_does_not_mean_complete=True,gripper_command_state=self.last_grip,
    table_evidence=p.get('table_evidence',{'status':'unknown','sites':[]}))
  for obj in objects:
   plane=state['table_evidence'].get('height_base')
   truth='unknown' if plane is None else ('true' if abs(obj['top_z_base']-obj['estimated_side']-plane)<.018 and self.last_grip=='open' else 'false')
   rels.append(dict(object=obj['name'],relation='on_table',destination='clear table',truth=truth,source='current RGBD cuboid and visible support plane'))
  self.capture_log.append(dict(frame=self.frame,physical_step=key,capture_io_seconds=capture,vision_ipc_seconds=vision,model_vision_seconds=p['seconds'],fresh_render=True))
  dump(self.out/('perception_'+str(self.frame)+'.json'),dict(state=state,raw=p));self.frame+=1
  self.cache=state;self.cache_key=key
  if state['missing_names'] and self.last_grip=='open' and not self.clearing_view:
   # Shared active-sensing adaptation, not model-specific or a success oracle.
   self.clearing_view=True
   try:
    dump(self.out/('active_view_'+str(self.frame)+'.json'),dict(reason='previously observed names absent from current RGBD after release',missing=state['missing_names'],action='original home_pose, gripper unchanged',first_frame=state['source_frame']))
    self.home_pose();self.cfg=None
    self.cache_key=None
    return self.observe_scene()
   finally:self.clearing_view=False
  return copy.deepcopy(state)
 def _object(self,name):
  os=[o for o in self.observe_scene()['objects'] if o['name']==name]
  if len(os)!=1:raise ValueError('Current camera cannot uniquely ground '+str(name))
  return os[0]
 def get_object_pose(self,object_name,return_bbox_extent=False):
  """Get current camera-derived XYZ position, down WXYZ quaternion, and full estimated cubical extent if requested (otherwise None). Returns three values. Missing/ambiguous detections raise ValueError."""
  o=self._object(object_name)
  return np.array(o['center_base']),np.array([0.,0.,1.,0.]),np.array(o['extent']) if return_bbox_extent else None
 def sample_grasp_pose(self,object_name):
  """Get current camera-derived grasp XYZ and downward WXYZ quaternion for a visible coloured cube. Cubic grasp prior: jaw centre 0.015m below observed top. Missing/ambiguous detections raise ValueError."""
  o=self._object(object_name);pos=np.array(o['center_base']);pos[2]=o['top_z_base']-.015
  return pos,np.array([0.,0.,1.,0.])
 def goto_pose(self,position,quaternion_wxyz,z_approach=0.0):
  """Move using the original CaP-X PyRoKi motion API. position is XYZ metres; quaternion_wxyz is WXYZ. Optional z_approach first approaches from that height and then descends to position."""
  t=time.perf_counter();before=self._env._sim_step_count
  super().goto_pose(np.array(position),np.array(quaternion_wxyz),z_approach=z_approach)
  self.motion_log.append(dict(api='goto_pose',seconds=time.perf_counter()-t,physical_steps=self._env._sim_step_count-before))
 def open_gripper(self):
  """Open gripper using the original controller and its unchanged 30-step wait."""
  t=time.perf_counter();before=self._env._sim_step_count;super().open_gripper();self.last_grip='open'
  self.motion_log.append(dict(api='open_gripper',seconds=time.perf_counter()-t,physical_steps=self._env._sim_step_count-before))
 def close_gripper(self):
  """Close gripper using the original controller and its unchanged 30-step wait."""
  t=time.perf_counter();before=self._env._sim_step_count;super().close_gripper();self.last_grip='closed'
  self.motion_log.append(dict(api='close_gripper',seconds=time.perf_counter()-t,physical_steps=self._env._sim_step_count-before))
 def lift_cube(self,object_name):
  """Shared reusable skill: camera-grounded pick up and keep a cube lifted 0.16m. Executes the same unmodified low-level APIs, observes afterwards, returns evidence without claiming task success."""
  namespace={'np':np};namespace.update(self.functions())
  exec(action_code(dict(skill='lift',object=object_name,destination='initial support')),namespace,namespace)
  return self.observe_scene()
 def stack_cube(self,object_name,destination_name):
  """Shared reusable skill: camera-grounded grasp, lift, gently place and release one cube on another. Same motion APIs, z_approach=.1, safe transfer=.2m, full-extent placement formula. Returns current evidence, not evaluator success."""
  namespace={'np':np};namespace.update(self.functions())
  exec(action_code(dict(skill='stack',object=object_name,destination=destination_name)),namespace,namespace)
  return self.observe_scene()
 def home_pose(self):
  """Move to original safe home joints without changing the gripper; used to clear camera occlusion. Full physical cost is included."""
  t=time.perf_counter();before=self._env._sim_step_count
  super().home_pose()
  self.motion_log.append(dict(api='home_pose',seconds=time.perf_counter()-t,physical_steps=self._env._sim_step_count-before))
 def grasp_blockers(self,object_name):
  """Current RGBD objects above and overlapping the requested object's grasp column. This is geometry evidence, not instruction permission to move them."""
  obj=self._object(object_name);blockers=[]
  for other in self.observe_scene()['objects']:
   if other['name']==object_name:continue
   delta=np.array(other['center_base'])-obj['center_base']
   if delta[2]>.4*min(other['estimated_side'],obj['estimated_side']) and np.all(abs(delta[:2])<.6*(other['estimated_side']+obj['estimated_side'])):
    blockers.append(other['name'])
  return blockers
 def move_cube_to_clear_table(self,object_name):
  """Temporarily place a visible cube on a measured clear tabletop patch. Caller must ensure the instruction permits moving it. Uses only current depth support and the same unchanged motion/30-step gripper waits. Raises if no supported visible free patch exists."""
  obj=self._object(object_name);sites=self.observe_scene()['table_evidence']['sites']
  if not sites:raise ValueError('No currently visible supported clear table placement')
  site=min(sites,key=lambda s:np.linalg.norm(np.array(s['center_base'][:2])-obj['center_base'][:2]))
  p,q=self.sample_grasp_pose(object_name);target=np.array(site['center_base']);target[2]+=obj['estimated_side']/2
  self.open_gripper();self.goto_pose(p,q,z_approach=.1);self.close_gripper()
  self.goto_pose(p+np.array([0.,0.,.2]),q);self.goto_pose(target,q,z_approach=.1)
  self.open_gripper();self.goto_pose(target+np.array([0.,0.,.12]),q)
  return self.observe_scene()
 def functions(self):
  f=super().functions()
  f.update(observe_scene=self.observe_scene,lift_cube=self.lift_cube,stack_cube=self.stack_cube,
   grasp_blockers=self.grasp_blockers,move_cube_to_clear_table=self.move_cube_to_clear_table)
  return f

def action_code(step):
 name=repr(step['object'])
 if step['skill']=='lift':
  return f"""import numpy as np
p,q=sample_grasp_pose({name})
open_gripper()
goto_pose(p+np.array([0.,0.,.14]),q)
goto_pose(p,q)
close_gripper()
goto_pose(p+np.array([0.,0.,.16]),q)
RESULT='continue'
"""
 if step['skill']=='stack':
  dest=repr(step['destination'])
  return f"""import numpy as np
p,q=sample_grasp_pose({name})
source_pos,source_q,source_extent=get_object_pose({name},return_bbox_extent=True)
dest_pos,dest_q,dest_extent=get_object_pose({dest},return_bbox_extent=True)
open_gripper()
goto_pose(p,q,z_approach=.1)
close_gripper()
goto_pose(p+np.array([0.,0.,.2]),q)
target=dest_pos.copy()
target[2]=dest_pos[2]+dest_extent[2]/2+source_extent[2]/2
goto_pose(target,q,z_approach=.1)
open_gripper()
goto_pose(target+np.array([0.,0.,.12]),q)
RESULT='continue'
"""
 raise ValueError('Unsupported native skill '+step['skill'])
