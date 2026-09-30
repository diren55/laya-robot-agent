"""Original CaP-X SpillWipe motion plus explicitly shared camera/skill adapter.
Raster trajectory follows the public franka_spill_wipe.py ORACLE_CODE pattern.
That program is NOT a planner input, and its competence is not our contribution.
Original goto_pose and SpillWipe max_steps=10 / tolerance=.005 are unchanged.
"""
import copy,time,numpy as np
from native_cube import CameraMotion
from common import send,dump
from robosuite.utils.camera_utils import get_real_depth_map
from motion import FrankaControlSpillWipeApi as OriginalWipe
class SpillMotion(CameraMotion):
 def __init__(self,env,worker):
  super().__init__(env,worker)
  # Exact official integration registration, not tuned from evaluation labels.
  self._TCP_OFFSET=np.array([0.,0.,-.0158])
 def goto_pose(self,position,quaternion_wxyz):
  """Original dedicated SpillWipe goto_pose using official sponge offset -0.0158m, unchanged default controller and waits. XYZ and WXYZ are in robot base frame."""
  t=time.perf_counter();before=self._env._sim_step_count
  OriginalWipe.goto_pose(self,np.asarray(position),np.asarray(quaternion_wxyz))
  self.motion_log.append(dict(api='goto_pose',seconds=time.perf_counter()-t,physical_steps=self._env._sim_step_count-before,official_spill_tcp_offset=-.0158))
 def reset_episode(self,out):
  super().reset_episode(out);self.spill_reference=None
 def observe_scene(self):
  key=int(self._env._sim_step_count)
  if self.cache_key==key:return copy.deepcopy(self.cache)
  t=time.perf_counter();cam=self._env.get_observation()['robot0_robotview']
  rgb,depth=self._env.robosuite_env.sim.render(camera_name='robot0_robotview',
    width=self._env._render_width,height=self._env._render_height,depth=True)
  packet=self.out/('camera_'+str(self.frame)+'.npz')
  np.savez_compressed(packet,rgb=rgb[::-1].copy(),depth=get_real_depth_map(self._env.robosuite_env.sim,depth[::-1]).copy(),
    intrinsics=cam['intrinsics'],pose_mat=cam['pose_mat'])
  capture=time.perf_counter()-t;t=time.perf_counter()
  p=send(self.worker,dict(packet=str(packet),reference=self.spill_reference,perception_domain='spill'))
  vision=time.perf_counter()-t
  if p.get('reference') is not None:self.spill_reference=p['reference']
  state=dict(objects=p['objects'],relation_evidence=[],source_frame=str(packet),observed_does_not_mean_complete=True,
    ambiguous_names=[],missing_names=[] if p['objects'] else ['brown spill'],
    gripper_command_state='sponge already attached',surface_visibility=p.get('visibility'),
    current_dirty_pixels=p.get('dirty_pixels'),vision_status=p['status'])
  if p['objects']:
   state['relation_evidence']=[dict(object='brown spill',relation='clean',destination='observed support surface',
     truth=p['clean_truth'],source='current RGBD colour and measured visibility of initial static surface region')]
  self.capture_log.append(dict(frame=self.frame,physical_step=key,capture_io_seconds=capture,
    vision_ipc_seconds=vision,model_vision_seconds=p['seconds'],fresh_render=True))
  dump(self.out/('perception_'+str(self.frame)+'.json'),dict(state=state,raw=p));self.frame+=1
  self.cache_key=key;self.cache=state
  return copy.deepcopy(state)
 def get_object_pose(self,object_name,return_bbox_extent=False):
  """Camera-grounded pose/extent of the currently visible brown spill in robot base frame. Returns XYZ, downward WXYZ, optional full XYZ extent. Static-region clean status is in observe_scene; not an object pose oracle."""
  o=self._object(object_name)
  if o['current_dirty_bounds'] is None:raise ValueError('Current camera does not show a dirty spill extent')
  return np.array(o['center_base']),np.array([0.,0.,1.,0.]),np.array(o['extent']) if return_bbox_extent else None
 def wipe_region(self,object_name):
  """Shared camera-grounded raster wipe skill. Uses current observed spill bounds with 5cm steps, original task-specified z=0 and downward WXYZ. Sponge already attached. Returns fresh visual completion evidence; does not assume motion means success."""
  obj=self._object(object_name);bounds=obj['current_dirty_bounds']
  if bounds is None:return self.observe_scene()
  lo,hi=np.array(bounds[0]),np.array(bounds[1]);q=np.array([0.,0.,1.,0.])
  xs=np.arange(lo[0],hi[0]+.05,.05);ys=np.arange(lo[1],hi[1]+.05,.05)
  self.goto_pose(np.array([xs[0],ys[0],.10]),q)
  for i,y in enumerate(ys):
   path=xs if i%2==0 else xs[::-1]
   self.goto_pose(np.array([path[0],y,.10]),q)
   self.goto_pose(np.array([path[0],y,0.]),q)
   for x in path[1:]:self.goto_pose(np.array([x,y,0.]),q)
   self.goto_pose(np.array([path[-1],y,.10]),q)
  self.home_pose();self.cfg=None
  return self.observe_scene()
 def functions(self):
  return dict(get_object_pose=self.get_object_pose,goto_pose=self.goto_pose,observe_scene=self.observe_scene,wipe_region=self.wipe_region,home_pose=self.home_pose)

def seed_wipe_reset(low,seed):
 # Wipe samples via WipeArena's aliased Generator, not placement_initializer.
 rng=low.robosuite_env.rng
 rng.bit_generator.state=type(rng.bit_generator)(seed).state
 np.random.seed(seed)
