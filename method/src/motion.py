# Adapted imports; selected control method bodies are from capgym/cap-x.
# Upstream revision 53e9966d7a8e2fa7494676772bccc35280f5c0ed.
# See THIRD_PARTY.md and licenses/. Motion, TCP and gripper waits are not original inventions.
import os
import bootstrap
import time
from typing import Any
import numpy as np
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as SciRotation
from capx.integrations.base_api import ApiBase
from capx.integrations.motion.pyroki import init_pyroki
from capx.integrations.franka.common import open_gripper as _open_gripper,close_gripper as _close_gripper
class FrankaControlApi(ApiBase):
    def functions(self) -> dict[str, Any]:
            fns = {
                "get_object_pose": self.get_object_pose,
                "sample_grasp_pose": self.sample_grasp_pose,
                "goto_pose": self.goto_pose,
                "open_gripper": self.open_gripper,
                "close_gripper": self.close_gripper,
                # "home_pose": self.home_pose,
                # "get_observation": self.get_observation,
            }
            if not self.real: # Only include home pose in simulation
                fns["home_pose"] = self.home_pose
            return fns
    def goto_pose(
            self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
        ) -> None:
            """Go to pose using Inverse Kinematics.
            There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
            Args:
                position: (3,) XYZ in meters.
                quaternion_wxyz: (4,) WXYZ unit quaternion.
                z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
            Returns:
                None
            """
            pos_str = np.array2string(np.asarray(position), precision=4)
            approach_info = f" (z_approach={z_approach:.3f})" if z_approach != 0.0 else ""
            self._log_step("goto_pose", f"Moving to position {pos_str}{approach_info} …")

            pos = np.asarray(position, dtype=np.float64).reshape(3)
            quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
            # Align with legacy env: apply TCP offset in end-effector frame
            quat_xyzw = np.array(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
            )
            rot = SciRotation.from_quat(quat_xyzw)
            offset_pos = pos + rot.apply(self._TCP_OFFSET)

            if self.real:
                quat_wxyz = (vtf.SO3(wxyz=quat_wxyz) @ vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi/4+np.pi/2)).wxyz

            if (
                z_approach != 0.0
            ):  # If z_approach is not 0.0, approach the object from above by z_approach meters
                z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))

                if self.cfg is None or self.real:
                    self.cfg = self.ik_solve_fn(
                        target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                    )
                else:
                    self.cfg = self.ik_solve_fn(
                        target_pose_wxyz_xyz=np.concatenate([quat_wxyz, z_offset_pos]),
                        prev_cfg=self.cfg,
                    )
                    # prev_cfg = self.cfg

                    # for i in range(15): # run w/ multiple iterations when using vel_cost ik solver
                    #     self.cfg = self.ik_solve_fn(
                    #         target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                    #         prev_cfg = prev_cfg,
                    #     )
                    #     if prev_cfg is not None:
                    #         # print(f"Error: {np.linalg.norm(self.cfg - prev_cfg)}", np.allclose(self.cfg, prev_cfg, atol=1e-3))
                    #         if np.allclose(self.cfg, prev_cfg, atol=1e-3):
                    #             break
                    #         else:
                    #             prev_cfg = self.cfg

                joints_z_offset = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)

                self._env.move_to_joints_blocking(joints_z_offset)

            if self.cfg is None or self.real:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                )
            else:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                    prev_cfg=self.cfg,
                )
                # prev_cfg = self.cfg

                # for i in range(15): # run w/ multiple iterations when using vel_cost ik solver
                #     self.cfg = self.ik_solve_fn(
                #         target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                #         prev_cfg = prev_cfg,
                #     )
                #     if prev_cfg is not None:
                #         # print(f"Error: {np.linalg.norm(self.cfg - prev_cfg)}", np.allclose(self.cfg, prev_cfg, atol=1e-3))
                #         if np.allclose(self.cfg, prev_cfg, atol=1e-3):
                #             break
                #         else:
                #             prev_cfg = self.cfg
            joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking(joints)
            self._log_step_update(text="Motion complete.")
    def home_pose(self) -> None:
            """
            Move the robot to a safe home pose.
            Args:
                None
            Returns:
                None
            """
            self._log_step("home_pose", "Moving robot to home configuration …")

            # joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
            joints = np.array(
                [
                    -2.95353726e-02,
                    1.69197371e-01,
                    2.39244731e-03,
                    -2.64089311e00,
                    -2.01237851e-03,
                    2.94565778e00,
                    8.31390616e-01,
                ]
            )
            self._env.move_to_joints_blocking(joints)
            self._log_step_update(text="Home position reached.")
    def open_gripper(self) -> None:
            """Open gripper fully.

            Args:
                None
            """
            self._log_step("open_gripper", "Opening gripper …")
            _open_gripper(self._env, steps=30)
            self._log_step_update(text="Gripper opened.")
    def close_gripper(self) -> None:
            """Close gripper fully.

            Args:
                None
            """
            self._log_step("close_gripper", "Closing gripper …")
            _close_gripper(self._env, steps=30)
            self._log_step_update(text="Gripper closed.")

class FrankaControlSpillWipeApi(ApiBase):
    def goto_pose(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
            """Go to pose using Inverse Kinematics.
            There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
            Args:
                position: (3,) XYZ in meters.
                quaternion_wxyz: (4,) WXYZ unit quaternion.
            Returns:
                None
            """

            pos = np.asarray(position, dtype=np.float64).reshape(3)
            quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
            quat_xyzw = np.array(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
            )
            rot = SciRotation.from_quat(quat_xyzw)
            offset_pos = pos + rot.apply(self._TCP_OFFSET)

            if self.cfg is None:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                )
            else:
                self.cfg = self.ik_solve_fn(
                    target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                    prev_cfg=self.cfg,
                )
            joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
            self._env.move_to_joints_blocking(joints)

class OfficialMotion(FrankaControlApi):
 def __init__(self,env):
  ApiBase.__init__(self,env)
  self._TCP_OFFSET=np.array([0.,0.,-.107])
  self.real=False;self.debug=False;self.cfg=None;self.calls=[]
  fn=init_pyroki(server_url=os.environ.get('LAYA_AGENT_IK_URL','http://127.0.0.1:8116'))
  def timed(**kwargs):
   t=time.perf_counter();q=fn(**kwargs)
   self.calls.append(dict(seconds=time.perf_counter()-t,target_pose=kwargs['target_pose_wxyz_xyz'].tolist(),joint_configuration=q.tolist(),has_prev=kwargs.get('prev_cfg') is not None))
   return q
  self.ik_solve_fn=timed
