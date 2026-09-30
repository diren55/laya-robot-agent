"""Frozen T4 no-generative-planner native mechanical-arm Agent.
Existing four-family implementation; not full paper reproduction or real-robot certification.
"""
import os,sys,json,time,traceback,subprocess,fcntl,copy,argparse
from pathlib import Path
import numpy as np
import bootstrap
from bootstrap import get_cube_env
from common import send,dump,worker_read
from native_cube import CameraMotion
from native_spill import SpillMotion
from skill_execution import SkillExecutionEnv
from capx.envs.tasks.franka.franka_lift import PROMPT as LIFT_PROMPT
from capx.envs.tasks.franka.franka_pick_place import PROMPT as STACK_PROMPT
from capx.envs.tasks.franka.franka_cube_restack import PROMPT as RESTACK_PROMPT
from capx.envs.tasks.franka.franka_spill_wipe import PROMPT as WIPE_PROMPT
from capx.envs.simulators.robosuite_spill_wipe import FrankaRobosuiteSpillWipeLowLevel
TASKS=[
 dict(name='CubeLift',env='franka_robosuite_cube_lift_low_level',prompt=LIFT_PROMPT,skills=['lift','stack']),
 dict(name='CubeStack',env='franka_robosuite_cubes_low_level',prompt=STACK_PROMPT,skills=['lift','stack']),
 dict(name='CubeRestack',env='franka_robosuite_cubes_restack_low_level',prompt=RESTACK_PROMPT,skills=['lift','stack']),
 dict(name='SpillWipe',env='franka_robosuite_spill_wipe_low_level',prompt=WIPE_PROMPT,skills=['wipe'])]
OUT=None
def get_env(name,**kwargs):
 if name!='franka_robosuite_spill_wipe_low_level':return get_cube_env(name,**kwargs)
 return FrankaRobosuiteSpillWipeLowLevel(**kwargs)
def record_initial_identity(low,out):
 # Evaluator only; never input to the semantic compiler or vision worker.
 s=low.robosuite_env.sim
 np.savez_compressed(out/'initial_state_evaluator_only.npz',qpos=s.data.qpos.copy(),
  qvel=s.data.qvel.copy(),ctrl=s.data.ctrl.copy(),geom_size=s.model.geom_size.copy(),body_pos=s.model.body_pos.copy())
 return {'recorded':True,'scope':'evaluator only'}
def expand_physical_prerequisite(step,plan,ctrl):
 return step,None

def seed_native_reset(low,seed):
 # Mutate Generator state in place: samplers retain aliases to this generator.
 # Robot reset also receives this exact rng. Global np.random.seed alone is insufficient.
 rng=low.robosuite_env.rng
 rng.bit_generator.state=type(rng.bit_generator)(seed).state
 sampler=getattr(low.robosuite_env,'placement_initializer',None)
 def bind(s):
  s.rng=rng
  for child in getattr(s,'samplers',{}).values():bind(child)
 if sampler is not None:bind(sampler)
 np.random.seed(seed)

def run_episode(task, seed, mode, low, worker):
    out = OUT / (task['name'] + '_' + str(seed) + '_' + mode)
    out.mkdir()
    start = time.perf_counter()
    ctrl = SpillMotion(low, worker) if task['name'] == 'SpillWipe' else CameraMotion(low, worker)
    ctrl.reset_episode(out)
    env = SkillExecutionEnv(low, ctrl, task['prompt'])
    messages = [dict(source='user', time_step=0, text=task['prompt'])]
    history = []
    decisions = []
    execution = []
    responses = []
    status = 'budget_exhausted'
    agent_done = False
    unknown = False
    try:
        t = time.perf_counter()
        seed_native_reset(low, seed)
        state, _ = env.reset(seed=seed)
        reset_seconds = time.perf_counter() - t
        initial_identity = record_initial_identity(low, out)
        send(worker, dict(reset=True, skills=task['skills'], model_variant='T4'))
        prompt = copy.deepcopy(env._full_prompt)
        prompt.append(dict(role='user', content='Initial current camera evidence: ' + json.dumps(state)))
        for turn in range(10):
            if time.perf_counter() - start > 1000:
                status = 'wall_budget'
                break
            t = time.perf_counter()
            z = send(worker, dict(plan=True, messages=messages, state=state, history=history))
            plan = z['plan']
            z['client_seconds'] = time.perf_counter() - t
            decisions.append(z)
            dump(out / ('decision_' + str(turn) + '.json'), z)
            if plan['status'] in ('done', 'clarify'):
                status = plan['status']
                agent_done = status == 'done'
                break
            if plan['status'] == 'observe':
                ctrl.cache_key = None
                state = ctrl.observe_scene()
                history.append(dict(tool='observe', time_step=turn, controller_returned=True, visual_placement_supported=False))
                if sum((h['tool'] == 'observe' for h in history)) >= 2:
                    status = 'unresolved_observation'
                    break
                continue
            step, intervention = expand_physical_prerequisite(plan['steps'][0], plan, ctrl)
            if intervention:
                dump(out / ('prerequisite_' + str(turn) + '.json'), intervention)
            if step is None:
                status = 'clarify'
                break
            if step['skill'] == 'unblock':
                code = 'move_cube_to_clear_table(' + repr(step['object']) + ')' + chr(10) + "RESULT='continue'"
            elif step['skill'] == 'wipe':
                code = 'wipe_region(' + repr(step['object']) + ')' + chr(10) + "RESULT='continue'"
            elif step['skill'] == 'lift':
                code = 'lift_cube(' + repr(step['object']) + ')' + chr(10) + "RESULT='continue'"
            elif step['skill'] == 'stack':
                code = 'stack_cube(' + repr(step['object']) + ',' + repr(step['destination']) + ')' + chr(10) + "RESULT='continue'"
            else:
                raise ValueError('Unsupported compiled skill')
            t = time.perf_counter()
            state, reward, terminated, truncated, info = env.step(code)
            elapsed = time.perf_counter() - t
            feedback = dict(stdout=info['stdout'][-12000:], stderr=info['stderr'][-12000:], sandbox_rc=info['sandbox_rc'], camera=state)
            execution.append(dict(turn=turn, code=code, seconds=elapsed, feedback=feedback, evaluation_only=dict(task_completed=info.get('task_completed'), reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))))
            dump(out / ('execution_' + str(turn) + '.json'), execution[-1])
            relation = 'clean' if step['skill'] == 'wipe' else 'raised' if step['skill'] == 'lift' else 'on_table' if step['skill'] == 'unblock' else 'on'
            truths = [r['truth'] for r in state['relation_evidence'] if r['object'] == step['object'] and r['destination'] == step['destination'] and (r['relation'] == relation)]
            truth = truths[0] if truths else 'unknown'
            history.append(dict(tool=step['skill'], object=step['object'], destination=step['destination'], time_step=turn, controller_returned=info['sandbox_rc'] == 0, visual_relation_truth=truth, visual_placement_supported=truth == 'true', error=info['stderr'][-500:] if info['sandbox_rc'] else None))
            if truncated:
                status = 'physics_budget'
                break
        physical_success = bool(low.task_completed())
    except Exception as e:
        status = 'infrastructure_error'
        unknown = True
        dump(out / 'ERROR.json', dict(error=repr(e), traceback=traceback.format_exc()))
        physical_success = bool(low.task_completed()) if ctrl.frame else False
        reset_seconds = locals().get('reset_seconds')
    row = dict(task=task['name'], seed=seed, mode=mode, status=status, success=physical_success, agent_declared_done=agent_done, full_success=physical_success and agent_done, unknown=unknown, wall_seconds=time.perf_counter() - start, reset_seconds=reset_seconds, initial_identity=locals().get('initial_identity'), decision_seconds=sum((x['client_seconds'] for x in decisions)), code_execution_seconds=sum((x['seconds'] for x in execution)), generated_code_blocks=0, llm_calls=len(responses), output_tokens=sum((x.get('usage', {}).get('output_tokens', 0) for x in responses)), physical_steps=low._sim_step_count, physical_seconds=low._sim_step_count * 0.05, capture_log=ctrl.capture_log, motion_log=ctrl.motion_log, ik_calls=ctrl.calls, generative_llm_calls=len(responses), feedback_history=history)
    dump(out / 'RESULT.json', row)
    print(json.dumps({k: v for k, v in row.items() if k not in ('capture_log', 'motion_log', 'ik_calls', 'feedback_history')}), flush=True)
    return row

def main():
 global OUT
 p=argparse.ArgumentParser()
 p.add_argument('--out',required=True)
 p.add_argument('--seeds',type=int,nargs='+',default=[1061201])
 p.add_argument('--tasks',nargs='+',choices=[t['name'] for t in TASKS],default=[t['name'] for t in TASKS])
 args=p.parse_args();OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=False)
 tasks=[t for t in TASKS if t['name'] in args.tasks];rows=[];worker=None
 dump(OUT/'PROTOCOL.json',dict(tasks=tasks,seeds=args.seeds,mode='laya',
  planned_episodes=len(tasks)*len(args.seeds),generative_llm_calls=0,
  checkpoint=os.environ.get('LAYA_AGENT_CHECKPOINT','/root/agp_model_store/laya_dynamic_semantic_train_v4'),
  budgets='10 turns, 1000s wall, original native horizon/dt/waits; same stable T4 behavior',
  boundary='Clean code export. No Codex/Phi/generative fallback. Not an original seven-family paper reproduction.'))
 lock_path=Path(os.environ.get('LAYA_AGENT_GPU_LOCK','/root/autodl-tmp/agp_laya_sim_20260930/dynamic_no_llm_v1/gpu_current_scope.lock'))
 with lock_path.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   py=os.environ.get('LAYA_AGENT_WORKER_PYTHON','/root/autodl-tmp/laya_wm_pilot_20260929/venv/bin/python')
   worker=subprocess.Popen([py,'-u',str(Path(__file__).parent/'worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
    stderr=(OUT/'worker.log').open('w'),text=True,bufsize=1)
   dump(OUT/'MODEL_READY.json',worker_read(worker))
   for task in tasks:
    t=time.perf_counter();low=get_env(task['env'],privileged=True,enable_render=True,viser_debug=False)
    dump(OUT/(task['name']+'_ENV_COLD.json'),dict(seconds=time.perf_counter()-t))
    try:
     for seed in args.seeds:
      row=run_episode(task,seed,'laya',low,worker);rows.append(row)
      dump(OUT/'PROGRESS.json',dict(completed=len(rows),planned=len(tasks)*len(args.seeds),last=row['status']))
      if row['unknown']:raise RuntimeError('Preserve runtime error before continuing')
    finally:low.robosuite_env.close()
   dump(OUT/'COMPLETE.json',dict(summary=dict(n=len(rows),full_success=sum(x['full_success'] for x in rows),
    physical_success=sum(x['success'] for x in rows),unknown=sum(x['unknown'] for x in rows),
    generative_llm_calls=sum(x['generative_llm_calls'] for x in rows)),rows=rows))
  except Exception as e:
   dump(OUT/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
  finally:
   if worker:
    if worker.poll() is None:
     worker.stdin.write(json.dumps({'stop':True})+'\n');worker.stdin.flush()
     try:worker.wait(timeout=5)
     except subprocess.TimeoutExpired:worker.terminate()
if __name__=='__main__':main()
