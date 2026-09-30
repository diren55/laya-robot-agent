import os,sys,types
from pathlib import Path
CAPX_REPO=Path(os.environ.get('LAYA_AGENT_CAPX_REPO','/root/capx_official_runtime_20260930/cap-x-53e9966d7a8e2fa7494676772bccc35280f5c0ed'))
sys.path.insert(0,str(CAPX_REPO));os.chdir(CAPX_REPO)
os.environ.setdefault('MUJOCO_GL','egl')
from capx.envs import get_env as get_cube_env
import capx
# Avoid the upstream eager registry; selected original motion methods are retained.
pkg=types.ModuleType('capx.integrations')
pkg.__path__=[str(CAPX_REPO/'capx/integrations')]
pkg.__package__='capx.integrations'
sys.modules['capx.integrations']=pkg
capx.integrations=pkg
