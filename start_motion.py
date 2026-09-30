"""Pinned CaP-X CPU PyRoKi service; only package and asset path isolation."""
import argparse,os,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
U=ROOT/'vendor/capx';sys.path.insert(0,str(U))
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('OMP_NUM_THREADS','2')
import capx
pkg=types.ModuleType('capx.integrations');pkg.__path__=[str(U/'capx/integrations')]
pkg.__package__='capx.integrations';sys.modules['capx.integrations']=pkg;capx.integrations=pkg
import robot_descriptions._cache as rd_cache
def pinned_asset(name,commit=None):
 if name=='example-robot-data':
  if commit is not None and commit!='d0d9098d752014aec3725b07766962acf06c5418':
   raise RuntimeError('Unexpected robot asset revision')
  return str(ROOT/'robot_assets/example-robot-data')
 raise RuntimeError('Bundle only contains Panda assets; will not download '+name)
rd_cache.clone_to_cache=pinned_asset
from capx.serving.launch_pyroki_server import main
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18116)
 args=parser.parse_args()
 main(port=args.port,host='127.0.0.1',robot='panda_description',target_link='panda_hand')
