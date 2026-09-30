"""Atomic skill-intent wording; success still checked using current clean relation.
Camera-only native Agent worker. DINO is an explicit SAM3/CGN substitute.
No simulator identifiers, evaluator labels, or object poses enter this process.
"""
import os,sys,json,time,inspect,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import semantic_core as dc
from semantic_agent import ScopeGraphAgent
from question_scope import ScopeCompiler
from question_direct import YES
dc.SKILLS['lift']=dict(source='block',target='reference',relation='raised',description='pick up {a} and keep it lifted above its initial support')
dc.REL_WORDS['raised']='is raised above'
dc.SKILLS['wipe']=dict(source='surface_region',target='reference',relation='clean',description='wipe up {a}')
dc.REL_WORDS['clean']='has been completely wiped off'
from vision_spill import spill_detect
class NativeCompiler(ScopeCompiler):
 def compile(self,messages,state,feedback=(),previous_graph=None):
  b=super().compile(messages,state,feedback,previous_graph)
  if 'lift' in self.enabled_skills:
   for o in state['objects']:
    if o['kind']!='block':continue
    a=dc.Action('lift:'+o['name'],'lift',o['name'],'initial support')
    b['actions'].append(dict(id=a.id,skill=a.skill,object=a.object,destination=a.destination))
    b['questions'][a.id]=dict(type='choice',criteria=copy.deepcopy(YES),
      instructions='According to the user, should the robot '+a.sentence()+'? A conditionally requested action counts as requested.')
    b['questions']['forbid:'+a.id]=dict(type='choice',criteria=copy.deepcopy(YES),
      instructions='Does the user explicitly forbid the robot to '+a.sentence()+'?')
  if 'wipe' in self.enabled_skills:
   for o in state['objects']:
    if o['kind']!='surface_region':continue
    a=dc.Action('wipe:'+o['name'],'wipe',o['name'],'observed support surface')
    b['actions'].append(dict(id=a.id,skill=a.skill,object=a.object,destination=a.destination))
    b['questions'][a.id]=dict(type='choice',criteria=copy.deepcopy(YES),
      instructions='According to the user, should the robot '+a.sentence()+'? A conditionally requested action counts as requested.')
    b['questions']['forbid:'+a.id]=dict(type='choice',criteria=copy.deepcopy(YES),
      instructions='Does the user explicitly forbid the robot to '+a.sentence()+'?')
    b['questions']['protect:'+o['name']]=dict(type='choice',criteria=copy.deepcopy(YES),
      instructions='Does the user require the robot to leave '+o['name']+' unchanged instead of cleaning it?')
  return b
t=time.perf_counter()
P=os.environ.get('LAYA_AGENT_DINO','/root/autodl-tmp/laya_wm_pilot_20260929/models/grounding_dino_tiny')
proc=AutoProcessor.from_pretrained(P,local_files_only=True)
model=AutoModelForZeroShotObjectDetection.from_pretrained(P,local_files_only=True,disable_custom_kernels=True).to('cuda').eval()
fn=proc.post_process_grounded_object_detection
threshold='threshold' if 'threshold' in inspect.signature(fn).parameters else 'box_threshold'
vision_cold=time.perf_counter()-t
agent=ScopeGraphAgent(('stack','lift'),checkpoint=Path(os.environ.get('LAYA_AGENT_CHECKPOINT','/root/agp_model_store/laya_dynamic_semantic_train_v4')))
agent.compiler=NativeCompiler(('stack','lift'))
from aggregation import GuardCheckedAggregator
agent.aggregator=GuardCheckedAggregator(max_failed_attempts=10)
print(json.dumps(dict(ready=True,vision_cold_seconds=vision_cold,laya_cold_seconds=agent.answers.cold_seconds)),flush=True)
def missing_color_components(packet,present):
 # Same-frame missing-object supplement, never evaluator poses or remembered coordinates.
 # Domain assumption (both methods): coloured cubic objects, not open-world semantics.
 from scipy.ndimage import label,find_objects
 rgb=packet['rgb'].astype(float);dep=packet['depth'].squeeze();K=packet['intrinsics'];T=packet['pose_mat']
 supplements=[];ambiguous=[]
 for color,ci in [('red',0),('green',1),('blue',2)]:
  name=color+' cube'
  if name in present:continue
  others=[j for j in range(3) if j!=ci]
  mask=(rgb[:,:,ci]>1.35*rgb[:,:,others[0]])&(rgb[:,:,ci]>1.35*rgb[:,:,others[1]])&(rgb[:,:,ci]>45)
  cc,count=label(mask);candidates=[]
  for index,sl in enumerate(find_objects(cc),1):
   if sl is None:continue
   yy,xx=np.where(cc[sl]==index);yy+=sl[0].start;xx+=sl[1].start
   z=dep[yy,xx];valid=np.isfinite(z)&(z>.05)&(z<3);yy,xx,z=yy[valid],xx[valid],z[valid]
   if len(z)<30:continue
   camera=np.stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z,np.ones_like(z)),axis=1)
   pts=(camera@T.T)[:,:3];lo,hi=np.percentile(pts,[5,95],axis=0)
   span=hi-lo;side=float(max(span[:2])/.9);top=float(np.percentile(pts[:,2],90));center=(lo+hi)/2;center[2]=top-side/2
   if np.max(span)>.14 or not .012<side<.12:continue
   if not (.25<center[0]<.85 and abs(center[1])<.35 and -.3<center[2]<.15):continue
   candidates.append(dict(name=name,kind='block',center_base=center.tolist(),lower_base=lo.tolist(),upper_base=hi.tolist(),
    estimated_side=side,extent=[side]*3,top_z_base=top,box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
    pixels=len(z),score=None,source='same current RGBD connected colour component supplement for DINO-missing known-colour cube; cubic prior and reachable workspace filter, not GT'))
  if len(candidates)==1:supplements.append(candidates[0])
  elif len(candidates)>1:ambiguous.append(name)
 return supplements,ambiguous

def visible_table_sites(packet,objects):
 # Measured workspace points only; no simulator object poses or table metadata.
 rgb=packet['rgb'];dep=packet['depth'].squeeze();K=packet['intrinsics'];T=packet['pose_mat']
 yy,xx=np.mgrid[0:dep.shape[0]:3,0:dep.shape[1]:3];z=dep[yy,xx]
 valid=np.isfinite(z)&(z>.05)&(z<3)
 for obj in objects:
  x0,y0,x1,y1=obj['box'];valid &= ~((xx>=x0-3)&(xx<=x1+3)&(yy>=y0-3)&(yy<=y1+3))
 yy,xx,z=yy[valid],xx[valid],z[valid]
 camera=np.stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z,np.ones_like(z)),axis=1)
 pts=(camera@T.T)[:,:3]
 pts=pts[(pts[:,0]>.28)&(pts[:,0]<.82)&(abs(pts[:,1])<.34)&(pts[:,2]>-.35)&(pts[:,2]<.15)]
 if len(pts)<100:return {'status':'unknown','sites':[],'reason':'insufficient current depth support'}
 bins=np.arange(-.35,.154,.004);counts,edges=np.histogram(pts[:,2],bins=bins);i=int(np.argmax(counts))
 near=pts[abs(pts[:,2]-(edges[i]+.002))<.006]
 if len(near)<100:return {'status':'unknown','sites':[],'reason':'no dominant horizontal surface'}
 height=float(np.median(near[:,2]));sites=[]
 for x in np.arange(.36,.741,.07):
  for y in np.arange(-.245,.246,.07):
   if any(np.linalg.norm(np.array([x,y])-o['center_base'][:2])<.075+o['estimated_side'] for o in objects):continue
   patch=pts[(abs(pts[:,0]-x)<.035)&(abs(pts[:,1]-y)<.035)]
   if len(patch)<30 or np.mean(abs(patch[:,2]-height)<.007)<.97:continue
   coverage=all(np.sum((abs(patch[:,0]-(x+dx))<.014)&(abs(patch[:,1]-(y+dy))<.014))>=2
    for dx in [-.022,0,.022] for dy in [-.022,0,.022])
   if coverage:sites.append({'center_base':[float(x),float(y),height],'visible_patch_width':.07,'depth_points':len(patch)})
 sites=sorted(sites,key=lambda s:abs(s['center_base'][1])+.2*abs(s['center_base'][0]-.5))[:8]
 return {'status':'measured' if sites else 'unknown','height_base':height,'sites':sites,
   'source':'current depth horizontal-plane mode plus fully visible clear support patches; fixed Panda reach envelope, no task truth'}

def detect(msg):
 packet=np.load(msg['packet']);rgb=packet['rgb'];dep=packet['depth'].squeeze();K=packet['intrinsics'];T=packet['pose_mat']
 h,w=rgb.shape[:2];t=time.perf_counter()
 x=proc(images=Image.fromarray(rgb),text='a red cube. a green cube. a blue cube.',return_tensors='pt').to('cuda')
 with torch.inference_mode():raw=model(**x)
 det=fn(raw,x.input_ids,**{threshold:.20,'text_threshold':.20,'target_sizes':[(h,w)]})[0]
 torch.cuda.synchronize()
 boxes=det['boxes'].cpu().tolist();scores=det['scores'].cpu().tolist()
 labels=det.get('text_labels',det.get('labels'));candidates=[]
 for box,score,label in zip(boxes,scores,labels):
  if not isinstance(label,str) or 'cube' not in label:continue
  x0,y0,x1,y1=box
  if (x1-x0)*(y1-y0)>.25*h*w:continue
  xa,xb=max(0,int(x0)),min(w,int(np.ceil(x1)));ya,yb=max(0,int(y0)),min(h,int(np.ceil(y1)))
  patch=rgb[ya:yb,xa:xb].astype(float)
  for color,ci in [('red',0),('green',1),('blue',2)]:
   other=[i for i in range(3) if i!=ci]
   mask=(patch[:,:,ci]>1.35*patch[:,:,other[0]])&(patch[:,:,ci]>1.35*patch[:,:,other[1]])&(patch[:,:,ci]>45)
   yy,xx=np.where(mask);yy+=ya;xx+=xa
   z=dep[yy,xx];valid=np.isfinite(z)&(z>.05)&(z<3.0);yy,xx,z=yy[valid],xx[valid],z[valid]
   if len(z)<20:continue
   camera=np.stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z,np.ones_like(z)),axis=1)
   pts=(camera@T.T)[:,:3];lo,hi=np.percentile(pts,[5,95],axis=0);span=hi-lo
   if np.max(span)>.14:continue
   side=float(max(span[:2])/.9)
   if not .012<side<.12:continue
   top=float(np.percentile(pts[:,2],90));center=(lo+hi)/2;center[2]=top-side/2
   candidates.append(dict(name=color+' cube',kind='block',center_base=center.tolist(),lower_base=lo.tolist(),upper_base=hi.tolist(),
     estimated_side=side,extent=[side]*3,top_z_base=top,box=box,pixels=len(z),score=score,
     source='current RGB-D, public DINO and colour mask inside detected box; cubic-shape prior'))
 objects=[];amb=[]
 for name in sorted({o['name'] for o in candidates}):
  os=sorted([o for o in candidates if o['name']==name],key=lambda o:-o['score'])
  distinct=any(np.linalg.norm(np.array(o['center_base'])-os[0]['center_base'])>.04 for o in os[1:])
  if distinct:amb.append(name)
  else:objects.append(os[0])
 supplements,extra_amb=missing_color_components(packet,{o['name'] for o in objects}|set(amb))
 objects+=supplements;amb+=extra_amb
 return dict(objects=objects,missing_colour_supplements=[o['name'] for o in supplements],table_evidence=visible_table_sites(packet,objects),ambiguous_names=amb,seconds=time.perf_counter()-t,raw_labels=labels,raw_boxes=boxes,raw_scores=scores)
for line in sys.stdin:
 try:
  msg=json.loads(line)
  if msg.get('stop'):break
  if msg.get('reset'):
   agent.graph=None;agent.answers.cache.clear()
   agent.compiler=NativeCompiler(tuple(msg['skills']))
   result={'reset':True}
  elif msg.get('plan'):
   plan,log=agent.plan(msg['messages'],msg['state'],msg.get('history',[]))
   result={'plan':plan,'log':log}
  else:result=spill_detect(msg,proc,model,fn,threshold) if msg.get('perception_domain')=='spill' else detect(msg)
  print(json.dumps(result),flush=True)
 except Exception as e:
  import traceback
  print(json.dumps(dict(error=repr(e),traceback=traceback.format_exc())),flush=True)
