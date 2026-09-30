"""Current-camera brown-spill adapter for the original CaP-X SpillWipe.
Public DINO identifies the initial region; subsequent colour/depth evidence is
scoped to that observed region. No marker poses, rewards or simulator task state.
This is a bounded brown-spill visual adapter, not an open-world segmenter.
"""
import time
import numpy as np
import torch
from PIL import Image

def brown_pixels(rgb):
 x=rgb.astype(float)
 return ((x[:,:,0]>25)&(x[:,:,0]<170)&(x[:,:,1]<150)&
  (x[:,:,0]>1.25*x[:,:,2])&(x[:,:,1]>1.15*x[:,:,2])&(x[:,:,0]>.9*x[:,:,1]))

def spill_detect(msg,proc,model,postprocess,threshold_name):
 began=time.perf_counter();p=np.load(msg['packet'])
 rgb=p['rgb'];depth=p['depth'].squeeze();K=p['intrinsics'];T=p['pose_mat']
 h,w=depth.shape;yy,xx=np.mgrid[:h,:w];z=depth
 valid=np.isfinite(z)&(z>.05)&(z<3.)
 xyz=(np.stack(((xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z,np.ones_like(z)),-1)@T.T)[:,:,:3]
 reachable=valid&(xyz[:,:,0]>.25)&(xyz[:,:,0]<.9)&(abs(xyz[:,:,1])<.45)&(xyz[:,:,2]>-.3)&(xyz[:,:,2]<.15)
 white=(rgb.min(axis=2)>150)&reachable
 plane=float(np.median(xyz[:,:,2][white])) if np.count_nonzero(white)>200 else None
 mask=brown_pixels(rgb)&reachable
 if plane is not None:mask &= abs(xyz[:,:,2]-plane)<.02
 labels=[];boxes=[];scores=[];reference=msg.get('reference')
 if reference is None:
  x=proc(images=Image.fromarray(rgb),text='brown spill. brown stain. dirty surface.',return_tensors='pt').to('cuda')
  with torch.inference_mode():raw=model(**x)
  det=postprocess(raw,x.input_ids,**{threshold_name:.20,'text_threshold':.20,'target_sizes':[(h,w)]})[0]
  torch.cuda.synchronize()
  boxes=det['boxes'].cpu().tolist();scores=det['scores'].cpu().tolist();labels=det.get('text_labels',det.get('labels'))
  support=np.zeros((h,w),bool)
  for b,lab in zip(boxes,labels):
   if not isinstance(lab,str) or not any(v in lab for v in ['spill','stain','dirty']):continue
   x0,y0,x1,y1=b
   support[max(0,int(y0)):min(h,int(np.ceil(y1))),max(0,int(x0)):min(w,int(np.ceil(x1)))]=True
  mask &= support
  if np.count_nonzero(mask)<30:
   return dict(status='unknown',reason='No unique supported initial spill region',objects=[],reference=None,
     seconds=time.perf_counter()-began,raw_labels=labels,raw_boxes=boxes,raw_scores=scores,plane=plane)
  points=xyz[mask];lo,hi=np.percentile(points,[.2,99.8],axis=0)
  reference=dict(xy_min=(lo[:2]-.012).tolist(),xy_max=(hi[:2]+.012).tolist(),
    plane_height=plane,source_frame=msg['packet'],kind='static observed surface region, not remembered moving-object coordinates')
 else:
  lo=np.array(reference['xy_min']);hi=np.array(reference['xy_max'])
  mask &= (xyz[:,:,0]>=lo[0])&(xyz[:,:,0]<=hi[0])&(xyz[:,:,1]>=lo[1])&(xyz[:,:,1]<=hi[1])
 dirty=int(np.count_nonzero(mask))
 lo=np.array(reference['xy_min']);hi=np.array(reference['xy_max']);plane_ref=reference['plane_height']
 clean_surface=reachable&(abs(xyz[:,:,2]-plane_ref)<.01)
 # Current measured coverage of the earlier observed fixed surface patch.
 cells=[]
 for x in np.arange(lo[0]+.006,hi[0],.012):
  for y in np.arange(lo[1]+.006,hi[1],.012):
   seen=(abs(xyz[:,:,0]-x)<.007)&(abs(xyz[:,:,1]-y)<.007)&clean_surface
   cells.append(bool(np.count_nonzero(seen)>=3))
 coverage=float(np.mean(cells)) if cells else 0.
 truth='false' if dirty>=10 else ('true' if coverage>=.95 else 'unknown')
 if dirty>=10:
  pts=xyz[mask];bl,bh=np.percentile(pts,[.2,99.8],axis=0)
  bounds=[bl.tolist(),bh.tolist()];center=((bl+bh)/2).tolist();extent=(bh-bl).tolist()
 else:
  bounds=None;center=[float((lo[0]+hi[0])/2),float((lo[1]+hi[1])/2),plane_ref];extent=[float(hi[0]-lo[0]),float(hi[1]-lo[1]),0.]
 obj=dict(name='brown spill',kind='surface_region',center_base=center,extent=extent,current_dirty_bounds=bounds,
  current_dirty_pixels=dirty,clean_truth=truth,current_surface_visibility=coverage,
  reference=reference,source='current RGBD of initially DINO-grounded static surface region; absence only counts with measured current visibility')
 return dict(status='observed',objects=[obj],reference=reference,clean_truth=truth,dirty_pixels=dirty,
  visibility=coverage,seconds=time.perf_counter()-began,raw_labels=labels,raw_boxes=boxes,raw_scores=scores,plane=plane)
