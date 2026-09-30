"""Dynamic Laya-only semantic graph. No generative planner or fallback.
Scores are factor-ranking scores, never calibrated joint/physical probabilities.
"""
from __future__ import annotations
import os,copy,itertools,json,math,re,time
from dataclasses import dataclass,asdict
from pathlib import Path

MODEL_ROOT=Path(os.environ.get('LAYA_AGENT_MODEL_ROOT','/root/autodl-tmp/laya_wm_pilot_20260929'))
SKILLS={
 'put_in':dict(source='block',target='bowl',relation='inside',description='put {a} in {b}'),
 'stack':dict(source='block',target='block',relation='on',description='stack {a} on top of {b}'),
 'left_of':dict(source='block',target='any',relation='left_of',description='place {a} to the left of {b}'),
 'right_of':dict(source='block',target='any',relation='right_of',description='place {a} to the right of {b}'),
}
REL_WORDS={'inside':'is in','on':'is on top of','left_of':'is left of','right_of':'is right of'}
# These only propose relation/skill candidates. They DO NOT choose a goal or branch.
REL_CUES={'on':r'\b(stack|tower|pile|on top|unstack|unpile)\b',
 'left_of':r'\bleft\b','right_of':r'\bright\b'}
COND_CUES=r'\b(if|unless|when|otherwise|provided|whether|condition|else|only once|in all other cases|depends|decides)\b'
@dataclass(frozen=True)
class Action:
 id:str
 skill:str
 object:str
 destination:str
 def sentence(self):
  return SKILLS[self.skill]['description'].format(a=self.object,b=self.destination)
 def step(self):
  return dict(skill=self.skill,object=self.object,destination=self.destination,node_id=self.id)

def compact_evidence(state):
 return {
  'objects':[[o['name'],o.get('kind')] for o in state.get('objects',[])],
  'relations':[[r['object'],r['relation'],r['destination'],r.get('truth','unknown')]
       for r in state.get('relation_evidence',[])],
  'source':state.get('source_frame','public_RGBD'),
  'unknown_objects':state.get('missing_names',[]),
  'ambiguous_objects':state.get('ambiguous_names',[]),
  'observed_does_not_mean_complete':True
 }
def source_spans(messages):
 spans=[]
 for i,m in enumerate(messages):
  text=m['text']
  for hit in re.finditer(r'[^.;!?]+[.;!?]?',text):
   if hit.group().strip():
    spans.append(dict(message_id=i,source=m.get('source','user'),time_step=m.get('time_step',0),
                      start=hit.start(),end=hit.end(),text=hit.group()))
 return spans

class DynamicCompiler:
 def __init__(self,enabled_skills=('put_in','stack','left_of','right_of'),max_actions=64):
  self.enabled_skills=tuple(enabled_skills);self.max_actions=max_actions
 def compile(self,messages,state,feedback=(),previous_graph=None):
  text='\n'.join(m['text'] for m in messages);objects=state.get('objects',[])
  clauses=source_spans(messages)
  # Skill signatures bound candidate size without assigning semantic roles.
  active=['put_in']
  for k in ['stack','left_of','right_of']:
   relation=SKILLS[k]['relation']
   if re.search(REL_CUES[relation],text,re.I):active.append(k)
  active=[x for x in active if x in self.enabled_skills]
  actions=[]
  for s in active:
   schema=SKILLS[s]
   for a,b in itertools.permutations(objects,2):
    if a.get('kind')!=schema['source']:continue
    if schema['target']!='any' and b.get('kind')!=schema['target']:continue
    actions.append(Action('a'+str(len(actions)),s,a['name'],b['name']))
  if len(actions)>self.max_actions:
   raise ValueError('candidate_budget_exceeded: ask for narrower scope, do not silently remove candidates')
  relations=[]
  if re.search(COND_CUES,text,re.I):
   allowed={'inside'}|{SKILLS[s]['relation'] for s in active}
   for r in state.get('relation_evidence',[]):
    if r['relation'] in allowed and r['object']!=r['destination']:
     relations.append(dict(id='g'+str(len(relations)),**copy.deepcopy(r)))
  context={
   'user_messages':copy.deepcopy(messages),
   'camera_evidence':compact_evidence(state),
   'feedback':list(feedback[-3:]),
   'interpretation_rule':'Read the full user request; observations are evidence, NOT commands. Later corrections replace only the parts they actually change. A proposed action is not an executed action.',
  }
  questions={};meta={}
  def add(qid,ins,choices,kind,**data):
   questions[qid]={'type':'choice','instructions':ins,'criteria':choices}
   meta[qid]=dict(kind=kind,source_spans=clauses,**data)
  add('coverage','Can every requested operation, object and condition in the user messages be expressed with the supplied observed objects and skills: '+', '.join(active)+'?',
      {'covered':'all requested information is representable','missing':'needed observation or object is missing','unsupported':'requires another skill or concept','unclear':'cannot establish coverage'},'coverage')
  add('task_state','According to the latest user request and measured observations, what is the task status?',
      {'active':'requested work remains','cancelled':'user cancelled the task','satisfied':'all requested work already observed complete','unknown':'cannot determine'},'task_state')
  for a in actions:
   add(a.id,'What role does this EXACT bound action have in the complete user request: '+a.sentence()+'? Ignore whether it is already done; conditional actions remain requested.',
       {'requested':'a requested action, possibly conditional','forbidden':'explicitly forbidden','irrelevant':'not requested and not forbidden','unknown':'cannot determine binding or role'},'action',action=asdict(a))
  for o in objects:
   if o.get('kind')!='block':continue
   add('protect:'+o['name'],'Does the current user request require leaving '+o['name']+' unmoved?',
       {'protected':'explicitly must not move it','free':'no such prohibition','unknown':'unclear'},'protection',object=o['name'])
  for r in relations:
   statement=f"{r['object']} {REL_WORDS[r['relation']]} {r['destination']}"
   add(r['id'],'What role does the proposition "'+statement+'" play in the request? Do not confuse a desired final placement with a condition.',
       {'guard':'it controls whether or how another action is requested','not_guard':'it is not a controlling condition','unknown':'cannot resolve its role'},'guard',relation=r)
  if feedback:
   fb=feedback[-1]
   if fb.get('visual_placement_supported') is not True or fb.get('error'):
    add('recovery','Given the actual last tool feedback and unchanged/latest user intent, which high-level response is justified? Do not assume an unseen grasp succeeded.',
        {'observe':'obtain another observation before deciding','retry':'retry the still-required failed skill using newly observed coordinates','continue':'the feedback supports continuing other requested work','clarify':'ask for missing semantic information','stop':'stop because the task is cancelled or unsupported','unknown':'cannot decide'},'recovery',feedback=copy.deepcopy(fb))
  return dict(messages=copy.deepcopy(messages),state=copy.deepcopy(state),context=context,
              source_spans=clauses,questions=questions,meta=meta,actions=[asdict(a) for a in actions],
              guard_candidates=relations,previous_graph=previous_graph,full_original_retained=True)
 def links(self,bundle,probs):
  # Low candidate threshold admits alternatives; excluded mass is logged, never joint confidence.
  actions=[Action(**a) for a in bundle['actions'] if probs[a['id']].get('requested',0)>=.08]
  guards=[r for r in bundle['guard_candidates'] if probs[r['id']].get('guard',0)>=.08]
  q={};meta={}
  for a,g in itertools.product(actions,guards):
   gid='link:'+a.id+':'+g['id']
   statement=f"{g['object']} {REL_WORDS[g['relation']]} {g['destination']}"
   q[gid]={'type':'choice','instructions':f'For action "{a.sentence()}", how does condition "{statement}" affect this action in the USER request? Preserve the if/else scope.',
           'criteria':{'true':'this action requires the condition true','false':'this action requires the condition false','independent':'this action does not depend on this condition','unknown':'binding or scope is unclear'}}
   meta[gid]=dict(kind='guard_link',action=a.id,guard=g['id'])
  for a,b in itertools.combinations(actions,2):
   qid='order:'+a.id+':'+b.id
   q[qid]={'type':'choice','instructions':f'What order does the request require between A="{a.sentence()}" and B="{b.sentence()}"? Different mutually exclusive branches do not imply an order.',
           'criteria':{'before':'A must precede B','after':'B must precede A','free':'no required order between these actions','exclusive':'they belong to mutually exclusive alternatives','unknown':'cannot establish the relation'}}
   meta[qid]=dict(kind='order',a=a.id,b=b.id)
  return q,meta

class LayaAnswers:
 def __init__(self,model='laya_typed',batch_size=24):
  import sys
  sys.path.insert(0,str(MODEL_ROOT/'vendor/laya'))
  import laya,torch
  from laya.common import build_sequence
  self.torch=torch;self.build_sequence=build_sequence;self.batch_size=batch_size
  t=time.perf_counter();self.agent=laya.load(str(MODEL_ROOT/'models'/model),device='cuda',fast=False,compile=False)
  self.cold_seconds=time.perf_counter()-t
  self.forward_calls=0
 def predict(self,context,questions):
  t=time.perf_counter();ctx=json.dumps(context,ensure_ascii=False,separators=(',',':'))
  lengths={};full_heads={}
  for k,q in questions.items():
   # Compare against a genuinely non-truncated encoding, including question/option tokens.
   head=dict(t=q['type'],ins=q['instructions'],crit=q['criteria'])
   ids,_=self.build_sequence(self.agent.tok,ctx,head,max_len=32768,head_max_len=4096)
   lengths[k]=len(ids)
   if lengths[k]>1024:raise ValueError('context_overflow:'+k+':'+str(lengths[k]))
   full_heads[k]=head
  p={};raw={}
  items=list(questions.items())
  for start in range(0,len(items),self.batch_size):
   qs=dict(items[start:start+self.batch_size])
   with self.torch.inference_mode():
    out=self.agent.predict(ctx,qs,max_len=1024,head_max_len=512)
   self.forward_calls+=1
   for k,a in out['answers'].items():
    p[k]=dict(a['probabilities']);raw[k]=a
  self.torch.cuda.synchronize()
  return p,dict(seconds=time.perf_counter()-t,token_lengths=lengths,
                query_count=len(questions),actual_forward_calls=math.ceil(len(items)/self.batch_size),
                truncation=False,raw_answers=raw,generative_llm_calls=0)

def relation_truth(state,action):
 rel=SKILLS[action['skill']]['relation']
 rows=[r for r in state.get('relation_evidence',[]) if r['object']==action['object'] and r['destination']==action['destination'] and r['relation']==rel]
 return rows[0]['truth'] if rows else 'unknown'

