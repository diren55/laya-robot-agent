"""Direct semantic propositions after DEV input-format diagnosis.
No LLM. Full vector fusion is explicit and is NOT a calibrated joint probability.
"""
import copy,json,math,time
from semantic_core import *
YES={'yes':'Yes.','no':'No.','unknown':'The supplied information does not determine this.'}
class DirectCompiler(DynamicCompiler):
 def compile(self,messages,state,feedback=(),previous_graph=None):
  b=super().compile(messages,state,feedback,previous_graph)
  q={}
  def add(k,ins,kind='language'):
   q[k]=dict(type='choice',instructions=ins,criteria=copy.deepcopy(YES),evidence_kind=kind)
  catalog=', '.join(o['name'] for o in state['objects'])
  add('coverage','Are all objects and operations needed by the user available in this list? Objects: '+catalog+'. Skills: '+', '.join(self.enabled_skills)+'.','inventory')
  add('cancelled','Has the latest user instruction cancelled the whole task?')
  add('task_state','Does the user request at least one robot action, even if it is conditional?')
  for a in b['actions']:
   action=Action(**a);s=action.sentence()
   add(a['id'],'According to the user, should the robot '+s+'? A conditionally requested action counts as requested.')
   add('forbid:'+a['id'],'Does the user explicitly forbid the robot to '+s+'?')
  for o in state['objects']:
   if o['kind']=='block':add('protect:'+o['name'],'Does the user require the robot to leave '+o['name']+' unmoved?')
  for g in b['guard_candidates']:
   s=f"{g['object']} {REL_WORDS[g['relation']]} {g['destination']}"
   add(g['id'],'Is "'+s+'" used as a condition for deciding whether or where another object should move, rather than just a desired final location?')
  if 'recovery' in b['questions']:
   prompts={
    'observe':'Does the last tool feedback indicate that another camera observation is needed before choosing an action?',
    'retry':'Does the last tool feedback justify retrying a still-required failed action using refreshed visual coordinates?',
    'continue':'Does the feedback confirm completion of the previous action so the robot can continue to other requested work?',
    'clarify':'Is required information about the user intention missing, so the robot needs to ask the user?',
    'stop':'Does the latest user instruction require stopping the whole task?'
   }
   for key,ins in prompts.items():add('recover:'+key,ins,'feedback')
  b['questions']=q
  return b
 def links(self,bundle,probs):
  oldq,meta=super().links(bundle,probs)
  actions={a['id']:Action(**a) for a in bundle['actions']}
  guards={g['id']:g for g in bundle['guard_candidates']}
  q={}
  for k,m in meta.items():
   if m['kind']=='guard_link':
    a=actions[m['action']];g=guards[m['guard']]
    statement=f"{g['object']} {REL_WORDS[g['relation']]} {g['destination']}"
    for polarity,assumption in [('T',statement),('F','it is NOT true that '+statement)]:
     q[k+':'+polarity]=dict(type='choice',
       instructions=f'Assume {assumption}. According to the full user instruction, should the robot {a.sentence()} in that case?',
       criteria=copy.deepcopy(YES),evidence_kind='language')
   else:
    a,b=actions[m['a']],actions[m['b']]
    for polarity,u,v in [('AB',a,b),('BA',b,a)]:
     q[k+':'+polarity]=dict(type='choice',
       instructions=f'Does the user require "{u.sentence()}" BEFORE "{v.sentence()}"? Do not infer an order just because they appear in that order in this question.',
       criteria=copy.deepcopy(YES),evidence_kind='language')
  return q,meta

def render_question_source(context,q):
 source='\n'.join(('The user says: ' if i==0 else 'Later the user says: ')+m['text']
                  for i,m in enumerate(context['user_messages']))
 kind=q.get('evidence_kind','language')
 if kind=='inventory':
  source+='\nObserved object names: '+', '.join(x[0] for x in context['camera_evidence']['objects'])
 if kind=='feedback':
  for fb in context['feedback']:
   source+=f"\nObserved feedback at step {fb.get('time_step')}: tool {fb.get('tool')} targeted {fb.get('object')} -> {fb.get('destination')}. Controller returned={fb.get('controller_returned')}; camera relation={fb.get('visual_relation_truth','true' if fb.get('visual_placement_supported') else 'not established')}; error={fb.get('error')}."
 return source
class DirectAnswers(LayaAnswers):
 def __init__(self):
  super().__init__('laya_en');self.cache={}
 def predict(self,context,questions):
  # All original source retained. Geometry is not repeated in questions about intent.
  # Facts are available to the compiler/graph executor, not treated as commands.
  groups={}
  for k,q in questions.items():
   s=render_question_source(context,q)
   groups.setdefault(s,{})[k]={kk:vv for kk,vv in q.items() if kk!='evidence_kind'}
  allp={};logs=[];t=time.perf_counter()
  for src,qs in groups.items():
   p,rec=self._text_predict(src,qs);allp.update(p);logs.append(dict(source=src,**rec))
  return allp,dict(seconds=time.perf_counter()-t,groups=logs,query_count=len(questions),
    generative_llm_calls=0,full_original_retained=True)
 def _text_predict(self,ctx,questions):
  t=time.perf_counter();lengths={}
  for k,q in questions.items():
   ids,_=self.build_sequence(self.agent.tok,ctx,dict(t=q['type'],ins=q['instructions'],crit=q['criteria']),max_len=32768,head_max_len=4096)
   lengths[k]=len(ids)
   if len(ids)>1024:raise ValueError('context_overflow:'+k)
  p={};pending=[];keys={}
  for k,q in questions.items():
   key=(ctx,json.dumps(q,sort_keys=True));keys[k]=key
   if key in self.cache:p[k]=copy.deepcopy(self.cache[key])
   else:pending.append((k,q))
  for start in range(0,len(pending),self.batch_size):
   with self.torch.inference_mode():
    out=self.agent.predict(ctx,dict(pending[start:start+self.batch_size]),max_len=1024,head_max_len=512)
   for k,a in out['answers'].items():
    p[k]=a['probabilities'];self.cache[keys[k]]=copy.deepcopy(p[k])
  self.torch.cuda.synchronize()
  return p,dict(seconds=time.perf_counter()-t,token_lengths=lengths,truncation=False,
      cached_questions=len(questions)-len(pending),actual_forward_calls=math.ceil(len(pending)/self.batch_size))
def fuse_primary(b,p):
 f={}
 f['coverage']=dict(covered=p['coverage']['yes'],missing=p['coverage']['no'],unclear=p['coverage']['unknown'])
 f['task_state']=dict(active=p['task_state']['yes'],cancelled=p['cancelled']['yes'],
                       satisfied=0.,unknown=max(p['task_state']['unknown'],p['cancelled']['unknown']))
 for a in b['actions']:
  aid=a['id'];r=p[aid];n=p['forbid:'+aid]
  f[aid]=dict(requested=min(r['yes'],n['no']),irrelevant=min(r['no'],n['no']),
     forbidden=min(n['yes'],r['no']),unknown=max(r['unknown'],n['unknown'],min(r['yes'],n['yes'])))
 for k,v in p.items():
  if k.startswith('protect:'):f[k]=dict(protected=v['yes'],free=v['no'],unknown=v['unknown'])
  if k.startswith('g') and k[1:].isdigit():f[k]=dict(guard=v['yes'],not_guard=v['no'],unknown=v['unknown'])
 if 'recover:observe' in p:
  f['recovery']={k.split(':')[1]:v['yes'] for k,v in p.items() if k.startswith('recover:')}
  f['recovery']['unknown']=max(v['unknown'] for k,v in p.items() if k.startswith('recover:'))
 return f
def fuse_links(p,meta):
 f={}
 for k,m in meta.items():
  if m['kind']=='guard_link':
   a,b=p[k+':T'],p[k+':F']
   f[k]={'true':min(a['yes'],b['no']),'false':min(a['no'],b['yes']),
         'independent':min(a['yes'],b['yes']),
         'unknown':max(a['unknown'],b['unknown'],min(a['no'],b['no']))}
  else:
   a,b=p[k+':AB'],p[k+':BA']
   f[k]={'before':min(a['yes'],b['no']),'after':min(a['no'],b['yes']),
         'free':min(a['no'],b['no']),'exclusive':0.,
         'unknown':max(a['unknown'],b['unknown'],min(a['yes'],b['yes']))}
 return f
