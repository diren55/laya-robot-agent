"""Uncertainty-preserving semantic factor beam, no generative fallback.
All numbers below are ranking potentials. Correlated answers are not multiplied
and advertised as calibrated joint or physical success probabilities.
"""
import copy,itertools,math,time
from semantic_core import Action,relation_truth

class GuardCheckedAggregator:
 def __init__(self,beam_width=24,plausibility_gap=.70,max_failed_attempts=2):
  self.width=beam_width;self.gap=plausibility_gap;self.max_failed_attempts=max_failed_attempts
 @staticmethod
 def alternatives(p,limit=2):
  # Unknown is a genuine assignment, not an option silently discarded.
  return sorted(p.items(),key=lambda z:-z[1])[:limit]
 def aggregate(self,b,probs,links,history=()):
  began=time.perf_counter()
  acts={a['id']:a for a in b['actions']};guards={g['id']:g for g in b['guard_candidates']}
  report=dict(all_probabilities=probs,score_type='sum log factor-ranking support, constant alternative gap; NOT a joint probability',
      unknown_retained=True,beam_width=self.width,pruned_assignments=0,generative_llm_calls=0)
  def finish(status,reason,graphs=(),steps=()):
   report.update(status=status,reason=reason,graphs=list(graphs),seconds=time.perf_counter()-began)
   return dict(steps=list(steps),status=status,complete_plan=status=='done',reason=reason,
       semantic_graphs=list(graphs),generative_llm_calls=0),report
  cp=probs['coverage']
  if max(cp,key=cp.get)!='covered' or cp.get('covered',0)<.55:
   return finish('clarify','coverage_'+max(cp,key=cp.get))
  tp=probs['task_state']
  if tp.get('cancelled',0)>.8:
   if tp.get('active',0)>.55:return finish('clarify','cancellation_action_conflict')
   return finish('done','explicit_cancellation')
  beams=[dict(roles={},protect={},guard_roles={},bindings={},orders={},score=0.,n=0)]
  def expand(key,p,field,relevant=lambda g:True):
   nonlocal beams
   nxt=[]
   for g in beams:
    # Irrelevant factors are still recorded/scored, but cannot constrain a
    # nonexistent node. Keep the same factor denominator for all graphs.
    variants=self.alternatives(p) if relevant(g) else self.alternatives(p,1)
    for label,value in variants:
     x=copy.deepcopy(g);x[field][key]=label
     x['score']+=math.log(max(float(value),1e-8));x['n']+=1
     nxt.append(x)
   report['pruned_assignments']+=max(0,len(nxt)-self.width)
   beams=sorted(nxt,key=lambda z:-z['score'])[:self.width]
  for aid in acts:expand(aid,probs[aid],'roles')
  for key in sorted(k for k in probs if k.startswith('protect:')):
   expand(key.split(':',1)[1],probs[key],'protect')
  # Inconsistent request/protection interpretations may not send a command.
  beams=[g for g in beams if not any(
    role=='requested' and g['protect'].get(acts[aid]['object'])=='protected'
    for aid,role in g['roles'].items())]
  if not beams:return finish('clarify','inconsistent_request_and_protection')
  for gid in guards:expand(gid,probs[gid],'guard_roles')
  for key,m in links.items():
   if m['kind']=='guard_link':
    expand(key,probs[key],'bindings',lambda g,m=m:g['roles'][m['action']]=='requested'
      and g['guard_roles'].get(m['guard']) in ('guard','unknown'))
   else:
    expand(key,probs[key],'orders',lambda g,m=m:g['roles'][m['a']]=='requested'
      and g['roles'][m['b']]=='requested')
  built=[]
  for g in beams:
   nodes=[aid for aid,r in g['roles'].items() if r=='requested']
   semantic_unknown=['role:'+aid for aid,r in g['roles'].items() if r=='unknown']
   physical_unknown=[];edges=[];inactive=set();bindings=[]
   for gid,label in g['guard_roles'].items():
    if label=='unknown':semantic_unknown.append('guard_role:'+gid)
   for key,label in g['bindings'].items():
    m=links[key];aid=m['action'];gid=m['guard']
    if aid not in nodes or g['guard_roles'].get(gid)!='guard':continue
    bindings.append(dict(action=aid,guard=gid,binding=label,probabilities=probs[key]))
    if label=='unknown':semantic_unknown.append(key)
    elif label in ('true','false'):
     # A positive prerequisite equal to this action's own goal cannot justify
     # declaring an unmet task complete. Preserve the contradiction for repair.
     ac=acts[aid];gd=guards[gid]
     from semantic_core import SKILLS
     same_goal=(ac['object']==gd['object'] and ac['destination']==gd['destination'] and SKILLS[ac['skill']]['relation']==gd['relation'])
     if label=='true' and same_goal and relation_truth(b['state'],ac)!='true':
      semantic_unknown.append('self_prerequisite:'+key)
     truth=guards[gid].get('truth','unknown')
     if truth=='unknown':physical_unknown.append(gid)
     elif (truth=='true')!=(label=='true'):inactive.add(aid)
   for key,label in g['orders'].items():
    m=links[key]
    if m['a'] not in nodes or m['b'] not in nodes:continue
    if label=='before':edges.append((m['a'],m['b']))
    elif label=='after':edges.append((m['b'],m['a']))
    elif label=='unknown':semantic_unknown.append(key)
    elif label=='exclusive' and m['a'] not in inactive and m['b'] not in inactive:
     semantic_unknown.append(key)
   done={aid for aid in nodes if relation_truth(b['state'],acts[aid])=='true'}
   remaining=[a for a in nodes if a not in done and a not in inactive]
   ready=[a for a in remaining if not any(v==a and u not in done and u not in inactive for u,v in edges)]
   for a in ready:
    if g['protect'].get(acts[a]['object'])=='unknown':
     semantic_unknown.append('protection:'+acts[a]['object'])
   if any(acts[a]['object']==acts[c]['object'] for a,c in itertools.combinations(ready,2)):
    semantic_unknown.append('incompatible_ready_destinations')
   if remaining and not ready:semantic_unknown.append('cycle_or_unsatisfied_dependency')
   if not nodes and tp.get('active',0)>.55:semantic_unknown.append('active_task_has_no_bound_action')
   if not nodes and tp.get('active',0)<=.55:semantic_unknown.append('no_explicit_completion_evidence')
   built.append(dict(nodes=[acts[a] for a in nodes],action_roles=g['roles'],
       guard_nodes=list(guards.values()),guard_roles=g['guard_roles'],guard_links=bindings,
       edge_labels=g['orders'],edges=edges,protection_roles=g['protect'],
       done=sorted(done),inactive=sorted(inactive),ready=ready,
       unknown=semantic_unknown+physical_unknown,semantic_unknown=semantic_unknown,
       physical_unknown=physical_unknown,score=g['score'],mean_log_support=g['score']/max(1,g['n']),factor_count=g['n']))
  if not built:return finish('clarify','no_retained_interpretation')
  built.sort(key=lambda z:-z['score'])
  plausible=[g for g in built if g['score']>=built[0]['score']-self.gap]
  if any(g['semantic_unknown'] for g in plausible):
   return finish('clarify','unresolved_semantic_binding',plausible)
  if any(g['physical_unknown'] for g in plausible):
   return finish('observe','unobserved_condition',plausible)
  ready_sets=[set(g['ready']) for g in plausible]
  common=set.intersection(*ready_sets)
  if not common:
   if all(not g['ready'] for g in plausible) and all(g['nodes'] for g in plausible):
    return finish('done','all_retained_graphs_complete_or_inactive',plausible)
   return finish('clarify','retained_graphs_disagree_on_next_action',plausible)
  chosen=sorted(common)[0]
  if 'recovery' in probs:
   rp=probs['recovery'];report['recovery_probabilities']=rp
   ranked=sorted(rp.items(),key=lambda z:-z[1]);route=ranked[0][0]
   if len(ranked)>1 and ranked[0][1]-ranked[1][1]<.10:
    return finish('clarify','ambiguous_recovery_interpretation',plausible)
   if route not in ('retry','continue'):
    return finish('observe' if route=='observe' else 'clarify','laya_recovery_'+route,plausible)
   if route=='retry' and history:
    last=history[-1]
    matches=[aid for aid in common if all(acts[aid][k]==last.get(k) for k in ('object','destination'))
             and acts[aid]['skill']==last.get('tool')]
    if matches:chosen=sorted(matches)[0]
   pair=(acts[chosen]['object'],acts[chosen]['destination'])
   failed=sum((h.get('object'),h.get('destination'))==pair
       and not h.get('visual_placement_supported',False) for h in history)
   if failed>=self.max_failed_attempts:return finish('clarify','bounded_laya_recovery_exhausted',plausible)
  return finish('ready','shared_next_action_across_retained_graphs',plausible,[Action(**acts[chosen]).step()])
