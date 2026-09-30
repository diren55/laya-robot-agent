"""Scope-explicit semantic questions. Original messages and source spans retained."""
import copy
from question_direct import DirectCompiler,DirectAnswers,YES,fuse_primary,render_question_source
from semantic_core import Action,REL_WORDS
class ScopeCompiler(DirectCompiler):
 def compile(self,*args,**kwargs):
  b=super().compile(*args,**kwargs)
  for g in b['guard_candidates']:
   s=f"{g['object']} {REL_WORDS[g['relation']]} {g['destination']}"
   b['questions'][g['id']]['instructions']=('Does the user use the truth of "'+s+
      '" as a prerequisite or a branch test? A requested output location alone is not a condition; a failed grasp is a different condition.')
  return b
 def links(self,bundle,probs):
  qs,meta=super().links(bundle,probs)
  actions={a['id']:Action(**a) for a in bundle['actions']}
  guards={g['id']:g for g in bundle['guard_candidates']}
  for k,m in meta.items():
   if m['kind']!='guard_link':continue
   a=actions[m['action']];g=guards[m['guard']]
   statement=f"{g['object']} {REL_WORDS[g['relation']]} {g['destination']}"
   for suffix,polarity in [('T','true'),('F','false')]:
    qs[k+':'+suffix]['instructions']=(
      f'Does the user explicitly require "{statement}" to be {polarity} before requesting "{a.sentence()}"? '
      'Judge only the dependency between these exact clauses. Do not treat the action goal as its own prerequisite.')
  return qs,meta

def fuse_scope_links(p,meta):
 from question_direct import fuse_links
 f=fuse_links(p,meta)
 for k,m in meta.items():
  if m['kind']=='guard_link':
   a,b=p[k+':T'],p[k+':F']
   f[k]=dict(true=min(a['yes'],b['no']),false=min(a['no'],b['yes']),
     independent=min(a['no'],b['no']),
     unknown=max(a['unknown'],b['unknown'],min(a['yes'],b['yes'])))
 return f
