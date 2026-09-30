"""Public/adapted Laya with retained semantic graphs. No generative planner."""
import os,json,time
from pathlib import Path
from question_scope import ScopeCompiler as DirectCompiler,DirectAnswers,fuse_primary,fuse_scope_links as fuse_links
from aggregation import GuardCheckedAggregator as RetainedGraphAggregator
CHECKPOINT=Path(os.environ.get('LAYA_AGENT_CHECKPOINT','/root/agp_model_store/laya_dynamic_semantic_train_v4'))
class ScopeGraphAgent:
 def __init__(self,skills,adapted=True,checkpoint=None):
  self.compiler=DirectCompiler(skills);self.answers=DirectAnswers()
  self.aggregator=RetainedGraphAggregator();self.graph=None
  self.kind='dynamic_adapted_laya_only' if adapted else 'dynamic_direct_graph_v3'
  self.training=None
  if adapted:
   start=time.perf_counter()
   self.training=json.loads(((Path(checkpoint) if checkpoint else CHECKPOINT)/'COMPLETE.json').read_text())
   selected=self.training['selected'];torch=self.answers.torch
   weight_path=Path(selected['weights'])
   if not weight_path.is_absolute():weight_path=(Path(checkpoint) if checkpoint else CHECKPOINT)/weight_path
   weights=torch.load(weight_path,map_location='cpu',weights_only=True)
   self.answers.agent.model.load_state_dict(weights,strict=True)
   self.answers.agent.model.eval()
   self.answers.agent.temperature_by_options['choice:3-5']=float(selected['temperature'])
   self.answers.cold_seconds+=time.perf_counter()-start
 def plan(self,messages,state,history):
  start=time.perf_counter();t=start
  b=self.compiler.compile(messages,state,history,self.graph);compile1=time.perf_counter()-t
  p,r1=self.answers.predict(b['context'],b['questions'])
  t=time.perf_counter();f=fuse_primary(b,p);fusion=time.perf_counter()-t
  t=time.perf_counter();qs,meta=self.compiler.links(b,f);compile2=time.perf_counter()-t
  p2,r2=self.answers.predict(b['context'],qs) if qs else ({},dict(seconds=0,query_count=0))
  t=time.perf_counter();f.update(fuse_links(p2,meta));fusion+=time.perf_counter()-t
  plan,agg=self.aggregator.aggregate(b,f,meta,history);self.graph=plan.get('semantic_graphs')
  return plan,dict(kind=self.kind,seconds=time.perf_counter()-start,
      question_compilation_seconds=compile1+compile2,fusion_seconds=fusion,
      questions={**b['questions'],**qs},source_spans=b['source_spans'],context=b['context'],
      raw_laya_probabilities={**p,**p2},fusion_potentials=f,
      training_selected=self.training['selected'] if self.training else None,
      fusion_rule='min-conjunction ranking potentials plus constrained semantic graph beam; NOT joint confidence',
      phase1=r1,phase2=r2,aggregation=agg,parsed=plan,generative_llm_calls=0,output_tokens=0)
