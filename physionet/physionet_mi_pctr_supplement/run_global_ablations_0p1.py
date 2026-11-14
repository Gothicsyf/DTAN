from pathlib import Path
import sys,subprocess,json,os
root=Path(__file__).resolve().parent
st=root/'logs/global_ablation_0p1_queue_status.json'
def save(value):
 tmp=st.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,st)
groups=[]
variants=['wo_pctr','wo_asre','wo_mtre','wo_dcsf','wo_alcw']
groups.extend([[(42,v,'source_global') for v in variants[i:i+2]] for i in range(0,len(variants),2)])
completed=[]
for group in groups:
 jobs=[]
 for seed,v,mode in group:
  tag=f'{v}_seed{seed}_pctr0p1';log=root/'logs'/f'{tag}_{mode}.log'
  with log.open('a',encoding='utf8') as f:
   proc=subprocess.Popen([sys.executable,'-u','run_experiment.py','--kind','formal','--device','cuda','--seed',str(seed),'--variant',v,'--tag',tag,'--mode',mode,'--pctr-weight','0.1'],cwd=root,stdout=f,stderr=subprocess.STDOUT)
  jobs.append((tag,mode,proc))
 save(dict(state='running',active=[(t,m,p.pid) for t,m,p in jobs],completed=completed))
 codes=[p.wait() for _,_,p in jobs]
 if any(codes):
  save(dict(state='failed',jobs=[(t,m,c) for (t,m,p),c in zip(jobs,codes)],completed=completed));sys.exit(1)
 completed.extend((t,m) for t,m,p in jobs)
save(dict(state='complete',completed=completed))


base=root/'results/full_seed42_pctr0p1/source_global'
report={}
for v in variants:
 folder=root/'results'/f'{v}_seed42_pctr0p1'/'source_global'
 scores=[]; gaps=[]
 for fold in range(1,11):
  rel=Path(f'subject{fold}/result.json')
  a,b=[json.loads((p/rel).read_text()) for p in (folder,base)]
  assert a['fold']==b['fold']==fold and a['target_test_trials']==b['target_test_trials']
  scores.append(a['metrics']['accuracy'])
  gaps.append(a['metrics']['accuracy']-b['metrics']['accuracy'])
 report[v]=dict(mean_accuracy=sum(scores)/10,mean_delta_pp=100*sum(gaps)/10,subject_delta_pp=[100*g for g in gaps])
(root/'results/global_ablation_0p1_comparison.json').write_text(json.dumps(report,indent=2))
