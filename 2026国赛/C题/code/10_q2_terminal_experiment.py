"""Reproduce Q2 main policies, then run a predeclared 2x2 terminal-target experiment.

Uses frozen copies of 02/05/08, original MILP and actual feedback. Outputs are isolated.
Run with E:/Anaconda/envs/math_modeling/python.exe code/10_q2_terminal_experiment.py
Completed policies can be resumed, provided input/config/source fingerprints agree.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_terminal_experiment_20260911'
OLD = ROOT / 'results/q2_quantile_experiment'
AUDIT = ROOT / 'results/q2_revision_audit_20260911'
SNAP = AUDIT / 'source_snapshot'
POLICIES = [
    dict(id='A_base', kind='fixed', theta='base', W=28, J=None, terminal='cycle'),
    dict(id='B_fixed_q80', kind='fixed', theta=.8, W=28, J=None, terminal='cycle'),
    dict(id='B_adaptive', kind='adaptive', theta=None, W=28, J=14, terminal='cycle'),
    dict(id='T_base_6000', kind='fixed', theta='base', W=28, J=None, terminal='6000'),
    dict(id='T_q80_6000', kind='fixed', theta=.8, W=28, J=None, terminal='6000'),
]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def module():
    spec = importlib.util.spec_from_file_location('frozen_q2', SNAP / 'code/08_q2_quantile_experiment.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.OUT = OUT
    return m


def archive(m, load, pv, terminal):
    class TerminalArchive(m.Archive):
        def solve_plan(self, protected_net_kWh, price, initial):
            self.solve_calls += 1
            (q, c, d, w, state), info = m.core.solve(
                np.asarray(protected_net_kWh)/m.DT, np.zeros(144), price,
                integer=True, initial_kWh=initial, terminal_kWh=6000.)
            assert abs(state[-1]-6000) < 1e-6
            return q, state, info
    cls = m.Archive if terminal == 'cycle' else TerminalArchive
    return cls(load, pv, len(load))


def run_one(policy, load, pv, price, warm, signature):
    m = module()
    marker = OUT / policy['id'] / 'completed.json'
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        assert saved['signature'] == signature
        assert all(digest(OUT / policy['id'] / k) == v for k, v in saved['files'].items())
        return saved['result']
    bundle = dict(archive=archive(m, load, pv, policy['terminal']), price=price, dates=m.bm.DATES,
                  warmup_states=warm, nu=float(np.median(price)/.9), a_monthly=None,
                  load=load, pv=pv, stop_day=None)
    result = m.run_policy(policy, bundle)
    result['totals']['terminal_rule'] = policy['terminal']
    pd.DataFrame(result.pop('selection')).to_csv(OUT / policy['id'] / 'selection_log.csv', index=False)
    assert result['totals']['solver_failures'] == 0
    assert max(result['validation']['checks'].values()) < 1e-6
    files = {p.name: digest(p) for p in marker.parent.glob('*.csv')}
    files['validation.json'] = digest(marker.parent / 'validation.json')
    save(marker, dict(signature=signature, result=result, files=files))
    return result


def independent(name):
    d = pd.read_csv(OUT/name/'dispatch.csv', low_memory=False)
    q,c,b,e,w = [d[x].to_numpy() for x in ['planned_kWh','charge_kWh','discharge_kWh','emergency_kWh','unused_kWh']]
    s0,s1,p = [d[x].to_numpy() for x in ['state_start_kWh','state_end_kWh','price_yuan_kWh']]
    fee=p*q+5*p*e
    checks = dict(balance=float(np.max(np.abs(q+d.pv_kW.to_numpy()/6+b+e-d.load_kW.to_numpy()/6-c-w))),
                  state=float(np.max(np.abs(s1-s0-.9*c+b/.9))),
                  continuous=float(np.max(np.abs(s0[1:]-s1[:-1]))),
                  capacity=float(max(0,1200-min(s0.min(),s1.min()),max(s0.max(),s1.max())-10800)),
                  power=float(max(0,c.max()-5000/6,b.max()-5000/6)),
                  nonnegative=float(max(0,-min(q.min(),c.min(),b.min(),e.min(),w.min()))),
                  mutex=float(np.minimum(c,b).max()))
    assert len(d)==48096 and d.date.nunique()==334
    assert max(checks.values())<1e-6
    loss=.1*c+(1/.9-1)*b
    energy=float(np.sum(q+e+d.pv_kW.to_numpy()/6-d.load_kW.to_numpy()/6-w-loss)-(s1[-1]-s0[0]))
    assert abs(energy)<1e-4
    daily=d.assign(total=fee).groupby('date').agg(total=('total','sum'),end=('state_end_kWh','last'))
    nominal=d.groupby('date').agg(initial=('state_start_kWh','first'),end=('nominal_state_end_kWh','last'))
    target=nominal.initial if name in ['A_base','B_fixed_q80','B_adaptive'] else 6000.
    nominal_error=float(np.max(np.abs(nominal.end-target)))
    assert nominal_error<1e-6
    return dict(strategy_id=name,total_cost_yuan=float(fee.sum()),energy_identity_residual_kWh=energy,
                nominal_terminal_max_error_kWh=nominal_error,checks=checks,
                final_kWh=float(s1[-1]),full_end_days=int((daily.end>=10800-1e-6).sum()))


def compare_original(name):
    a=pd.read_csv(OLD/name/'dispatch.csv',low_memory=False)
    b=pd.read_csv(OUT/name/'dispatch.csv',low_memory=False)
    numeric=['planned_kWh','charge_kWh','discharge_kWh','emergency_kWh','unused_kWh',
             'state_start_kWh','state_end_kWh','nominal_state_end_kWh',
             'residual_adjustment_kWh','planned_cost_yuan','emergency_cost_yuan']
    diffs={x:float(np.max(np.abs(a[x]-b[x]))) for x in numeric}
    assert a[['date','slot']].equals(b[['date','slot']])
    assert a.theta.astype(str).equals(b.theta.astype(str))
    assert max(diffs.values())<1e-6, diffs
    return dict(max_differences=diffs,theta_matches=True,all_within_1e_6=True)


def diagnostic(name, folder):
    d=pd.read_csv(folder/name/'dispatch.csv',low_memory=False)
    d['hour']=d.slot//6
    em=d.emergency_kWh>1e-6;unused=d.unused_kWh>1e-6
    energy=d.state_end_kWh<=1200+1e-6;power=d.discharge_kWh>=5000/6-1e-6
    capacity=d.state_end_kWh>=10800-1e-6;charge=d.charge_kWh>=5000/6-1e-6
    rows=[]
    # Categories are exclusive; a tie is explicitly recorded.
    for kind,active,left,right,flow in [('emergency',em,energy,power,'emergency_kWh'),
                                      ('unused',unused,capacity,charge,'unused_kWh')]:
        for label,mask in [('state_only',left&~right),('power_only',right&~left),('both',left&right),('neither',~left&~right)]:
            x=d[active&mask]
            rows.append(dict(strategy_id=name,event=kind,limit=label,slots=len(x),energy_kWh=float(x[flow].sum()),emergency_cost_yuan=float(x.emergency_cost_yuan.sum())))
        assert sum(x['slots'] for x in rows if x['event']==kind)==int(active.sum())
    hourly=d.groupby('hour')[['planned_kWh','emergency_kWh','unused_kWh','emergency_cost_yuan']].sum().reset_index()
    hourly.insert(0,'strategy_id',name)
    return rows,hourly


def causal_checks(m,load,pv,price,warm):
    rows=[]
    for name,k,W,terminal in [('B_adaptive',31,28,'cycle'),('B_adaptive',200,28,'cycle'),
                              ('S_adaptive_W56',31,56,'cycle'),('T_q80_6000',200,28,'6000')]:
        folder=OLD if name=='S_adaptive_W56' else OUT
        daily=pd.read_csv(folder/name/'daily_summary.csv')
        history=dict(warm)
        history.update({int((pd.Timestamp(r.date)-m.bm.DATES[0]).days):float(r.initial_kWh) for r in daily.itertuples()})
        previous=None if k==31 else daily.iloc[k-32].theta
        if previous is not None and previous!='base':previous=float(previous)
        l2=load.copy();v2=pv.copy();l2[k:]=load[k:]*2+1234;v2[k:]=pv[k:]*.1
        a=archive(m,load,pv,terminal);b=archive(m,l2,v2,terminal)
        if name.startswith('T_'):
            theta1=theta2=.8;score_error=0.
        else:
            theta1,s1,_=m.select_theta(k,W,14,history,a,price,m.bm.DATES,float(np.median(price)/.9),previous)
            theta2,s2,_=m.select_theta(k,W,14,history,b,price,m.bm.DATES,float(np.median(price)/.9),previous)
            score_error=max(abs(s1[t]['score_yuan']-s2[t]['score_yuan']) for t in s1)
        q1,_,info1,_,r1,_,_=a.plan_day(k,theta1,W,price,history[k])
        q2,_,info2,_,r2,_,_=b.plan_day(k,theta2,W,price,history[k])
        row=dict(policy=name,day=str(m.bm.DATES[k].date()),theta_same=theta1==theta2,score_error=score_error,
                 forecast_error=float(np.max(np.abs(a.n_forecast[k]-b.n_forecast[k]))),
                 correction_error=float(np.max(np.abs(r1-r2))),plan_error=float(np.max(np.abs(q1-q2))))
        assert row['theta_same'] and all(row[x]==0 for x in ['score_error','forecast_error','correction_error','plan_error'])
        assert not info1['fallback'] and not info2['fallback']
        rows.append(row)
    return rows


def main():
    assert Path(sys.prefix).name=='math_modeling'
    OUT.mkdir(exist_ok=True);AUDIT.mkdir(exist_ok=True)
    start=time.time()
    inputs=['code/02_q1_baseline.py','code/05_q2_baseline.py','code/08_q2_quantile_experiment.py',
            'code/09_q2_quantile_diagnostics.py','code/10_q2_terminal_experiment.py',
            '附件/附件1.xlsx','附件/附件2.xlsx','results/audit/audit_summary.json']
    hashes={p:digest(ROOT/p) for p in inputs}
    signature=hashlib.sha256(json.dumps(dict(hashes=hashes,policies=POLICIES),sort_keys=True).encode()).hexdigest()
    for rel in inputs:
        target=SNAP/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert digest(target)==hashes[rel],rel
        else:shutil.copy2(ROOT/rel,target)
    protected=[p for tree in ['附件','results/q1_milp','results/q2_baseline','results/q2_quantile_experiment'] for p in (ROOT/tree).rglob('*') if p.is_file() and not p.name.startswith('~$')]
    protected_hashes={str(p.relative_to(ROOT)):digest(p) for p in protected}
    save(AUDIT/'protected_before.json',protected_hashes)
    old_manifest=json.loads((OLD/'run_manifest.json').read_text(encoding='utf-8-sig'))
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='running',signature=signature,
                source_input_sha256=hashes,old08_sha256=old_manifest['code_sha256']['code/08_q2_quantile_experiment.py'],
                provenance_note='Original 08 content not recovered. Frozen current source reproduced independently; do not relabel original manifest.',
                policies=POLICIES,executable=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
                fixed_conditions='lag predictor, W28, same January warmup, actual greedy feedback, .9/.9, same MILP, normal q paid in full, no extra surplus penalty',
                primary_contrast='T_q80_6000 minus B_fixed_q80; base pair diagnoses interaction; B_adaptive is reproduction only',
                registration='reports/问题二_终端目标对照实验方案.md')
    save(OUT/'run_manifest.json',record)
    m=module();load,pv,price,_=m.bm.read_sources()
    warm,frame,end,checks=m.run_warmup(m.Archive(load,pv,len(load)),price,m.bm.DATES)
    assert abs(end-m.REFERENCE['evaluation_initial_kWh'])<1e-6
    frame.to_csv(OUT/'warmup_january.csv',index=False)
    save(OUT/'config.json',dict(policies=POLICIES,common=m.CONFIG,terminal6000_is_model_assumption=True,
                              evaluation_initial_kWh=end,selection='fixed groups do not select parameters',
                              checkpoint='completed policy only, not within-policy day checkpoints'))
    pre=[];hours=[]
    for name in ['A_base','B_fixed_q80','B_adaptive']:
        x,h=diagnostic(name,OLD);pre.extend(x);hours.append(h)
    pd.DataFrame(pre).to_csv(OUT/'pre_experiment_limits.csv',index=False)
    pd.concat(hours).to_csv(OUT/'pre_experiment_hourly.csv',index=False)
    results=[]
    # Verify current-version reproduction before launching new treatment policies.
    for batch in [POLICIES[:3], POLICIES[3:]]:
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs={pool.submit(run_one,p,load,pv,price,warm,signature):p['id'] for p in batch}
            for job in as_completed(jobs):
                result=job.result();results.append(result)
                print('Completed',jobs[job],result['totals']['total_cost_yuan'],flush=True)
        if batch[0]['id']=='A_base':
            save(AUDIT/'reproduction.json',{name:compare_original(name) for name in ['A_base','B_fixed_q80','B_adaptive']})
            print('Current-version main-policy reproduction passed; starting registered treatments.',flush=True)
    comparison=pd.DataFrame([x['totals'] for x in results]).set_index('strategy_id').reindex([p['id'] for p in POLICIES]).reset_index()
    comparison['delta_vs_A_yuan']=comparison.total_cost_yuan-comparison.total_cost_yuan.iloc[0]
    comparison.to_csv(OUT/'comparison.csv',index=False)
    independent_results=[independent(p['id']) for p in POLICIES]
    for r in independent_results:
        expected=float(comparison.set_index('strategy_id').loc[r['strategy_id'],'total_cost_yuan'])
        assert abs(expected-r['total_cost_yuan'])<1e-4
    save(OUT/'independent_checks.json',independent_results)
    save(AUDIT/'reproduction.json',{name:compare_original(name) for name in ['A_base','B_fixed_q80','B_adaptive']})
    save(OUT/'future_perturbation_checks.json',causal_checks(m,load,pv,price,warm))
    rows=[];hours=[]
    for p in POLICIES:
        x,h=diagnostic(p['id'],OUT);rows.extend(x);hours.append(h)
    pd.DataFrame(rows).to_csv(OUT/'limits.csv',index=False);pd.concat(hours).to_csv(OUT/'hourly.csv',index=False)
    assert all(digest(ROOT/p)==h for p,h in hashes.items()),'Sources changed during run'
    assert all(digest(ROOT/p)==h for p,h in protected_hashes.items()),'Protected files changed'
    record.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-start,
                  source_inputs_unchanged=True,protected_files_unchanged=True,protected_file_count=len(protected_hashes),
                  result_sha256={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='run_manifest.json'})
    save(OUT/'run_manifest.json',record)
    print(comparison[['strategy_id','total_cost_yuan','delta_vs_A_yuan']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
