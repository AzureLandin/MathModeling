"""Past-only PV support gate and matched full-year scheduling contrasts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_pv_night_gate'
FIG = ROOT / 'figures/q2_pv_night_gate'
OLD = ROOT / 'results/q2_ridge_forecast'
PARAMETERS = dict(history_days=7, activity_threshold_kW=1.0, padding_slots=3,
                  early_fallback='unchanged if fewer than 7 days or no active history',
                  forecast_treatment='postprocess issued PV only; original ridge fits/lambda unchanged',
                  calibration='own issued residuals, W28 q80, negative values retained',
                  terminal_kWh=6000, evaluation_start='2025-02-01')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def module():
    spec = importlib.util.spec_from_file_location('ridge_gate_parent', ROOT/'code/14_q2_ridge_forecast_experiment.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.OUT = OUT
    return m


def support_mask(pv):
    mask = np.ones(pv.shape, dtype=bool)
    for k in range(7, len(pv)):
        active = np.flatnonzero(np.max(pv[k-7:k], axis=0) > 1.0)
        if len(active):
            lo, hi = max(0, int(active[0])-3), min(143, int(active[-1])+3)
            mask[k] = False
            mask[k, lo:hi+1] = True
    return mask


def treatment(forecasts, mask):
    result = dict(forecasts)
    result['issued'] = dict(forecasts['issued'])
    values = forecasts['issued']['pv']
    result['issued']['pv'] = np.where(mask[:len(values)], values, 0.0)
    return result


def independent_check(result, load, pv, price, predictions, group):
    d = result['dispatch']
    fields = ['planned_kWh','charge_kWh','discharge_kWh','emergency_kWh','unused_kWh',
              'state_start_kWh','state_end_kWh']
    q,c,dis,e,w,start,end = [d[f].to_numpy() for f in fields]
    actual = (load-pv)/6
    pred_l = predictions['naive']['load'] if group['load']=='naive' else predictions['issued']['load']
    pred_v = predictions['issued']['pv']
    net = (pred_l-pred_v)/6
    eps = actual-net
    expected_q80 = np.array([np.sort(eps[k-28:k], axis=0)[22] for k in range(31,365)])
    q80_error = float(np.max(abs(expected_q80.ravel()-d.residual_adjustment_kWh)))
    loss = .1*c+(1/.9-1)*dis
    energy_error = float(np.sum(q+e-actual[31:].ravel()-w-loss)-(end[-1]-start[0]))
    state = float(start[0]); errors = []
    for i, net_actual in enumerate(actual[31:].ravel()):
        surplus=q[i]-net_actual
        ch=dd=ee=ww=0.0
        if surplus>=0:
            ch=min(surplus,5000/6,(10800-state)/.9); ww=surplus-ch
        else:
            dd=min(-surplus,5000/6,.9*(state-1200)); ee=-surplus-dd
        state+=.9*ch-dd/.9
        errors.append(max(abs(ch-c[i]),abs(dd-dis[i]),abs(ee-e[i]),abs(ww-w[i]),abs(state-end[i])))
    cash=float(np.sum(np.tile(price,334)*(q+5*e)))
    check=dict(q80_max_kWh=q80_error, energy_identity_kWh=energy_error,
               independent_control_max_kWh=float(max(errors)),
               fee_difference_yuan=cash-result['totals']['total_cost_yuan'])
    assert q80_error<1e-6 and abs(energy_error)<1e-4
    assert max(errors)<1e-6 and abs(check['fee_difference_yuan'])<1e-4
    return check


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    m=module()
    protected = [p for folder in [ROOT/'附件',OLD,ROOT/'results/q2_revision_audit_20260911/source_snapshot']
                 for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and not p.name.startswith('~$')]
    protected += [ROOT/'code/14_q2_ridge_forecast_experiment.py',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in protected}
    signature=hashlib.sha256(json.dumps(dict(parameters=PARAMETERS,hashes=hashes),sort_keys=True).encode()).hexdigest()
    reg=OUT/'registration.json'
    if reg.exists():
        assert json.loads(reg.read_text(encoding='utf-8'))['signature']==signature, 'Existing registration differs'
    else:
        save(reg,dict(registered_utc=datetime.now(timezone.utc).isoformat(),signature=signature,
                      parameters=PARAMETERS,source_hashes=hashes,
                      limitation='same-year follow-up designed after observing ridge results; not blind test'))
    snapshot=OUT/'source_archive'; snapshot.mkdir(exist_ok=True)
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(ROOT/'code/14_q2_ridge_forecast_experiment.py',snapshot/'14_q2_ridge_forecast_experiment.py')
    save(OUT/'run_manifest.json',dict(status='running',signature=signature))
    frozen=m.frozen()
    load,pv,price,_=frozen.bm.read_sources(); dates=frozen.bm.DATES
    print('Rebuilding original causal forecasts and public January ...',flush=True)
    forecasts=m.build_forecasts(load,pv,dates)
    original=np.load(OLD/'candidate_forecasts.npz')
    for target in ['load','pv']:
        np.testing.assert_allclose(forecasts['issued'][target],original[f'issued_{target}'],atol=1e-6,rtol=1e-10,equal_nan=True)
    warm,frame,initial,_=frozen.m08.run_warmup(frozen.m08.Archive(load,pv,len(dates)),price,dates)
    assert abs(initial-8801.462273333342)<1e-6
    frame.to_csv(OUT/'warmup_january.csv',index=False)
    mask=support_mask(pv); gated=treatment(forecasts,mask)
    pd.DataFrame(dict(date=np.repeat(dates.strftime('%Y-%m-%d'),144),slot=np.tile(np.arange(144),365),
        allow_pv=mask.ravel(),actual_pv_kW=pv.ravel(),original_forecast_kW=forecasts['issued']['pv'].ravel(),
        gated_forecast_kW=gated['issued']['pv'].ravel())).to_csv(OUT/'gate_forecasts.csv',index=False)
    groups=[dict(id='C2_pv_original',load='naive',pv='ridge'),dict(id='C3_joint_original',load='ridge',pv='ridge'),
            dict(id='G2_pv_gate',load='naive',pv='ridge'),dict(id='G3_joint_gate',load='ridge',pv='ridge')]
    results=[]; checks={}
    for i,group in enumerate(groups):
        pred=forecasts if i<2 else gated
        print('Running '+group['id'],flush=True)
        result=m.run_group(group,load,pv,price,dates,pred,warm)
        results.append(result)
        checks[group['id']]=independent_check(result,load,pv,price,pred,group)
        if i<2:
            old=pd.read_csv(OLD/('R2_pv_ridge' if i==0 else 'R3_both_ridge')/'dispatch.csv')
            cols=['planned_kWh','charge_kWh','discharge_kWh','emergency_kWh','state_end_kWh','residual_adjustment_kWh']
            error=float(np.max(abs(result['dispatch'][cols].to_numpy()-old[cols].to_numpy())))
            assert error<1e-6
            checks[group['id']]['reference_dispatch_max_kWh']=error
    print('Verifying future perturbations and gate boundaries ...',flush=True)
    perturbations=[]
    for k in [31,171,354]:
        for kind in ['load','pv']:
            ll,vv=load.copy(),pv.copy()
            if kind=='load': ll[k:]*=1.2
            else: vv[k:]*=.7
            rebuilt=m.build_forecasts(ll,vv,dates,upto=k)
            rebuilt=treatment(rebuilt,support_mask(vv))
            error=max(float(np.max(abs(rebuilt['issued'][target][k]-gated['issued'][target][k]))) for target in ['load','pv'])
            a=m.make_group_archive(groups[3],ll,vv,dates,rebuilt)
            clean=results[3]['archive']
            adj=a.adjustment(k,28,.8)[0]; adj_clean=clean.adjustment(k,28,.8)[0]
            q80error=float(np.max(abs(adj-adj_clean)))
            state=float(results[3]['daily'].iloc[k-31].initial_kWh)
            plan=a.plan_day(k,.8,28,price,state)[0]
            old_plan=results[3]['dispatch'].iloc[(k-31)*144:(k-30)*144]
            planerror=float(np.max(abs(plan-old_plan.planned_kWh.to_numpy())))
            assert error<1e-6 and q80error<1e-6 and planerror<1e-6
            perturbations.append(dict(day=k,kind=kind,forecast_max_kW=error,q80_max_kWh=q80error,plan_max_kWh=planerror))
    toy=np.zeros((10,144)); toy[:,40:101]=10
    assert support_mask(toy)[7,37:104].all() and not support_mask(toy)[7,:37].any()
    assert support_mask(toy)[:7].all() and support_mask(np.zeros((10,144))).all()
    future=toy.copy(); future[7:]=100
    assert np.array_equal(support_mask(toy)[7],support_mask(future)[7])
    checks['future_perturbations']=perturbations
    checks['gate_boundaries_passed']=True
    rows=[]; monthly=[]; windows=[]
    for i,result in enumerate(results):
        base=results[i%2]; d=result['dispatch']; bd=base['dispatch']
        t=dict(result['totals']); t['comparison_strategy']=base['group']['id']
        for field in ['total_cost_yuan','planned_cost_yuan','emergency_cost_yuan','planned_kWh','emergency_kWh','unused_kWh','loss_kWh','final_kWh']:
            t['delta_'+field]=t[field]-base['totals'][field]
        v=pv[31:].ravel(); predicted=d.pv_forecast_kW.to_numpy()
        t.update(pv_mae_kW=float(np.mean(abs(v-predicted))),zero_pv_prediction_kW=float(predicted[v==0].mean()),
                 gated_actual_energy_kWh=float(v[~mask[31:].ravel()].sum()/6),
                 gated_actual_max_kW=float(v[~mask[31:].ravel()].max()),
                 gated_actual_gt100_slots=int(np.sum((v>100)&~mask[31:].ravel())))
        diff=result['monthly'].set_index('month').total_cost_yuan-base['monthly'].set_index('month').total_cost_yuan
        t['improved_months']=int((diff < -1e-6).sum())
        rows.append(t)
        for month,delta in diff.items(): monthly.append(dict(strategy_id=t['strategy_id'],month=month,delta_cost_yuan=float(delta)))
        for lo,hi in [(0,6),(6,19),(19,21),(21,24),(19,24)]:
            use=(d.slot>=lo*6)&(d.slot<hi*6)
            windows.append(dict(strategy_id=t['strategy_id'],hours=f'{lo}-{hi}',actual_pv_kW=float(d.loc[use,'pv_kW'].mean()),
                forecast_pv_kW=float(d.loc[use,'pv_forecast_kW'].mean()),
                removed_forecast_kWh=float((bd.pv_forecast_kW-d.pv_forecast_kW)[use].sum()/6),
                delta_protected_kWh=float((d.protected_net_kWh-bd.protected_net_kWh)[use].sum()),
                delta_emergency_cost_yuan=float((d.emergency_cost_yuan-bd.emergency_cost_yuan)[use].sum())))
    comparison=pd.DataFrame(rows); comparison.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_contrasts.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'window_contrasts.csv',index=False)
    # Original cross-predictor differences are observations, not a night-only intervention.
    b=pd.read_csv(OLD/'R0_naive/dispatch.csv'); r=results[0]['dispatch']; use=r.slot>=114
    attribution=dict(extra_pv_forecast_19_24_kWh=float((r.pv_forecast_kW-b.pv_forecast_kW)[use].sum()/6),
        delta_protected_19_24_kWh=float((r.protected_net_kWh-b.protected_net_kWh)[use].sum()),
        delta_emergency_cost_19_24_yuan=float((r.emergency_cost_yuan-b.emergency_cost_yuan)[use].sum()),
        emergency_increase_fraction_of_plan_savings=float((r.emergency_cost_yuan.sum()-b.emergency_cost_yuan.sum())/(b.planned_cost_yuan.sum()-r.planned_cost_yuan.sum())))
    save(OUT/'attribution_check.json',attribution)
    checks['protected_changes']=[rel for rel,h in hashes.items() if digest(ROOT/rel)!=h]
    assert not checks['protected_changes']
    save(OUT/'checks.json',checks)
    plot(comparison,pd.DataFrame(monthly))
    save(OUT/'run_manifest.json',dict(status='complete',signature=signature,finished_utc=datetime.now(timezone.utc).isoformat(),
        verification='reference replay; independently coded execution/q80/accounting; six full-prefix prediction perturbations',
        protected_unchanged=True,adoption='physical forecast correction candidate; judge cash effects from comparison.csv'))
    print(comparison[['strategy_id','total_cost_yuan','delta_total_cost_yuan','emergency_cost_yuan','zero_pv_prediction_kW','gated_actual_energy_kWh','gated_actual_gt100_slots']].to_string(index=False),flush=True)


def plot(comparison,monthly):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    labels=['PV ridge','PV + gate','Joint ridge','Joint + gate']
    order=[0,2,1,3]; data=comparison.iloc[order]
    axes[0].bar(labels,data.zero_pv_prediction_kW,color=['#64748b','#16a085','#64748b','#16a085'])
    axes[0].set(ylabel='Mean forecast (kW)',title='Forecast when observed PV is zero')
    for strategy,color in [('G2_pv_gate','#16a085'),('G3_joint_gate','#c05b43')]:
        part=monthly[monthly.strategy_id==strategy]
        axes[1].plot(part.month.str[5:],part.delta_cost_yuan,marker='o',label=strategy,color=color)
    axes[1].axhline(0,color='#777777',linewidth=.8)
    axes[1].set(xlabel='Month (2025)',ylabel='Cash cost difference (yuan)',title='Gate minus matched original')
    axes[1].legend(); fig.savefig(FIG/'pv_gate_effects.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
