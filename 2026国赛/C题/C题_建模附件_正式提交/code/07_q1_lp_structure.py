"""Verify the exact LP projection under free, unrestricted surplus disposal.

This structural analysis does not overwrite the locked Q1/Q2 strategies.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, csr_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/q1_structure'
CONFIG = dict(eta_c=.9, eta_d=.9, dt_hours=1/6, power_max_kW=5000.,
              state_min_kWh=1200., state_max_kWh=10800., initial_kWh=6000.,
              terminal_kWh=6000., tolerance=1e-6)


def compact_lp(net_kWh, price):
    n = len(net_kWh)
    ec,ed = CONFIG['eta_c'],CONFIG['eta_d']
    cap = CONFIG['power_max_kW']*CONFIG['dt_hours']
    # Decision vector: q[0:n], E[0:n+1]. State increments are affine expressions.
    a = lil_matrix((4*n,2*n+1))
    b = np.r_[-net_kWh,-net_kWh,np.full(n,ec*cap),np.full(n,cap/ed)]
    for t in range(n):
        for block,slope in [(0,1/ec),(1,ed)]:
            a[block*n+t,t] = -1
            a[block*n+t,n+t] = -slope
            a[block*n+t,n+t+1] = slope
        a[2*n+t,n+t],a[2*n+t,n+t+1] = -1,1
        a[3*n+t,n+t],a[3*n+t,n+t+1] = 1,-1
    bounds = [(0,None)]*n+[(CONFIG['state_min_kWh'],CONFIG['state_max_kWh'])]*(n+1)
    bounds[n] = (CONFIG['initial_kWh'],CONFIG['initial_kWh'])
    bounds[-1] = (CONFIG['terminal_kWh'],CONFIG['terminal_kWh'])
    result = linprog(np.r_[price,np.zeros(n+1)],A_ub=csr_matrix(a),b_ub=b,bounds=bounds,method='highs')
    assert result.success,result.message
    q,state = result.x[:n],result.x[n:]
    u = np.diff(state)
    c,d = np.maximum(u,0)/ec,ed*np.maximum(-u,0)
    w = q-net_kWh-c+d
    checks = dict(balance_kWh=float(np.max(np.abs(q+d-net_kWh-c-w))),
                  state_recursion_kWh=float(np.max(np.abs(u-ec*c+d/ed))),
                  nonnegative_kWh=float(max(0,-min(q.min(),c.min(),d.min(),w.min()))),
                  state_bounds_kWh=float(max(0,CONFIG['state_min_kWh']-state.min(),state.max()-CONFIG['state_max_kWh'])),
                  power_bounds_kWh=float(max(0,c.max()-cap,d.max()-cap)),
                  terminal_error_kWh=float(max(abs(state[0]-CONFIG['initial_kWh']),abs(state[-1]-CONFIG['terminal_kWh']))),
                  simultaneous_kWh=float(np.max(np.minimum(c,d))))
    assert max(checks.values())<CONFIG['tolerance'],checks
    return q,c,d,w,state,dict(cost_yuan=float(price@q),grid_kWh=float(q.sum()),
                              variables=2*n+1,inequalities=4*n,checks=checks,solver_message=result.message)


def one_slot(surplus_penalty,integer=False):
    # Synthetic case: 1 kWh PV surplus, zero load, unchanged SOC; flows capped at 10 kWh.
    ec=ed=.9
    cap=10.
    a=np.array([[1.,-1.,1.,-1.],[0.,ec,-1/ed,0.]])
    b=np.array([-1.,0.])
    objective=np.array([1.,0.,0.,surplus_penalty])
    if integer:
        matrix=np.zeros((4,5))
        matrix[:2,:4]=a
        matrix[2,1],matrix[2,4]=1,-cap
        matrix[3,2],matrix[3,4]=1,cap
        result=milp(np.r_[objective,0.],integrality=[0,0,0,0,1],
                    bounds=Bounds(np.zeros(5),[np.inf,cap,cap,np.inf,1.]),
                    constraints=LinearConstraint(matrix,np.r_[b,-np.inf,-np.inf],np.r_[b,0.,cap]))
    else:
        result=linprog(objective,A_eq=a,b_eq=b,bounds=[(0,None),(0,cap),(0,cap),(0,None)],method='highs')
    assert result.success,result.message
    q,c,d,w=result.x[:4]
    return dict(cost=float(result.fun),grid_kWh=float(q),charge_kWh=float(c),discharge_kWh=float(d),unused_kWh=float(w))


def main():
    assert Path(sys.prefix).name=='math_modeling',sys.prefix
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'results/audit/q1_normalized.csv')
    net=(data.load_kW-data.pv_kW).to_numpy()*CONFIG['dt_hours']
    price=data.price_yuan_kWh.to_numpy()
    q,c,d,w,state,summary=compact_lp(net,price)
    locked=json.loads((ROOT/'results/q1_milp/validation_summary.json').read_text(encoding='utf-8'))['summary']
    summary['cost_difference_vs_locked_milp_yuan']=summary['cost_yuan']-locked['cost_yuan']
    assert abs(summary['cost_difference_vs_locked_milp_yuan'])<CONFIG['tolerance']
    data=data.assign(grid_kWh=q,charge_kWh=c,discharge_kWh=d,unused_kWh=w,
                     state_start_kWh=state[:-1],state_end_kWh=state[1:])
    data.to_csv(OUT/'compact_lp_dispatch.csv',index=False,encoding='utf-8-sig')
    examples={f'{name}_{mode}':one_slot(penalty,integer=mode=='milp')
              for name,penalty in [('free_surplus',0.),('penalized_surplus',1.)] for mode in ['lp','milp']}
    assert abs(examples['free_surplus_lp']['cost']-examples['free_surplus_milp']['cost'])<1e-8
    assert abs(examples['penalized_surplus_lp']['cost'])<1e-8
    assert abs(examples['penalized_surplus_milp']['cost']-1.)<1e-8
    # An explicitly constructed optimal LP cycle demonstrates that not every optimum is physical.
    r=.81
    cycle=np.array([0.,1/(1-r),r/(1-r),0.])
    delta=min(cycle[1],cycle[2]/r)
    cleaned=cycle+np.array([0.,-delta,-r*delta,(1-r)*delta])
    assert np.allclose(cleaned,[0,0,0,1],rtol=0,atol=1e-12)
    examples['free_surplus_nonphysical_optimum']=dict(zip(['grid_kWh','charge_kWh','discharge_kWh','unused_kWh'],cycle.tolist()))
    examples['cleaned_same_cost_optimum']=dict(zip(['grid_kWh','charge_kWh','discharge_kWh','unused_kWh'],cleaned.tolist()))
    output=dict(config=CONFIG,environment=sys.executable,scipy=scipy.__version__,compact_lp=summary,
                synthetic_examples=examples,
                note='One-slot examples are mathematical counterexamples, not attachment observations. No locked result is replaced.')
    (OUT/'verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
