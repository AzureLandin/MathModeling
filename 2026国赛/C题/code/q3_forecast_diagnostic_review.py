"""Read-only metrics review using saved arrays; run in math_modeling."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q3_forecast_diagnostic_review'


def main():
    assert Path(sys.prefix).name == 'math_modeling'
    with np.load(ROOT/'results/q3_forecast_interpolation_diagnostic/forecast_archive.npz') as f:
        pv = f['truth_pv'][31:]
        truth = (f['truth_load'][31:] - pv)/6
        result = {'scope':'Saved array metrics only, no training/interpolation/MILP', 'D1':{},'D3':[]}
        for name in ('Q2','Linear'):
            forecast=f['pv_'+name][31:,0,1:]
            actual=pv[:,1:]
            err=forecast-actual
            result['D1'][name]={'n':int(err.size),'pv_mae_kW':float(abs(err).mean()),
                'pv_bias_kW':float(err.mean()),'false_positive_kWh':float(forecast[actual==0].sum()/6)}
        for v,h in enumerate((1,36,72,108)):
            row={'hour':v*6,'n':int(truth[:,h:].size)}
            for name in ('Linear','PCHIP'):
                pred=f['pv_'+name][31:,v,h:]
                net=f['net_'+name][31:,v,h:]
                protected=net+f['rho_'+name][31:,v,h:]
                row[name]={'pv_mae_kW':float(abs(pred-pv[:,h:]).mean()),
                    'net_mae_kWh':float(abs(net-truth[:,h:]).mean()),
                    'protected_shortfall_kWh':float(np.maximum(truth[:,h:]-protected,0).mean()),
                    'protected_surplus_kWh':float(np.maximum(protected-truth[:,h:],0).mean())}
            result['D3'].append(row)
        hours=(np.arange(1,145)//6)%24
        other=(hours<5)|(hours>=20)
        actual=pv[:,1:][:,other]
        result['other_window_actual']={'max_kW':float(actual.max()),'positive_intervals':int((actual>0).sum())}
        delta=f['pv_Linear'][31:,3,108:]-f['pv_Linear'][31:,2,108:]
        result['18_vs_12_raw_change']={'max_kW':float(abs(delta).max()),
            'mean_abs_kW':float(abs(delta).mean()),'nonzero_intervals':int((abs(delta)>1e-8).sum())}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
