"""Known-price Q4 baseline. Frozen forecast inputs; new shared January; no training.

Run with conda run -n math_modeling python code/q4_known_price_experiment.py.
Writes numeric evidence only. Report generation is a separate read-only script.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import openpyxl
import scipy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q4_known_price'
BASE = pd.Timestamp('2025-01-01')
DT = 1 / 6
TOL = 1e-6


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def inputs():
    paths = ['code/29_q2_time_mapping_experiment.py', 'code/q3_rolling_baseline_experiment.py',
             'code/q4_known_price_experiment.py', '附件/附件2.xlsx', '附件/附件4.xlsx',
             'results/q2_time_mapping/archive_float.npz',
             'results/q2_time_mapping/naive_load.npy', 'results/q2_time_mapping/naive_pv.npy',
             'results/q3_rolling_baseline/prediction_protection.npz',
             'results/q3_rolling_baseline/B0_dispatch.csv',
             'results/q3_rolling_baseline/B2_dispatch.csv']
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {s: digest(ROOT / s) for s in paths}
    snap = OUT / 'source_snapshot'
    snap.mkdir(exist_ok=True)
    for s in paths[:3]:
        shutil.copy2(ROOT / s, snap / Path(s).name)
    q2 = module(snap / Path(paths[0]).name, 'q4_q2_kernel')
    q3 = module(snap / Path(paths[1]).name, 'q4_q3_kernel')
    q3.OUT = OUT
    wb = openpyxl.load_workbook(ROOT / '附件/附件2.xlsx', read_only=True, data_only=True)
    rows = [list(s.values) for s in wb]
    wb.close()
    wb = openpyxl.load_workbook(ROOT / '附件/附件4.xlsx', read_only=True, data_only=True)
    prices = list(wb.worksheets[0].values)
    wb.close()
    dates = pd.date_range(BASE, periods=365)
    assert all(list(pd.to_datetime([r[0] for r in x[1:]])) == list(dates)
               for x in [*rows, prices])
    assert all([q3.label_text(v) for v in x[0][1:]] ==
               [q3.label_text(v) for v in prices[0][1:]] for x in rows)
    load, pv = (np.asarray([r[1:] for r in x[1:]], dtype=float) for x in rows)
    p = np.asarray([r[1:] for r in prices[1:]], dtype=float)
    assert load.shape == pv.shape == p.shape == (365, 144)
    assert np.isfinite(p).all() and (p > 0).all()
    assert np.isfinite(load).all() and np.isfinite(pv).all()
    nl, nv = q3.natural_arrays(load, pv)
    tl, tv = q3.build_truth(load, pv)
    nat_p = np.full((366, 144), np.nan)
    nat_p[:365, 1:] = p[:, :143]
    nat_p[1:, 0] = p[:, 143]
    archive = np.load(ROOT / paths[5])
    protected2 = (archive['issued_load'] - archive['issued_pv']) * DT + archive['protection']
    q3archive = np.load(ROOT / paths[8])
    protected3 = q3archive['protected'].copy()
    assert np.array_equal(archive['issued_load'], q3archive['issued_load'], equal_nan=True)
    naive = (np.load(ROOT / paths[6]) - np.load(ROOT / paths[7])) * DT
    save('registration.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
         hashes=hashes, environment=dict(python=sys.executable, numpy=np.__version__,
         scipy=scipy.__version__, pandas=pd.__version__, openpyxl=openpyxl.__version__),
         price_information='00:00 knows its complete 144-slot template prices, including next midnight',
         january='Jan1 idle, Jan2 zero plans with feedback; Jan3-31 naive/no protection/fixed S143=6000',
         formal='Q42=Q2 frozen forecasts; Q43=Q3 B2 forecasts and revisions; free terminal; greedy feedback',
         controls='F42/F43 freeze Q2/Q3 purchase plans, replay feedback from new common January state',
         validation='input timestamp mapping, all physical/cash ledgers, one paired single-day price adapter test',
         training_calls=0, expected_main_solves=29+334+334*4))
    return q2, q3, p, nat_p, nl, nv, tl, tv, naive, protected2, protected3, hashes


def row(q3, group, stamp, price, load, pv, q0, q, state, owner, version, battery=True):
    net = (load - pv) * DT
    c, d, e, w, after = q3.feedback_step(q - net, state) if battery else (0., 0., max(net-q, 0.), max(q-net, 0.), state)
    return dict(group=group, interval_start=stamp, interval_end=stamp+pd.Timedelta(minutes=10),
                owner_date=owner, effective_version=version, price_yuan_kWh=float(price),
                actual_load_kW=float(load), actual_pv_kW=float(pv), net_kWh=float(net),
                q0_kWh=float(q0), q_eff_kWh=float(q), charge_kWh=c, discharge_kWh=d,
                emergency_kWh=e, unused_kWh=w, state_start_kWh=state, state_end_kWh=after,
                ordinary_cost_yuan=price*q, adjustment_cost_yuan=.5*price*abs(q-q0),
                emergency_cost_yuan=5*price*e), after


def january(q2, q3, p, nat_p, load, pv, naive):
    state, carry = 6000., 0.
    records, plans, logs = [], [], []
    for k in range(31):
        if k < 2:
            q = np.zeros(144)
        else:
            _, _, _, _, estimate = q3.feedback_step(carry-naive[k, 0], state)
            q, c, d, w, s, log, _ = q2.solve_day(naive[k, 1:], p[k], estimate, 'fixed6000')
            logs.append(dict(day=k, **log))
            assert abs(s[143]-6000) < TOL
        plans.append(q)
        for h in range(144):
            if k == 0 and h == 0:
                continue  # absent observation; idle, unknown cash, never fabricate a record
            amount = carry if h == 0 else q[h-1]
            stamp = BASE + pd.Timedelta(days=k, minutes=10*h)
            r, state = row(q3, 'PUBLIC_JAN', stamp, nat_p[k, h], load[k, h], pv[k, h],
                           amount, amount, state, str(stamp.date()), 'public_January', k > 0)
            records.append(r)
        carry = float(q[-1])
    frame = pd.DataFrame(records)
    q3.physics_residual(frame)
    frame.to_csv(OUT / 'public_january_dispatch.csv', index=False)
    np.savez_compressed(OUT / 'public_january_plans.npz', plans=np.asarray(plans))
    save('public_january_solver.json', logs)
    meta = dict(initial_2025_01_01_kWh=6000., feb1_state_kWh=state, feb1_carry_kWh=carry,
                feb1_carry_original_kWh=carry, solves=len(logs),
                missing_segment='2025-01-01 00:00-00:10: idle state; unknown supply/demand and cash',
                january_cost_excluded_from_formal=True)
    save('public_initialization.json', meta)
    print('Public January:', meta, flush=True)
    return state, carry


def run(q3, name, protected, p, nat_p, tl, tv, initial, first_carry, rolling):
    state, carry, carry0 = initial, first_carry, first_carry
    rows, plans, decisions, solves, nominal = [], [], [], [], []
    carry_owner, carry_version = '2025-01-31', 'public_January'
    net = (tl-tv)*DT
    for k in range(31, 365):
        day = BASE+pd.Timedelta(days=k)
        _, _, _, _, estimated = q3.feedback_step(carry-protected[k, 0, 0], state)
        q0, c, d, w, s, log = q3.solve_interval(protected[k, 0, 1:], p[k], estimated)
        solves.append(dict(group=name, date=str(day.date()), hour=0, **log))
        effective = q0.copy()
        versions = np.full(144, f'{day.date()} 00:00', dtype=object)

        def archive(hour, idx, candidate, old, old_version, accepted, c, d, w, s):
            pub = day+pd.Timedelta(hours=hour)
            for j in range(len(candidate)):
                start = day+pd.Timedelta(minutes=10*(idx+j+1))
                plans.append(dict(group=name, published_at=pub, interval_start=start,
                    q0_kWh=q0[idx+j], previous_kWh=old[j], previous_version=str(old_version[j]),
                    candidate_kWh=candidate[j], accepted=accepted))
                nominal.append(dict(group=name, published_at=pub, interval_start=start,
                    q_kWh=candidate[j], c_kWh=c[j], d_kWh=d[j], w_kWh=w[j],
                    state_start_kWh=s[j], state_end_kWh=s[j+1], price_yuan_kWh=p[k, idx+j],
                    protected_net_kWh=protected[k, hour//6, idx+j+1]))

        archive(0, 0, q0, q0, versions, True, c, d, w, s)
        for h in range(144):
            if rolling and h in (36, 72, 108):
                idx, version = h-1, h//36
                n = protected[k, version, h:]
                old = effective[idx:].copy()
                old_parts = q3.score_components(old, q0[idx:], p[k, idx:], n, state)
                cand, c, d, w, s, log = q3.solve_interval(n, p[k, idx:], state, q0[idx:])
                new_parts = q3.score_components(cand, q0[idx:], p[k, idx:], n, state)
                accepted = bool(sum(new_parts) < sum(old_parts)-q3.SELECT_TOL_YUAN)
                solves.append(dict(group=name, date=str(day.date()), hour=h//6, **log))
                decisions.append(dict(group=name, published_at=day+pd.Timedelta(minutes=10*h),
                    accepted=accepted, state_kWh=state, old_total_yuan=sum(old_parts),
                    new_total_yuan=sum(new_parts), old_ordinary_yuan=old_parts[0],
                    old_adjustment_yuan=old_parts[1], old_emergency_yuan=old_parts[2],
                    new_ordinary_yuan=new_parts[0], new_adjustment_yuan=new_parts[1],
                    new_emergency_yuan=new_parts[2]))
                archive(h//6, idx, cand, old, versions[idx:], accepted, c, d, w, s)
                if accepted:
                    effective[idx:] = cand
                    versions[idx:] = f'{day.date()} {h//6:02d}:00'
            q, original = (carry, carry0) if h == 0 else (effective[h-1], q0[h-1])
            owner, ver = (carry_owner, carry_version) if h == 0 else (str(day.date()), versions[h-1])
            r, state = row(q3, name, day+pd.Timedelta(minutes=10*h), nat_p[k, h], tl[k, h],
                           tv[k, h], original, q, state, owner, ver)
            rows.append(r)
        carry, carry0 = float(effective[-1]), float(q0[-1])
        carry_owner, carry_version = str(day.date()), versions[-1]
        if k % 30 == 0 or k == 364:
            print(name, day.date(), 'solves', len(solves), flush=True)
    r, state = row(q3, name, pd.Timestamp('2026-01-01'), p[-1,-1], tl[-1,-1], tv[-1,-1],
                   carry0, carry, state, carry_owner, carry_version)
    rows.append(r)
    for suffix, data in [('plan_versions', plans), ('nominal', nominal), ('decisions', decisions)]:
        if data:
            pd.DataFrame(data).to_csv(OUT / f'{name}_{suffix}.csv', index=False)
    save(f'{name}_solver.json', solves)
    return pd.DataFrame(rows)


def frozen(q3, group, source, p, tl, tv, initial, carry):
    f = pd.read_csv(ROOT/source, parse_dates=['interval_start', 'interval_end'])
    states, records = initial, []
    for x in f.itertuples():
        stamp = x.interval_start
        k = (stamp.normalize()-BASE).days
        h = stamp.hour*6+stamp.minute//10
        pubk, j = (k-1, 143) if h == 0 else (k, h-1)
        original, q = (carry, carry) if k == 31 and h == 0 else (x.q0_kWh, x.q_eff_kWh)
        # Frozen ordinary commitments are an explicit benchmark, not a fresh forecast policy.
        r, states = row(q3, group, stamp, p[pubk,j], x.actual_load_kW, x.actual_pv_kW,
                        original, q, states, x.owner_date, x.effective_version)
        records.append(r)
    return pd.DataFrame(records)


def main():
    started = time.perf_counter()
    q2,q3,p,pnat,nl,nv,tl,tv,naive,prot2,prot3,hashes = inputs()
    initial, carry = january(q2,q3,p,pnat,nl,nv,naive)
    # Verify daily price selection with intentionally distinguishable day/slot tokens.
    tokens = np.arange(365*144).reshape(365,144)
    mapping = []
    for k,h in [(31,0),(31,1),(171,36),(171,72),(364,143),(365,0)]:
        srcday,col = (k-1,143) if h == 0 else (k,h-1)
        expected = p[srcday,col]
        observed = pnat[k,h]
        assert expected == observed
        mapping.append(dict(natural_day=k, natural_slot=h, source_day=srcday,
                            source_slot=col, token=int(tokens[srcday,col]), price=float(observed)))
    save('price_mapping_checks.json', mapping)
    frames = {}
    summaries, months, days = [], [], []
    for group, source in [('F42','results/q3_rolling_baseline/B0_dispatch.csv'),
                          ('F43','results/q3_rolling_baseline/B2_dispatch.csv')]:
        frames[group] = frozen(q3,group,source,p,tl,tv,initial,carry)
    p2 = np.repeat(prot2[:,None,:], 4, axis=1)
    frames['Q42'] = run(q3,'Q42',p2,p,pnat,tl,tv,initial,carry,False)
    frames['Q43'] = run(q3,'Q43',prot3,p,pnat,tl,tv,initial,carry,True)
    checks = {}
    for name, frame in frames.items():
        summary, monthly, daily, frame = q3.finalize(name,frame,(tl-tv)*DT)
        assert abs(summary['initial_state_kWh']-initial) < TOL
        stamp = frame.interval_start
        k = (stamp.dt.normalize()-BASE).dt.days.to_numpy()
        h = (stamp.dt.hour*6+stamp.dt.minute//10).to_numpy()
        pk, pj = np.where(h == 0,k-1,k), np.where(h == 0,143,h-1)
        assert np.array_equal(frame.price_yuan_kWh.to_numpy(),p[pk,pj])
        bridge = summary['template_total_yuan']-summary['natural_total_yuan']
        assert abs(bridge-(frame.total_cost_yuan.iloc[-1]-frame.total_cost_yuan.iloc[0])) < 1e-5
        checks[name] = dict(physics_kWh=summary['physics_residual_kWh'],
                           settlement_yuan=summary['settlement_residual_yuan'], bridge_yuan=bridge,
                           prices_match=True, common_initial=True)
        summaries.append(summary); months.append(monthly); days.append(daily)
        frame[frame.interval_start.dt.strftime('%Y-%m-%d').isin(q3.SELECTED_DATES)].to_csv(
            OUT/f'{name}_selected_dates.csv',index=False)
        frames[name] = frame
    # Directly compare new saved Q42 plan with Q2 kernel for its actual day initial state.
    k = 171
    daynew = frames['Q42'][frames['Q42'].interval_start.dt.strftime('%Y-%m-%d') == '2025-06-21']
    state0 = float(daynew.state_start_kWh.iloc[0]); carry0 = float(daynew.q_eff_kWh.iloc[0])
    estimated = q3.feedback_step(carry0-prot2[k,0],state0)[-1]
    q,*rest = q2.solve_day(prot2[k,1:],p[k],estimated,'free')
    saved = frames['Q42'].iloc[(k-31)*144+1:(k-31)*144+145].q0_kWh.to_numpy()
    diff = float(abs(p[k] @ (q-saved)))
    assert diff < 1e-4
    checks['single_day_q2_kernel'] = dict(date='2025-06-21', objective_difference_yuan=diff,
         max_plan_difference_kWh=float(np.max(np.abs(q-saved))), additional_solves=1,
         note='Same inputs, independently assembled existing Q2 kernel; objective equality allows tied plans.')
    # Verify run-time nominal energy constraints from saved trajectories as well.
    for name in ['Q42','Q43']:
        n = pd.read_csv(OUT/f'{name}_nominal.csv')
        bal = (n.q_kWh+n.d_kWh-n.c_kWh-n.w_kWh-n.protected_net_kWh).abs().max()
        rec = (n.state_end_kWh-n.state_start_kWh-.9*n.c_kWh+n.d_kWh/.9).abs().max()
        assert max(bal,rec) < TOL
        checks[name]['saved_nominal_max_residual_kWh'] = float(max(bal,rec))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.concat(months).to_csv(OUT/'monthly.csv',index=False)
    pd.concat(days).to_csv(OUT/'daily.csv',index=False)
    assert all(digest(ROOT/s)==v for s,v in hashes.items())
    save('checks.json',checks)
    manifest = dict(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                    seconds=time.perf_counter()-started, main_solves=1699, extra_solves=1,
                    training_calls=0, input_hashes_unchanged=True,
                    outputs={str(f.relative_to(OUT)):digest(f) for f in OUT.rglob('*')
                             if f.is_file() and f.name!='run_manifest.json'})
    save('run_manifest.json',manifest)
    print(pd.DataFrame(summaries)[['group','natural_total_yuan','emergency_cost_yuan','initial_state_kWh']],flush=True)


if __name__ == '__main__':
    main()
