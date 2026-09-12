"""Independent audit of the Q2 ridge-regression residual forecast experiment (code/14).

Read-only over the audited artifacts. Every decisive quantity is rebuilt from the raw
attachments and the per-segment archives:

  * 15 load / 8 PV features and the per-day standardisation rebuilt from the specification
  * ridge coefficients solved by the Cholesky factorisation of the normal equations
    (Z^T Z + N*lambda I) beta = Z^T u_c -- a different linear-algebra path from the
    experiment's SVD -- plus the optimality residual ||Z^T(Z beta - u_c)/N + lambda beta||
  * the five fixed-lambda candidate forecasts, the 14-day forward selection and its scores
  * the per-group q80 correction from that group's own issued net-demand forecast
  * per-segment physics, costs, losses, energy identity and reconciliation for all four groups
  * sampled independent MILP re-solves at the plan-specified dates
  * future-input perturbation and controller-prefix causality from a clean rebuild
  * forecast metrics (annual / monthly / hourly / PV subsets) and q80 coverage

Run:
    E:/Anaconda/envs/math_modeling/python.exe code/15_q2_ridge_forecast_audit.py

Outputs: results/q2_ridge_forecast_audit/
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.linalg

ROOT = Path(__file__).resolve().parents[1]
RIDGE = ROOT / 'results/q2_ridge_forecast'
AUDIT = ROOT / 'results/q2_ridge_forecast_audit'
REF = ROOT / 'results/q2_terminal_sensitivity/T_q80_6000'
SNAP = ROOT / 'results/q2_revision_audit_20260911/source_snapshot'

DT = 1 / 6
ETA = 0.9
CAP = 5000 * DT
STATE_MIN, STATE_MAX = 1200.0, 10800.0
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)
DEFAULT_LAMBDA = 0.1
TRAIN_WINDOW = 56
MIN_TRAIN_DAYS = 14
VALIDATION_WINDOW = 14
MIN_SELECTION_DAYS = 7
TIE_RELATIVE = 1e-10
RESIDUAL_WINDOW = 28
THETA = 0.80
TERMINAL_KWH = 6000.0
WARMUP_DAYS = 31
REFERENCE_TOTAL = 14158360.487139747
TOL_SEG = 1e-6            # kWh, per-segment
TOL_COST = 1e-4           # yuan
TOL_ENERGY = 1e-4         # kWh, cumulative
TOL_FORECAST_ABS = 1e-6   # kW
TOL_FORECAST_REL = 1e-10
DATES = pd.date_range('2025-01-01', '2025-12-31')
EVAL_DATES = [str(d.date()) for d in DATES[WARMUP_DAYS:]]

GROUPS = [('R0_naive', 'naive', 'naive'), ('R1_load_ridge', 'ridge', 'naive'),
          ('R2_pv_ridge', 'naive', 'ridge'), ('R3_both_ridge', 'ridge', 'ridge')]
LOAD_FEATURES = ['l_base', 'l_yesterday_minus_base', 'l_known_week_change',
                 'l_sameweekday_mean_minus_base', 'l_daily_week_change',
                 'weekday_tue', 'weekday_wed', 'weekday_thu', 'weekday_fri',
                 'weekday_sat', 'weekday_sun', 'sin1', 'cos1', 'sin2', 'cos2']
PV_FEATURES = ['v_base', 'v_recent_change', 'v_week_mean_minus_base', 'v_daily_change',
               'sin1', 'cos1', 'sin2', 'cos2']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jload(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name, obj):
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_def),
                              encoding='utf-8')


def _def(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def f(x):
    return float(x)


# ======================================================================================
# independent input reader (does not use code/05)
# ======================================================================================
def read_sources():
    import openpyxl
    wb = openpyxl.load_workbook(ROOT / '附件/附件1.xlsx', read_only=True, data_only=True)
    price = np.array([r[1] for r in list(wb.worksheets[0].values)[1:]], dtype=float)
    wb.close()
    wb = openpyxl.load_workbook(ROOT / '附件/附件2.xlsx', read_only=True, data_only=True)
    tables, axis = [], None
    for ws in wb.worksheets:
        rows = list(ws.values)
        a = pd.DatetimeIndex([r[0] for r in rows[1:]])
        axis = a if axis is None else axis
        assert a.equals(axis)
        tables.append(np.asarray([r[1:] for r in rows[1:]], dtype=float))
    wb.close()
    assert axis.equals(DATES)
    assert price.shape == (144,) and (price > 0).all()
    load, pv = tables[0], tables[1]
    assert load.shape == pv.shape == (365, 144) and np.isfinite(load).all() and np.isfinite(pv).all()
    return load, pv, price


# ======================================================================================
# independent feature / ridge / selection rebuild (from the specification)
# ======================================================================================
def harmonics():
    t = np.arange(144, dtype=float)
    return np.column_stack([np.sin(2 * np.pi * t / 144), np.cos(2 * np.pi * t / 144),
                            np.sin(4 * np.pi * t / 144), np.cos(4 * np.pi * t / 144)])


HARM = harmonics()


def naive_load(load, k):
    return load[k - 7] if k >= 7 else load[k - 1]


def weekday_cols(dates, k):
    wd = dates[k].weekday()          # Monday = 0 is the reference level
    return np.array([1.0 if wd == d else 0.0 for d in range(1, 7)])


def load_feat(load, dates, k):
    out = np.empty((144, 15))
    base = naive_load(load, k)
    out[:, 0] = base
    out[:, 1] = load[k - 1] - base
    out[:, 2] = load[k - 1] - load[k - 8]
    ids = [r for r in (1, 2, 3, 4) if k - 7 * r >= 0]
    out[:, 3] = load[[k - 7 * r for r in ids]].mean(axis=0) - base
    out[:, 4] = load[k - 1].mean() - load[k - 8].mean()
    out[:, 5:11] = weekday_cols(dates, k)
    out[:, 11:15] = HARM
    return out


def pv_feat(pv, dates, k):
    out = np.empty((144, 8))
    out[:, 0] = pv[k - 1]
    out[:, 1] = pv[k - 1] - pv[k - 2]
    out[:, 2] = pv[k - 7:k].mean(axis=0) - pv[k - 1]
    out[:, 3] = pv[k - 1].mean() - pv[k - 2].mean()
    out[:, 4:8] = HARM
    return out


def label(actual, char, k):
    return (actual[k] - naive_load(actual, k)) if char == 'load' else (actual[k] - actual[k - 1])


def standardise(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return mean, scale, (X - mean) / scale


def ridge_cholesky(Z, y, lam):
    """beta solving (Z^T Z + N*lam I) beta = Z^T (y - mean(y)) via Cholesky."""
    N = Z.shape[0]
    yc = y - y.mean()
    gram = Z.T @ Z
    A = gram + N * lam * np.eye(Z.shape[1])
    rhs = Z.T @ yc
    c, low = scipy.linalg.cho_factor(A, lower=True, check_finite=True)
    return scipy.linalg.cho_solve((c, low), rhs, check_finite=True), gram, rhs, N


def build_audit_forecasts(load, pv, dates, upto=None, verbose=False):
    """Independent rebuild of the 5 candidate forecasts and the issued forecast."""
    n = len(dates) if upto is None else min(len(dates), int(upto) + 1)
    out = {}
    for char, threshold, feat_fn, actual in (('load', 8, load_feat, load), ('pv', 7, pv_feat, pv)):
        feat = {k: feat_fn(actual, dates, k) for k in range(threshold, n)}
        naive = np.full((n, 144), np.nan)
        for k in range(1, n):
            naive[k] = naive_load(actual, k) if char == 'load' else actual[k - 1]
        cand = {lam: np.full((n, 144), np.nan) for lam in LAMBDAS}
        issued = np.full((n, 144), np.nan)
        trained = np.zeros(n, bool)
        rec = {}
        for k in range(threshold, n):
            train = [j for j in range(max(0, k - TRAIN_WINDOW), k) if j >= threshold]
            base = naive_load(actual, k) if char == 'load' else actual[k - 1]
            if len(train) < MIN_TRAIN_DAYS:
                for lam in LAMBDAS:
                    cand[lam][k] = base
                rec[k] = dict(trained=False, n_train=len(train), train=train, scores=None,
                              chosen=DEFAULT_LAMBDA, mean=None, scale=None, beta={}, singular=None)
                continue
            X = np.concatenate([feat[j] for j in train], axis=0)
            y = np.concatenate([label(actual, char, j) for j in train], axis=0)
            mean, scale, Z = standardise(X)
            z = (feat[k] - mean) / scale
            lm = float(y.mean())
            singular = np.linalg.svd(Z, compute_uv=False)
            betas, opt_res = {}, {}
            for lam in LAMBDAS:
                beta, gram, rhs, N = ridge_cholesky(Z, y, lam)
                betas[lam] = beta
                # optimality residual: Z^T(Z beta - u_c)/N + lambda beta == 0
                opt_res[lam] = float(np.max(np.abs((Z.T @ (Z @ beta - (y - lm))) / N + lam * beta)))
                cand[lam][k] = np.maximum(0.0, base + lm + z @ beta)
            trained[k] = True
            rec[k] = dict(trained=True, n_train=len(train), train=train, base=base, lm=lm,
                          mean=mean, scale=scale, singular=singular, beta=betas, scores=None,
                          chosen=None, opt_res=opt_res)
        # forward selection
        for k in range(threshold, n):
            window = [j for j in range(max(0, k - VALIDATION_WINDOW), k) if trained[j]]
            if len(window) < MIN_SELECTION_DAYS:
                chosen, scores, tied = DEFAULT_LAMBDA, None, []
            else:
                scores = {lam: float(np.mean([(actual[j] - cand[lam][j]) ** 2 for j in window]))
                          for lam in LAMBDAS}
                best = min(scores.values())
                tied = [lam for lam in LAMBDAS if scores[lam] <= best + TIE_RELATIVE * max(1.0, best)]
                chosen = max(tied)
            issued[k] = cand[chosen][k].copy()
            if k in rec:
                rec[k]['scores'] = scores
                rec[k]['chosen'] = chosen
                rec[k]['tied'] = tied
                rec[k]['window'] = window
                rec[k]['used_default'] = bool(scores is None)
        out[char] = dict(feat=feat, cand=cand, issued=issued, naive=naive, trained=trained, rec=rec)
        if verbose:
            first = int(np.flatnonzero(trained)[0]) if trained.any() else -1
            print(f'  {char}: first trained day index {first}', flush=True)
    return out


# ======================================================================================
# geometry: does one group's issued forecast reproduce the cached archive?
# ======================================================================================
def cached_npz():
    return np.load(RIDGE / 'candidate_forecasts.npz')


def evidence_forecast(load, pv, dates, mine, z):
    """Compare the independent rebuild against the archived mean/scale/singular/beta/candidates."""
    rows, summary = [], {}
    for char in ('load', 'pv'):
        ns = len(LOAD_FEATURES) if char == 'load' else len(PV_FEATURES)
        max_mean = max_scale = max_sing = max_beta = max_beta0 = max_cand = max_issued = 0.0
        start = 8 if char == 'load' else 7
        for k in range(start, len(dates)):
            r = mine[char]['rec'].get(k)
            if r is None or not r['trained']:
                continue
            max_mean = max(max_mean, f(np.max(np.abs(r['mean'] - z[f'mean_{char}'][k]))))
            max_scale = max(max_scale, f(np.max(np.abs(r['scale'] - z[f'scale_{char}'][k]))))
            max_sing = max(max_sing, f(np.max(np.abs(np.sort(r['singular'])[::-1]
                                                     - np.sort(z[f'singular_{char}'][k])[::-1]))))
            for i, lam in enumerate(LAMBDAS):
                max_beta = max(max_beta, f(np.max(np.abs(r['beta'][lam] - z[f'beta_{char}'][k, i]))))
                max_beta0 = max(max_beta0, f(abs(r['lm'] - z[f'beta0_{char}'][k, i])))
                max_cand = max(max_cand, f(np.max(np.abs(
                    mine[char]['cand'][lam][k] - z[f'candidate_{char}_{lam:g}'.replace('.', 'p')][k]))))
            max_issued = max(max_issued, f(np.nanmax(np.abs(
                mine[char]['issued'][k] - z[f'issued_{char}'][k]))))
        summary[char] = dict(max_mean_diff=max_mean, max_scale_diff=max_scale,
                             max_singular_diff=max_sing, max_beta_diff=max_beta,
                             max_beta0_diff=max_beta0, max_candidate_diff_kW=max_cand,
                             max_issued_diff_kW=max_issued,
                             passed=bool(max(max_mean, max_scale, max_sing, max_beta, max_beta0,
                                             max_cand, max_issued) <= TOL_FORECAST_ABS))
        rows.extend(evidence_feature_details(char, mine, z))
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / 'feature_ridge_evidence.csv', index=False, encoding='utf-8-sig')
    return summary, frame


def evidence_feature_details(char, mine, z):
    rows = []
    for k in range(15 if char == 'load' else 8, len(DATES)):
        r = mine[char]['rec'].get(k)
        if r is None or not r['trained']:
            continue
        cand = {lam: f(np.max(np.abs(mine[char]['cand'][lam][k]
                                     - z[f'candidate_{char}_{lam:g}'.replace(".", "p")][k])))
                for lam in LAMBDAS}
        rows.append(dict(target=char, day_index=k, date=str(DATES[k].date()),
                         n_train=len(r['train']),
                         max_mean_diff=f(np.max(np.abs(r['mean'] - z[f'mean_{char}'][k]))),
                         max_scale_diff=f(np.max(np.abs(r['scale'] - z[f'scale_{char}'][k]))),
                         max_beta_diff=f(max(np.max(np.abs(r['beta'][lam] - z[f'beta_{char}'][k, i]))
                                             for i, lam in enumerate(LAMBDAS))),
                         max_candidate_diff_kW=f(max(cand.values())),
                         max_opt_residual=f(max(r['opt_res'].values()))))
    return rows


def evidence_selection(mine, z):
    sel = pd.read_csv(RIDGE / 'selection_log.csv')
    fit = pd.read_csv(RIDGE / 'fit_log.csv')
    rows, max_score_diff, mismatch = [], 0.0, []
    for char in ('load', 'pv'):
        s = sel[sel.target == char].set_index('day_index')
        for k, r in sorted(mine[char]['rec'].items()):
            if k not in s.index:
                continue
            line = s.loc[k]
            mine_choice = r['chosen']
            stored_choice = float(line['chosen_lambda']) if pd.notna(line['chosen_lambda']) else None
            score_diff = 0.0
            if r['scores'] is not None and pd.notna(line.get('score_lambda_0.1', np.nan)):
                for lam in LAMBDAS:
                    score_diff = max(score_diff, f(abs(r['scores'][lam] - float(line[f'score_lambda_{lam:g}']))))
                max_score_diff = max(max_score_diff, score_diff)
            if stored_choice != mine_choice:
                mismatch.append(dict(target=char, day_index=k, date=str(DATES[k].date()),
                                     mine=mine_choice, stored=stored_choice))
            rows.append(dict(target=char, day_index=k, date=str(DATES[k].date()),
                             score_days=int(line['score_days']),
                             mine_score_days=len(r.get('window', [])),
                             chosen_lambda=mine_choice, stored_lambda=stored_choice,
                             max_score_abs_diff=score_diff,
                             used_default=bool(r['used_default'])))
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / 'selection_evidence.csv', index=False, encoding='utf-8-sig')
    return dict(rows=len(frame), max_score_abs_diff=max_score_diff,
                n_lambda_mismatch=len(mismatch), mismatches=mismatch[:20],
                passed=bool(len(mismatch) == 0 and max_score_diff < 1e-6))


# ======================================================================================
# per-group archive: q80 from that group's own issued forecasts
# ======================================================================================
def group_net_forecasts(load, pv, mine, l_kind, v_kind):
    n = load.shape[0]
    pred_l = mine['load']['issued' if l_kind == 'ridge' else 'naive'][:n]
    pred_v = mine['pv']['issued' if v_kind == 'ridge' else 'naive'][:n]
    n_actual = (load - pv) * DT
    n_forecast = (pred_l - pred_v) * DT
    return n_forecast, n_actual - n_forecast


def q80_from_eps(eps, k):
    lo = max(1, k - RESIDUAL_WINDOW)
    m = k - lo
    if m < MIN_SELECTION_DAYS:
        return np.zeros(144), m
    ordered = np.sort(eps[lo:k], axis=0)
    return ordered[int(math.ceil(m * THETA)) - 1].copy(), m


def audit_q80(load, pv, mine):
    """Rebuild per-group q80 for every evaluation day and compare with the archives."""
    out, rows = {}, []
    for sid, l_kind, v_kind in GROUPS:
        n_forecast, eps = group_net_forecasts(load, pv, mine, l_kind, v_kind)
        d = pd.read_csv(RIDGE / sid / 'dispatch.csv', low_memory=False)
        res = pd.read_csv(RIDGE / sid / 'forecast_residuals.csv')
        max_adj = max_prot = max_net = max_eps = 0.0
        days = 0
        for k in range(WARMUP_DAYS, len(DATES)):
            date = str(DATES[k].date())
            block = d[d.date == date]
            r, m = q80_from_eps(eps, k)
            max_adj = max(max_adj, f(np.max(np.abs(r - block.residual_adjustment_kWh.to_numpy()))))
            max_net = max(max_net, f(np.max(np.abs(n_forecast[k] - block.net_forecast_kWh.to_numpy()))))
            max_prot = max(max_prot, f(np.max(np.abs(n_forecast[k] + r
                                                     - block.protected_net_kWh.to_numpy()))))
            days += 1
        rb = res[res.date.isin(EVAL_DATES)]
        idx = np.array([(pd.Timestamp(x) - DATES[0]).days for x in rb.date])
        max_eps = f(np.max(np.abs(rb.residual_kWh.to_numpy()
                                  - eps[idx, rb.slot.to_numpy()])))
        out[sid] = dict(days=days, max_net_forecast_diff_kWh=max_net,
                        max_adjustment_diff_kWh=max_adj, max_protected_diff_kWh=max_prot,
                        stored_eps_check=max_eps,
                        passed=bool(max(max_adj, max_prot, max_net) <= TOL_SEG))
        rows.append(out[sid])
    pd.DataFrame(rows).to_csv(AUDIT / 'q80_evidence.csv', index=False, encoding='utf-8-sig')
    return out


def group_ablation(load, pv, mine):
    """Only the predictor may differ. Ridge components must be shared between the groups that
    use the same predictor; q80 must be rebuilt per group and may therefore differ."""
    sids = [sid for sid, _, _ in GROUPS]
    d = {sid: pd.read_csv(RIDGE / sid / 'dispatch.csv', low_memory=False) for sid in sids}

    def maxdiff(a, b, col):
        return f(np.max(np.abs(d[a][col].to_numpy() - d[b][col].to_numpy())))

    naive_load_same = maxdiff('R0_naive', 'R2_pv_ridge', 'load_forecast_kW')
    naive_pv_same = maxdiff('R0_naive', 'R1_load_ridge', 'pv_forecast_kW')
    ridge_load_same = maxdiff('R1_load_ridge', 'R3_both_ridge', 'load_forecast_kW')
    ridge_pv_same = maxdiff('R2_pv_ridge', 'R3_both_ridge', 'pv_forecast_kW')
    load_treatment_active = maxdiff('R0_naive', 'R1_load_ridge', 'load_forecast_kW')
    pv_treatment_active = maxdiff('R0_naive', 'R2_pv_ridge', 'pv_forecast_kW')
    adj_pairs = {f'{a}__{b}': f(np.max(np.abs(d[a].residual_adjustment_kWh.to_numpy()
                                              - d[b].residual_adjustment_kWh.to_numpy())))
                 for a, b in (('R0_naive', 'R1_load_ridge'), ('R0_naive', 'R2_pv_ridge'),
                              ('R1_load_ridge', 'R3_both_ridge'), ('R2_pv_ridge', 'R3_both_ridge'))}
    out = dict(
        naive_load_identical_R0_R2=naive_load_same,
        naive_pv_identical_R0_R1=naive_pv_same,
        ridge_load_identical_R1_R3=ridge_load_same,
        ridge_pv_identical_R2_R3=ridge_pv_same,
        load_treatment_max_diff=load_treatment_active,
        pv_treatment_max_diff=pv_treatment_active,
        adjustment_max_diff_pairs=adj_pairs,
        q80_shared_between_groups=bool(all(v == 0.0 for v in adj_pairs.values())),
        passed=bool(naive_load_same <= TOL_SEG and naive_pv_same <= TOL_SEG
                    and ridge_load_same <= TOL_SEG and ridge_pv_same <= TOL_SEG
                    and load_treatment_active > 1.0 and pv_treatment_active > 1.0))
    return out


# ======================================================================================
# physics / costs / energy for one group
# ======================================================================================
def audit_group_physics(sid):
    d = pd.read_csv(RIDGE / sid / 'dispatch.csv', low_memory=False)
    nom = pd.read_csv(RIDGE / sid / 'nominal_dispatch.csv', low_memory=False)
    daily = pd.read_csv(RIDGE / sid / 'daily_summary.csv')
    monthly = pd.read_csv(RIDGE / sid / 'monthly_summary.csv')
    events = pd.read_csv(RIDGE / sid / 'emergency_events.csv')
    solvers = pd.read_csv(RIDGE / sid / 'solver_log.csv')
    q = d.planned_kWh.to_numpy(); c = d.charge_kWh.to_numpy(); dd = d.discharge_kWh.to_numpy()
    e = d.emergency_kWh.to_numpy(); w = d.unused_kWh.to_numpy()
    s0 = d.state_start_kWh.to_numpy(); s1 = d.state_end_kWh.to_numpy()
    p = d.price_yuan_kWh.to_numpy(); l_all = d.load_kW.to_numpy(); v_all = d.pv_kW.to_numpy()
    balance = q + v_all * DT + dd + e - l_all * DT - c - w
    state_rec = (s1 - s0) - (ETA * c - dd / ETA)
    continuity = np.abs(s0[1:] - s1[:-1])
    bounds = float(max(0.0, STATE_MIN - min(s0.min(), s1.min()),
                       max(s0.max(), s1.max()) - STATE_MAX))
    power = float(max(0.0, c.max() - CAP, dd.max() - CAP))
    nonneg = float(max(0.0, -min(q.min(), c.min(), dd.min(), e.min(), w.min())))
    mutex = float(np.minimum(c, dd).max())
    loss = 0.1 * c + (1 / ETA - 1) * dd
    planned = f((p * q).sum()); emergency = f((5 * p * e).sum())
    # nominal side
    nc = nom.nominal_charge_kWh.to_numpy(); nd = nom.nominal_discharge_kWh.to_numpy()
    nw = nom.nominal_unused_kWh.to_numpy(); nb = nom.nominal_binary_mode.to_numpy()
    n0 = nom.nominal_state_start_kWh.to_numpy(); n1 = nom.nominal_state_end_kWh.to_numpy()
    nrec = (n1 - n0) - (ETA * nc - nd / ETA)
    nterminal = np.abs(nom.groupby('date', sort=False).nominal_state_end_kWh.last().to_numpy()
                       - TERMINAL_KWH)
    nominal = dict(
        recursion=f(np.max(np.abs(nrec))),
        binary=f(np.max(np.abs(nb - np.rint(nb)))),
        binary_gate=f(max(0.0, np.max(nc - CAP * nb), np.max(nd - CAP * (1 - nb)))),
        mutex=f(np.minimum(nc, nd).max()),
        balance=f(np.max(np.abs(nc + nw - nd - (q - d.protected_net_kWh.to_numpy())))),
        terminal=f(np.max(nterminal)),
        nonneg=f(max(0.0, -min(nc.min(), nd.min(), nw.min()))),
        power=f(max(0.0, nc.max() - CAP, nd.max() - CAP)),
        planned_cost_vs_dispatch=f(abs(f((p * q).sum()) - f(nom.nominal_planned_cost_yuan.sum()))))
    identity = f(q.sum() + e.sum() + v_all.sum() * DT - l_all.sum() * DT - w.sum() - loss.sum()
                 - (s1[-1] - s0[0]))
    morning = d.slot.to_numpy() < 60
    out = dict(
        strategy_id=sid, segments=int(len(d)), days=int(d.date.nunique()),
        initial_kWh=f(s0[0]), final_kWh=f(s1[-1]),
        planned_cost_yuan=planned, emergency_cost_yuan=emergency, total_cost_yuan=planned + emergency,
        planned_kWh=f(q.sum()), emergency_kWh=f(e.sum()), unused_kWh=f(w.sum()),
        charge_kWh=f(c.sum()), discharge_kWh=f(dd.sum()), loss_kWh=f(loss.sum()),
        morning_0_10h_emergency_cost_yuan=f((5 * p * e)[morning].sum()),
        emergency_slots=int((e > 1e-6).sum()),
        emergency_days=int(pd.Series(e > 1e-6).groupby(d.date.to_numpy()).any().sum()),
        emergency_events=int(len(events)),
        full_end_days=int((s1.reshape(-1, 144)[:, -1] >= STATE_MAX - 1e-6).sum()),
        physical_max_kWh=f(max(np.max(np.abs(balance)), np.max(np.abs(state_rec)), continuity.max(),
                               bounds, power, nonneg, mutex)),
        nominal=nominal, energy_identity_kWh=identity,
        cost_reconciliation_yuan=f(abs(planned + emergency - daily.total_cost_yuan.sum())),
        monthly_reconciliation_yuan=f(abs(planned + emergency - monthly.total_cost_yuan.sum())),
        event_energy_reconciliation_kWh=f(abs(events.emergency_kWh.sum() - e.sum())),
        event_cost_reconciliation_yuan=f(abs(events.emergency_cost_yuan.sum() - emergency)),
        solver_rows=int(len(solvers)), solver_fallbacks=int(solvers.fallback.sum()),
        max_mip_gap=f(solvers.mip_gap.max()),
        nominal_terminal_error_reported=f(daily.nominal_terminal_error_kWh.max()),
        daily_fallback_days=int((daily.fallback_reason.fillna('') != '').sum()),
    )
    out['physical_passed'] = bool(out['physical_max_kWh'] <= TOL_SEG)
    out['nominal_passed'] = bool(max(v for k, v in nominal.items()
                                     if k != 'planned_cost_vs_dispatch') <= TOL_SEG)
    out['accounting_passed'] = bool(max(out['cost_reconciliation_yuan'],
                                        out['monthly_reconciliation_yuan'],
                                        out['event_energy_reconciliation_kWh'],
                                        out['event_cost_reconciliation_yuan'],
                                        abs(identity)) <= TOL_ENERGY)
    out['solver_passed'] = bool(out['solver_fallbacks'] == 0 and out['max_mip_gap'] <= 1e-9)
    return out


def energy_contrasts_check():
    c = pd.read_csv(RIDGE / 'energy_contrasts.csv').set_index('strategy_id')
    comp = pd.read_csv(RIDGE / 'comparison.csv').set_index('strategy_id')
    base = comp.loc['R0_naive']
    rows, worst = [], 0.0
    for sid in comp.index:
        t = comp.loc[sid]
        dq = f(t.planned_kWh - base.planned_kWh)
        de = f(t.emergency_kWh - base.emergency_kWh)
        dw = f(t.unused_kWh - base.unused_kWh)
        dl = f(t.loss_kWh - base.loss_kWh)
        dend = f(t.final_kWh - base.final_kWh)
        rhs = -de + dw + dl + dend
        rows.append(dict(strategy_id=sid, recomputed_delta_planned=dq, recomputed_lhs=rhs,
                         residual=f(rhs - dq), file_delta_planned=f(c.loc[sid].delta_planned_kWh),
                         file_residual=f(c.loc[sid].identity_residual_kWh)))
        worst = max(worst, abs(rhs - dq), abs(dq - c.loc[sid].delta_planned_kWh))
    pd.DataFrame(rows).to_csv(AUDIT / 'energy_contrasts_check.csv', index=False,
                              encoding='utf-8-sig')
    return dict(max_recomputed_residual_kWh=worst, passed=bool(worst <= TOL_ENERGY))


# ======================================================================================
# sampled MILP re-solve
# ======================================================================================
def load_kernel():
    spec = importlib.util.spec_from_file_location('audit_core', SNAP / 'code/02_q1_baseline.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def milp_resample(price, core):
    sample_spec = ['2025-02-01', '2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
    rows = []
    for sid, _, _ in GROUPS:
        d = pd.read_csv(RIDGE / sid / 'dispatch.csv', low_memory=False)
        daily = pd.read_csv(RIDGE / sid / 'daily_summary.csv')
        dates = list(dict.fromkeys(d.date))
        worst_e = daily.loc[daily.emergency_cost_yuan.idxmax(), 'date']
        worst_p = daily.loc[daily.planned_cost_yuan.idxmax(), 'date']
        wanted = list(dict.fromkeys(sample_spec + [worst_e, worst_p]))
        for date in wanted:
            block = d[d.date == date]
            initial = float(block.state_start_kWh.iloc[0])
            protected = block.protected_net_kWh.to_numpy()
            (qq, cc, ddx, ww, st), summary = core.solve(protected / DT, np.zeros(144), price,
                                                        integer=True, initial_kWh=initial,
                                                        terminal_kWh=TERMINAL_KWH)
            recorded = f(block.planned_cost_yuan.sum())
            rows.append(dict(strategy_id=sid, date=date, day_index=dates.index(date),
                             recorded_cost_yuan=recorded, resolved_cost_yuan=f(price @ qq),
                             objective_difference_yuan=f(abs(price @ qq - recorded)),
                             mip_gap=f(summary['mip_gap']),
                             plan_max_abs_diff_kWh=f(np.max(np.abs(qq - block.planned_kWh.to_numpy())))))
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / 'milp_resample.csv', index=False, encoding='utf-8-sig')
    return dict(n_resolved=int(len(frame)), max_objective_difference_yuan=f(frame.objective_difference_yuan.max()),
                max_mip_gap=f(frame.mip_gap.max()),
                note='sampled re-solve only; not a full-period re-solve',
                passed=bool(frame.objective_difference_yuan.max() <= TOL_COST and frame.mip_gap.max() <= 1e-9))


# ======================================================================================
# forecast metrics and q80 coverage
# ======================================================================================
def metrics_check():
    annual = pd.read_csv(RIDGE / 'forecast_metrics.csv').set_index(['strategy_id', 'scope'])
    subsets = pd.read_csv(RIDGE / 'pv_subsets.csv').set_index(['strategy_id', 'subset'])
    monthly = pd.read_csv(RIDGE / 'forecast_metrics_monthly.csv')
    hourly = pd.read_csv(RIDGE / 'forecast_metrics_hourly.csv')
    rows, worst = [], 0.0
    for sid, _, _ in GROUPS:
        d = pd.read_csv(RIDGE / sid / 'dispatch.csv', low_memory=False)
        lerr = d.load_kW.to_numpy() - d.load_forecast_kW.to_numpy()
        verr = d.pv_kW.to_numpy() - d.pv_forecast_kW.to_numpy()
        nerr = (d.load_kW.to_numpy() - d.pv_kW.to_numpy()) - (d.load_forecast_kW.to_numpy()
                                                              - d.pv_forecast_kW.to_numpy())
        for name, err in (('load', lerr), ('pv', verr), ('net_demand', nerr)):
            a = annual.loc[(sid, name)]
            got = (f(np.abs(err).mean()), f(np.sqrt((err ** 2).mean())), f(err.mean()))
            exp = (f(a.mae), f(a.rmse), f(a.mean_signed))
            worst = max(worst, max(abs(x - y) for x, y in zip(got, exp)))
            rows.append(dict(strategy_id=sid, scope=name, mae_recomputed=got[0], mae_file=exp[0],
                             rmse_recomputed=got[1], rmse_file=exp[1],
                             bias_recomputed=got[2], bias_file=exp[2]))
        actual = d.pv_kW.to_numpy(); fc = d.pv_forecast_kW.to_numpy()
        for lab, mask in (('all', np.ones(len(actual), bool)), ('actual_gt_100kW', actual > 100),
                          ('actual_le_100kW', actual <= 100), ('actual_zero', actual == 0)):
            s = subsets.loc[(sid, lab)]
            rows.append(dict(strategy_id=sid, scope=f'pv_{lab}', mae_recomputed=f(np.abs(verr[mask]).mean()),
                             mae_file=f(s.pv_mae_kW), rmse_recomputed=f(np.sqrt((verr[mask] ** 2).mean())),
                             rmse_file=f(s.pv_rmse_kW), bias_recomputed=f(verr[mask].mean()),
                             bias_file=f(s.pv_bias_kW),
                             mean_positive_recomputed=f(fc[mask].clip(min=0).mean()),
                             mean_positive_file=f(s.mean_positive_forecast_kW),
                             samples_recomputed=int(mask.sum()), samples_file=int(s.samples)))
            worst = max(worst, abs(f(np.abs(verr[mask]).mean()) - f(s.pv_mae_kW)),
                        abs(f(fc[mask].clip(min=0).mean()) - f(s.mean_positive_forecast_kW)))
    pd.DataFrame(rows).to_csv(AUDIT / 'forecast_metrics_check.csv', index=False, encoding='utf-8-sig')
    return dict(max_abs_difference=worst, n_rows=len(rows), passed=bool(worst <= TOL_SEG),
                monthly_rows=int(len(monthly)), hourly_rows=int(len(hourly))), rows


def coverage_check():
    cov = pd.read_csv(RIDGE / 'q80_coverage.csv')
    annual = cov[cov.scope == 'annual'].set_index('strategy_id')
    rows, worst = [], 0.0
    for sid, _, _ in GROUPS:
        d = pd.read_csv(RIDGE / sid / 'dispatch.csv', low_memory=False)
        actual = (d.load_kW.to_numpy() - d.pv_kW.to_numpy()) / 6
        protected = d.protected_net_kWh.to_numpy()
        covered = (actual <= protected + 1e-9).mean()
        adj = d.residual_adjustment_kWh.to_numpy()
        got = (f(covered), f(adj.mean()), f((adj > 0).mean()))
        exp = (f(annual.loc[sid].coverage), f(annual.loc[sid].mean_adjustment_kWh),
               f(annual.loc[sid].mean_adjustment_positive_share))
        worst = max(worst, max(abs(x - y) for x, y in zip(got, exp)))
        rows.append(dict(strategy_id=sid, coverage_recomputed=got[0], coverage_file=exp[0],
                         mean_adj_recomputed=got[1], mean_adj_file=exp[1],
                         positive_share_recomputed=got[2], positive_share_file=exp[2]))
    pd.DataFrame(rows).to_csv(AUDIT / 'q80_coverage_check.csv', index=False, encoding='utf-8-sig')
    return dict(max_abs_difference=worst, passed=bool(worst <= TOL_SEG))


# ======================================================================================
# causality: perturbation and controller prefix, from the audit's own rebuild
# ======================================================================================
def perturbation_audit(load, pv, dates, mine, price, core):
    out, rows = {}, []
    for k, label in ((31, '2025-02-01'), (171, '2025-06-21'), (354, '2025-12-21')):
        for kind in ('load_x1.2', 'pv_x0.7'):
            pl, pv2 = load.copy(), pv.copy()
            if kind == 'load_x1.2':
                pl[k:] = load[k:] * 1.2
            else:
                pv2[k:] = pv[k:] * 0.7
            rebuilt = build_audit_forecasts(pl, pv2, dates, upto=k)
            row = dict(case=f'{label}:{kind}', day_index=k)
            worst = 0.0
            for char in ('load', 'pv'):
                worst = max(worst, f(np.nanmax(np.abs(mine[char]['issued'][k] - rebuilt[char]['issued'][k]))))
                for lam in LAMBDAS:
                    worst = max(worst, f(np.nanmax(np.abs(mine[char]['cand'][lam][k]
                                                          - rebuilt[char]['cand'][lam][k]))))
                mrec, rrec = mine[char]['rec'][k], rebuilt[char]['rec'][k]
                row[f'{char}_chosen_unchanged'] = bool(mrec['chosen'] == rrec['chosen'])
                row[f'{char}_n_train_unchanged'] = bool(mrec['n_train'] == rrec['n_train'])
                if mrec['scores'] is not None and rrec['scores'] is not None:
                    worst = max(worst, max(abs(mrec['scores'][lam] - rrec['scores'][lam])
                                           for lam in LAMBDAS))
                if mrec['trained']:
                    worst = max(worst, f(np.max(np.abs(mrec['mean'] - rrec['mean']))),
                                f(np.max(np.abs(mrec['scale'] - rrec['scale']))))
            row['max_difference'] = worst
            # q80 for each group at day k must also be unchanged
            for sid, l_kind, v_kind in GROUPS:
                nf_c, eps_c = group_net_forecasts(load[:k + 1], pv[:k + 1], mine, l_kind, v_kind)
                nf_p, eps_p = group_net_forecasts(pl[:k + 1], pv2[:k + 1], rebuilt, l_kind, v_kind)
                rc, _ = q80_from_eps(eps_c, k)
                rp, _ = q80_from_eps(eps_p, k)
                row[f'{sid}_q80_diff_kWh'] = f(np.max(np.abs(rc - rp)))
                row[f'{sid}_protected_diff_kWh'] = f(np.max(np.abs((nf_c[k] + rc) - (nf_p[k] + rp))))
                worst = max(worst, row[f'{sid}_protected_diff_kWh'])
            row['max_difference'] = worst
            row['passed'] = bool(worst < 1e-9 and row['load_chosen_unchanged']
                                 and row['pv_chosen_unchanged'])
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / 'future_perturbation_audit.csv', index=False, encoding='utf-8-sig')
    return dict(cases=rows, all_passed=bool(frame.passed.all()))


def feedback_prefix_audit(load, pv, core):
    sid = 'R0_naive'
    d = pd.read_csv(RIDGE / sid / 'dispatch.csv', low_memory=False)
    date = '2025-06-21'
    block = d[d.date == date]
    k = list(dict.fromkeys(d.date)).index(date) + WARMUP_DAYS
    initial = float(block.state_start_kWh.iloc[0])
    plan = block.planned_kWh.to_numpy()
    ml, mp = load[k].copy(), pv[k].copy()
    ml[72:] = ml[72:] * 1.5 + 500.0
    mp[72:] = mp[72:] * 0.5
    c1, d1, e1, w1, s1 = greedy(plan, load[k], pv[k], initial)
    c2, d2, e2, w2, s2 = greedy(plan, ml, mp, initial)
    rows = []
    for name, a, b in (('charge', c1, c2), ('discharge', d1, d2), ('emergency', e1, e2),
                       ('unused', w1, w2)):
        rows.append(dict(quantity=name, prefix_max_diff=f(np.max(np.abs(a[:72] - b[:72]))),
                         full_day_max_diff=f(np.max(np.abs(a - b)))))
    rows.append(dict(quantity='state_prefix', prefix_max_diff=f(np.max(np.abs(s1[:73] - s2[:73]))),
                     full_day_max_diff=f(np.max(np.abs(s1 - s2)))))
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / 'feedback_prefix_audit.csv', index=False, encoding='utf-8-sig')
    return dict(date=date, rows=rows,
                prefix_unchanged=bool(frame.prefix_max_diff.max() < 1e-12),
                later_does_change=bool(frame.full_day_max_diff.max() > 1e-9),
                passed=bool(frame.prefix_max_diff.max() < 1e-12
                            and frame.full_day_max_diff.max() > 1e-9))


def greedy(plan, load_day, pv_day, initial, battery=True):
    n = len(plan)
    c = np.zeros(n); dd = np.zeros(n); e = np.zeros(n); w = np.zeros(n)
    st = np.r_[float(initial), np.zeros(n)]
    cap = CAP if battery else 0.0
    for t in range(n):
        surplus = plan[t] + (pv_day[t] - load_day[t]) * DT
        if surplus >= 0:
            c[t] = min(surplus, cap, max(0.0, (STATE_MAX - st[t]) / ETA))
            w[t] = surplus - c[t]
        else:
            dd[t] = min(-surplus, cap, max(0.0, (st[t] - STATE_MIN) * ETA))
            e[t] = -surplus - dd[t]
        st[t + 1] = st[t] + ETA * c[t] - dd[t] / ETA
    return c, dd, e, w, st


# ======================================================================================
# provenance / protected assets / January
# ======================================================================================
def selection_distribution():
    """The report quotes a lambda distribution 'over the 334 evaluation days'; check both scopes."""
    s = pd.read_csv(RIDGE / 'selection_log.csv')
    out = {}
    for char, threshold in (('load', 8), ('pv', 7)):
        b = s[(s.target == char) & (s.day_index >= threshold)]
        ev = b[b.day_index >= 31]
        out[char] = dict(
            all_days_n=int(len(b)),
            all_days={f'{k:g}': int(v) for k, v in b.chosen_lambda.value_counts().items()},
            evaluation_days_n=int(len(ev)),
            evaluation_days={f'{k:g}': int(v) for k, v in ev.chosen_lambda.value_counts().items()},
            evaluation_default_days=int((ev.score_days < 7).sum()),
            lambda_10_never_chosen=bool((b.chosen_lambda != 10.0).all()))
    out['report_claims'] = {
        'label': '评价期334天', 'load': {'0.1': 179, '1': 75, '0.001': 75, '0.01': 28},
        'pv': {'0.001': 139, '0.1': 87, '0.01': 72, '1': 60}}
    out['report_load_claim_matches_evaluation_scope'] = bool(
        out['load']['evaluation_days'] == {'0.1': 179, '1': 75, '0.001': 75, '0.01': 28})
    out['report_pv_claim_matches_evaluation_scope'] = bool(
        out['pv']['evaluation_days'] == {'0.001': 139, '0.1': 87, '0.01': 72, '1': 60})
    out['report_load_claim_matches_all_days'] = bool(
        out['load']['all_days'] == {'0.1': 179, '1': 75, '0.001': 75, '0.01': 28})
    out['report_pv_claim_matches_all_days'] = bool(
        out['pv']['all_days'] == {'0.001': 139, '0.1': 87, '0.01': 72, '1': 60})
    return out


def monthly_check():
    mc = pd.read_csv(RIDGE / 'monthly_contrasts.csv')
    daily = {sid: pd.read_csv(RIDGE / sid / 'daily_summary.csv') for sid, _, _ in GROUPS}
    base = daily['R0_naive'].assign(month=lambda x: x.date.str[:7]).groupby('month').total_cost_yuan.sum()
    rows = []
    worst = 0.0
    for sid, _, _ in GROUPS:
        m = daily[sid].assign(month=lambda x: x.date.str[:7]).groupby('month').total_cost_yuan.sum()
        for month in m.index:
            delta = f(m[month] - base[month])
            stored = mc[(mc.strategy_id == sid) & (mc.month == month)]
            stored_delta = f(stored.delta_total_cost_yuan.iloc[0]) if len(stored) else None
            worst = max(worst, abs(delta - stored_delta)) if stored_delta is not None else worst
            rows.append(dict(strategy_id=sid, month=month, total_cost_recomputed=f(m[month]),
                             delta_recomputed=delta, delta_file=stored_delta))
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDIT / 'monthly_check.csv', index=False, encoding='utf-8-sig')
    summary = {}
    for sid, _, _ in GROUPS:
        b = mc[mc.strategy_id == sid]
        summary[sid] = dict(months=int(len(b)), improved=int((b.delta_total_cost_yuan < 0).sum()),
                            worse=int((b.delta_total_cost_yuan > 0).sum()),
                            worst_month=str(b.loc[b.delta_total_cost_yuan.idxmax(), 'month']),
                            worst_delta_yuan=f(b.delta_total_cost_yuan.max()))
    return dict(max_abs_difference_yuan=worst, rows=len(frame), per_group=summary,
                passed=bool(worst <= TOL_COST))


def naive_diag_check(load, pv):
    eps = np.full((365, 144), np.nan)
    for k in range(1, 365):
        eps[k] = (load[k] - pv[k]) * DT - my_naive_net(load, pv, k)
    e = eps[WARMUP_DAYS:]
    db = np.nanmean(e, axis=1)
    adj = np.array([np.corrcoef(e[i], e[i + 1])[0, 1] for i in range(len(e) - 1)])
    file = pd.read_csv(RIDGE / 'naive_residual_summary.csv').iloc[0]
    got = dict(mean_bias_kWh=f(np.mean(db)), mean_absolute_day_bias_kWh=f(np.mean(np.abs(db))),
               worst_positive_day_bias_kWh=f(np.max(db)), worst_negative_day_bias_kWh=f(np.min(db)),
               adjacent_day_same_slot_mean_correlation=f(np.mean(adj[np.isfinite(adj)])))
    exp = {k: f(file[k]) for k in got}
    worst = max(abs(got[k] - exp[k]) for k in got)
    return dict(recomputed=got, file=exp, max_abs_difference=worst,
                passed=bool(worst <= 1e-6))


def provenance(load, pv, price):
    reg = jload(RIDGE / 'registration.json')
    man = jload(RIDGE / 'run_manifest.json')
    out = {'registered_utc': reg['registered_utc'],
           'original_registered_utc': reg.get('original_registered_utc'),
           'n_amendments': len(reg.get('amendments', [])),
           'finished_utc': man['finished_utc'], 'wall_seconds': man['wall_seconds'],
           'code_sha256_registered': reg['code_sha256'],
           'code_sha256_live': digest(ROOT / 'code/14_q2_ridge_forecast_experiment.py'),
           'spec_sha256_registered': reg['specification_sha256'],
           'spec_sha256_live': digest(ROOT / 'reports/问题二_岭回归残差预测实验方案.md'),
           'input_hash_match': {rel: bool(digest(ROOT / rel) == h)
                                for rel, h in reg['input_sha256'].items()},
           'snapshot_hash_match': {rel: bool(digest(SNAP / rel) == h)
                                   for rel, h in reg['snapshot_sha256'].items()},
           'protected_file_count_registered': reg['protected_file_count'],
           'protected_unchanged_selfreported': man['protected_unchanged'],
           'registry_reproduce_command': reg['reproduce_command']}
    out['all_input_hashes_match'] = bool(all(out['input_hash_match'].values()))
    out['all_snapshot_hashes_match'] = bool(all(out['snapshot_hash_match'].values()))
    out['code_matches_registration'] = bool(out['code_sha256_live'] == out['code_sha256_registered'])
    out['spec_matches_registration'] = bool(out['spec_sha256_live'] == out['spec_sha256_registered'])
    # recompute the registration signature exactly as code/14 does
    hashes = {rel: digest(ROOT / rel) for rel in reg['input_sha256']}
    snap = {rel: digest(SNAP / rel) for rel in reg['snapshot_sha256']}
    sig = hashlib.sha256(json.dumps(dict(parameters=reg['parameters'], input_sha256=hashes,
                                         snapshot_sha256=snap,
                                         code_sha256=out['code_sha256_live']),
                                    sort_keys=True).encode()).hexdigest()
    out['recomputed_signature'] = sig
    out['signature_reproduced'] = bool(sig == reg['signature'])
    # protected assets: re-hash the before-list
    before = jload(RIDGE / 'protected_before.json')
    changed, missing = [], []
    for rel, h in before.items():
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif digest(p) != h:
            changed.append(rel)
    out['protected_rehash'] = dict(n=len(before), n_changed=len(changed), n_missing=len(missing),
                                   changed=sorted(changed)[:20], missing=sorted(missing)[:20])
    # artifact_hashes vs live (excluding the self-hash excluded files)
    ah = jload(RIDGE / 'artifact_hashes.json')
    bad = [rel for rel, h in ah.items() if not (RIDGE / rel).exists() or digest(RIDGE / rel) != h]
    out['artifact_hashes_verified'] = dict(n=len(ah), n_mismatch=len(bad), mismatch=sorted(bad)[:20])
    # January: independent replay of the public warm-up
    state = 6000.0
    rows = []
    for k in range(31):
        initial = state
        if k == 0:
            plan = np.zeros(144)
        else:
            netf = my_naive_net(load, pv, k)
            (plan, cc, ddx, ww, st), _ = load_kernel_solve(netf, price, initial)
        c, dd, e, w, st = greedy(plan, load[k], pv[k], initial, battery=(k > 0))
        rows.append(dict(day_index=k, date=str(DATES[k].date()), initial=initial, final=f(st[-1]),
                         planned_cost=f((plan * price).sum()),
                         emergency_cost=f(5 * (e * price).sum())))
        state = f(st[-1])
    jan = pd.DataFrame(rows)
    warm = pd.read_csv(RIDGE / 'warmup_january.csv')
    out['january'] = dict(feb1_state_recomputed=f(state), feb1_state_recorded=f(warm.final_kWh.iloc[-1]),
                          feb1_abs_diff=f(abs(state - 8801.462273333342)),
                          matches_8801=bool(abs(state - 8801.462273333342) < 1e-9),
                          max_diff_vs_recorded=f(np.max(np.abs(jan.final.to_numpy()
                                                               - warm.final_kWh.to_numpy()))))
    return out


_KERNEL = {}


def load_kernel_solve(netf, price, initial):
    if 'core' not in _KERNEL:
        _KERNEL['core'] = load_kernel()
    core = _KERNEL['core']
    return core.solve(netf / DT, np.zeros(144), price, integer=True, initial_kWh=initial,
                      terminal_kWh=initial)


def my_naive_net(load, pv, k):
    return (naive_load(load, k) - pv[k - 1]) * DT


# ======================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-milp', action='store_true')
    args = parser.parse_args()
    assert Path(sys.prefix).name == 'math_modeling', sys.prefix
    AUDIT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    audit_inputs = ['code/14_q2_ridge_forecast_experiment.py',
                    'reports/问题二_岭回归残差预测实验方案.md',
                    'results/q2_ridge_forecast/registration.json',
                    'results/q2_ridge_forecast/run_manifest.json',
                    'results/q2_ridge_forecast/candidate_forecasts.npz',
                    'results/q2_ridge_forecast/issued_forecasts.csv',
                    'results/q2_ridge_forecast/selection_log.csv',
                    'results/q2_ridge_forecast/comparison.csv']
    start_hashes = {rel: digest(ROOT / rel) for rel in audit_inputs}
    save('audit_inputs_manifest.json', dict(
        audit_script_sha256=digest(Path(__file__)), start_hashes=start_hashes,
        executable=sys.executable, python=sys.version, numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__))

    load, pv, price = read_sources()
    print('rebuilding features / ridge / selection independently ...', flush=True)
    mine = build_audit_forecasts(load, pv, DATES, verbose=True)
    z = cached_npz()
    fr_summary, fr_frame = evidence_forecast(load, pv, DATES, mine, z)
    save('feature_ridge_summary.json', fr_summary)
    print('  forecast evidence:', json.dumps(fr_summary, ensure_ascii=False), flush=True)
    sel = evidence_selection(mine, z)
    save('selection_summary.json', sel)
    print('  selection:', json.dumps({k: v for k, v in sel.items() if k != 'mismatches'},
                                     ensure_ascii=False), flush=True)

    print('rebuilding q80 per group ...', flush=True)
    q80 = audit_q80(load, pv, mine)
    save('q80_summary.json', q80)
    ablation = group_ablation(load, pv, mine)
    save('group_ablation.json', ablation)

    print('auditing physics / costs / energy for the four groups ...', flush=True)
    physics = {sid: audit_group_physics(sid) for sid, _, _ in GROUPS}
    save('group_physics_costs.json', physics)
    energy = energy_contrasts_check()
    save('energy_identity_check.json', energy)

    metrics, metric_rows = metrics_check()
    save('metrics_summary.json', metrics)
    coverage = coverage_check()
    save('coverage_summary.json', coverage)
    monthly = monthly_check()
    save('monthly_summary_check.json', monthly)
    lam_dist = selection_distribution()
    save('lambda_distribution.json', lam_dist)
    naive_diag = naive_diag_check(load, pv)
    save('naive_diagnostics_check.json', naive_diag)

    print('causality checks ...', flush=True)
    pert = perturbation_audit(load, pv, DATES, mine, price, None)
    save('future_perturbation_audit.json', pert)
    prefix = feedback_prefix_audit(load, pv, None)
    save('feedback_prefix_audit.json', prefix)

    prov = provenance(load, pv, price)
    save('provenance.json', prov)
    print('provenance:', json.dumps({k: prov[k] for k in
                                     ['signature_reproduced', 'all_input_hashes_match',
                                      'protected_rehash', 'artifact_hashes_verified',
                                      'january']}, ensure_ascii=False), flush=True)

    milp = None if args.skip_milp else milp_resample(price, load_kernel())
    if milp:
        save('milp_resample_summary.json', milp)

    # ---------------- consolidated verdict ----------------
    checks = {
        'baseline_reproduction': dict(
            max_segment_diff_kWh=max(fr0 for fr0 in
                                     [np.max(np.abs(
                                         pd.read_csv(RIDGE / 'R0_naive/dispatch.csv', low_memory=False)[c].to_numpy()
                                         - pd.read_csv(REF / 'dispatch.csv', low_memory=False)[c].to_numpy()))
                                      for c in ['planned_kWh', 'charge_kWh', 'discharge_kWh',
                                                'emergency_kWh', 'unused_kWh', 'state_start_kWh',
                                                'state_end_kWh', 'load_forecast_kW', 'pv_forecast_kW']]),
            cost_diff_yuan=None),
        'features_and_ridge': fr_summary,
        'selection': {k: v for k, v in sel.items() if k != 'mismatches'},
        'q80': q80,
        'group_ablation': ablation,
        'physics': {sid: dict(physical_max_kWh=physics[sid]['physical_max_kWh'],
                              nominal_max_kWh=max(v for k, v in physics[sid]['nominal'].items()
                                                  if k != 'planned_cost_vs_dispatch'),
                              physical_passed=physics[sid]['physical_passed'],
                              nominal_passed=physics[sid]['nominal_passed'],
                              accounting_passed=physics[sid]['accounting_passed'],
                              solver_passed=physics[sid]['solver_passed'])
                    for sid, _, _ in GROUPS},
        'energy_contrasts': energy,
        'metrics': metrics,
        'coverage': coverage,
        'monthly_contrasts': monthly,
        'lambda_distribution': lam_dist,
        'naive_diagnostics': naive_diag,
        'future_perturbation': dict(all_passed=pert['all_passed']),
        'feedback_prefix': prefix,
        'provenance': dict(signature_reproduced=prov['signature_reproduced'],
                           all_input_hashes_match=prov['all_input_hashes_match'],
                           code_matches_registration=prov['code_matches_registration'],
                           spec_matches_registration=prov['spec_matches_registration'],
                           protected=prov['protected_rehash'],
                           artifacts=prov['artifact_hashes_verified'],
                           january=prov['january']),
        'milp_resample': milp,
    }
    # R0 vs reference: full per-column comparison
    ref = pd.read_csv(REF / 'dispatch.csv', low_memory=False)
    r0 = pd.read_csv(RIDGE / 'R0_naive/dispatch.csv', low_memory=False)
    cols = ['load_forecast_kW', 'pv_forecast_kW', 'net_forecast_kWh', 'residual_adjustment_kWh',
            'protected_net_kWh', 'planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh',
            'unused_kWh', 'state_start_kWh', 'state_end_kWh', 'nominal_state_end_kWh',
            'planned_cost_yuan', 'emergency_cost_yuan']
    base_rep = dict(axis_match=bool(ref[['date', 'slot']].equals(r0[['date', 'slot']])),
                    max_abs_diff={c: f(np.max(np.abs(ref[c].to_numpy() - r0[c].to_numpy())))
                                  for c in cols})
    base_rep['max_overall_kWh'] = max(v for c, v in base_rep['max_abs_diff'].items()
                                      if c.endswith('_kWh') or c.endswith('_kW'))
    base_rep['cost_diff_yuan'] = f(abs((ref.planned_cost_yuan.sum() + ref.emergency_cost_yuan.sum())
                                       - (r0.planned_cost_yuan.sum() + r0.emergency_cost_yuan.sum())))
    base_rep['r0_total_cost_yuan'] = f(r0.planned_cost_yuan.sum() + r0.emergency_cost_yuan.sum())
    base_rep['registered_reference_yuan'] = REFERENCE_TOTAL
    base_rep['cost_diff_vs_registered_yuan'] = f(abs(base_rep['r0_total_cost_yuan'] - REFERENCE_TOTAL))
    base_rep['passed'] = bool(base_rep['axis_match'] and base_rep['max_overall_kWh'] <= TOL_SEG
                              and base_rep['cost_diff_vs_registered_yuan'] <= TOL_COST)
    checks['baseline_reproduction'] = base_rep

    end_hashes = {rel: digest(ROOT / rel) for rel in audit_inputs}
    save('checks.json', checks)
    verdict = {
        'implementation_baseline_reproduced': bool(base_rep['passed']),
        'features_ridge_reproduced': bool(all(v['passed'] for v in fr_summary.values())),
        'selection_reproduced': bool(sel['passed']),
        'q80_reproduced': bool(all(v['passed'] for v in q80.values())),
        'four_groups_differ_only_in_predictor': bool(ablation['passed']),
        'physics_and_accounting': bool(all(physics[s]['physical_passed'] and physics[s]['nominal_passed']
                                           and physics[s]['accounting_passed']
                                           and physics[s]['solver_passed'] for s, _, _ in GROUPS)),
        'energy_contrasts_identity': bool(energy['passed']),
        'metrics_recomputed': bool(metrics['passed']),
        'coverage_recomputed': bool(coverage['passed']),
        'monthly_contrasts_recomputed': bool(monthly['passed']),
        'naive_diagnostics_recomputed': bool(naive_diag['passed']),
        'report_lambda_distribution_scope_consistent': bool(
            lam_dist['report_load_claim_matches_evaluation_scope']
            and lam_dist['report_pv_claim_matches_evaluation_scope']),
        'future_perturbation_passed': bool(pert['all_passed']),
        'feedback_prefix_passed': bool(prefix['passed']),
        'provenance_intact': bool(prov['signature_reproduced'] and prov['all_input_hashes_match']
                                  and prov['protected_rehash']['n_changed'] == 0
                                  and prov['protected_rehash']['n_missing'] == 0
                                  and prov['january']['matches_8801']),
        'milp_sample_resolved': None if milp is None else bool(milp['passed']),
        'audit_inputs_unchanged_during_audit': bool(start_hashes == end_hashes),
    }
    summary = dict(generated_utc=pd.Timestamp.now('UTC').isoformat(),
                   wall_seconds=f(time.perf_counter() - t0),
                   thresholds=dict(per_segment_kWh=TOL_SEG, cost_yuan=TOL_COST,
                                   cumulative_energy_kWh=TOL_ENERGY,
                                   forecast_abs_kW=TOL_FORECAST_ABS,
                                   forecast_rel=TOL_FORECAST_REL),
                   verdict=verdict, checks=checks)
    save('summary.json', summary)
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)
    print('wall seconds', summary['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
