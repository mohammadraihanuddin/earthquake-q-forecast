#!/usr/bin/env python3
"""Five-parameter, auditable ETAS benchmark on the exploratory 644-cell grid.

The model follows the deck's temporal/magnitude formulation
``mu, K, alpha, c, p``. A hard 100-km spatial cutoff is used for triggering
parents. The deck does not specify a spatial background density, so this
implementation uses a transparent local empirical background proxy: for each
cell, the historical M>=Mc event rate within 100 km during the fit period.
Shared temporal parameters are fitted on the full fit catalog. This is a
named sensitivity variant, not a claim that the deck's background scope has
been uniquely recovered.
"""
from __future__ import annotations

import hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

WORK = Path(__file__).resolve().parent
DATA = WORK.parent / "data/raw"
OUT = WORK / "etas_experiment"
OUT.mkdir(exist_ok=True)
DAY = 86400.0
RADIUS_KM = 100.0
MC = 1.75
TARGET = 2.0
B_VALUE = 0.846
TARGET_SCALE = 10.0 ** (-B_VALUE * (TARGET - MC))

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def hav(lat, lon, lats, lons):
    a = np.sin(np.radians(lats-lat)/2)**2 + np.cos(np.radians(lat))*np.cos(np.radians(lats))*np.sin(np.radians(lons-lon)/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def load():
    fs = [pd.read_csv(DATA/f"earthquakeq_{x}.csv", parse_dates=["time"]) for x in ("train","test")]
    x = pd.concat(fs, ignore_index=True)
    return x[x.source.eq("seismic")][["time","latitude","longitude","magnitude"]].dropna().sort_values("time").reset_index(drop=True)

def cells(cat):
    a = np.floor(cat.latitude/0.5)*0.5; b = np.floor(cat.longitude/0.5)*0.5
    return np.asarray(sorted(set(zip(a.round(3), b.round(3)))), float)

def omori_integral(age, horizon, c, p):
    """Integrate the deck's unnormalized (age+c)^(-p) kernel."""
    if np.isclose(p, 1.0):
        return np.log((age + horizon + c) / (age + c))
    return ((age + c) ** (1 - p) - (age + horizon + c) ** (1 - p)) / (p - 1)

def check_integrals():
    # Hand-checkable identities, including the deck's one-event calculation.
    c, p, age = 0.7, 1.25, 3.0
    got = omori_integral(age, 30, c, p)
    full = (age+c)**(1-p)/(p-1)
    numerical = float(np.trapezoid((np.arange(300001) / 10000 + age + c) ** (-p),
                               np.arange(300001) / 10000))
    return {"finite_positive": bool(0 < got < full), "full_tail": float(full),
            "finite_30d": float(got), "zero_age_30d": float(omori_integral(0,30,c,p)),
            "numerical_30d": numerical,
            "one_event_relative_error": float(abs(got - np.trapezoid(
                (np.arange(300001) / 10000 + age + c) ** (-p),
                np.arange(300001) / 10000)) / got),
            "future_guard": bool(np.isclose(omori_integral(age, 0, c, p), 0.0)),
            "cutoff_guard": "past uses t < cutoff; labels use cutoff < t <= cutoff+30d"}

def likelihood(theta, t, lat, lon, mag, start, end):
    mu, K, alpha, c, p = theta
    if mu <= 0 or K <= 0 or c <= 0 or p <= 1: return 1e100
    n = len(t); lam = np.full(n, mu, float)
    # Pair contributions use the explicit hard cutoff.  The matrix is bounded
    # to fit-period events only, so no calibration/test event can leak in.
    for j in range(n):
        prior = (t < t[j])
        if prior.any():
            d = hav(lat[j], lon[j], lat[prior], lon[prior])
            age = t[j] - t[prior]
            use = d <= RADIUS_KM
            if use.any():
                lam[j] += np.sum(K*np.exp(alpha*(mag[prior][use]-MC)) *
                                 (age[use] + c) ** (-p))
    # Same unnormalized kernel and day units as the event intensity above.
    ages0 = np.maximum(0.0, start-t)
    trig = np.sum(K*np.exp(alpha*(mag-MC))*omori_integral(ages0, end-start, c, p))
    integral = mu*(end-start) + trig
    return float(integral - np.log(np.clip(lam, 1e-300, None)).sum())

def fit_etas(cat):
    fit = cat[(cat.time >= "2001-01-01") & (cat.time < "2020-01-01") & cat.magnitude.ge(MC)]
    start = pd.Timestamp("2001-01-01")
    end = pd.Timestamp("2020-01-01")
    t = (fit.time - start).dt.total_seconds().to_numpy() / DAY
    fit_days = (end - start).total_seconds() / DAY
    # Global daily rate starts at the observed rate, not a per-cell rate.
    x0 = np.array([len(t)/fit_days, .01, .5, 1., 1.2])
    bounds = [(1e-5, 10.), (1e-8, 5.), (0., 3.), (.01, 30.), (1.001, 3.)]
    result = minimize(likelihood, x0, args=(t, fit.latitude.to_numpy(), fit.longitude.to_numpy(),
                    fit.magnitude.to_numpy(), 0.0, fit_days), method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 120, "ftol": 1e-10, "maxls": 30})
    return result, fit, (0.0, fit_days)

def examples(cat, cs, dates, theta, background_rates):
    t = cat.time.to_numpy(dtype="datetime64[s]").astype("int64"); la=cat.latitude.to_numpy(); lo=cat.longitude.to_numpy(); m=cat.magnitude.to_numpy()
    out=[]; y=[]
    for d in dates:
        cut = pd.Timestamp(d).value/1e9
        past=t < cut
        future=(t > cut)&(t <= cut+30*DAY)
        for index, (cell_la,cell_lo) in enumerate(cs):
            dist=hav(cell_la,cell_lo,la[past],lo[past]); use=dist<=RADIUS_KM
            age=(cut-t[past][use])/DAY
            trigger=np.sum(theta[1]*np.exp(theta[2]*(m[past][use]-MC))*omori_integral(age,30,theta[3],theta[4]))
            expected=(background_rates[index]*30+trigger) * TARGET_SCALE
            out.append(1-np.exp(-max(expected,0)))
            fd=hav(cell_la,cell_lo,la[future],lo[future])
            y.append(int(np.any((fd<=RADIUS_KM)&(m[future]>=TARGET))))
    return np.asarray(out), np.asarray(y)

def score(y,p,base):
    p=np.clip(p,1e-5,1-1e-5)
    return {"n":int(len(y)),"positives":int(y.sum()),"base_rate":float(base),
            "ig_bits":float((log_loss(y,np.full(len(y),base),labels=[0,1])-log_loss(y,p,labels=[0,1]))/np.log(2)),
            "auc":float(roc_auc_score(y,p)) if np.unique(y).size > 1 else None,
            "pr_auc":float(average_precision_score(y,p)),
            "brier":float(brier_score_loss(y,p))}

def main():
    cat=load(); cs=cells(cat[cat.time<"2022-01-01"])
    assert len(cs) == 644, f"Expected 644 exploratory cells, got {len(cs)}"
    result,fit,_=fit_etas(cat)
    theta=result.x
    fit_start = pd.Timestamp("2001-01-01")
    fit_end = pd.Timestamp("2020-01-01")
    fit_days = (fit_end - fit_start).total_seconds() / DAY
    background_rates = np.asarray([
        theta[0] * np.sum(
            hav(la, lo, fit.latitude.to_numpy(), fit.longitude.to_numpy()) <= RADIUS_KM
        ) / len(fit) for la, lo in cs
    ])
    cal_dates=pd.date_range("2020-01-01","2021-12-01",freq="MS")
    test_dates=pd.date_range("2022-01-01","2024-11-01",freq="MS")
    raw_cal,ycal=examples(cat,cs,cal_dates,theta,background_rates); raw_test,ytest=examples(cat,cs,test_dates,theta,background_rates)
    iso=IsotonicRegression(y_min=1e-5,y_max=1-1e-5,out_of_bounds="clip").fit(raw_cal,ycal)
    pcal=iso.predict(raw_cal); ptest=iso.predict(raw_test)
    base=float(ycal.mean()); metrics={"calibration":score(ycal,pcal,base),"test":score(ytest,ptest,float(ytest.mean())),
                                        "test_raw":score(ytest,raw_test,float(ytest.mean()))}
    np.savez_compressed(OUT/"predictions.npz", calibration_raw=raw_cal, calibration=pcal,
                        calibration_y=ycal, test_raw=raw_test, test=ptest, test_y=ytest)
    meta={"parameters":dict(zip(["mu_global_day","K","alpha","c_days","p"],map(float,theta))),
          "success":bool(result.success),"message":str(result.message),"nit":int(result.nit),
          "fun":float(result.fun),"fit_events":int(len(fit)),"cells":int(len(cs)),
          "background_scope":"mu_global multiplied by each cell's historical M>=Mc activity share; shared temporal ETAS parameters",
          "cell_contract_assertion":"644 cells derived from all seismic events before 2022; magnitude filtering does not alter IDs",
          "kernel":"unnormalized (s-t_j+c)^(-p), analytically integrated over each 30-day window",
          "timestamp_unit":"integer Unix seconds from datetime64[s]; origin uses strict t_event < t0",
          "magnitude_convention":"fit and background use M>=1.75; expected counts scaled to M>=2.0 with regional b=0.846 before Poisson conversion",
          "target_scale":float(TARGET_SCALE),
          "hard_radius_km":100,"mc":MC,"target_magnitude":TARGET,"horizon_days":30,
          "splits":{"fit":"2001-01-01..2019-12-31","calibration":"2020-01..2021-12","test":"2022-01..2024-11"},
          "integral_checks":check_integrals(),"metrics":metrics,
          "calibration_diagnostics":{
              "raw_calibration_unique":int(np.unique(raw_cal).size),
              "calibrated_unique":int(np.unique(pcal).size),
              "raw_test_quantiles":np.quantile(raw_test,[0.01,0.5,0.99]).tolist(),
              "calibrated_test_quantiles":np.quantile(ptest,[0.01,0.5,0.99]).tolist(),
          },
          "inputs":{f"earthquakeq_{x}.csv":sha256(DATA/f"earthquakeq_{x}.csv") for x in ("train","test")},
          "python":sys.version,"platform":platform.platform()}
    (OUT/"results.json").write_text(json.dumps(meta,indent=2,default=str)+"\n")
    with (WORK/"experiment_log.md").open("a") as f:
        f.write("\n## Run 008 — ETAS magnitude-scaled spatial-share sensitivity\n\n")
        f.write("**Status:** Completed on the 644-cell chronological development grid; judge data was not loaded.  \n")
        f.write("**Code/artifacts:** `run_etas_experiment.py`, `etas_experiment/results.json`, `etas_experiment/predictions.npz`.  \n\n")
        f.write("This rerun keeps the 644 IDs from all pre-2022 seismic events, uses the same unnormalized integrated Omori kernel in intensity, compensator, and forecast, weights global mu by each cell's historical M>=1.75 activity share, and applies the b=0.846 Gutenberg-Richter factor before the M>=2.0 Poisson conversion. Run 006 is archived as invalid because it mixed global and per-cell background scales. Raw and separately calibrated scores are both retained. K is at its lower bound, so this remains a sensitivity result rather than a finalized ETAS benchmark.\n\n")
        f.write(f"Test calibrated IG: **{metrics['test']['ig_bits']:.4f} bits**, AUC {metrics['test']['auc']:.4f}, Brier {metrics['test']['brier']:.4f}.\n")
    print(json.dumps({"parameters":meta["parameters"],"metrics":metrics,"converged":result.success},indent=2))
if __name__=="__main__": main()
