"""Compare common and stage-specific EASI estimators on identical patient splits.

Run from any directory with --root or T2T_ROOT. The input contains private health
records; only aggregate results and model coefficients belong in a public package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import lstsq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

MARKERS = ["log2_CCL17", "log2_CCL26", "log2_IL-5"]
FEATURES = ["bsa"] + MARKERS
PRIVATE_COLUMNS = ["patient_uid", "visit_week", "observed", "treatment", "center", "dose_ambiguous", "source_contradiction"]


def safe_float(value):
    return float(value) if np.isfinite(value) else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Model:
    def __init__(self, frame, structure, alpha):
        self.structure, self.alpha = structure, float(alpha)
        self.features = ["bsa"] if structure.startswith("bsa_") else MARKERS if structure == "serum_linear" else FEATURES
        self.initial_scale = StandardScaler().fit(frame[self.features])
        self.zero_anchor = structure in ("unified_area_interaction", "bsa_area")
        self.last_scale = StandardScaler(with_mean=not self.zero_anchor)
        x = self.last_scale.fit_transform(self.design(frame))
        y = frame.observed.to_numpy()
        weight = 1 / frame.patient_uid.map(frame.patient_uid.value_counts()).to_numpy()
        self.weight = weight / weight.mean()
        self.x_mean = np.zeros(x.shape[1]) if self.zero_anchor else np.average(x, axis=0, weights=self.weight)
        self.y_mean = 0.0 if self.zero_anchor else np.average(y, weights=self.weight)
        root_w = np.sqrt(self.weight)
        matrix = np.vstack([(x - self.x_mean) * root_w[:, None], np.sqrt(alpha) * np.eye(x.shape[1])])
        target = np.r_[(y - self.y_mean) * root_w, np.zeros(x.shape[1])]
        self.beta = lstsq(matrix, target, lapack_driver="gelsd")[0]
        self.intercept = self.y_mean - np.einsum("i,i->", self.x_mean, self.beta)

    def design(self, frame):
        values = frame[self.features].to_numpy(float)
        z = self.initial_scale.transform(frame[self.features])
        if self.structure == "unified_area_interaction":
            return np.column_stack([values[:, 0], values[:, :1] * z[:, 1:]])
        if self.structure == "bsa_area":
            return values
        if self.structure in ("stage_interaction", "bsa_stage"):
            stage = frame.stage.to_numpy()[:, None]
            return np.column_stack([z, stage, z * stage])
        if self.structure == "unified_quadratic":
            return np.column_stack([z, values[:, 0] ** 2])
        return z

    def predict_raw(self, frame):
        x = self.last_scale.transform(self.design(frame))
        return self.intercept + np.einsum("ij,j->i", x, self.beta, optimize=False)

    def predict(self, frame):
        return np.clip(self.predict_raw(frame), 0, 72)

    def formula(self):
        beta = self.beta / self.last_scale.scale_
        origin = self.intercept - np.einsum("i,i->", beta, self.last_scale.mean_) if self.last_scale.with_mean else self.intercept
        if self.structure == "unified_area_interaction":
            slopes = beta[1:] / self.initial_scale.scale_[1:]
            constant = beta[0] - np.einsum("i,i->", slopes, self.initial_scale.mean_[1:])
            return {"expression": "clip(BSA * (constant + sum(coefficient * log2_concentration)), 0, 72)", "intercept": 0.0, "bsa_factor": float(constant), "bsa_log2_coefficients": dict(zip(MARKERS, map(float, slopes)))}
        if self.structure == "bsa_area":
            return {"expression": "clip(coefficient * BSA, 0, 72)", "intercept": 0.0, "coefficients": {"bsa": float(beta[0])}}
        size = len(self.features)
        formulas = {}
        stages = [0, 1] if self.structure in ("stage_interaction", "bsa_stage") else [0]
        for stage in stages:
            slopes = beta[:size].copy()
            offset = origin
            if self.structure in ("stage_interaction", "bsa_stage"):
                slopes += stage * beta[size + 1:]
                offset += stage * beta[size]
            slopes = slopes / self.initial_scale.scale_
            offset -= np.einsum("i,i->", slopes, self.initial_scale.mean_)
            coefficients = dict(zip(self.features, map(float, slopes)))
            if self.structure == "unified_quadratic":
                coefficients["bsa_squared"] = float(beta[-1])
            formulas[str(stage)] = {"intercept": float(offset), "coefficients": coefficients}
        return {"expression": "clip(intercept + sum(coefficient * input), 0, 72)", "by_stage": formulas} if len(stages) == 2 else {"expression": "clip(intercept + sum(coefficient * input), 0, 72)", **formulas["0"]}


def evaluate_formula(frame, formula):
    if "by_stage" in formula:
        out = np.zeros(len(frame))
        for stage, subformula in formula["by_stage"].items():
            mask = frame.stage.eq(int(stage)).to_numpy()
            out[mask] = evaluate_formula(frame.loc[mask], subformula)
        return out
    if "bsa_factor" in formula:
        factor = np.full(len(frame), formula["bsa_factor"])
        for col, coefficient in formula["bsa_log2_coefficients"].items():
            factor += coefficient * frame[col].to_numpy()
        return np.clip(frame.bsa.to_numpy() * factor, 0, 72)
    output = np.full(len(frame), formula["intercept"], dtype=float)
    for col, coefficient in formula["coefficients"].items():
        value = frame.bsa.to_numpy() ** 2 if col == "bsa_squared" else frame[col].to_numpy()
        output += coefficient * value
    return np.clip(output, 0, 72)


def select_model(frame, structures, cfg):
    ids = np.array(sorted(frame.patient_uid.unique()))
    splits = list(KFold(cfg["inner_folds"], shuffle=True, random_state=cfg["seed"]).split(ids))
    scores = []
    for structure in structures:
        for alpha in cfg["ridge_alpha"]:
            errors = []
            for train, validation in splits:
                a = frame[frame.patient_uid.isin(ids[train])]
                b = frame[frame.patient_uid.isin(ids[validation])]
                assert set(a.patient_uid).isdisjoint(b.patient_uid)
                fitted = Model(a, structure, alpha)
                result = pd.DataFrame({"patient_uid": b.patient_uid, "error": abs(b.observed - fitted.predict(b))})
                errors.extend(result.groupby("patient_uid").error.mean().tolist())
            scores.append({"structure": structure, "alpha": alpha, "inner_patient_MAE": float(np.mean(errors))})
    chosen = min(scores, key=lambda row: row["inner_patient_MAE"])
    return Model(frame, chosen["structure"], chosen["alpha"]), scores


def scalar_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    e = p - y
    if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0:
        rho = spearmanr(y, p).statistic
        slope = np.mean((p - p.mean()) * (y - y.mean())) / np.var(p)
        intercept = y.mean() - slope * p.mean()
        ccc = 2 * np.mean((y - y.mean()) * (p - p.mean())) / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    else:
        rho, slope, intercept, ccc = np.nan, np.nan, np.nan, np.nan
    sd = np.std(e, ddof=1) if len(e) > 1 else np.nan
    high = y > 7
    return dict(MAE=float(np.mean(abs(e))), RMSE=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()), rho=rho, CCC=ccc,
                R2=1 - np.sum(e**2) / np.sum((y - y.mean())**2) if np.var(y) else np.nan,
                calibration_intercept=intercept, calibration_slope=slope,
                LoA_low=e.mean() - 1.96 * sd, LoA_high=e.mean() + 1.96 * sd,
                band_accuracy=np.mean(np.digitize(y, [7, 21, 50], right=True) == np.digitize(p, [7, 21, 50], right=True)),
                within1=np.mean(abs(e) <= 1), within2=np.mean(abs(e) <= 2),
                sensitivity_gt7=np.mean(p[high] > 7) if high.any() else np.nan,
                specificity_le7=np.mean(p[~high] <= 7) if (~high).any() else np.nan)


def bootstrap_mean(values, cfg):
    x = np.asarray(values, float)
    if len(x) < 2:
        return np.array([np.nan, np.nan])
    rng = np.random.default_rng(cfg["seed"])
    draws = rng.choice(x, (cfg["bootstrap_replicates"], len(x)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def metric(frame, cfg):
    by_repeat = pd.DataFrame([scalar_metrics(g.observed, g.prediction) for _, g in frame.groupby("repeat")])
    per_patient = frame.assign(error=abs(frame.prediction - frame.observed)).groupby("patient_uid").error.mean()
    ci = bootstrap_mean(per_patient, cfg)
    output = by_repeat.mean().to_dict()
    output.update(n_patients=int(frame.patient_uid.nunique()), n_visits=len(frame[["patient_uid", "visit_week"]].drop_duplicates()), patient_MAE=per_patient.mean(), MAE_lo=ci[0], MAE_hi=ci[1])
    return output


def subsets(frame):
    yield "overall", "all", frame
    for name, mask in [("EASI_le7", frame.observed.le(7)), ("EASI_7_to21", frame.observed.gt(7) & frame.observed.le(21)), ("EASI_gt21", frame.observed.gt(21))]:
        if mask.any():
            yield "severity", name, frame.loc[mask]
    for name, sub in frame.loc[~frame.dose_ambiguous.astype(bool)].groupby("treatment"):
        yield "treatment", name, sub
    for name, sub in frame.groupby("center"):
        yield "center", name, sub
    consistent = frame.loc[~frame.source_contradiction.astype(bool)]
    if len(consistent):
        yield "source_consistency", "recorded_total_consistent", consistent


def summary(prediction, cfg):
    rows = []
    for (model, week), data in prediction.groupby(["model", "visit_week"]):
        for kind, name, sub in subsets(data):
            rows.append(dict(model=model, week=week, group_type=kind, group=name, **metric(sub, cfg)))
    return pd.DataFrame(rows)


def paired_comparisons(prediction, cfg):
    rows = []
    keys = ["patient_uid", "visit_week", "repeat", "fold"]
    contrasts = [("unified_selected", "stage_interaction"), ("unified_linear", "stage_interaction"),
                 ("unified_quadratic", "stage_interaction"), ("unified_area_interaction", "stage_interaction"),
                 ("unified_linear", "bsa_linear"), ("unified_area_interaction", "bsa_area"), ("stage_interaction", "bsa_stage")]
    for candidate, reference in contrasts:
        a = prediction[prediction.model.eq(candidate)]
        b = prediction[prediction.model.eq(reference)]
        joined = a.merge(b[keys + ["prediction"]], on=keys, validate="one_to_one", suffixes=("", "_reference"))
        joined["error_difference"] = abs(joined.prediction - joined.observed) - abs(joined.prediction_reference - joined.observed)
        for week, sub in joined.groupby("visit_week"):
            patient_delta = sub.groupby("patient_uid").error_difference.mean()
            lo, hi = bootstrap_mean(patient_delta, cfg)
            rows.append(dict(candidate=candidate, reference=reference, week=week, n_patients=len(patient_delta), MAE_difference=patient_delta.mean(), difference_lo=lo, difference_hi=hi, interpretation="positive means candidate has greater absolute error"))
    return pd.DataFrame(rows)


def repeat_audit(prediction):
    rows = []
    for (model, week), g in prediction.groupby(["model", "visit_week"]):
        correct = g.assign(error=abs(g.prediction - g.observed)).groupby("patient_uid").error.mean().mean()
        averaged = g.groupby("patient_uid").agg(observed=("observed", "first"), prediction=("prediction", "mean"))
        shortcut = abs(averaged.prediction - averaged.observed).mean()
        rows.append(dict(model=model, week=week, mean_repeat_absolute_error=correct, absolute_error_after_averaging_predictions=shortcut, apparent_error_reduction_from_averaging=correct-shortcut, reported_method="mean_repeat_absolute_error"))
    return pd.DataFrame(rows)


def changes(prediction, cfg):
    outputs, rates = [], []
    keys = ["patient_uid", "repeat", "fold"]
    for model, data in prediction.groupby("model"):
        for start, end in [(0, 12), (0, 24), (12, 24)]:
            a = data[data.visit_week.eq(start)]
            b = data[data.visit_week.eq(end)]
            pair = b.merge(a[keys + ["observed", "prediction"]], on=keys, validate="one_to_one", suffixes=("", "_start"))
            if not len(pair):
                continue
            delta = pair.copy()
            delta["observed"] = pair.observed - pair.observed_start
            delta["prediction"] = pair.prediction - pair.prediction_start
            values = metric(delta, cfg)
            outputs.append(dict(model=model, window=f"W{start}_W{end}", **{k: v for k, v in values.items() if k not in ("band_accuracy", "sensitivity_gt7", "specificity_le7")}))
            if start == 0:
                eligible = pair[(pair.observed_start > 0) & (pair.prediction_start > 0)].copy()
                for threshold in [0.50, 0.75, 0.90]:
                    statistics, denominators = [], []
                    for repeat, sub in eligible.groupby("repeat"):
                        truth = (1 - sub.observed / sub.observed_start) >= threshold
                        estimate = (1 - sub.prediction / sub.prediction_start) >= threshold
                        denominators.append(len(sub))
                        statistics.append(dict(sensitivity=estimate[truth].mean() if truth.any() else np.nan, specificity=(~estimate[~truth]).mean() if (~truth).any() else np.nan, agreement=(truth == estimate).mean()))
                    result = pd.DataFrame(statistics).mean().to_dict()
                    rates.append(dict(model=model, window=f"W0_W{end}", endpoint=f"EASI{round(threshold*100)}", n_pairs=int(pair.patient_uid.nunique()), n_evaluable=int(eligible.patient_uid.nunique()), n_evaluable_min_per_repeat=min(denominators), n_evaluable_max_per_repeat=max(denominators), **result))
    return pd.DataFrame(outputs), pd.DataFrame(rates)


def published_comparator(data, prediction, cfg):
    data = data.copy()
    for column in ["log2_IL-22", "log2_IL2RA"]:
        if column not in data:
            data[column] = np.nan
    available = data.dropna(subset=["log2_CCL17", "log2_IL-22", "log2_IL2RA"]).copy()
    pre = available.visit_week.eq(0)
    ccl17_log10 = available["log2_CCL17"] * np.log10(2)
    il22, il2ra = 2.0**available["log2_IL-22"], 2.0**available["log2_IL2RA"]
    available["published_prediction"] = np.where(pre, -36.12 + 18.49*ccl17_log10 + .009*il22 - .009*il2ra, -5.82 + 4.04*ccl17_log10 + .003*il22 - .003*il2ra)
    # Original rounded coefficients are applied unchanged. IL2RA is a nominal
    # local assay proxy for sIL-2R; cross-platform equivalence is unestablished.
    keys = ["patient_uid", "visit_week"]
    common = prediction[prediction.model.isin(["unified_selected", "unified_linear", "unified_area_interaction", "stage_interaction"])].merge(available[keys + ["published_prediction"]], on=keys, validate="many_to_one")
    rows = []
    for (model, week), sub in common.groupby(["model", "visit_week"]):
        own = metric(sub, cfg)
        published = metric(sub.assign(prediction=sub.published_prediction), cfg)
        for name, value in [(model, own), ("published_original_raw", published)]:
            rows.append(dict(matched_candidate=model, model=name, week=week, **value))
    return pd.DataFrame(rows), common


def held_out_groups(data, cfg):
    predictions, audit = [], []
    for dimension in ["center", "treatment"]:
        eligible = data.loc[~data.dose_ambiguous.astype(bool)] if dimension == "treatment" else data
        for group in sorted(eligible[dimension].dropna().unique()):
            test = eligible[eligible[dimension].eq(group)]
            test_ids = set(test.patient_uid)
            train = data.loc[~data.patient_uid.isin(test_ids) & data.visit_week.isin(cfg["development_weeks"])]
            if train.patient_uid.nunique() < 20 or test.patient_uid.nunique() < 3:
                continue
            fit, _ = select_model(train, cfg["unified_selector_candidates"], cfg)
            assert set(train.patient_uid).isdisjoint(test.patient_uid)
            result = test[PRIVATE_COLUMNS].copy()
            result["prediction"] = fit.predict(test)
            result["repeat"] = 0
            result["fold"] = 0
            result["model"] = "unified_selected"
            result["held_out_dimension"] = dimension
            result["held_out_group"] = group
            predictions.append(result)
            audit.append(dict(held_out_dimension=dimension, held_out_group=group, structure=fit.structure, alpha=fit.alpha, n_training_patients=train.patient_uid.nunique(), n_test_patients=test.patient_uid.nunique(), overlap=0))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit)


def audit_historical_targets(root, out, data):
    clinical_path = root / "results/tables/update_20260925/clinical.csv"
    source = root / "results/tables/clinical_alignment"
    if not clinical_path.exists() or not (source / "stage_performance.csv").exists():
        return []
    clinical = pd.read_csv(clinical_path)
    joined = data.merge(clinical[["patient_uid", "visit_week", "easi_total", "bsa"]], on=["patient_uid", "visit_week"], validate="one_to_one", suffixes=("", "_clinical"))
    checks = []
    for input_field, current_field in [("observed", "easi_total"), ("bsa", "bsa_clinical")]:
        for week, sub in joined.groupby("visit_week"):
            mismatched = ~np.isclose(sub[input_field], sub[current_field], atol=1e-10, equal_nan=True)
            assert not mismatched.any()
            checks.append(dict(input_field=input_field, current_field=current_field, week=week, n_checked=len(sub), n_discrepant=int(mismatched.sum()), max_abs_difference=float(abs(sub[input_field]-sub[current_field]).max())))
    pd.DataFrame(checks).to_csv(out / "clinical_input_reconciliation.csv", index=False)
    audits, sources = [], [clinical_path]
    for target in ["pp_nrs", "dlqi", "iga"]:
        path = source / f"stage_predictions_{target}.csv"
        split_path = source / f"stage_splits_{target}.csv"
        prediction, splits = pd.read_csv(path), pd.read_csv(split_path)
        q = prediction[prediction.scope.isin(["nested_OOF", "W24_patient_held_out"]) & prediction.model.eq("stage_small")]
        unique = q[["patient_uid", "visit_week", "observed", "age"]].drop_duplicates()
        joined = unique.merge(clinical[["patient_uid", "visit_week", target]], on=["patient_uid", "visit_week"], how="left", validate="one_to_one")
        valid = joined[target].notna()
        mismatched = valid & ~np.isclose(joined.observed, joined[target], atol=1e-8, equal_nan=True)
        probabilities = prediction[[f"p{i}" for i in range(6)]] if target == "iga" else None
        notes = {"pp_nrs": "Historical phase/drug-specific fit; recall window requires confirmation.",
                 "dlqi": "Historical age>16 scope includes 17-year-olds; not the current >=18 primary scope; questionnaire identity requires confirmation.",
                 "iga": "Recorded six-category ordinal model; actual IGA instrument identity requires confirmation."}
        audits.append(dict(target=target, status="HISTORICAL_EXPLORATORY_NOT_UPDATED_FINAL", independent_model=True,
                           split_rows=len(splits), patient_overlap_max=max(len(set(r.train_ids.split("|")) & set(r.test_ids.split("|"))) for r in splits.itertuples()),
                           max_training_week=int(splits.max_training_week.max()), predictions_finite=bool(np.isfinite(prediction.prediction).all()),
                           duplicate_prediction_count=int(prediction.duplicated(["scope", "model", "patient_uid", "visit_week", "repeat"]).sum()),
                           current_truth_checked_rows=int(valid.sum()), current_truth_mismatch_rows=int(mismatched.sum()),
                           old_truth_currently_missing_rows=int((~valid).sum()), min_prediction_age=float(q.age.min()),
                           under18_visits_in_historical_dlqi=int((unique.age < 18).sum()) if target == "dlqi" else 0,
                           ordinal_probabilities_valid=bool((probabilities >= 0).all().all() and np.allclose(probabilities.sum(1),1)) if target == "iga" else None,
                           notes=notes[target], source=str(path.relative_to(root)), sha256=sha(path)))
        sources.extend([path, split_path])
    pd.DataFrame(audits).to_csv(out / "other_targets_audit.csv", index=False)
    for name, destination in [("stage_performance.csv", "other_targets_historical_metrics.csv"), ("stage_stratified_performance.csv", "other_targets_historical_by_week.csv")]:
        path = source / name
        frame = pd.read_csv(path)
        keep = frame.target.isin(["pp_nrs", "dlqi", "iga"])
        if name == "stage_performance.csv":
            keep &= frame.model.isin(["stage_small", "stage_best", "state_only", "CCL17"])
        else:
            keep &= frame.model.eq("stage_small") & frame.stratum.eq("visit_week")
        frame = frame.loc[keep].copy()
        frame["evidence_status"] = "HISTORICAL_EXPLORATORY_NOT_UPDATED_FINAL"
        frame.to_csv(out / destination, index=False)
        sources.append(path)
    return sources


def assay_audit_and_sensitivity(root, out, data, prediction, cfg):
    assay_path = root / "results/tables/redevelopment/assay_long.csv"
    flags_path = root / "results/tables/refinement/data_forensics_msd_sample_flags.csv"
    if not assay_path.exists() or not flags_path.exists():
        return []
    assay, flags = pd.read_csv(assay_path), pd.read_csv(flags_path)
    rows = []
    for marker in ["CCL17", "CCL26", "IL-5"]:
        col = "log2_" + marker
        joined = data[["patient_uid", "visit_week", col]].merge(assay[assay.analyte.eq(marker)], on=["patient_uid", "visit_week"], how="left", validate="one_to_one")
        assert joined.state.eq("observed").all()
        assert np.allclose(joined[col], np.log2(joined.concentration), atol=1e-12)
        for week, sub in joined.groupby("visit_week"):
            rows.append(dict(marker=marker, week=week, n_rows=len(sub), n_exported_positive=len(sub), exported_min=float(sub.concentration.min()), exported_max=float(sub.concentration.max()), log2_source_max_abs=float(np.max(abs(sub[col]-np.log2(sub.concentration)))), half_min_or_pseudocount_used=False, source_matrix="Summary/4.Integration/Summary.xlsx, not Filter matrix"))
    pd.DataFrame(rows).to_csv(out / "assay_input_reconciliation.csv", index=False)
    il5 = flags[flags.analyte.eq("IL-5")][["patient_uid", "visit_week", "software_calc_low"]]
    bounded = data.merge(il5, on=["patient_uid", "visit_week"], how="left", validate="one_to_one")
    threshold = np.log2(2 * bounded.software_calc_low)
    affected = bounded["log2_IL-5"] < threshold
    coverage = bounded.assign(below_software_2x=affected).groupby("visit_week").agg(n_input=("patient_uid","size"), n_boundary_available=("software_calc_low","count"), n_below_software_2x=("below_software_2x","sum")).reset_index()
    coverage["qualification"] = "2x reported software Calc.Low; not validated LLOQ; positive exports can be below this boundary"
    coverage.to_csv(out / "assay_software_boundary_coverage.csv", index=False)
    bounded.loc[affected,"log2_IL-5"] = threshold[affected]
    ids = np.array(sorted(bounded.patient_uid.unique()))
    outputs, audit = [], []
    for repeat in range(cfg["outer_repeats"]):
        for fold, (tr, te) in enumerate(KFold(cfg["outer_folds"], shuffle=True, random_state=cfg["seed"]+repeat).split(ids)):
            train = bounded[bounded.patient_uid.isin(ids[tr]) & bounded.visit_week.isin(cfg["development_weeks"])]
            test = bounded[bounded.patient_uid.isin(ids[te])]
            fit, _ = select_model(train, cfg["unified_selector_candidates"], cfg)
            assert set(train.patient_uid).isdisjoint(test.patient_uid)
            sub = test[PRIVATE_COLUMNS].copy()
            sub["prediction"] = fit.predict(test)
            sub["model"], sub["repeat"], sub["fold"] = "unified_selected_software_bound_2x", repeat, fold
            outputs.append(sub)
            audit.append(dict(repeat=repeat, fold=fold, selected_structure=fit.structure, alpha=fit.alpha, max_training_week=int(train.visit_week.max()), patient_overlap=0))
    sensitivity = pd.concat(outputs,ignore_index=True)
    sensitivity.to_csv(out / "assay_sensitivity_predictions_private.csv", index=False)
    summary(sensitivity,cfg).to_csv(out / "assay_sensitivity_metrics.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "assay_sensitivity_fit_audit.csv", index=False)
    keys = ["patient_uid","visit_week","repeat","fold"]
    joined = sensitivity.merge(prediction[prediction.model.eq("unified_selected")][keys+["prediction"]], on=keys,validate="one_to_one",suffixes=("","_reported"))
    joined["loss_difference"] = abs(joined.prediction-joined.observed)-abs(joined.prediction_reported-joined.observed)
    rows = []
    for week, sub in joined.groupby("visit_week"):
        differences = sub.groupby("patient_uid").loss_difference.mean()
        lo, hi = bootstrap_mean(differences,cfg)
        rows.append(dict(week=week,n_patients=len(differences),MAE_difference_boundary_minus_reported=differences.mean(),difference_lo=lo,difference_hi=hi,qualification="Alternative censor representation only; primary exported values and formula unchanged"))
    pd.DataFrame(rows).to_csv(out / "assay_sensitivity_comparison.csv",index=False)
    return [assay_path,flags_path,root/"src/cytot2t/stages/stage_01_data_lock.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("T2T_ROOT", Path(__file__).resolve().parents[3])))
    parser.add_argument("--config", default="config/unified_model_20261017.json")
    args = parser.parse_args()
    root = args.root.resolve()
    cfg_path = root / args.config
    cfg = json.loads(cfg_path.read_text())
    out = root / cfg["output_path"]
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(root / cfg["input_path"])
    assert not data.duplicated(["patient_uid", "visit_week"]).any()
    assert data[FEATURES + ["observed"]].notna().all().all()
    assert np.isfinite(data[FEATURES + ["observed"]]).all().all()
    assert data.bsa.between(0, 100).all() and data.observed.between(0, 72).all()
    assert data.visit_week.isin([0, 12, 24]).all()
    ids = np.array(sorted(data.patient_uid.unique()))
    predictions, audits, inner_scores = [], [], []
    for repeat in range(cfg["outer_repeats"]):
        for fold, (train_indices, test_indices) in enumerate(KFold(cfg["outer_folds"], shuffle=True, random_state=cfg["seed"]+repeat).split(ids)):
            train = data[data.patient_uid.isin(ids[train_indices]) & data.visit_week.isin(cfg["development_weeks"])]
            test = data[data.patient_uid.isin(ids[test_indices])]
            assert set(train.patient_uid).isdisjoint(test.patient_uid)
            for name in cfg["models"] + ["unified_selected"]:
                structures = cfg["unified_selector_candidates"] if name == "unified_selected" else [name]
                fit, scores = select_model(train, structures, cfg)
                q = test[PRIVATE_COLUMNS].copy()
                q["prediction"] = fit.predict(test)
                q["prediction_raw"] = fit.predict_raw(test)
                q["repeat"], q["fold"], q["model"] = repeat, fold, name
                predictions.append(q)
                audits.append(dict(repeat=repeat, fold=fold, model=name, selected_structure=fit.structure, alpha=fit.alpha, n_training_visits=len(train), n_training_patients=train.patient_uid.nunique(), max_training_week=int(train.visit_week.max()), features="|".join(fit.features), train_ids="|".join(sorted(train.patient_uid.unique())), test_ids="|".join(sorted(test.patient_uid.unique())), overlap=0))
                inner_scores.extend(dict(repeat=repeat, fold=fold, model=name, **score) for score in scores)
            print(f"Completed repeat {repeat+1}, fold {fold+1}", flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    assert not predictions.duplicated(["model", "repeat", "patient_uid", "visit_week"]).any()
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "fit_audit_private.csv", index=False)
    pd.DataFrame(audits).drop(columns=["train_ids", "test_ids"]).to_csv(out / "fit_audit.csv", index=False)
    pd.DataFrame(inner_scores).to_csv(out / "inner_selection.csv", index=False)
    metrics = summary(predictions, cfg)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    paired_comparisons(predictions, cfg).to_csv(out / "paired_model_comparisons.csv", index=False)
    repeat_audit(predictions).to_csv(out / "repeat_aggregation_audit.csv", index=False)
    delta, target = changes(predictions, cfg)
    delta.to_csv(out / "paired_change_metrics.csv", index=False)
    target.to_csv(out / "improvement_classification.csv", index=False)
    published, common = published_comparator(data, predictions, cfg)
    published.to_csv(out / "published_matched_metrics.csv", index=False)
    common.to_csv(out / "published_matched_predictions_private.csv", index=False)
    stress, stress_audit = held_out_groups(data, cfg)
    stress.to_csv(out / "held_out_group_predictions_private.csv", index=False)
    stress_audit.to_csv(out / "held_out_group_fit_audit.csv", index=False)
    stress_rows = []
    for (dimension, group, week), sub in stress.groupby(["held_out_dimension", "held_out_group", "visit_week"]):
        stress_rows.append(dict(held_out_dimension=dimension, held_out_group=group, week=week, **metric(sub, cfg)))
    pd.DataFrame(stress_rows).to_csv(out / "held_out_group_metrics.csv", index=False)
    development = data[data.visit_week.isin(cfg["development_weeks"])]
    final, scores = select_model(development, cfg["unified_selector_candidates"], cfg)
    pd.DataFrame(scores).to_csv(out / "full_development_inner_selection.csv", index=False)
    formulas, verification = {}, {}
    for structure in cfg["models"]:
        fitted, _ = select_model(development, [structure], cfg)
        formula = fitted.formula()
        reconstruction = np.max(abs(evaluate_formula(data, formula)-fitted.predict(data)))
        assert reconstruction < 1e-9
        # Numerical implementation is independently checked against sklearn's
        # SVD ridge on exactly the same transformed features and weights.
        x = fitted.last_scale.transform(fitted.design(development))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            reference = Ridge(alpha=fitted.alpha, fit_intercept=not fitted.zero_anchor, solver="svd").fit(x, development.observed, sample_weight=fitted.weight)
        sx = fitted.last_scale.transform(fitted.design(data))
        reference_prediction = reference.intercept_ + np.einsum("ij,j->i", sx, reference.coef_, optimize=False)
        assert np.isfinite(reference_prediction).all()
        difference = np.max(abs(reference_prediction-fitted.predict_raw(data)))
        assert difference < 1e-8
        formulas[structure] = dict(structure=structure, alpha=fitted.alpha, formula=formula, reconstruction_max_abs=float(reconstruction), sklearn_reference_max_abs=float(difference), sklearn_reference_warnings=[str(w.message) for w in caught])
    artifact = dict(version="BSA-three-common-v1-20261017", status=cfg["status"], selected_structure=final.structure, alpha=final.alpha,
                    inputs=FEATURES, formula=final.formula(), training_weeks=cfg["development_weeks"], n_training_visits=len(development), n_training_patients=int(development.patient_uid.nunique()),
                    units="BSA percentage from 0 to 100; molecular inputs log2 concentrations in recorded pg/ml units from Summary.xlsx. Assay bridge has not been established.",
                    assay_processing="Positive Summary.xlsx exports were log2-transformed unchanged; no half-min or pseudocount replacement. < or > marked concentrations and nonpositive exports were missing, not exact inputs. Numeric positivity does not establish validated quantification; IL-5 software-bound sensitivity is reported separately.",
                    missing_policy="Reject incomplete/nonfinite inputs; no imputation of clinical outcome or deployment input.", output_range=[0,72],
                    no_stage_drug_history_inputs=True, selection="3-fold patient-inner mean per-patient absolute error; candidate structure and ridge penalty chosen together; all fitting uses W0/W12 only",
                    performance_scope="Repeated patient-held-out nested validation of a selection procedure; the single final full-development formula has not been independently validated.",
                    range_at_development={col:[float(development[col].min()),float(development[col].max())] for col in FEATURES})
    (out / "formula.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n")
    (root / "config/unified_bsa_three_v1_20261017.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n")
    (out / "all_candidate_formulas.json").write_text(json.dumps(formulas, ensure_ascii=False, indent=2)+"\n")
    old_path = root / "results/tables/update_20261002/frozen_bsa_three/validation_predictions.csv"
    old_difference = None
    if old_path.exists():
        old = pd.read_csv(old_path)
        new = predictions[predictions.model.eq("stage_interaction")]
        joined = new.merge(old[["patient_uid", "visit_week", "repeat", "fold", "prediction"]], on=["patient_uid", "visit_week", "repeat", "fold"], validate="one_to_one", suffixes=("", "_old"))
        assert len(joined) == len(new) == len(old)
        old_difference = float(np.max(abs(joined.prediction-joined.prediction_old)))
        assert old_difference < 1e-8
    verification.update(patient_disjoint=True, same_patient_same_folds=True, n_unique_patients=int(data.patient_uid.nunique()), development_patients=int(development.patient_uid.nunique()),
                        outcome_not_imputed=True, validation_not_training_fit=True, W24_absent_from_fitting_selection=True, repeats_scored_before_averaging=True,
                        zero_anchor_held_for_area_model=True, frozen_stage_predictions_reproduced=old_difference is not None, maximum_frozen_prediction_difference=old_difference,
                        patient_overlap_was_checked_in_inner_and_outer_folds=True)
    (out / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    examples = pd.DataFrame([{"bsa":10.,"log2_CCL17":10.,"log2_CCL26":7.,"log2_IL-5":1.}, {"bsa":0.,"log2_CCL17":10.,"log2_CCL26":7.,"log2_IL-5":1.}])
    examples["EASI_estimate"] = evaluate_formula(examples, final.formula())
    examples.to_csv(out / "calculator_examples.csv", index=False)
    metrics.query("group_type == 'overall'").to_csv(out / "overall_metrics.csv", index=False)
    historical_sources = audit_historical_targets(root, out, data)
    assay_sources = assay_audit_and_sensitivity(root, out, data, predictions, cfg)
    manifest = {"version":"2026-10-17", "seed":cfg["seed"], "config":str(cfg_path.relative_to(root)), "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
                "private_files":["predictions.csv","fit_audit_private.csv","published_matched_predictions_private.csv","held_out_group_predictions_private.csv","assay_sensitivity_predictions_private.csv"],
                "source_hashes":{str(path.relative_to(root)):sha(path) for path in [root/cfg["input_path"],cfg_path,Path(__file__),root/"src/cytot2t/curation/unified_bsa_three_calculator.py"] + ([old_path] if old_path.exists() else []) + historical_sources + assay_sources},
                "outputs":[{"path":str(path.relative_to(root)),"sha256":sha(path)} for path in sorted(out.iterdir()) if path.is_file() and path.name != "model_manifest.json"],
                "selected_structure":final.structure,"validation_checks":verification}
    (out / "model_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(metrics.query("group_type == 'overall'")[["model","week","n_patients","MAE","RMSE","rho","CCC","bias"]].to_string(index=False), flush=True)
    print("Final structure:",final.structure,"alpha:",final.alpha,flush=True)


if __name__ == "__main__":
    main()
