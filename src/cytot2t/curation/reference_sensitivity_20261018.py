"""Refit the locked EASI procedure after training-only reference exclusions.

This is robustness to exclusion, not correction of clinical truth. The primary
formula, results, predictions and outer patient folds are never overwritten.
Only aggregate outputs are written; patient-level predictions remain in memory.

Run with PYTHONPATH=src python3 -m cytot2t.curation.reference_sensitivity_20261018.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from cytot2t.curation import unified_model_20261017 as primary


REGIONS = ["easi_head_neck", "easi_upper_limb", "easi_trunk", "easi_lower_limb"]
MODELS = ["unified_selected", "bsa_area", "bsa_linear"]
METRICS = ["MAE", "RMSE", "bias", "rho", "CCC", "calibration_intercept", "calibration_slope"]
KEYS = ["patient_uid", "visit_week", "repeat", "fold"]
INTERVAL_SCOPE = "patient percentile bootstrap conditional on saved sensitivity/primary predictions; no fitting or selection repeated in bootstrap"


def batch_metrics(y, p):
    """Rows are bootstrap samples, columns are sampled patients at one visit."""
    e = p - y
    ym, pm = y.mean(axis=1), p.mean(axis=1)
    yc, pc = y - ym[:, None], p - pm[:, None]
    cov = (yc * pc).mean(axis=1)
    vy, vp = (yc ** 2).mean(axis=1), (pc ** 2).mean(axis=1)
    yr, pr = rankdata(y, axis=1), rankdata(p, axis=1)
    yr -= yr.mean(axis=1)[:, None]
    pr -= pr.mean(axis=1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = cov / vp
        rho = (yr * pr).mean(axis=1) / np.sqrt((yr ** 2).mean(axis=1) * (pr ** 2).mean(axis=1))
        ccc = 2 * cov / (vy + vp + (ym - pm) ** 2)
    return {"MAE": abs(e).mean(axis=1), "RMSE": np.sqrt((e ** 2).mean(axis=1)),
            "bias": e.mean(axis=1), "rho": rho, "CCC": ccc,
            "calibration_intercept": ym - slope * pm, "calibration_slope": slope}


def prediction_arrays(frame, ids):
    wide = frame.pivot(index="patient_uid", columns="repeat", values="prediction").reindex(ids)
    truth = frame.groupby("patient_uid")["observed"].first().reindex(ids).to_numpy()
    assert frame.groupby("patient_uid")["observed"].nunique().eq(1).all()
    assert np.isfinite(wide.to_numpy()).all() and np.isfinite(truth).all()
    return truth, wide.to_numpy().T


def summarize(predictions, reference, cfg):
    metrics, per_repeat, intervals, contrasts = [], [], [], []
    for evaluation, wanted in [("all_original_evaluable", reference.index),
                               ("strict_reference_consistent", reference.index[reference.strict_consistent])]:
        selected = predictions.merge(reference.loc[wanted, ["patient_uid", "visit_week"]],
                                     on=["patient_uid", "visit_week"], validate="many_to_one")
        for week, visit in selected.groupby("visit_week"):
            ids = np.array(sorted(visit.patient_uid.unique()))
            rng = np.random.default_rng(cfg["seed"])
            draws = rng.choice(np.arange(len(ids)), (cfg["bootstrap_replicates"], len(ids)), replace=True)
            estimates, samples = {}, {}
            for (analysis, model), frame in visit.groupby(["analysis", "model"]):
                y, p = prediction_arrays(frame, ids)
                repeat_values = [primary.scalar_metrics(y, pp) for pp in p]
                value = {name: float(np.mean([r[name] for r in repeat_values])) for name in METRICS}
                boot = {name: np.zeros(len(draws)) for name in METRICS}
                for pp in p:
                    vectorized = batch_metrics(y[None, :], pp[None, :])
                    for name in METRICS:
                        assert np.allclose(vectorized[name][0], primary.scalar_metrics(y, pp)[name], rtol=0, atol=1e-10, equal_nan=True)
                    values = batch_metrics(y[draws], pp[draws])
                    for name in METRICS:
                        boot[name] += values[name] / len(p)
                estimates[(analysis, model)], samples[(analysis, model)] = value, boot
                row = dict(evaluation=evaluation, week=int(week), analysis=analysis, model=model,
                           n_patients=len(ids), n_repeats=len(p), **value)
                for name in METRICS:
                    lo, hi = np.nanquantile(boot[name], [.025, .975])
                    intervals.append(dict(evaluation=evaluation, week=int(week), analysis=analysis, model=model,
                                          metric=name, estimate=value[name], ci_low=lo, ci_high=hi,
                                          n_patients=len(ids), n_repeats=len(p), bootstrap_replicates=len(draws),
                                          valid_bootstrap_replicates=int(np.isfinite(boot[name]).sum()), interval_scope=INTERVAL_SCOPE))
                    if name == "MAE":
                        row["MAE_lo"], row["MAE_hi"] = lo, hi
                metrics.append(row)
                for repeat, vals in enumerate(repeat_values):
                    per_repeat.append(dict(evaluation=evaluation, week=int(week), analysis=analysis, model=model,
                                           repeat=repeat, n_patients=len(ids), **{name: vals[name] for name in METRICS}))
            pairs = [(("reference_exclusion", model), ("primary", model), "training_exclusion_minus_primary") for model in MODELS]
            pairs += [((analysis, "unified_selected"), (analysis, control), "selected_minus_matched_control")
                      for analysis in ["primary", "reference_exclusion"] for control in ["bsa_area", "bsa_linear"]]
            for candidate, control, kind in pairs:
                for name in METRICS:
                    delta = samples[candidate][name] - samples[control][name]
                    lo, hi = np.nanquantile(delta, [.025, .975])
                    contrasts.append(dict(evaluation=evaluation, week=int(week), contrast_type=kind,
                                          candidate_analysis=candidate[0], candidate=candidate[1],
                                          reference_analysis=control[0], reference=control[1], metric=name,
                                          difference=estimates[candidate][name]-estimates[control][name],
                                          ci_low=lo, ci_high=hi, n_patients=len(ids),
                                          n_repeats=cfg["outer_repeats"], bootstrap_replicates=len(draws),
                                          difference_definition="candidate minus reference; positive MAE/RMSE is greater error", interval_scope=INTERVAL_SCOPE))
    return [pd.DataFrame(x) for x in [metrics, per_repeat, intervals, contrasts]]


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x) for x in row) + " |")
    return "\n".join(lines)


def run(root):
    root = Path(root).resolve()
    out = root / "results/tables/update_20261017/reference_sensitivity"
    report = root / "reports/manuscript_review/reference_sensitivity.md"
    cfg_path = root / "config/unified_model_20261017.json"
    cfg = json.loads(cfg_path.read_text())
    original_dir = root / cfg["output_path"]
    # Guard all existing primary artifacts, its deployed configuration and inputs.
    protected = sorted(p for p in original_dir.rglob("*") if p.is_file())
    protected += [cfg_path, root / cfg["input_path"], root / "config/unified_bsa_three_v1_20261017.json"]
    protected_hashes = {str(p.relative_to(root)): primary.sha(p) for p in protected}
    source = pd.read_csv(root / cfg["input_path"])
    assert not source.duplicated(["patient_uid", "visit_week"]).any()
    complete = source[REGIONS].notna().all(axis=1)
    gap = source[REGIONS].sum(axis=1, min_count=4) - source.observed
    source["known_inconsistent"] = complete & gap.abs().gt(.15)
    source["strict_consistent"] = complete & gap.abs().le(.01)
    assert np.array_equal(source.known_inconsistent, source.source_contradiction.astype(bool))
    reference = source[["patient_uid", "visit_week", "known_inconsistent", "strict_consistent"]].copy()
    exclusions = []
    for week, g in source.groupby("visit_week"):
        exclusions.append(dict(week=int(week), n_original_visits=len(g),
                               n_complete_regional_reference=int(complete.loc[g.index].sum()),
                               n_known_inconsistent=int(g.known_inconsistent.sum()),
                               n_missing_regional_reference=int((~complete.loc[g.index]).sum()),
                               n_strict_reference_consistent=int(g.strict_consistent.sum()),
                               n_development_excluded=int(g.known_inconsistent.sum()) if week in cfg["development_weeks"] else 0,
                               max_absolute_reference_gap=float(gap.loc[g.index].abs().max())))
    assert [r["n_known_inconsistent"] for r in exclusions] == [8, 2, 0]
    saved = pd.read_csv(original_dir / "predictions.csv")
    saved = saved[saved.model.isin(MODELS)].copy()
    saved["analysis"] = "primary"
    joined = saved.merge(source[["patient_uid", "visit_week", "observed"]], on=["patient_uid", "visit_week"], validate="many_to_one", suffixes=("", "_source"))
    assert len(joined) == len(saved) and np.allclose(joined.observed, joined.observed_source, rtol=0, atol=0)
    ids = np.array(sorted(source.patient_uid.unique()))
    outputs, audit, scores = [], [], []
    reproduction_max = 0.0
    for repeat in range(cfg["outer_repeats"]):
        for fold, (tr, te) in enumerate(KFold(cfg["outer_folds"], shuffle=True, random_state=cfg["seed"]+repeat).split(ids)):
            train_original = source[source.patient_uid.isin(ids[tr]) & source.visit_week.isin(cfg["development_weeks"])]
            test = source[source.patient_uid.isin(ids[te])]
            expected = saved[saved.model.eq("unified_selected") & saved["repeat"].eq(repeat) & saved.fold.eq(fold)]
            assert set(test.patient_uid) == set(expected.patient_uid)
            assert set(zip(test.patient_uid, test.visit_week)) == set(zip(expected.patient_uid, expected.visit_week))
            train_restricted = train_original.loc[~train_original.known_inconsistent]
            assert set(train_restricted.patient_uid).isdisjoint(test.patient_uid)
            assert train_restricted.visit_week.max() == 12
            assert train_restricted.known_inconsistent.sum() == 0
            for analysis, train in [("primary_reproduction", train_original), ("reference_exclusion", train_restricted)]:
                for model in MODELS:
                    structures = cfg["unified_selector_candidates"] if model == "unified_selected" else [model]
                    fitted, inner = primary.select_model(train, structures, cfg)
                    q = test[primary.PRIVATE_COLUMNS].copy()
                    q["prediction"] = fitted.predict(test)
                    q["repeat"], q["fold"], q["model"], q["analysis"] = repeat, fold, model, analysis
                    if analysis == "primary_reproduction":
                        comparator = saved[saved.model.eq(model) & saved["repeat"].eq(repeat) & saved.fold.eq(fold)]
                        check = q.merge(comparator[KEYS + ["prediction"]], on=KEYS, validate="one_to_one", suffixes=("", "_saved"))
                        assert len(check) == len(q)
                        difference = float(abs(check.prediction-check.prediction_saved).max())
                        reproduction_max = max(reproduction_max, difference)
                        assert difference < 1e-10
                    else:
                        outputs.append(q)
                        scores.extend(dict(repeat=repeat, fold=fold, model=model, **row) for row in inner)
                    audit.append(dict(analysis=analysis, repeat=repeat, fold=fold, model=model,
                                      selected_structure=fitted.structure, alpha=fitted.alpha,
                                      n_training_visits=len(train), n_training_patients=int(train.patient_uid.nunique()),
                                      n_original_training_visits=len(train_original),
                                      n_excluded_training_visits=len(train_original)-len(train),
                                      n_unknown_regional_reference_retained=int(train[REGIONS].isna().any(axis=1).sum()),
                                      max_training_week=int(train.visit_week.max()), n_test_visits=len(test),
                                      patient_overlap=0, test_outcomes_unchanged=True))
            print(f"Reference sensitivity completed repeat {repeat+1}, fold {fold+1}", flush=True)
    sensitivity = pd.concat(outputs, ignore_index=True)
    prediction = pd.concat([saved, sensitivity], ignore_index=True)
    assert not prediction.duplicated(["analysis", "model"] + KEYS).any()
    assert sensitivity.groupby(["model", "repeat"]).size().eq(len(source)).all()
    metrics, repeats, uncertainty, contrasts = summarize(prediction, reference, cfg)
    audit = pd.DataFrame(audit)
    structure_counts = audit[audit.model.eq("unified_selected")].groupby(["analysis", "selected_structure"]).size().rename("outer_fits").reset_index()
    for p in protected:
        assert primary.sha(p) == protected_hashes[str(p.relative_to(root))], f"Protected artifact changed: {p}"
    out.mkdir(parents=True, exist_ok=True)
    tables = {"reference_exclusion_counts": pd.DataFrame(exclusions), "fit_audit": audit,
              "inner_selection": pd.DataFrame(scores), "selected_structure_counts": structure_counts,
              "metrics": metrics, "metrics_per_repeat": repeats, "uncertainty": uncertainty, "paired_differences": contrasts}
    for name, table in tables.items():
        assert not set(table.columns) & {"patient_uid", "record_id", "adverse_event_text"}
        table.to_csv(out / f"{name}.csv", index=False)
    checks = dict(status="PASS", known_inconsistent_counts={str(r["week"]): r["n_known_inconsistent"] for r in exclusions},
                  development_original_visits=int(source.visit_week.isin(cfg["development_weeks"]).sum()),
                  development_excluded_visits=int((source.known_inconsistent & source.visit_week.isin(cfg["development_weeks"])).sum()),
                  all_original_evaluation_visits_retained=True, original_patient_outer_folds_verified=True,
                  training_only_exclusion=True, unknown_regional_reference_retained=True,
                  clinical_truth_unchanged=True, marker_panel_fixed=True, patient_disjoint=True,
                  preprocessing_and_structure_penalty_selection_repeated=True,
                  primary_reproduction_max_absolute_prediction_difference=reproduction_max,
                  all_existing_primary_artifact_hashes_unchanged=True,
                  primary_protected_file_count=len(protected), public_outputs_are_aggregate_only=True,
                  bootstrap_replicates=cfg["bootstrap_replicates"], bootstrap_seed=cfg["seed"], interval_scope=INTERVAL_SCOPE)
    (out / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    main = metrics[metrics.model.eq("unified_selected")]
    delta = contrasts[(contrasts.metric == "MAE") & (contrasts.contrast_type == "training_exclusion_minus_primary") & contrasts.candidate.eq("unified_selected")]
    control = contrasts[(contrasts.metric == "MAE") & (contrasts.contrast_type == "selected_minus_matched_control") & contrasts.reference.eq("bsa_area")]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n\n".join([
        "# Training-only EASI reference exclusion sensitivity",
        "Completed 18 October 2026. This analysis measures robustness to excluding known inconsistent development records. It neither corrects clinical truth nor replaces the frozen primary equation or predictions.",
        "The exclusion was fixed from complete four-region records with absolute regional-sum minus recorded-total discrepancy greater than 0.15 EASI points. Exactly eight W0 and two W12 visits were excluded from fitting only. Missing regional references were retained because they are not known inconsistent. All original held-out W0/W12 and W24 outcomes remain in evaluation. A separate strict evaluation uses complete four-region references with discrepancy at most 0.01, matching the earlier evaluation-only sensitivity.",
        "The original patient outer folds were verified against saved primary predictions. The same fixed marker panel, candidate structures, penalties, inner patient folds, patient fitting weights and training-only preprocessing were reused through unified_model_20261017.select_model. W24 never enters fitting or selection. Original fits for all three evaluated procedures were independently reproduced before sensitivity fitting. No primary artifact was overwritten.",
        "## Exclusions and evaluation populations", markdown_table(tables["reference_exclusion_counts"], ["week", "n_original_visits", "n_complete_regional_reference", "n_known_inconsistent", "n_missing_regional_reference", "n_strict_reference_consistent", "n_development_excluded"]),
        "## Selected-procedure performance", markdown_table(main, ["evaluation", "week", "analysis", "n_patients", "MAE", "MAE_lo", "MAE_hi", "RMSE", "bias", "rho", "CCC", "calibration_slope"]),
        "## Paired MAE changes from primary", markdown_table(delta, ["evaluation", "week", "difference", "ci_low", "ci_high"]),
        "Positive differences mean greater error after training exclusion. These comparisons retain identical evaluation patients and truth.",
        "## Matched BSA-only area comparison", markdown_table(control, ["evaluation", "week", "candidate_analysis", "difference", "ci_low", "ci_high"]),
        "The difference is selected procedure minus BSA-only area MAE within the named analysis. Both procedures are refitted under the same training exclusion. The BSA-linear sensitivity, all requested metric estimates and their paired differences are retained in the CSV collection.",
        "## Outer-fold structure selection", markdown_table(structure_counts, ["analysis", "selected_structure", "outer_fits"]),
        "## Uncertainty and limits", f"All metrics are calculated within each of two outer repetitions before averaging. Paired 95% percentile intervals use {cfg['bootstrap_replicates']} patient resamples with seed {cfg['seed']}; the same sampled patients are carried across repetitions, analyses and comparators. The intervals condition on the saved predictions generated by each run and do not repeat model fitting or selection. They do not cover earlier adaptive marker exploration, resolve which clinical reference field is correct, establish an assay bridge or provide independent external validation.",
        "## Verification", f"Original prediction reproduction maximum absolute difference was {reproduction_max:.3g}. All {len(protected)} guarded primary artifacts and inputs retained identical SHA256 hashes. Outputs contain aggregate tables only. No source EASI value was changed, no residual-based exclusion was made and no new final formula was released.",
        "## Endpoint and bootstrap source verification",
        "PP-NRS decrease of at least four points requires baseline PP-NRS >=4, then compares baseline minus current PP-NRS with >=4. Missing follow-up yields unknown via rule(). The implemented rule is clinical_evidence_20261017.py line 90, with rule() at lines 55 to 58. This is an eligibility restriction, not imputation.",
        "The primary weighted clinical bootstrap uses seed 20261017 (clinical_evidence_20261017.py line 33; sampling and propensity refitting at lines 168 to 173). The W2 baseline-itch sensitivity reruns all three model labels with seed 20261018 (t2t_paper/revision_checks_20261018.py line 26; model labels at lines 158 to 160; sampling/refitting at lines 170 to 176). Both default to 1,000 treatment-stratified patient resamples, refit standardized multinomial logistic C=1, use normal-Wald P values from bootstrap standard errors, and use a Holm family of three pairwise comparisons within the endpoint/visit. Primary family code is at clinical_evidence_20261017.py lines 190 to 207; sensitivity family code is at revision_checks_20261018.py lines 191 to 198. Thus original_full_cohort in the latter table is a distinct Monte Carlo seed rerun of the same point estimate, not a new Holm family. Existing numbers were not edited.",
        "Run command: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m cytot2t.curation.reference_sensitivity_20261018 --root /Users/yanax/Desktop/yao"
    ]) + "\n")
    manifest = dict(status="PASS", purpose="robustness to training-only reference exclusion, not corrected truth",
                    source_hashes={str(p.relative_to(root)): primary.sha(p) for p in [Path(__file__), cfg_path, root / cfg["input_path"], original_dir / "predictions.csv", Path(primary.__file__)]},
                    protected_primary_hashes=protected_hashes, config=cfg, verification=checks,
                    outputs=[dict(path=str(p.relative_to(root)), sha256=primary.sha(p)) for p in sorted(out.glob("*")) if p.is_file() and p.name != "manifest.json"],
                    report=dict(path=str(report.relative_to(root)), sha256=primary.sha(report)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(main[["evaluation", "week", "analysis", "MAE", "RMSE", "bias", "rho", "CCC", "calibration_slope"]].to_string(index=False), flush=True)
    print("Verification PASS. Primary artifacts unchanged; aggregate outputs only.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.root)
