"""Observed molecular summaries and explicitly assumed EOS interpolation.

All estimates are derived outputs. The clinical registry and assay files remain
unchanged. Week-4 dupilumab EOS is unobserved and cannot validate a time curve.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ARMS = ["dupilumab", "abrocitinib_100mg", "abrocitinib_200mg"]
WEEKS = [0, 12, 24]
SHAPES = ["log1p_linear_time", "linear_original_scale",
          "early_log_time_log1p", "late_quadratic_time_log1p"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpolation(before, after, fraction, shape, endpoint_week=12):
    before, after = np.asarray(before, float), np.asarray(after, float)
    if shape == "linear_original_scale":
        return (1 - fraction) * before + fraction * after
    if shape == "early_log_time_log1p":
        fraction = np.log1p(endpoint_week * fraction) / np.log1p(endpoint_week)
    elif shape == "late_quadratic_time_log1p":
        fraction = fraction ** 2
    elif shape != "log1p_linear_time":
        raise ValueError(shape)
    return np.expm1((1 - fraction) * np.log1p(before) + fraction * np.log1p(after))


def bootstrap_summary(x, cfg, seed):
    x = np.asarray(x, float)
    assert np.isfinite(x).all() and len(x)
    draws = np.random.default_rng(seed).integers(len(x), size=(cfg["bootstrap_replicates"], len(x)))
    medians = np.median(x[draws], axis=1)
    means = x[draws].mean(axis=1)
    return dict(n=len(x), median=float(np.median(x)), q25=float(np.quantile(x, .25)),
                q75=float(np.quantile(x, .75)), mean=float(x.mean()),
                median_ci_low=float(np.quantile(medians, .025)),
                median_ci_high=float(np.quantile(medians, .975)),
                mean_ci_low=float(np.quantile(means, .025)),
                mean_ci_high=float(np.quantile(means, .975)))


def eos_estimates(c, cfg, out):
    coverage, trajectories, estimated, backchecks = [], [], [], []
    seed = cfg["seed"]
    for arm, g in c[c.visit_week.isin([0, 4, 12, 24])].groupby("treatment"):
        for (center, week), h in g.groupby(["center", "visit_week"]):
            coverage.append(dict(treatment=arm, center=center, week=week,
                                 n_initial=h.patient_uid.nunique(), n_observed=int(h.eo_count_numeric.notna().sum())))
    d = c[c.treatment.eq("dupilumab") & c.visit_week.isin([0, 4, 12, 24])]
    pivot = d.pivot(index="patient_uid", columns="visit_week", values="eo_count_numeric")
    assert pivot[4].notna().sum() == 0
    for pi, (population, weeks) in enumerate([("paired_W0_W12", [0, 12]),
                                             ("complete_W0_W12_W24", WEEKS)]):
        selected = pivot[weeks].dropna()
        assert len(selected) and selected.ge(0).all().all()
        for week in weeks:
            trajectories.append(dict(population=population, treatment="dupilumab", week=week,
                                     kind="observed", method="observed endpoint-complete patients",
                                     **bootstrap_summary(selected[week], cfg, seed + pi)))
        for shape in SHAPES:
            x = interpolation(selected[0], selected[12], 1/3, shape)
            row = dict(population=population, treatment="dupilumab", week=4, kind="estimated_not_observed",
                       method=shape, primary=shape == SHAPES[0],
                       **bootstrap_summary(x, cfg, seed + pi),
                       units="recorded EOS units", conditioning="endpoint-complete selection and assumed temporal shape")
            estimated.append(row)
            if shape == SHAPES[0]:
                trajectories.append(row)
    def backcheck(z, arm, target, last, population):
        for shape in SHAPES:
            pred = interpolation(z[0], z[last], target / last, shape, last)
            error = pred - z[target].to_numpy()
            lohi = bootstrap_summary(np.abs(error), cfg, seed + target + last)
            backchecks.append(dict(treatment=arm, population=population, masked_week=target,
                                   endpoint_week=last, method=shape, n=len(z), MAE=float(np.abs(error).mean()),
                                   RMSE=float(np.sqrt(np.mean(error ** 2))), bias=float(error.mean()),
                                   MAE_ci_low=lohi["mean_ci_low"], MAE_ci_high=lohi["mean_ci_high"],
                                   scope="observed-time interpolation check; not validation of dupilumab W4"))
    backcheck(pivot[WEEKS].dropna(), "dupilumab", 12, 24, "complete_W0_W12_W24")
    for arm in ARMS[1:]:
        p = c[c.treatment.eq(arm)].pivot(index="patient_uid", columns="visit_week", values="eo_count_numeric")
        z = p[[0, 4, 12]].dropna()
        if len(z):
            backcheck(z, arm, 4, 12, "complete_W0_W4_W12")
    pd.DataFrame(coverage).to_csv(out / "eos_coverage.csv", index=False)
    pd.DataFrame(estimated).to_csv(out / "eos_w4_estimates.csv", index=False)
    pd.DataFrame(trajectories).to_csv(out / "eos_endpoint_complete_trajectory.csv", index=False)
    pd.DataFrame(backchecks).to_csv(out / "eos_interpolation_backchecks.csv", index=False)
    return pd.DataFrame(estimated), pd.DataFrame(backchecks)


def marker_baseline(c, assay, registry, cfg, out):
    baseline = c[c.visit_week.eq(0)][["patient_uid", "treatment"]]
    assert not baseline.patient_uid.duplicated().any()
    a = assay[assay.visit_week.eq(0)].merge(baseline, on="patient_uid", validate="many_to_one")
    assert not a.duplicated(["patient_uid", "analyte"]).any()
    initial = baseline.treatment.value_counts()
    rows = []
    for marker in registry.analyte:
        for arm in ARMS:
            g = a[a.analyte.eq(marker) & a.treatment.eq(arm)]
            h = g[g.state.eq("observed") & np.isfinite(g.concentration) & g.concentration.gt(0)]
            x = h.concentration.to_numpy(float)
            row = dict(marker=marker, treatment=arm, unit="recorded pg/mL", n_initial=int(initial[arm]),
                       n_assay_records=len(g), n_numeric=len(x),
                       n_no_assay_record=int(initial[arm]) - len(g),
                       n_above_uloq=int(g.state.eq("above_uloq").sum()),
                       n_below_lloq=int(g.state.eq("below_lloq").sum()),
                       n_not_measured=int(g.state.eq("not_measured").sum()),
                       n_assay_failure=int(g.state.eq("assay_failure").sum()),
                       n_other_state=int((~g.state.isin(["observed", "above_uloq", "below_lloq", "not_measured", "assay_failure"])).sum()),
                       median=float(np.median(x)) if len(x) else np.nan,
                       q25=float(np.quantile(x, .25)) if len(x) else np.nan,
                       q75=float(np.quantile(x, .75)) if len(x) else np.nan,
                       minimum=float(np.min(x)) if len(x) else np.nan,
                       maximum=float(np.max(x)) if len(x) else np.nan)
            assert (row["n_numeric"] + sum(row[k] for k in ["n_above_uloq", "n_below_lloq", "n_not_measured", "n_assay_failure", "n_other_state"])) == len(g)
            rows.append(row)
    result = pd.DataFrame(rows)
    display = []
    for marker in registry.analyte:
        row = dict(marker=marker, unit="recorded pg/mL")
        for arm in ARMS:
            z = result[result.marker.eq(marker) & result.treatment.eq(arm)].iloc[0]
            row[arm + "_n"] = int(z.n_numeric)
            row[arm + "_median_q1_q3"] = f"{z['median']:.2f} ({z.q25:.2f}–{z.q75:.2f})" if z.n_numeric else "Not estimable"
            row[arm + "_outside_quantification"] = int(z.n_above_uloq + z.n_below_lloq)
        display.append(row)
    result.to_csv(out / "baseline_marker_summary.csv", index=False)
    pd.DataFrame(display).to_csv(out / "baseline_marker_display.csv", index=False)
    return result


def signflip_p(x, rng, exact_refinement=False, monte_carlo_draws=49999):
    x = np.asarray(x, float)
    if np.all(x == 0):
        return 1., "all_changes_zero"
    n = len(x)
    observed = abs(np.mean(x))
    if exact_refinement:
        count, total = 0, 2 ** (n - 1)
        for start in range(0, total, 50000):
            codes = np.arange(start, min(start + 50000, total), dtype=np.uint32)[:, None]
            signs = 2 * ((codes >> np.arange(n - 1)) & 1).astype(float) - 1
            statistic = np.abs((x[0] + np.einsum('ij,j->i', signs, x[1:])) / n)
            count += np.sum(statistic >= observed - 1e-12)
        return float(count / total), "exact_sign_flip_global_symmetry"
    if n <= 15:
        codes = np.arange(2 ** n, dtype=np.uint32)[:, None]
        signs = 2 * ((codes >> np.arange(n)) & 1).astype(float) - 1
        return float(np.mean(np.abs(np.einsum('ij,j->i', signs, x) / n) >= observed - 1e-12)), "exact_sign_flip"
    count, total = 0, monte_carlo_draws
    for start in range(0, total, 5000):
        signs = rng.choice([-1., 1.], size=(min(5000, total - start), n))
        count += np.sum(np.abs(np.einsum('ij,j->i', signs, x) / n) >= observed - 1e-12)
    return float((1 + count) / (total + 1)), f"Monte_Carlo_sign_flip_{total}"


def marker_triplets(c, molecular, registry, cfg, out):
    d = c[c.visit_week.isin(WEEKS)][["patient_uid", "visit_week", "treatment"]].merge(
        molecular, on=["patient_uid", "visit_week"], validate="one_to_one")
    rows, changes = [], {}
    rng = np.random.default_rng(cfg["seed"])
    for marker in registry.analyte:
        for arm in ARMS:
            p = d[d.treatment.eq(arm)].pivot(index="patient_uid", columns="visit_week", values="log2_" + marker)
            p = p.reindex(columns=WEEKS).dropna()
            n = len(p)
            for week in WEEKS:
                change = (p[week] - p[0]).to_numpy(float)
                mean = float(change.mean()) if n else np.nan
                width = stats.t.ppf(.975, n - 1) * change.std(ddof=1) / np.sqrt(n) if n >= 5 else np.nan
                pvalue, method = (np.nan, "no_inference")
                if week and n >= 5:
                    pvalue, method = signflip_p(change, rng)
                row = dict(marker=marker, treatment=arm, week=week, n=n, effect=mean,
                           ci_low=mean - width, ci_high=mean + width,
                           geometric_pair_ratio=2 ** mean if n else np.nan,
                           baseline_geometric_concentration=2 ** p[0].mean() if n else np.nan,
                           baseline_median_concentration=float(np.median(np.exp2(p[0]))) if n else np.nan,
                           p=pvalue, test_method=method,
                           status=("baseline_reference" if week == 0 else "estimated") if n >= 5 else ("descriptive_n_lt5" if n else "no_complete_triplet"),
                           family="23_markers_x_3_initial_treatments_x_2_baseline_contrasts",
                           family_size=138, q=np.nan, significant=False,
                           definition="Identical complete three-visit patients within marker and treatment; later minus own W0 log2 concentration")
                changes[len(rows)] = change
                rows.append(row)
    result = pd.DataFrame(rows)
    test_idx = result.index[result.week.ne(0)]
    assert len(test_idx) == 138
    q = multipletests(result.loc[test_idx, "p"].fillna(1), method="fdr_bh")[1]
    result.loc[test_idx, "q"] = np.where(result.loc[test_idx, "p"].notna(), q, np.nan)
    result['p_initial'] = result.p
    result['q_initial'] = result.q
    refine = result.index[result.q.between(.04, .06) & result.test_method.str.startswith('Monte_Carlo')]
    for index in refine:
        x = changes[index]
        value, method = signflip_p(x, rng, exact_refinement=len(x) <= 24, monte_carlo_draws=1999999)
        result.loc[index, ['p', 'test_method']] = [value, method]
    q = multipletests(result.loc[test_idx, "p"].fillna(1), method="fdr_bh")[1]
    result.loc[test_idx, "q"] = np.where(result.loc[test_idx, "p"].notna(), q, np.nan)
    result["significant"] = result.q.lt(.05)
    result["star"] = np.where(result.significant, "*", "")
    result.to_csv(out / "molecular_three_visit_changes.csv", index=False)
    return result


def joint_low_activity(c, cfg, out):
    """Observed single-visit IEC EASI-and-itch states; no outcome imputation."""
    settings = cfg["joint_low_activity"]
    cohort = c[c.visit_week.eq(0)][["patient_uid", "treatment"]].copy()
    assert not cohort.patient_uid.duplicated().any() and len(cohort) == 163
    visits = c[c.visit_week.isin(settings["weeks"])][["patient_uid", "visit_week", "easi_total", "pp_nrs"]]
    assert not visits.duplicated(["patient_uid", "visit_week"]).any()
    complete_grid = cohort.merge(pd.DataFrame({"visit_week": settings["weeks"]}), how="cross")
    complete_grid = complete_grid.merge(visits, on=["patient_uid", "visit_week"], how="left", validate="one_to_one")
    for col, maximum in [("easi_total", 72), ("pp_nrs", 10)]:
        values = complete_grid[col].dropna()
        assert np.isfinite(values).all() and values.between(0, maximum).all()
    rows = []
    for arm in ARMS + ["all_initial_treatments"]:
        population = complete_grid if arm == "all_initial_treatments" else complete_grid[complete_grid.treatment.eq(arm)]
        for week, d in population.groupby("visit_week", sort=True):
            for state, thresholds in settings["states"].items():
                easi, itch = d.easi_total, d.pp_nrs
                both = easi.notna() & itch.notna()
                success = both & easi.le(thresholds["easi_max"]) & itch.le(thresholds["pp_nrs_max"])
                # An observed failed component proves AND=false, even if its partner is missing.
                failed = (easi.notna() & easi.gt(thresholds["easi_max"])) | (itch.notna() & itch.gt(thresholds["pp_nrs_max"]))
                unknown = ~(success | failed)
                total, n, k = len(d), int(both.sum()), int(success.sum())
                ci = stats.binomtest(k, n).proportion_ci(confidence_level=.95, method="wilson") if n else None
                rows.append(dict(treatment=arm, week=int(week), state=state,
                                 easi_threshold=thresholds["easi_max"], pp_nrs_threshold=thresholds["pp_nrs_max"],
                                 n_success=k, n_joint_observed=n, N_initial=total,
                                 proportion=k/n if n else np.nan,
                                 ci_low=float(ci.low) if ci else np.nan, ci_high=float(ci.high) if ci else np.nan,
                                 missing_easi=int(easi.isna().sum()), missing_pp_nrs=int(itch.isna().sum()),
                                 missing_both=int((easi.isna() & itch.isna()).sum()), n_not_joint=total-n,
                                 known_false=int(failed.sum()), unknown=int(unknown.sum()),
                                 full_cohort_lower_bound=k/total,
                                 full_cohort_upper_bound=(k+int(unknown.sum()))/total,
                                 analysis="post_hoc_single_visit_IEC_compatible_EASI_and_itch",
                                 denominator="both EASI and PP-NRS observed at this visit",
                                 interval="95% Wilson score interval for the observed-joint proportion",
                                 bounds="All unresolved AND states fail versus succeed; observed failed components remain failures",
                                 interpretation="Single-visit state, not full AHEAD MDA or sustained disease control",
                                 source_doi=settings["source_doi"]))
    result = pd.DataFrame(rows)
    group_rows = result[result.treatment.isin(ARMS)]
    overall_rows = result[~result.treatment.isin(ARMS)]
    assert len(group_rows) == 24 and len(overall_rows) == 8
    assert (result.n_success + result.known_false + result.unknown == result.N_initial).all()
    assert (result.n_success <= result.n_joint_observed).all()
    assert (result.unknown <= result.n_not_joint).all()
    assert (result.full_cohort_lower_bound <= result.full_cohort_upper_bound).all()
    for (arm, week), g in result.groupby(["treatment", "week"]):
        x = g.set_index("state")
        assert x.loc["vLDA", "n_success"] <= x.loc["LDA", "n_success"]
        assert x.n_joint_observed.nunique() == 1
    group_rows.to_csv(out / "joint_low_activity_rates.csv", index=False)
    overall_rows.to_csv(out / "joint_low_activity_rates_overall.csv", index=False)
    return result


def verify(c, assay, molecular, baseline, triplets, cfg, root):
    checks = {}
    assert c.patient_uid.nunique() == 163
    assert not c.duplicated(["patient_uid", "visit_week"]).any()
    checks["clinical_cohort_163_unique_patients"] = True
    source = pd.read_csv(root / "results/tables/update_20261017/clinical/molecular_three_visit_paired.csv")
    merged = triplets.merge(source, on=["marker", "treatment", "week"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(merged) == 207 and (merged.n_new == merged.n_old).all()
    for column in ["effect", "ci_low", "ci_high", "geometric_pair_ratio"]:
        assert np.allclose(merged[column + "_new"], merged[column + "_old"], equal_nan=True, atol=1e-11)
    checks["all_207_triplet_rows_reproduce_existing_population_and_geometry"] = True
    assert triplets.groupby(["marker", "treatment"]).n.nunique().eq(1).all()
    assert triplets[triplets.week.eq(0)].effect.dropna().eq(0).all()
    assert not triplets[triplets.n.lt(5)].p.notna().any()
    checks["patient_sets_fixed_across_three_visits_and_no_small_n_inference"] = True
    a = assay[assay.state.eq("observed") & assay.concentration.gt(0) & assay.visit_week.isin(WEEKS)].copy()
    long = molecular.melt(id_vars=["patient_uid", "visit_week"], value_vars=["log2_" + m for m in baseline.marker.unique()], var_name="analyte", value_name="log_value")
    long["analyte"] = long.analyte.str[5:]
    comp = a.merge(long, on=["patient_uid", "visit_week", "analyte"], validate="one_to_one")
    assert np.allclose(np.log2(comp.concentration), comp.log_value, equal_nan=False, atol=1e-12)
    checks["raw_concentrations_match_molecular_log_values"] = int(len(comp))
    existing = pd.read_csv(root / "results/tables/update_20261017/clinical/molecular_distributions.csv")
    b = baseline.merge(existing[existing.week.eq(0)], on=["marker", "treatment"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(b) == 69 and (b.n_numeric == b.n).all()
    checks["all_69_baseline_counts_match_existing_figure"] = True
    for method in SHAPES:
        assert np.allclose(interpolation([.1, .2], [.9, .8], 0, method), [.1, .2])
        assert np.allclose(interpolation([.1, .2], [.9, .8], 1, method), [.9, .8])
        mid = interpolation([.1, .8], [.9, .2], 1/3, method)
        assert np.all(mid >= [.1, .2]) and np.all(mid <= [.9, .8])
    checks["interpolation_has_correct_endpoints_and_nonnegative_bounds"] = True
    p, _ = signflip_p(np.ones(5), np.random.default_rng(1))
    assert p == 2/32
    checks["exact_signflip_known_case"] = True
    return checks


def run(root):
    plan = root / "config/laboratory_molecular_revision_20261107.json"
    cfg = json.loads(plan.read_text())
    plan_hash = digest(plan)
    for path, wanted in cfg["inputs"].items():
        assert digest(root / path) == wanted, path
    out, report = root / cfg["outputs"], root / cfg["report"]
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(root / "results/tables/update_20260925/clinical.csv")
    c = raw[raw.age.ge(12) & ~raw.dose_ambiguous].copy()
    assay = pd.read_csv(root / "results/tables/redevelopment/assay_long.csv")
    molecular = pd.read_csv(root / "results/tables/redevelopment/molecular.csv")
    registry = pd.read_csv(root / "config/marker_registry.csv")
    estimates, backchecks = eos_estimates(c, cfg, out)
    baseline = marker_baseline(c, assay, registry, cfg, out)
    triplets = marker_triplets(c, molecular, registry, cfg, out)
    joint_rates = joint_low_activity(c, cfg, out)
    checks = verify(c, assay, molecular, baseline, triplets, cfg, root)
    assert digest(plan) == plan_hash
    for path, wanted in cfg["inputs"].items():
        assert digest(root / path) == wanted, path
    audit = dict(status="PASS", plan=str(plan.relative_to(root)), plan_sha256=plan_hash,
                 source_sha256=digest(Path(__file__)), inputs_unchanged=True, checks=checks,
                 new_tests=dict(planned_family=138, estimable=int(triplets.p.notna().sum()),
                                q_below_05=int(triplets.significant.sum()),
                                no_tests_at_W0=True, q_not_reused_from_old_different_patient_sets=True),
                 eos_estimates=estimates.to_dict("records"),
                 joint_low_activity=dict(status="post_hoc_descriptive", treatment_rows=24, overall_rows=8,
                                         no_clinical_outcomes_imputed=True, no_new_significance_tests=True,
                                         source_doi=cfg["joint_low_activity"]["source_doi"]),
                 cautions=["Dupilumab W4 is not observed: all estimates depend on a chosen temporal shape.",
                           "Bootstrap intervals describe sampling uncertainty conditional on observed endpoints; they do not identify the missing shape or remove complete-case selection.",
                           "Do not use the interpolated point as a clinical record, a treatment-effect test or a model-validation outcome.",
                           "An early transient rise is not ruled out by interpolation.",
                           "Triplet molecular tests are exploratory within-group tests. Stars do not establish a between-treatment difference."])
    (report / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (report / "methods.txt").write_text(
        "EOS week-4 display estimate. No dupilumab patient has an observed week-4 eosinophil count. "
        "For an explicitly defined endpoint-complete group, each patient's week-4 value was estimated by linear interpolation "
        "of log(1+EOS) between observed baseline and week 12, followed by back-transformation. The group median and "
        "a 95% percentile interval were calculated using 2,000 patient bootstrap samples. The baseline/week-12 group "
        "and complete three-visit group were reported separately. These are retrospective interpolation estimates "
        "conditional on temporal shape, not observed data or validated individual predictions. Raw-scale linear, "
        "early logarithmic-time and delayed quadratic-time interpolation were sensitivity scenarios. None permits "
        "an unobserved transient rise, so the scenarios do not span all clinically plausible paths. Masking observed "
        "intermediate visits supplied limited interpolation backchecks, not validation of dupilumab week 4. "
        "No estimated point enters treatment comparisons or clinical prediction validation.\n\n"
        "Baseline molecular table. All 23 markers were summarized in recorded pg/mL directly from positive observed "
        "patient-level assay concentrations. Median and quartiles use type-7 linear interpolation on the original scale. "
        "No additional dilution multiplier or substitution for censored results was used. Each marker-treatment cell "
        "reports its own numeric denominator, with counts outside quantification and other unavailable states retained.\n\n"
        "Fixed-patient molecular changes. Within each marker and initial treatment, the same patients had observed "
        "positive concentrations at weeks 0, 12 and 24. Baseline log2 concentration was subtracted from each "
        "patient's follow-up value, then averaged. Baseline is zero by definition. "
        "Pointwise Student-t 95% intervals and two-sided paired sign-flip tests were used only for at least five pairs; "
        "sign flips were enumerated for at most 15 patients and otherwise used 49,999 Monte Carlo draws with plus-one "
        "correction. Simulated tests with provisional q between 0.04 and 0.06 were refined by exact enumeration "
        "when n was at most 24, or 1,999,999 Monte Carlo draws otherwise, before recalculating the same family. "
        "This numerical precision check did not change the hypotheses, effects or populations. "
        "Benjamini-Hochberg adjustment covered 138 planned marker-treatment-time comparisons (23 x 3 x 2). "
        "Non-estimable hypotheses contributed p=1 to that family and have no displayed q value. Stars indicate q<0.05 "
        "within a treatment; they do not indicate treatment differences. These new tests use the fixed triplet population "
        "and do not reuse q values from the prior two-visit analyses.\n\n"
        "Post hoc IEC-compatible EASI-and-itch states. At each of weeks 0, 4, 12 and 24, low disease activity "
        "was EASI <=7 AND PP-NRS <=4; very low disease activity was EASI <=3 AND PP-NRS <=1 "
        "(IEC consensus, DOI 10.1001/jamadermatol.2026.0662). Recorded scores were not rounded or imputed. "
        "The primary proportion and Wilson 95% confidence interval use only patients with both components "
        "observed at that visit. All initial-treatment patients remain in the missingness accounting. A missing "
        "component leaves the AND state unknown unless an observed component already exceeds its threshold, "
        "which establishes failure. Full-cohort bounds count all unresolved states as failures or successes, "
        "while preserving these known failures. Bounds are not confidence intervals. The 24 treatment-specific "
        "cells and eight pooled cells are reported without new significance testing. These are single-visit "
        "EASI-branch states, not the full AHEAD MDA definition or sustained control.\n")
    print(estimates[["population", "method", "n", "median", "median_ci_low", "median_ci_high"]].to_string(index=False))
    print(backchecks.to_string(index=False))
    print(joint_rates[["treatment", "week", "state", "n_success", "n_joint_observed", "N_initial", "unknown"]].to_string(index=False))
    print(json.dumps({"status": "PASS", "baseline_cells": len(baseline), "triplet_rows": len(triplets), "tests": audit["new_tests"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    run(parser.parse_args().root.resolve())
