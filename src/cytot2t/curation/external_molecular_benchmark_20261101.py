"""Apply existing molecular equations to OMIX002850 without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PLAN = "config/external_molecular_benchmark_20261101.json"
ANALYTES = {
    "CCL17": "CCL17/TARC",
    "CCL26": "CCL26/Eotaxin-3",
    "IL-5": "IL-5",
    "IL22": "IL-22",
    "sIL2Ra": "CD25/IL-2Rα",
}
NAMES = {
    "core_all__serum3": "既有三分子模型",
    "core_all__ccl17": "既有单CCL17模型",
    "peasi_2017_precise": "p-EASI原全精度公式",
    "peasi_2020_rounded": "p-EASI发表舍入公式",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_hashes(root: Path, plan: dict) -> dict:
    actual = {name: sha256(root / name) for name in plan["protected_sha256"]}
    if actual != plan["protected_sha256"]:
        raise ValueError("A protected source differs from the fixed analysis plan")
    return actual


def local_prediction(frame: pd.DataFrame, definition: dict) -> np.ndarray:
    formula = definition["formula"]
    prediction = np.full(len(frame), formula["intercept"], dtype=float)
    for name, coefficient in formula["coefficients"].items():
        if not name.startswith("log2_"):
            raise ValueError("This benchmark accepts only the specified molecular inputs")
        prediction += coefficient * np.log2(frame[name[5:]].to_numpy(float))
    return prediction


def peasi_prediction(frame: pd.DataFrame, coefficients: dict) -> np.ndarray:
    return (
        coefficients["intercept"]
        + coefficients["log10_CCL17"] * np.log10(frame["CCL17"].to_numpy(float))
        + coefficients["IL22"] * frame["IL22"].to_numpy(float)
        + coefficients["sIL2Ra"] * frame["sIL2Ra"].to_numpy(float)
    )


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(np.asarray(numerator, dtype=float), np.nan),
        where=np.asarray(denominator) > 1e-15,
    )


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics along the final axis; calibration is diagnostic, never applied."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    error = p - y
    ym, pm = y.mean(axis=-1), p.mean(axis=-1)
    yc, pc = y - ym[..., None], p - pm[..., None]
    yv, pv = (yc**2).mean(axis=-1), (pc**2).mean(axis=-1)
    covariance = (yc * pc).mean(axis=-1)
    yr, pr = rankdata(y, axis=-1), rankdata(p, axis=-1)
    yr -= yr.mean(axis=-1)[..., None]
    pr -= pr.mean(axis=-1)[..., None]
    rho = ratio((yr * pr).sum(axis=-1), np.sqrt((yr**2).sum(axis=-1) * (pr**2).sum(axis=-1)))
    slope = ratio(covariance, pv)
    return {
        "MAE": np.abs(error).mean(axis=-1),
        "RMSE": np.sqrt((error**2).mean(axis=-1)),
        "bias_prediction_minus_observed": error.mean(axis=-1),
        "Spearman": rho,
        "CCC": ratio(2 * covariance, yv + pv + (ym - pm) ** 2),
        "R2": 1 - ratio((error**2).mean(axis=-1), yv),
        "within_1": (np.abs(error) <= 1).mean(axis=-1),
        "within_2": (np.abs(error) <= 2).mean(axis=-1),
        "calibration_intercept": ym - slope * pm,
        "calibration_slope": slope,
    }


def interval(values: np.ndarray | None) -> tuple[float, float, int]:
    if values is None:
        return np.nan, np.nan, 0
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(len(finite))


def verification(plan: dict) -> dict:
    frame = pd.DataFrame({"CCL17": [100.0, 10000.0], "CCL26": [4.0, 64.0], "IL-5": [0.25, 16.0], "IL22": [5.0, 20.0], "sIL2Ra": [200.0, 1000.0]})
    for model, definition in plan["local_models"].items():
        scalar = []
        for _, row in frame.iterrows():
            scalar.append(definition["formula"]["intercept"] + sum(c * math.log2(row[k[5:]]) for k, c in definition["formula"]["coefficients"].items()))
        np.testing.assert_allclose(local_prediction(frame, definition), scalar, rtol=0, atol=1e-12)
    for _, stages in plan["peasi"].items():
        for coefficients in stages.values():
            scalar = [coefficients["intercept"] + coefficients["log10_CCL17"] * math.log10(row.CCL17) + coefficients["IL22"] * row.IL22 + coefficients["sIL2Ra"] * row.sIL2Ra for row in frame.itertuples()]
            np.testing.assert_allclose(peasi_prediction(frame, coefficients), scalar, rtol=0, atol=1e-12)
    perfect = metrics(np.array([0., 1., 3.]), np.array([0., 1., 3.]))
    for key in ["MAE", "RMSE", "bias_prediction_minus_observed", "calibration_intercept"]:
        np.testing.assert_allclose(perfect[key], 0, atol=1e-12)
    for key in ["Spearman", "CCC", "R2", "within_1", "within_2", "calibration_slope"]:
        np.testing.assert_allclose(perfect[key], 1, atol=1e-12)
    shifted = metrics(np.array([1., 2., 3.]), np.array([3., 4., 5.]))
    np.testing.assert_allclose([shifted["MAE"], shifted["RMSE"], shifted["bias_prediction_minus_observed"]], [2, 2, 2])
    np.testing.assert_allclose([shifted["CCC"], shifted["R2"], shifted["calibration_intercept"], shifted["calibration_slope"]], [.25, -5, -2, 1])
    constant = metrics(np.array([1., 2., 3.]), np.array([2., 2., 2.]))
    assert np.isnan(constant["Spearman"]) and np.isnan(constant["calibration_slope"])
    np.testing.assert_array_equal(np.clip([-1., 0., 2., 72., 80.], 0, 72), [0., 0., 2., 72., 72.])
    return {"status": "PASS", "checks": ["All four formula implementations agree with independent scalar calculations", "Perfect-prediction metrics", "Known two-point bias and calibration", "Constant predictions yield undefined correlation/slope", "Common [0,72] bounds"]}


def load_visits(root: Path, plan: dict) -> tuple[pd.DataFrame, dict]:
    source = pd.read_excel(root / plan["source_data"], header=1)
    if source["*ID"].duplicated().any():
        raise ValueError("Duplicate source participant IDs")
    ad = source.loc[source["*experiment_group"].eq("Case set")].copy()
    visits, counts = [], []
    for stage, week in [("before", 0), ("after", 16)]:
        d = pd.DataFrame({"external_patient_id": ad["*ID"].astype(str), "week": week})
        d["observed_easi"] = pd.to_numeric(ad[f"Eczema Area and Severity Index score {stage} treatment"], errors="coerce")
        for analyte, label in ANALYTES.items():
            d[analyte] = pd.to_numeric(ad[f"serum {label} level {stage} treatment (pg/ml)"], errors="coerce")
        observed = np.isfinite(d.observed_easi) & d.observed_easi.between(0, 72)
        valid_molecules = np.isfinite(d[list(ANALYTES)]).all(axis=1) & d[list(ANALYTES)].gt(0).all(axis=1)
        selected = d.loc[observed & valid_molecules].copy()
        for model, definition in plan["local_models"].items():
            selected[f"{model}__raw"] = local_prediction(selected, definition)
        for model, stages in plan["peasi"].items():
            selected[f"{model}__raw"] = peasi_prediction(selected, stages[stage])
        for model in NAMES:
            selected[f"{model}__clip"] = selected[f"{model}__raw"].clip(0, 72)
        counts.append({"week": week, "AD_rows": len(ad), "observed_EASI": int(observed.sum()), "five_positive_finite_molecules": int(valid_molecules.sum()), "matched_visits": len(selected), "excluded_missing_or_invalid": int((~(observed & valid_molecules)).sum())})
        visits.append(selected)
    complete = pd.concat(visits, ignore_index=True).sort_values(["external_patient_id", "week"])
    if complete.duplicated(["external_patient_id", "week"]).any():
        raise ValueError("Repeated patient-visit rows")
    audit = {"source_rows": len(source), "AD_patients": len(ad), "healthy_rows_excluded": int(source["*experiment_group"].eq("Health people").sum()), "coverage": counts, "contains_BSA": any("bsa" in str(c).lower() or "body surface" in str(c).lower() for c in source.columns), "unique_included_patients": int(complete.external_patient_id.nunique())}
    return complete, audit


def paired_changes(visits: pd.DataFrame) -> pd.DataFrame:
    before = visits.loc[visits.week.eq(0)].set_index("external_patient_id")
    after = visits.loc[visits.week.eq(16)].set_index("external_patient_id")
    shared = sorted(set(before.index) & set(after.index))
    columns = ["observed_easi"] + [f"{model}__{mode}" for model in NAMES for mode in ["raw", "clip"]]
    result = after.loc[shared, columns] - before.loc[shared, columns]
    result["baseline_observed_easi"] = before.loc[shared, "observed_easi"]
    result = result.reset_index()
    assert len(result) == len(shared) and result.external_patient_id.is_unique
    for column in columns:
        np.testing.assert_allclose(result[column], after.loc[shared, column].to_numpy() - before.loc[shared, column].to_numpy(), atol=0, rtol=0)
    return result


def summarize(data: pd.DataFrame, scope: str, stratum: str, plan: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    if not data.external_patient_id.is_unique:
        raise ValueError("Each resampling unit must be one patient")
    n = len(data)
    if not n:
        return [], []
    cfg = plan["confidence_intervals"]
    indices = rng.integers(0, n, size=(cfg["replicates"], n)) if n >= cfg["minimum_patients_for_interval"] else None
    y = data.observed_easi.to_numpy(float)
    result, comparisons = [], []
    for mode in ["clip", "raw"]:
        for model in NAMES:
            p = data[f"{model}__{mode}"].to_numpy(float)
            point = metrics(y, p)
            samples = metrics(y[indices], p[indices]) if indices is not None else None
            for name, value in point.items():
                low, high, valid = interval(samples[name] if samples else None)
                result.append({"scope": scope, "stratum": stratum, "prediction_scale": mode, "model": model, "n_patients": n, "metric": name, "estimate": float(value), "ci_low": low, "ci_high": high, "valid_bootstrap_replicates": valid, "ci_status": "percentile_patient_bootstrap" if indices is not None else "not_estimated_n_lt_5"})
        for local in plan["local_models"]:
            for reference in plan["peasi"]:
                difference = np.abs(data[f"{local}__{mode}"].to_numpy(float) - y) - np.abs(data[f"{reference}__{mode}"].to_numpy(float) - y)
                low, high, valid = interval(difference[indices].mean(axis=1) if indices is not None else None)
                comparisons.append({"scope": scope, "stratum": stratum, "prediction_scale": mode, "local_model": local, "reference_model": reference, "n_patients": n, "paired_MAE_difference_local_minus_reference": float(difference.mean()), "ci_low": low, "ci_high": high, "valid_bootstrap_replicates": valid, "direction": "negative_favors_local", "ci_status": "percentile_patient_bootstrap" if indices is not None else "not_estimated_n_lt_5"})
    return result, comparisons


def json_clean(value):
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(root: Path) -> dict:
    plan_path = root / PLAN
    plan = json.loads(plan_path.read_text())
    plan_hash = sha256(plan_path)
    before_hashes = checked_hashes(root, plan)
    tests = verification(plan)
    visits, audit = load_visits(root, plan)
    changes = paired_changes(visits)
    audit["paired_change_patients"] = len(changes)
    rng = np.random.default_rng(plan["confidence_intervals"]["seed"])
    records, comparisons = [], []
    for week in plan["visits"]:
        at_visit = visits.loc[visits.week.eq(week)].copy()
        strata = {
            "all": np.ones(len(at_visit), dtype=bool),
            "EASI_0_to_7": at_visit.observed_easi.le(7),
            "EASI_gt7_to_21": at_visit.observed_easi.gt(7) & at_visit.observed_easi.le(21),
            "EASI_gt21_to_72": at_visit.observed_easi.gt(21),
        }
        for stratum, mask in strata.items():
            a, b = summarize(at_visit.loc[mask], f"W{week}", stratum, plan, rng)
            records.extend(a)
            comparisons.extend(b)
    a, b = summarize(changes, "W16_minus_W0_same_patients", "all", plan, rng)
    records.extend(a)
    comparisons.extend(b)
    metrics_table, comparison_table = pd.DataFrame(records), pd.DataFrame(comparisons)
    table_dir = root / plan["output_table_directory"]
    report_dir = root / plan["output_report_directory"]
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    visits.to_csv(table_dir / "molecular_predictions_PRIVATE.csv", index=False)
    changes.to_csv(table_dir / "molecular_paired_changes_PRIVATE.csv", index=False)
    metrics_table.to_csv(table_dir / "molecular_metrics.csv", index=False)
    comparison_table.to_csv(table_dir / "molecular_paired_MAE_differences.csv", index=False)
    flags = []
    for week in plan["visits"]:
        d = visits.loc[visits.week.eq(week)]
        for model in NAMES:
            raw = d[f"{model}__raw"]
            flags.append({"week": week, "model": model, "n_patients": len(d), "raw_below_0": int(raw.lt(0).sum()), "raw_above_72": int(raw.gt(72).sum()), "observed_min": float(d.observed_easi.min()), "observed_max": float(d.observed_easi.max()), "raw_prediction_min": float(raw.min()), "raw_prediction_max": float(raw.max())})
    pd.DataFrame(flags).to_csv(table_dir / "molecular_output_bounds.csv", index=False)
    checked_hashes(root, plan)
    if sha256(plan_path) != plan_hash:
        raise ValueError("The benchmark plan changed during execution")
    tests["checks"].extend(["Unique participant-visit rows", "Same-patient change arithmetic checked for all methods", "Original data, prior formulas and selected hybrid hashes unchanged", "Fixed plan unchanged during execution"])
    report = {
        "status": "COMPLETED_RETROSPECTIVE_TRANSPORT_BENCHMARK",
        "analysis_plan": PLAN,
        "analysis_plan_sha256": plan_hash,
        "source_code": str(Path(__file__).resolve().relative_to(root)),
        "source_code_sha256": sha256(Path(__file__)),
        "protected_inputs_unchanged": before_hashes,
        "coverage": audit,
        "verification": tests,
        "limitations": plan["limitations"],
        "primary_metrics": metrics_table.loc[metrics_table.stratum.eq("all") & metrics_table.prediction_scale.eq("clip")].to_dict("records"),
        "primary_paired_differences": comparison_table.loc[comparison_table.stratum.eq("all") & comparison_table.prediction_scale.eq("clip") & comparison_table.reference_model.eq("peasi_2017_precise")].to_dict("records"),
        "private_prediction_exports": [str((table_dir / name).relative_to(root)) for name in ["molecular_predictions_PRIVATE.csv", "molecular_paired_changes_PRIVATE.csv"]],
        "source_urls": ["https://ngdc.cncb.ac.cn/omix/release/OMIX002850", "https://pubmed.ncbi.nlm.nih.gov/36756674/"],
        "uncertainty_scope": "Patient sampling uncertainty conditional on fixed formulas and this cohort; excludes assay calibration uncertainty, model-selection uncertainty and unverified participant overlap",
        "calibration_note": "Observed EASI regressed on fixed predictions only as a diagnostic metric. Diagnostic slope/intercept were never used to modify predictions.",
    }
    (report_dir / "molecular_benchmark_audit.json").write_text(json.dumps(json_clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    brief = ["# 既有纯分子模型与 p-EASI：OMIX002850 回顾性应用", "", "本次没有训练或修改任何公式。这里的三分子模型是之前单独拟合的分子对照，不是当前 BSA＋三分子主模型；公开表缺 BSA，不能验证主模型。所有方法用于相同患者、相同访视。主比较统一将预测限制在 0–72；原始未限制预测另行完整报告。", "", "| 访视 | 模型 | 人数 | MAE（95%CI） | RMSE | Spearman | CCC | 误差≤2分 |", "|---|---|---:|---|---:|---:|---:|---:|"]
    primary = metrics_table.loc[metrics_table.stratum.eq("all") & metrics_table.prediction_scale.eq("clip")]
    for scope in ["W0", "W16", "W16_minus_W0_same_patients"]:
        for model in NAMES:
            d = primary.loc[primary.scope.eq(scope) & primary.model.eq(model)].set_index("metric")
            mae = d.loc["MAE"]
            brief.append(f"| {scope} | {NAMES[model]} | {int(mae.n_patients)} | {mae.estimate:.2f}（{mae.ci_low:.2f}–{mae.ci_high:.2f}） | {d.loc['RMSE','estimate']:.2f} | {d.loc['Spearman','estimate']:.3f} | {d.loc['CCC','estimate']:.3f} | {d.loc['within_2','estimate']:.1%} |")
    brief += ["", "正偏差表示高估，负偏差表示低估。MAE差为本地模型减去p-EASI，负数才表示本地误差较小。变化是同一患者 W16 减 W0，先配对、再相减，不把125人基线均值与60人随访均值直接相减。", "", "| 比较范围 | 三分子模型 MAE − p-EASI MAE | 95%患者bootstrap区间 |", "|---|---:|---|"]
    for scope in ["W0", "W16", "W16_minus_W0_same_patients"]:
        d = comparison_table.loc[comparison_table.scope.eq(scope) & comparison_table.stratum.eq("all") & comparison_table.prediction_scale.eq("clip") & comparison_table.local_model.eq("core_all__serum3") & comparison_table.reference_model.eq("peasi_2017_precise")].iloc[0]
        brief.append(f"| {scope} | {d.paired_MAE_difference_local_minus_reference:.2f} | {d.ci_low:.2f}–{d.ci_high:.2f} |")
    brief += ["", "完整结果包括 raw/clip、严重度分层、95%患者bootstrap区间、校准斜率与截距、R²、误差≤1/2分比例，见同目录结果CSV。小于5人的分层不计算bootstrap区间。", "", "这不是干净的独立外部验证：OMIX此前已被本项目探索，患者重叠尚未排除，检测平台校准尚未建立。不得据此单独宣布哪种分子公式普遍更好。预测文件带 PRIVATE 标记，不应进入公开仓库或文章结果包。", ""]
    (report_dir / "molecular_benchmark_results.md").write_text("\n".join(brief))
    print("\n".join(brief[:18]))
    print("Protected source hashes unchanged; verification PASS")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    run(args.root.resolve())
