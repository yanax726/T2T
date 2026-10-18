"""Audit display-coefficient rounding without changing the frozen EASI estimator."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import itertools
import json
import os
import platform

import numpy as np
import pandas as pd

from cytot2t.curation.unified_model_20261017 import (
    FEATURES, MARKERS, Model, evaluate_formula, scalar_metrics,
)

POLICIES = {
    "full_precision": (None, 1),
    "direct_2dp": (2, 1),
    "direct_3dp": (3, 1),
    "direct_4dp": (4, 1),
    "scaled100_2dp": (2, 100),
}
SCOPES = {
    "frozen_deployment": "Numerical approximation of the final full-development formula; W0/W12 fit performance is apparent, not held-out validation.",
    "fixed_structure_OOF": "Patient-held-out predictions using the fixed area-interaction structure and the originally selected fold penalties; diagnostic conditional on that structure.",
    "nested_selection_OOF": "Patient-held-out predictions using each fold's originally selected structure and penalty; the primary nested development procedure. Rounding occurs only after fitting.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coefficient_rounding(value, policy):
    digits, scale = POLICIES[policy]
    return float(value) if digits is None else round(float(value) * scale, digits) / scale


def rounded_formula(formula, policy):
    result = deepcopy(formula)
    for key in ["intercept", "bsa_factor"]:
        if key in result:
            result[key] = coefficient_rounding(result[key], policy)
    for key in ["bsa_log2_coefficients", "coefficients"]:
        if key in result:
            result[key] = {name: coefficient_rounding(value, policy) for name, value in result[key].items()}
    if "by_stage" in result:
        result["by_stage"] = {stage: rounded_formula(value, policy) for stage, value in result["by_stage"].items()}
    return result


def groups(frame):
    yield "overall", "all", frame
    for label, mask in [
        ("EASI_le7", frame.observed.le(7)),
        ("EASI_7_to21", frame.observed.gt(7) & frame.observed.le(21)),
        ("EASI_gt21", frame.observed.gt(21)),
    ]:
        if mask.any():
            yield "severity", label, frame.loc[mask]


def metrics(frame):
    repeated = pd.DataFrame([scalar_metrics(g.observed, g.prediction) for _, g in frame.groupby("repeat")])
    original = pd.DataFrame([scalar_metrics(g.observed, g.reference) for _, g in frame.groupby("repeat")])
    output = repeated.mean().to_dict()
    for name in ["MAE", "RMSE", "bias", "rho", "CCC", "band_accuracy"]:
        output[name + "_change_from_full"] = float((repeated[name] - original[name]).mean())
    difference = frame.prediction - frame.reference
    output.update(n_patients=int(frame.patient_uid.nunique()), n_prediction_evaluations=len(frame),
                  n_repeats=int(frame["repeat"].nunique()), prediction_change_mean=float(difference.mean()),
                  absolute_prediction_change_mean=float(abs(difference).mean()),
                  absolute_prediction_change_p95=float(np.quantile(abs(difference), .95)),
                  absolute_prediction_change_max=float(abs(difference).max()))
    return output


def decision_row(frame, original, approximate, observed, target):
    original, approximate, observed = map(np.asarray, (original, approximate, observed))
    changed = original != approximate
    count = frame.assign(changed=changed).groupby("repeat").changed.sum()
    return dict(target=target, n_patients=int(frame.patient_uid.nunique()), n_evaluations=len(frame),
                changed_evaluations=int(changed.sum()), changed_fraction=float(changed.mean()),
                n_changed_unique_patients=int(frame.loc[changed, "patient_uid"].nunique()),
                changed_min_per_repeat=int(count.min()), changed_max_per_repeat=int(count.max()),
                original_accuracy=float(np.mean(original == observed)), rounded_accuracy=float(np.mean(approximate == observed)))


def decision_tables(predictions):
    absolute, response = [], []
    for (scope, policy, week), g in predictions.groupby(["scope", "policy", "visit_week"]):
        for kind, label, sub in groups(g):
            key = dict(scope=scope, policy=policy, week=int(week), group_type=kind, group=label)
            for threshold in [3, 5, 7]:
                absolute.append(dict(**key, **decision_row(sub, sub.reference.le(threshold), sub.prediction.le(threshold), sub.observed.le(threshold), f"EASI_le{threshold}")))
            absolute.append(dict(**key, **decision_row(sub, np.digitize(sub.reference, [7, 21, 50], right=True), np.digitize(sub.prediction, [7, 21, 50], right=True), np.digitize(sub.observed, [7, 21, 50], right=True), "severity_band_7_21_50")))
    pair_keys = ["patient_uid", "repeat", "fold"]
    for (scope, policy), g in predictions.groupby(["scope", "policy"]):
        baseline = g[g.visit_week.eq(0)][pair_keys + ["reference", "prediction"]].rename(columns={"reference": "reference_W0", "prediction": "prediction_W0"})
        for week in [12, 24]:
            followup = g[g.visit_week.eq(week)]
            for baseline_type in ["observed_W0", "estimated_W0"]:
                if baseline_type == "observed_W0":
                    valid = followup[followup.baseline_easi.gt(0)].copy()
                    ref_base, pred_base = valid.baseline_easi, valid.baseline_easi
                else:
                    valid = followup.merge(baseline, on=pair_keys, validate="one_to_one")
                    valid = valid[valid.baseline_easi.gt(0) & valid.reference_W0.gt(0) & valid.prediction_W0.gt(0)]
                    ref_base, pred_base = valid.reference_W0, valid.prediction_W0
                if not len(valid):
                    continue
                valid = valid.assign(reference_baseline=ref_base, approximate_baseline=pred_base)
                for kind, label, sub in groups(valid):
                    for threshold in [.50, .75, .90]:
                        truth = sub.observed.le(sub.baseline_easi * (1-threshold))
                        original = sub.reference.le(sub.reference_baseline * (1-threshold))
                        approximate = sub.prediction.le(sub.approximate_baseline * (1-threshold))
                        response.append(dict(scope=scope, policy=policy, week=week, group_type=kind, group=label,
                                             baseline_reference=baseline_type, **decision_row(sub, original, approximate, truth, f"EASI{int(threshold*100)}")))
    return pd.DataFrame(absolute), pd.DataFrame(response)


def frozen_domain_bound(artifact, rounded):
    original = artifact["formula"]
    delta_constant = rounded["bsa_factor"] - original["bsa_factor"]
    delta = {name: rounded["bsa_log2_coefficients"][name] - original["bsa_log2_coefficients"][name] for name in MARKERS}
    corners = itertools.product(*(artifact["range_at_development"][name] for name in MARKERS))
    max_factor_error = max(abs(delta_constant + sum(delta[name] * value for name, value in zip(MARKERS, corner))) for corner in corners)
    return dict(max_abs_change_bound_at_BSA79=max_factor_error*artifact["range_at_development"]["bsa"][1],
                max_abs_change_bound_at_BSA100=max_factor_error*100,
                domain="All combinations within the W0/W12 development molecular ranges; BSA upper bounds as named. Clipping cannot enlarge this bound. Not a clinical-accuracy guarantee.")


def run(root):
    root = Path(root).resolve()
    out = root / "results/tables/update_20261017/model/rounding"
    report = root / "reports/update_20261017/model/rounding"
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    inputs = [root/"config/unified_bsa_three_v1_20261017.json", root/"config/unified_model_20261017.json",
              root/"results/tables/update_20261002/stage_refinement/input.csv",
              root/"results/tables/update_20261017/model/fit_audit_private.csv",
              root/"results/tables/update_20261017/model/predictions.csv",
              root/"results/tables/update_20261017/model/overall_metrics.csv"]
    before = {str(p.relative_to(root)): sha(p) for p in inputs}
    artifact = json.loads(inputs[0].read_text()); cfg = json.loads(inputs[1].read_text())
    data, audit, old = map(pd.read_csv, inputs[2:5])
    assert not data.duplicated(["patient_uid", "visit_week"]).any()
    assert np.isfinite(data[FEATURES + ["observed", "baseline_easi"]]).all().all()
    records, fits, formulas = [], [], []
    keep = ["patient_uid", "visit_week", "observed", "baseline_easi"]

    def append(frame, formula, scope, repeat, fold):
        exact = evaluate_formula(frame, formula)
        for policy in POLICIES:
            rounded = rounded_formula(formula, policy)
            q = frame[keep].copy()
            q["reference"] = exact
            q["prediction"] = evaluate_formula(frame, rounded)
            q["scope"], q["policy"], q["repeat"], q["fold"] = scope, policy, repeat, fold
            records.append(q)
            formulas.append(dict(scope=scope, repeat=repeat, fold=fold, policy=policy, formula=rounded))
        return exact

    append(data, artifact["formula"], "frozen_deployment", 0, -1)
    scope_names = {"unified_area_interaction": "fixed_structure_OOF", "unified_selected": "nested_selection_OOF"}
    for row in audit[audit.model.isin(scope_names)].itertuples():
        train_ids, test_ids = set(row.train_ids.split("|")), set(row.test_ids.split("|"))
        assert train_ids.isdisjoint(test_ids)
        train = data[data.patient_uid.isin(train_ids) & data.visit_week.isin(cfg["development_weeks"])]
        test = data[data.patient_uid.isin(test_ids)]
        assert len(train) == row.n_training_visits and train.patient_uid.nunique() == row.n_training_patients
        fitted = Model(train, row.selected_structure, row.alpha)
        exact = append(test, fitted.formula(), scope_names[row.model], row.repeat, row.fold)
        reference = old[old.model.eq(row.model) & old["repeat"].eq(row.repeat) & old.fold.eq(row.fold)]
        joined = test[keep].assign(reconstructed=exact).merge(reference[["patient_uid", "visit_week", "prediction"]], on=["patient_uid", "visit_week"], validate="one_to_one")
        difference = float(abs(joined.reconstructed-joined.prediction).max())
        assert len(joined)==len(test) and difference < 1e-10
        fits.append(dict(scope=scope_names[row.model], repeat=row.repeat, fold=row.fold, structure=row.selected_structure,
                         alpha=row.alpha, n_training_patients=len(train_ids), n_test_patients=len(test_ids),
                         patient_overlap=0, max_training_week=int(train.visit_week.max()), reconstruction_max_absolute=difference))
    predictions = pd.concat(records, ignore_index=True)
    assert not predictions.duplicated(["scope", "policy", "repeat", "patient_uid", "visit_week"]).any()
    predictions.to_csv(out/"predictions_private.csv", index=False)
    rows=[]
    for (scope, policy, week), g in predictions.groupby(["scope", "policy", "visit_week"]):
        for kind, label, sub in groups(g):
            rows.append(dict(scope=scope, policy=policy, week=int(week), group_type=kind, group=label, **metrics(sub)))
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(out/"metrics.csv", index=False)
    metrics_table[metrics_table.group_type.eq("overall")].to_csv(out/"overall_metrics.csv", index=False)
    absolute, response = decision_tables(predictions)
    absolute.to_csv(out/"absolute_target_changes.csv", index=False)
    response.to_csv(out/"response_target_changes.csv", index=False)
    pd.DataFrame(fits).to_csv(out/"fold_reconstruction_audit.csv", index=False)
    (out/"all_formulas.json").write_text(json.dumps(formulas, indent=2)+"\n")
    coefficients, bounds = [], []
    for policy in POLICIES:
        formula = rounded_formula(artifact["formula"], policy)
        digits, scale = POLICIES[policy]
        for name, value in {"constant":formula["bsa_factor"], **formula["bsa_log2_coefficients"]}.items():
            coefficients.append(dict(policy=policy, term=name, effective_coefficient=value, displayed_coefficient=value*scale, external_divisor=scale, display_decimal_places=digits))
        bounds.append(dict(policy=policy, **frozen_domain_bound(artifact, formula)))
    pd.DataFrame(coefficients).to_csv(out/"frozen_display_coefficients.csv", index=False)
    pd.DataFrame(bounds).to_csv(out/"development_domain_bounds.csv", index=False)
    original_metrics = pd.read_csv(inputs[5]); checks=[]
    for model, scope in scope_names.items():
        for week in [0,12,24]:
            a = metrics_table[(metrics_table.scope==scope)&metrics_table.policy.eq("full_precision")&metrics_table.week.eq(week)&metrics_table.group_type.eq("overall")].iloc[0]
            b = original_metrics[original_metrics.model.eq(model)&original_metrics.week.eq(week)].iloc[0]
            discrepancy=max(abs(a[key]-b[key]) for key in ["MAE","RMSE","rho","CCC"])
            assert discrepancy < 1e-10
            checks.append(dict(scope=scope,week=week,max_metric_difference=discrepancy))
    unchanged = {str(p.relative_to(root)): sha(p)==before[str(p.relative_to(root))] for p in inputs}
    assert all(unchanged.values())
    primary = metrics_table[metrics_table.scope.eq("nested_selection_OOF") & metrics_table.group_type.eq("overall")]
    primary.to_csv(out/"primary_validation_comparison.csv", index=False)
    display_columns = ["policy", "week", "group_type", "group", "n_patients", "prediction_change_mean",
                       "absolute_prediction_change_mean", "absolute_prediction_change_p95", "absolute_prediction_change_max"]
    deployment = metrics_table[metrics_table.scope.eq("frozen_deployment")]
    deployment[display_columns].to_csv(out/"display_approximation_effect.csv", index=False)
    recommended = "scaled100_2dp"
    recommendation = dict(
        policy=recommended, reference_model=artifact["version"], calculator_default="unchanged full precision",
        display_expression="EASI approximately clip{(BSA/100) * [51.27 + 0.98 log2(CCL17) + 0.96 log2(CCL26) - 0.83 log2(IL-5)], 0, 72}",
        BSA_input="Percentage from 0 to 100; enter 10 for 10% affected skin, not 0.10.",
        molecular_input="Original positive concentration in the same recorded pg/ml assay units; base-2 log. Do not round concentrations or log values as part of this display audit.",
        reason="Two decimal places in the rescaled display preserve four effective decimal places and avoid dropping small molecular coefficients. Chosen for numerical fidelity and readability, not an apparent reduction in clinical MAE.",
        deployment_approximation=deployment[deployment.policy.eq(recommended)&deployment.group_type.eq("overall")][display_columns].to_dict("records"),
        primary_validation=primary[primary.policy.isin(["full_precision",recommended])].to_dict("records"),
        frozen_absolute_target_changes=int(absolute[absolute.scope.eq("frozen_deployment")&absolute.policy.eq(recommended)&absolute.group_type.eq("overall")].changed_evaluations.sum()),
        response_target_changes=int(response[response.policy.eq(recommended)&response.group_type.eq("overall")].changed_evaluations.sum()),
        threshold_note="One W24 patient-repeat crosses the EASI 7 threshold in each held-out analysis; this is one decision among 150 evaluations per scope. It is the same event counted by the severity-band row. No assessed EASI50/75/90 decisions changed. Near a cut-off, use the full-precision calculator.",
        interpretation="The displayed formula is a numerical approximation, not a newly validated model. Main reported validation remains the original full-precision nested procedure; apparent W0/W12 performance of the final deployment formula must not replace it.")
    (out/"recommendation.json").write_text(json.dumps(recommendation,indent=2)+"\n")
    lines=["# 推荐的易读公式", "",
           "EASI估计 ≈ BSA / 100 × [51.27 + 0.98 × log₂(CCL17) + 0.96 × log₂(CCL26) − 0.83 × log₂(IL-5)]。结果小于0取0，大于72取72。", "",
           "BSA仍输入百分数：10%填10，不填0.10。分子浓度使用原检测单位，先取以2为底的对数。这个写法把系数先放大100倍、最后除以100，显示两位小数，实际保留了原系数四位精度。冻结计算器继续使用全部精度。", "",
           "## 正式验证流程的影响", "", "| 时点 | 原MAE | 简化MAE | 原/简化Spearman | 原/简化CCC |", "|---|---:|---:|---:|---:|"]
    for week in [0,12,24]:
        exact=primary[primary.week.eq(week)&primary.policy.eq("full_precision")].iloc[0]
        approx=primary[primary.week.eq(week)&primary.policy.eq(recommended)].iloc[0]
        lines.append(f"| W{week} | {exact.MAE:.6f} | {approx.MAE:.6f} | {exact.rho:.6f} / {approx.rho:.6f} | {exact.CCC:.6f} / {approx.CCC:.6f} |")
    lines += ["", "这里每个验证折先使用原先的训练患者、结构和参数还原公式，再取整后预测未参加训练的患者；没有重新调参。误差在每轮内计算后汇总，没有先平均多轮预测。W24仍是已看过的同队列时间迁移。", "",
              "## 冻结公式用于计算时的偏差", "", "| 时点 | 平均绝对变化 | 95%绝对变化分位数 | 最大绝对变化 |", "|---|---:|---:|---:|"]
    for row in deployment[deployment.policy.eq(recommended)&deployment.group_type.eq("overall")].sort_values("week").itertuples():
        lines.append(f"| W{row.week} | {row.absolute_prediction_change_mean:.6f} | {row.absolute_prediction_change_p95:.6f} | {row.absolute_prediction_change_max:.6f} |")
    lines += ["", "这些是同一个冻结公式简化前后的计算差，不是预测临床EASI的误差。不能把最终全样本公式在训练数据上的拟合误差当独立验证。", "",
              "不建议把原式直接截成0.51、0.01、0.01、−0.01。虽然本次最终公式总体误差变化不大，但某些患者会变动更多；逐折审计中，小系数直接取两位会造成明显预测跳变。三位/四位对照和全部严重度分层均保留。", "",
              "推荐写法没有改变冻结公式的任何已核验绝对达标或EASI50/75/90判断。逐折验证中，W24有1/150个患者-轮次的预测跨过EASI=7，分类表中同一件事也体现为严重度档位变化；其他已核验达标判断未变。临界值附近使用精确计算器。", "",
              "这次审计说明如何更清楚地显示公式，不说明重症误差已经解决，也不把精度简化当作新模型或新增验证人群。"]
    (report/"recommendation.md").write_text("\n".join(lines)+"\n")
    manifest=dict(status="PASS",version="2026-10-17 rounding audit",input_hashes=before,inputs_unchanged=unchanged,
                  source_hashes={str(Path(__file__).relative_to(root)):sha(__file__),"src/cytot2t/curation/unified_model_20261017.py":sha(root/"src/cytot2t/curation/unified_model_20261017.py")},
                  policies={k:dict(decimal_places=v[0],internal_multiplier=v[1]) for k,v in POLICIES.items()},scopes=SCOPES,
                  rounding="Python round (nearest, ties to even), applied to physical-unit coefficients after fitting; logs, clinical inputs and final scores are not rounded by this audit.",
                  validation_checks=checks,folds_reconstructed=len(fits),max_prediction_reconstruction_error=max(x['reconstruction_max_absolute'] for x in fits),
                  selection="No tuning, reselection or refitting to improve outcomes. Existing fold membership, structure and penalty reused solely to reconstruct saved formulas.",
                  limits="W24 remains previously inspected same-cohort temporal transfer. Frozen deployment W0/W12 performance is apparent; it is not interchangeable with OOF validation. Coefficient rounding does not remedy severe-disease model error.",
                  private_files=["predictions_private.csv"],software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
                  outputs=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'])
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (report/"methods.md").write_text("# Coefficient display precision audit\n\n"+
        "The frozen estimator and all prior result files remain unchanged. Raw concentrations retain their original units, positive-value handling and full-precision log2 transform. BSA remains a percentage from 0 to 100. Predicted EASI is clipped to [0,72] after calculation.\n\n"+
        "We compared full precision, direct two/three/four decimal coefficients, and two decimals after multiplying the coefficients by 100 and dividing the resulting expression by 100. This last representation preserves four effective decimals; it is not the same as directly rounding small molecular coefficients to two decimals.\n\n"+
        "Three distinct estimands are retained: (1) numerical approximation of the frozen formula, whose W0/W12 errors are apparent; (2) the fixed-area-structure held-out diagnostic; (3) the primary nested-selection procedure. For both held-out analyses, saved patient membership, structure and penalty were used to reconstruct each training-fold formula, then rounded before predicting the original held-out patients. No W24 outcomes entered fitting and no parameters were reselected. Exact reconstructions were checked against saved predictions and existing metrics.\n\n"+
        "Metrics are computed within each repeat before averaging; prediction errors are never computed after averaging repeated predictions. Rounding shifts use all held-out evaluations. Severity strata use observed EASI <=7, >7 to21 and >21. Absolute targets and EASI50/75/90 decision changes are counted per patient-repeat; response targets separately use the recorded observed baseline or the model-estimated baseline. The latter requires both visits and pairs within patient, repeat and fold. Decision changes describe agreement with the unrounded model, not proof of clinical interchangeability.\n\n"+
        "The precision audit is an implementation check. A smaller apparent MAE after rounding is not used to select a new clinical model. The full-precision formula remains the reproducibility reference and calculator default.\n")
    print(metrics_table[metrics_table.group_type.eq('overall')][['scope','policy','week','MAE','rho','CCC','absolute_prediction_change_p95','absolute_prediction_change_max']].to_string(index=False))
    print('PASS: original predictions/metrics reconstructed; frozen inputs unchanged.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('T2T_ROOT',Path(__file__).resolve().parents[3])))
    run(parser.parse_args().root)
