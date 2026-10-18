"""Independent PP-NRS, adult DLQI and recorded ordinal-grade estimation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import check_grad, minimize
from scipy.special import expit
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from cytot2t.curation.unified_model_20261017 import Model, bootstrap_mean, scalar_metrics

MARKERS = ["log2_CCL17", "log2_CCL26", "log2_IL-5"]
FEATURES = ["bsa"] + MARKERS
UPPER = {"pp_nrs": 10, "dlqi": 30, "iga": 5}
CATEGORIES = 6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(model):
    return ["bsa"] if model == "BSA_only" else MARKERS if model == "molecules_only" else FEATURES


def ordinal_objective(x, y, weight, alpha):
    n, p = x.shape
    def objective(values):
        beta, a = values[:p], values[p:]
        thresholds = np.r_[a[0], a[0] + np.cumsum(np.exp(a[1:]))]
        linear = np.einsum("ij,j->i", x, beta, optimize=False)
        cumulative = expit(thresholds[None, :] - linear[:, None])
        derivative = cumulative * (1 - cumulative)
        padded = np.column_stack([np.zeros(n), cumulative, np.ones(n)])
        probability = np.maximum(padded[np.arange(n),y+1] - padded[np.arange(n),y], 1e-14)
        padded_derivative = np.column_stack([np.zeros(n), derivative, np.zeros(n)])
        delta = padded_derivative[np.arange(n),y] - padded_derivative[np.arange(n),y+1]
        grad_beta = -np.einsum("ij,i->j", x, weight*delta/probability, optimize=False) + alpha*beta
        grad_threshold = np.array([-np.sum(weight*derivative[:,j]*((y==j).astype(float)-(y==j+1).astype(float))/probability) for j in range(CATEGORIES-1)])
        grad_a = np.r_[grad_threshold.sum(), np.exp(a[1:])*np.cumsum(grad_threshold[:0:-1])[::-1]]
        loss = -np.sum(weight*np.log(probability)) + alpha*.5*np.sum(beta**2)
        return loss, np.r_[grad_beta, grad_a]
    return objective


def ordinal_probability(x, beta, thresholds):
    linear = np.einsum("ij,j->i", x, beta, optimize=False)
    cumulative = expit(thresholds[None,:]-linear[:,None])
    probability = np.diff(np.column_stack([np.zeros(len(x)),cumulative,np.ones(len(x))]), axis=1)
    assert np.isfinite(probability).all() and (probability >= -1e-12).all()
    assert np.allclose(probability.sum(axis=1),1,atol=1e-12)
    return probability


class Fit:
    def __init__(self, frame, target, model, alpha=1., link="identity"):
        self.target, self.model, self.alpha, self.link = target, model, float(alpha), link
        self.features = [] if model == "constant" else features(model)
        weight = 1/frame.patient_uid.map(frame.patient_uid.value_counts()).to_numpy()
        self.weight = weight/weight.mean()
        self.converged = True
        if model == "constant":
            if target == "iga":
                counts = np.bincount(frame.observed.astype(int), weights=self.weight,minlength=CATEGORIES)+.25
                self.probability = counts/counts.sum()
            else:
                values = frame.observed.to_numpy()
                order = np.argsort(values)
                self.offset = float(values[order[np.searchsorted(np.cumsum(self.weight[order])/self.weight.sum(),.5)]])
            return
        if target != "iga":
            structure = {"BSA_three":"unified_linear","BSA_only":"bsa_linear","molecules_only":"serum_linear"}[model]
            train = frame.copy()
            if link == "log1p":
                train["observed"] = np.log1p(train.observed)
            self.regression = Model(train,structure,alpha)
            return
        self.scale = StandardScaler().fit(frame[self.features])
        x = self.scale.transform(frame[self.features])
        y = frame.observed.to_numpy(int)
        counts = np.bincount(y,weights=self.weight,minlength=CATEGORIES)+.25
        cumulative = np.clip(np.cumsum(counts/counts.sum())[:-1],1e-6,1-1e-6)
        theta = np.log(cumulative/(1-cumulative))
        initial = np.r_[np.zeros(x.shape[1]),theta[0],np.log(np.diff(theta))]
        objective = ordinal_objective(x,y,self.weight,alpha)
        bounds = [(None,None)]*x.shape[1]+[(None,None)]+[(-8,8)]*(CATEGORIES-2)
        fitted = minimize(objective,initial,jac=True,method="L-BFGS-B",bounds=bounds,options={"maxiter":1000,"ftol":1e-11,"gtol":1e-6})
        if not fitted.success:
            raise ValueError("Ordinal optimization failed: "+str(fitted.message))
        self.beta = fitted.x[:x.shape[1]]
        a = fitted.x[x.shape[1]:]
        self.thresholds = np.r_[a[0],a[0]+np.cumsum(np.exp(a[1:]))]
        self.optimizer = dict(success=bool(fitted.success),iterations=int(fitted.nit),objective=float(fitted.fun),max_abs_gradient=float(np.max(abs(fitted.jac))))

    def predict(self, frame):
        if self.model == "constant":
            return np.tile(self.probability,(len(frame),1)) if self.target == "iga" else np.full(len(frame),self.offset)
        if self.target == "iga":
            return ordinal_probability(self.scale.transform(frame[self.features]),self.beta,self.thresholds)
        raw = self.regression.predict_raw(frame)
        if self.link == "log1p":
            raw = np.expm1(np.clip(raw,-30,30))
        return np.clip(raw,0,UPPER[self.target])

    def artifact(self):
        result = dict(target=self.target,model=self.model,features=self.features,alpha=self.alpha,link=self.link,input_units="BSA percentage 0..100; log2 positive Summary.xlsx concentrations in recorded pg/ml",missing_policy="reject incomplete inputs",training_weeks=[0,12],stage_drug_history_inputs=False)
        if self.target == "iga":
            slopes = self.beta/self.scale.scale_
            thresholds = self.thresholds+np.einsum("i,i->",slopes,self.scale.mean_)
            result.update(family="proportional_odds_ordinal_logistic",categories=list(range(6)),coefficients=dict(zip(self.features,map(float,slopes))),thresholds=list(map(float,thresholds)),
                          formula="P(recorded_grade <= k)=sigmoid(threshold[k]-sum(coefficient*input)); category probabilities are adjacent cumulative differences",decision="argmax category probability; expected grade is a secondary continuous summary",instrument_status="Recorded 0..5 grade. Exact IGA instrument identity pending; not validated vIGA-AD.",optimizer=self.optimizer)
        else:
            formula=self.regression.formula()
            formula["expression"]="clip(intercept+sum(coefficient*input),0,target_max)" if self.link=="identity" else "clip(expm1(intercept+sum(coefficient*input)),0,target_max)"
            result.update(family="ridge_regression",formula=formula,output_range=[0,UPPER[self.target]],log1p_interpretation="Inverse log1p without mean-smearing; candidate optimized for original-scale absolute error" if self.link=="log1p" else None)
        return result


def loss_per_row(frame,prediction,target):
    y=frame.observed.to_numpy()
    if target != "iga":
        return abs(prediction-y)
    truth=(y[:,None] <= np.arange(CATEGORIES-1)[None,:]).astype(float)
    cumulative=np.cumsum(prediction,axis=1)[:,:-1]
    return np.mean((cumulative-truth)**2,axis=1)


def choose(frame,target,model,cfg):
    if model=="constant":
        return Fit(frame,target,model),[]
    ids=np.array(sorted(frame.patient_uid.unique()))
    candidates=[(alpha,link) for alpha in cfg["ridge_alpha"] for link in (["identity"] if target=="iga" else cfg["continuous_links"])]
    results=[]
    for alpha,link in candidates:
        errors=[]
        for train,test in KFold(cfg["inner_folds"],shuffle=True,random_state=cfg["seed"]).split(ids):
            a=frame[frame.patient_uid.isin(ids[train])];b=frame[frame.patient_uid.isin(ids[test])]
            assert set(a.patient_uid).isdisjoint(b.patient_uid)
            fit=Fit(a,target,model,alpha,link)
            values=pd.DataFrame({"patient_uid":b.patient_uid,"loss":loss_per_row(b,fit.predict(b),target)})
            errors.extend(values.groupby("patient_uid").loss.mean())
        results.append(dict(alpha=alpha,link=link,inner_patient_loss=float(np.mean(errors)),loss="normalized_RPS" if target=="iga" else "MAE"))
    selected=min(results,key=lambda x:x["inner_patient_loss"])
    return Fit(frame,target,model,selected["alpha"],selected["link"]),results


def metrics(frame,target,cfg):
    outputs=[]
    for _,part in frame.groupby("repeat"):
        y=part.observed.to_numpy();prediction=part.prediction.to_numpy()
        values=scalar_metrics(y,prediction)
        for field in ["band_accuracy","sensitivity_gt7","specificity_le7"]:
            values.pop(field,None)
        if target=="iga":
            probability=part[[f"p{i}" for i in range(CATEGORIES)]].to_numpy()
            grade=probability.argmax(1);actual=y.astype(int)
            p01=probability[:,:2].sum(1);actual01=actual<=1
            values.update(normalized_RPS=float(np.mean(loss_per_row(part,probability,target))),exact_category=float(np.mean(grade==actual)),within_one_category=float(np.mean(abs(grade-actual)<=1)),
                          category_MAE=float(np.mean(abs(grade-actual))),quadratic_kappa=cohen_kappa_score(actual,grade,labels=list(range(CATEGORIES)),weights="quadratic") if len(np.unique(np.r_[actual,grade]))>1 else np.nan,
                          Brier_IGA01=float(np.mean((p01-actual01)**2)),AUROC_IGA01=roc_auc_score(actual01,p01) if len(np.unique(actual01))==2 else np.nan,
                          sensitivity_IGA01=float(np.mean(p01[actual01]>=.5)) if actual01.any() else np.nan,
                          specificity_IGA01=float(np.mean(p01[~actual01]<.5)) if (~actual01).any() else np.nan)
        outputs.append(values)
    result=pd.DataFrame(outputs).mean().to_dict()
    for name,col in [("MAE","absolute_error")]+([("normalized_RPS","RPS")] if target=="iga" else []):
        per=frame.groupby("patient_uid")[col].mean();lo,hi=bootstrap_mean(per,cfg);result[name+"_lo"],result[name+"_hi"]=lo,hi
    result.update(n_patients=int(frame.patient_uid.nunique()),n_visits=len(frame[["patient_uid","visit_week"]].drop_duplicates()),primary_metric="normalized_RPS" if target=="iga" else "MAE",evidence_status="CURRENT_RECORDED_GRADE_EXPLORATORY" if target=="iga" else "CURRENT_RECORDED_TARGET_EXPLORATORY")
    return result


def formula_predictions(frame,artifact):
    if artifact["target"]=="iga":
        linear=sum(value*frame[name].to_numpy() for name,value in artifact["coefficients"].items())
        cumulative=expit(np.array(artifact["thresholds"])[None,:]-linear[:,None])
        return np.diff(np.column_stack([np.zeros(len(frame)),cumulative,np.ones(len(frame))]),axis=1)
    formula=artifact["formula"];linear=np.full(len(frame),formula["intercept"])
    for name,value in formula["coefficients"].items():linear+=value*frame[name].to_numpy()
    if artifact["link"]=="log1p":linear=np.expm1(np.clip(linear,-30,30))
    return np.clip(linear,*artifact["output_range"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(os.environ.get("T2T_ROOT",Path(__file__).resolve().parents[3])))
    parser.add_argument("--config",default="config/auxiliary_targets_20261017.json")
    args=parser.parse_args();root=args.root.resolve();cfg=json.loads((root/args.config).read_text());out=root/cfg["output_path"];out.mkdir(parents=True,exist_ok=True)
    clinical=pd.read_csv(root/cfg["clinical_path"]);molecular=pd.read_csv(root/cfg["molecular_path"])
    available=clinical[clinical.visit_week.isin([0,12,24])&clinical.age.ge(12)].merge(molecular,on=["patient_uid","visit_week"],validate="one_to_one")
    all_predictions=[];audits=[];inner=[];cohort=[];formula_checks=[];confusions=[];flow=[]
    for target in cfg["targets"]:
        age_min=18 if target=="dlqi" else 12
        eligible=available[available.age.ge(age_min)]
        data=eligible.dropna(subset=FEATURES+[target]).copy().rename(columns={target:"observed"})
        assert np.isfinite(data[FEATURES+["observed"]]).all().all()
        assert data.observed.between(0,UPPER[target]).all() and data.bsa.between(0,100).all()
        if target=="iga":assert np.equal(data.observed,np.floor(data.observed)).all()
        ids=np.array(sorted(data.patient_uid.unique()))
        for week,g in eligible.groupby("visit_week"):
            flow.append(dict(target=target,week=week,age_min=age_min,n_age_eligible_assay_linked=int(g.patient_uid.nunique()),n_target_observed_assay_linked=int(g[target].notna().sum()),n_complete_four_inputs_assay_linked=int(g[FEATURES].notna().all(axis=1).sum()),n_analyzed=int(data.visit_week.eq(week).sum())))
        for repeat in range(cfg["outer_repeats"]):
            for fold,(tr,te) in enumerate(KFold(cfg["outer_folds"],shuffle=True,random_state=cfg["seed"]+repeat).split(ids)):
                train=data[data.patient_uid.isin(ids[tr])&data.visit_week.isin(cfg["development_weeks"])]
                test=data[data.patient_uid.isin(ids[te])]
                assert set(train.patient_uid).isdisjoint(test.patient_uid)
                for model in cfg["models"]:
                    fit,scores=choose(train,target,model,cfg);prediction=fit.predict(test)
                    q=test[["patient_uid","visit_week","observed","age","treatment","center","dose_ambiguous"]].copy()
                    q["prediction"]=np.einsum("ij,j->i",prediction,np.arange(CATEGORIES)) if target=="iga" else prediction
                    q["target"],q["model"],q["repeat"],q["fold"]=target,model,repeat,fold
                    q["absolute_error"]=abs(q.prediction-q.observed)
                    if target=="iga":
                        for i in range(CATEGORIES):q[f"p{i}"]=prediction[:,i]
                        q["RPS"]=loss_per_row(test,prediction,target)
                    all_predictions.append(q)
                    audits.append(dict(target=target,model=model,repeat=repeat,fold=fold,alpha=fit.alpha,link=fit.link,n_training_patients=int(train.patient_uid.nunique()),n_training_visits=len(train),max_training_week=int(train.visit_week.max()),min_training_age=float(train.age.min()),features="|".join(fit.features),patient_overlap=0,train_ids="|".join(sorted(train.patient_uid.unique())),test_ids="|".join(sorted(test.patient_uid.unique()))))
                    inner.extend(dict(target=target,model=model,repeat=repeat,fold=fold,**score) for score in scores)
                print(target,"repeat",repeat+1,"fold",fold+1,flush=True)
        development=data[data.visit_week.isin(cfg["development_weeks"])]
        fitted,scores=choose(development,target,"BSA_three",cfg);artifact=fitted.artifact()
        artifact.update(version="2026-10-17",status=cfg["status"],age_min=age_min,n_training_patients=int(development.patient_uid.nunique()),n_training_visits=len(development),selection="3-fold patient-inner MAE for continuous targets; normalized RPS for ordinal target. W24 absent from selection.",validation="2 repeats of 5-fold patient-held-out outer validation; W24 previously inspected same-cohort temporal transfer",assay_processing="Same positive Summary.xlsx exports as EASI input; no half-min replacement; below software detection bounds can remain positive. Assay bridge not established.")
        reconstruction=float(np.max(abs(formula_predictions(data,artifact)-fitted.predict(data))));assert reconstruction<1e-9
        formula_checks.append(dict(target=target,reconstruction_max_abs=reconstruction))
        (out/f"formula_{target}.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+"\n")
        pd.DataFrame(scores).to_csv(out/f"full_development_selection_{target}.csv",index=False)
        cohort.append(data[["patient_uid","visit_week","observed","age"]+FEATURES].assign(target=target))
    prediction=pd.concat(all_predictions,ignore_index=True)
    assert not prediction.duplicated(["target","model","repeat","patient_uid","visit_week"]).any()
    prediction.to_csv(out/"predictions_private.csv",index=False)
    pd.concat(cohort,ignore_index=True).to_csv(out/"input_private.csv",index=False)
    pd.DataFrame(audits).to_csv(out/"fit_audit_private.csv",index=False)
    pd.DataFrame(audits).drop(columns=["train_ids","test_ids"]).to_csv(out/"fit_audit.csv",index=False)
    pd.DataFrame(inner).to_csv(out/"inner_selection.csv",index=False)
    pd.DataFrame(flow).to_csv(out/"cohort_flow.csv",index=False)
    tables=[];comparisons=[];repeat_checks=[]
    for (target,model,week),part in prediction.groupby(["target","model","visit_week"]):
        subsets=[("overall","all",part)]
        subsets += [("treatment",label,g) for label,g in part.loc[~part.dose_ambiguous.astype(bool)].groupby("treatment")]
        subsets += [("center",label,g) for label,g in part.groupby("center")]
        for kind,label,g in subsets:
            tables.append(dict(target=target,model=model,week=week,group_type=kind,group=label,**metrics(g,target,cfg)))
        averaged=part.groupby("patient_uid").agg(observed=("observed","first"),prediction=("prediction","mean"))
        repeat_checks.append(dict(target=target,model=model,week=week,reported_mean_repeat_MAE=float(part.absolute_error.mean()),MAE_after_averaging_predictions=float(abs(averaged.prediction-averaged.observed).mean()),reported_method="Error calculated in each repeat before averaging"))
        if target=="iga":
            for repeat,g in part.groupby("repeat"):
                predicted=g[[f"p{i}" for i in range(CATEGORIES)]].to_numpy().argmax(1)
                actual=g.observed.to_numpy(int)
                for i in range(CATEGORIES):
                    for j in range(CATEGORIES):confusions.append(dict(model=model,week=week,repeat=repeat,observed_grade=i,predicted_grade=j,n=int(((actual==i)&(predicted==j)).sum())))
    summary=pd.DataFrame(tables);summary.to_csv(out/"metrics.csv",index=False);summary[summary.group_type.eq("overall")].to_csv(out/"overall_metrics.csv",index=False)
    for (target,week),part in prediction.groupby(["target","visit_week"]):
        keys=["patient_uid","visit_week","repeat","fold"]
        metric_name="RPS" if target=="iga" else "absolute_error"
        base=part[part.model.eq("BSA_three")]
        for comparator in ["BSA_only","molecules_only","constant"]:
            joined=base.merge(part[part.model.eq(comparator)][keys+[metric_name]],on=keys,validate="one_to_one",suffixes=("","_reference"))
            difference=(joined[metric_name]-joined[metric_name+"_reference"]).groupby(joined.patient_uid).mean()
            lo,hi=bootstrap_mean(difference,cfg)
            comparisons.append(dict(target=target,week=week,model="BSA_three",comparator=comparator,n_patients=len(difference),metric="normalized_RPS" if target=="iga" else "MAE",difference=float(difference.mean()),difference_lo=lo,difference_hi=hi,interpretation="negative favors BSA plus three molecules"))
    pd.DataFrame(comparisons).to_csv(out/"paired_comparisons.csv",index=False)
    pd.DataFrame(repeat_checks).to_csv(out/"repeat_aggregation_audit.csv",index=False)
    pd.DataFrame(confusions).to_csv(out/"IGA_confusion_by_repeat.csv",index=False)
    rng=np.random.default_rng(cfg["seed"]);x=rng.normal(size=(42,4));y=np.tile(np.arange(6),7);w=np.ones(42);objective=ordinal_objective(x,y,w,1.)
    v=np.r_[rng.normal(size=4)*.1,-1.5,np.zeros(4)]
    gradient_error=float(check_grad(lambda z:objective(z)[0],lambda z:objective(z)[1],v))
    assert gradient_error<1e-4
    probabilities=prediction.loc[prediction.target.eq("iga"),[f"p{i}" for i in range(6)]].to_numpy()
    checks=dict(patient_disjoint=True,inner_patient_disjoint=True,W24_absent_from_training_and_selection=True,no_current_target_inputs=True,no_stage_drug_or_history_inputs=True,DLQI_min_age18=True,IGA_ordinal_probabilities_sum1=bool(np.allclose(probabilities.sum(1),1)),IGA_probabilities_nonnegative=bool((probabilities>=0).all()),IGA_gradient_finite_difference_error=gradient_error,formulas_reconstruct=formula_checks,clinical_interpretation="Exploratory recorded targets; IGA instrument identity and PP-NRS recall window require source confirmation")
    (out/"verification.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n")
    sources=[root/args.config,Path(__file__),root/"src/cytot2t/curation/unified_model_20261017.py",root/cfg["clinical_path"],root/cfg["molecular_path"]]
    manifest=dict(version="2026-10-17",status=cfg["status"],private_files=["predictions_private.csv","input_private.csv","fit_audit_private.csv"],software={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},source_hashes={str(path.relative_to(root)):sha(path) for path in sources},outputs=[dict(path=str(path.relative_to(root)),sha256=sha(path)) for path in sorted(out.iterdir()) if path.is_file() and path.name!="manifest.json"],checks=checks)
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(summary[summary.group_type.eq("overall") & summary.model.eq("BSA_three")][["target","week","n_patients","MAE","RMSE","rho","CCC","normalized_RPS","exact_category"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
