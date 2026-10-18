"""Research-only calculation from separately fitted target-specific formulas."""
import argparse
import csv
import json
import math
import os
from pathlib import Path


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def calculate(row, artifact, input_scale="log2"):
    age, bsa = float(row["age"]), float(row["bsa"])
    if not math.isfinite(age) or age < artifact["age_min"]:
        raise ValueError("Age is outside this model's development population")
    if not math.isfinite(bsa) or not 0 <= bsa <= 100:
        raise ValueError("BSA must be a finite percentage from 0 to 100")
    values = {"bsa": bsa}
    for name in ["CCL17", "CCL26", "IL-5"]:
        value = float(row[name if input_scale == "raw" else "log2_" + name])
        if not math.isfinite(value):
            raise ValueError("Molecular inputs must be finite; missing inputs are not imputed")
        if input_scale == "raw":
            if value <= 0:
                raise ValueError("Raw concentrations must be positive")
            value = math.log2(value)
        elif input_scale != "log2":
            raise ValueError("input_scale must be raw or log2")
        values["log2_" + name] = value
    if artifact["target"] == "iga":
        linear = sum(coefficient * values[name] for name, coefficient in artifact["coefficients"].items())
        cumulative = [0.] + [sigmoid(threshold-linear) for threshold in artifact["thresholds"]] + [1.]
        probabilities = [cumulative[i+1]-cumulative[i] for i in range(6)]
        if min(probabilities) < -1e-12 or abs(sum(probabilities)-1) > 1e-12:
            raise ValueError("Invalid ordinal formula probabilities")
        return {"expected_recorded_grade": sum(i*p for i,p in enumerate(probabilities)), "predicted_recorded_grade": max(range(6),key=lambda i:probabilities[i]), **{f"p{i}":p for i,p in enumerate(probabilities)}}
    formula = artifact["formula"]
    value = formula["intercept"] + sum(coefficient*values[name] for name,coefficient in formula["coefficients"].items())
    if artifact["link"] == "log1p":
        value = math.expm1(min(30,max(-30,value)))
    low,high = artifact["output_range"]
    return {artifact["target"] + "_estimate": min(high,max(low,value))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--target", choices=["pp_nrs","dlqi","iga"], required=True)
    parser.add_argument("--input-scale", choices=["log2","raw"], default="log2")
    parser.add_argument("--formula", type=Path)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("T2T_ROOT",Path(__file__).resolve().parents[3])))
    args=parser.parse_args()
    path=args.formula or args.root/f"results/tables/update_20261017/model/auxiliary_targets/formula_{args.target}.json"
    artifact=json.loads(path.read_text())
    if artifact["target"]!=args.target:
        raise ValueError("Requested target does not match supplied formula")
    with args.input_csv.open(newline="",encoding="utf-8-sig") as handle:
        reader=csv.DictReader(handle);fields=list(reader.fieldnames or []);rows=list(reader)
    outputs=[]
    for number,row in enumerate(rows,start=2):
        try:outputs.append(dict(row,**calculate(row,artifact,args.input_scale)))
        except (KeyError,ValueError,TypeError) as error:raise ValueError(f"Input row {number}: {error}") from error
    extra=[] if not outputs else [key for key in outputs[0] if key not in fields]
    with args.output_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields+extra);writer.writeheader();writer.writerows(outputs)


if __name__=="__main__":
    main()
