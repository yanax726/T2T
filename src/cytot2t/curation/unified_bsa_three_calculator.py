"""Calculate the versioned common EASI estimate from BSA and three molecules."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

LOG_INPUTS = ["log2_CCL17", "log2_CCL26", "log2_IL-5"]


def calculate(row, artifact, input_scale="log2"):
    if artifact["selected_structure"] != "unified_area_interaction":
        raise ValueError("This calculator requires the frozen common area-interaction formula")
    bsa = float(row["bsa"])
    if not math.isfinite(bsa) or not 0 <= bsa <= 100:
        raise ValueError("BSA must be a finite percentage between 0 and 100")
    values = {}
    for name in LOG_INPUTS:
        if input_scale == "raw":
            concentration = float(row[name.removeprefix("log2_")])
            if not math.isfinite(concentration) or concentration <= 0:
                raise ValueError("Raw concentrations must be positive and finite")
            values[name] = math.log2(concentration)
        elif input_scale == "log2":
            values[name] = float(row[name])
        else:
            raise ValueError("input_scale must be log2 or raw")
        if not math.isfinite(values[name]):
            raise ValueError(f"{name} must be finite; missing inputs are not imputed")
    formula = artifact["formula"]
    factor = formula["bsa_factor"] + sum(formula["bsa_log2_coefficients"][name] * value for name, value in values.items())
    return min(72.0, max(0.0, bsa * factor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("T2T_ROOT", Path(__file__).resolve().parents[3])))
    parser.add_argument("--formula", type=Path)
    parser.add_argument("--input-scale", choices=["log2", "raw"], default="log2")
    args = parser.parse_args()
    artifact_path = args.formula or args.root / "config/unified_bsa_three_v1_20261017.json"
    artifact = json.loads(artifact_path.read_text())
    with args.input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = ["bsa"] + (LOG_INPUTS if args.input_scale == "log2" else [name.removeprefix("log2_") for name in LOG_INPUTS])
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Required columns: " + ", ".join(required))
        fields = list(reader.fieldnames)
        rows = list(reader)
    for number, row in enumerate(rows, start=2):
        try:
            row["EASI_estimate"] = calculate(row, artifact, args.input_scale)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Input row {number}: {error}") from error
    if "EASI_estimate" not in fields:
        fields.append("EASI_estimate")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
