#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
if [[ -n "${T2T_PYTHON:-}" ]]; then python_bin="$T2T_PYTHON";
elif [[ -x .venv/bin/python ]]; then python_bin="$project_root/.venv/bin/python";
else python_bin=python3; fi
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$project_root/results/audit/update_20261017/mpl"
mkdir -p "$MPLCONFIGDIR" reports/update_20261017/logs
mode="${1:-rebuild}"
case "$mode" in
  demo)
    "$python_bin" -m t2t_paper.demo
    ;;
  test)
    "$python_bin" -m unittest discover -s tests -p 'test_*calculator.py'
    "$python_bin" -m unittest discover -s tests -p 'test_unified_bsa_three.py'
    "$python_bin" -m unittest discover -s tests -p 'test_rounding_20261017.py'
    "$python_bin" -m unittest discover -s tests -p 'test_redesign_evidence_20261018.py'
    "$python_bin" -m unittest discover -s tests -p 'test_public_package.py'
    ;;
  figures)
    "$python_bin" -m t2t_paper.figures
    ;;
  manuscript)
    "$python_bin" -m t2t_paper.build --compile
    ;;
  guide)
    "$python_bin" -m t2t_paper.figure_script_zh
    "$python_bin" -m t2t_paper.export_script_pdf
    ;;
  rebuild)
    "$python_bin" -m t2t_paper.figures
    "$python_bin" -m t2t_paper.build --compile
    ;;
  sensitivity)
    "$python_bin" -m t2t_paper.preflight
    if [[ ! -f results/tables/update_20261017/model/predictions.csv ]]; then
      echo 'The frozen development predictions are missing. Run ./run.sh analyse with authorized inputs first.' >&2
      exit 2
    fi
    "$python_bin" -m cytot2t.curation.clinical_evidence_20261017
    "$python_bin" -m cytot2t.curation.ige_missingness_20261025
    "$python_bin" -m cytot2t.curation.model_ablation_20261025
    "$python_bin" -m t2t_paper.figures
    "$python_bin" -m t2t_paper.build --compile
    ;;
  analyse)
    "$python_bin" -m t2t_paper.preflight
    "$python_bin" -m cytot2t.curation.clinical_evidence_20261017
    "$python_bin" -m cytot2t.curation.regional_evidence_20261017
    "$python_bin" -m cytot2t.curation.unified_model_20261017
    "$python_bin" -m cytot2t.curation.rounding_20261017
    "$python_bin" -m cytot2t.curation.published_clipping_sensitivity_20261017
    "$python_bin" -m cytot2t.curation.auxiliary_targets_20261017
    "$python_bin" -m cytot2t.curation.unified_bsa_three_calculator examples/unified_easi_input.csv results/tables/update_20261017/model/calculator_raw_example_output.csv --input-scale raw
    for score_target in pp_nrs dlqi iga; do
      "$python_bin" -m cytot2t.curation.auxiliary_targets_calculator_20261017 examples/auxiliary_targets_input.csv "results/tables/update_20261017/model/auxiliary_targets/example_${score_target}.csv" --input-scale raw --target "$score_target"
    done
    "$python_bin" -m cytot2t.curation.reference_sensitivity_20261018
    "$python_bin" -m t2t_paper.redesign_evidence_20261018
    "$python_bin" -m t2t_paper.revision_checks_20261018
    "$python_bin" -m cytot2t.curation.ige_missingness_20261025
    "$python_bin" -m cytot2t.curation.model_ablation_20261025
    "$python_bin" -m t2t_paper.figures
    "$python_bin" -m t2t_paper.build --compile
    ;;
  *)
    echo 'Usage: ./run.sh [demo|rebuild|figures|manuscript|guide|test|sensitivity|analyse]'
    exit 2
    ;;
esac
