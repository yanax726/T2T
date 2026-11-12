#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
if [[ -n "${T2T_PYTHON:-}" ]]; then python_bin="$T2T_PYTHON";
elif [[ -x .venv/bin/python ]]; then python_bin="$project_root/.venv/bin/python";
else python_bin=python3; fi
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$project_root/results/.cache/matplotlib"
export XDG_CACHE_HOME="$project_root/results/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig" reports/update_20261017/logs
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

run_update_analysis() {
  "$python_bin" -m cytot2t.curation.laboratory_molecular_revision_20261107
  "$python_bin" -m cytot2t.curation.molecular_upgrade_20261107
  "$python_bin" -m cytot2t.curation.lab_display_20261112
  "$python_bin" -m cytot2t.curation.local_paired_benchmark_20261101
  if [[ -f data/clinical_alignment/external/OMIX002850-01.xlsx ]]; then
    "$python_bin" -m cytot2t.curation.external_molecular_benchmark_20261101
  else
    echo "SKIP: OMIX002850 input unavailable; external benchmark was not rerun. Other update analyses completed." >&2
  fi
}
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
    "$python_bin" -m unittest discover -s tests -p 'test_response_intervals_20261031.py'
    "$python_bin" -m unittest discover -s tests -p 'test_palette_semantics_20261031.py'
    ;;
  figures)
    "$python_bin" -m t2t_paper.figures
    ;;
  lab-update)
    "$python_bin" -m t2t_paper.preflight
    "$python_bin" -m cytot2t.curation.lab_display_20261112
    "$python_bin" -m t2t_paper.figures
    "$python_bin" -m t2t_paper.build --compile
    ;;
  itch-update)
    "$python_bin" -m t2t_paper.preflight
    "$python_bin" -m cytot2t.curation.itch_prediction_20261112
    "$python_bin" -m cytot2t.curation.verify_itch_prediction_20261112
    "$python_bin" -m t2t_paper.build --compile
    ;;
  manuscript)
    "$python_bin" -m t2t_paper.build --compile
    ;;
  guide)
    echo 'The legacy Chinese guide uses retired figure numbers and is archived. Current figure explanations are in manuscript/figure_book.pdf and the notes of manuscript/figure_deck.pptx.' >&2
    exit 2
    ;;
  rebuild)
    "$python_bin" -m t2t_paper.figures
    "$python_bin" -m t2t_paper.build --compile
    ;;
  update-analysis)
    "$python_bin" -m t2t_paper.preflight
    run_update_analysis
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
    "$python_bin" -m cytot2t.curation.manuscript_audit_20261031
    "$python_bin" -m cytot2t.curation.iga_definition_sensitivity_20261031
    "$python_bin" -m t2t_paper.model_display_evidence_20261018
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
    "$python_bin" -m cytot2t.curation.manuscript_audit_20261031
    "$python_bin" -m cytot2t.curation.iga_definition_sensitivity_20261031
    run_update_analysis
    "$python_bin" -m t2t_paper.figures
    "$python_bin" -m t2t_paper.build --compile
    ;;
  *)
    echo 'Usage: ./run.sh [demo|rebuild|figures|manuscript|guide|test|sensitivity|update-analysis|lab-update|itch-update|analyse]'
    exit 2
    ;;
esac
