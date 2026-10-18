"""Generate aggregate evidence used by the redesigned figures after analysis."""
from pathlib import Path
import argparse
import os
from . import cohort_flow_20261018, model_display_evidence_20261018, balance_support_20261018, plot_source_aggregates_20261018


def run(root):
    cohort_flow_20261018.run(root)
    model_display_evidence_20261018.run(root)
    balance_support_20261018.run(root)
    plot_source_aggregates_20261018.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('T2T_ROOT', Path(__file__).resolve().parents[2])))
    run(parser.parse_args().root)
