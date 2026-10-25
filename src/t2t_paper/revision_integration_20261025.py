"""Retain population, confounding and detailed prediction evidence in the appendix."""
from . import figures as base
from . import main_redesign_20261018 as previous
from .clinical_framework_20261018 import cohort_flow


def supporting_figures(root):
    cohort_flow(root, base)
    previous.balance_diagnostics(root)
    original_save = base.save

    def supplementary_save(fig, name, title, caption, sources, root=root):
        renamed = {'Figure4': 'FigureS44', 'Figure5': 'FigureS45'}[name]
        original_save(fig, renamed, title, caption, sources, root)

    try:
        base.save = supplementary_save
        previous.model_readout(root)
        previous.monitoring(root)
    finally:
        base.save = original_save
