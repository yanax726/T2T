"""Semantic colours shared by the manuscript and presentation figures."""
from pathlib import Path
import json
import re
import numpy as np
from matplotlib.colors import ListedColormap, to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.collections import Collection
from matplotlib.patches import Patch
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / 'config/palette_v4_registry.json').read_text())


def treatment_label(key):
    return REGISTRY['display_labels'][key]


def display_text(text):
    """Use unambiguous treatment names without altering analytical identifiers."""
    text = str(text)
    text = re.sub(r'\b(?:Pooled|pooled)[ _](?:abrocitinib|A)\b', 'pooled abro', text)
    text = re.sub(r'\b(?:Dupilumab|dupilumab)\b', 'dupi', text)
    text = re.sub(r'\b(?:Abrocitinib|abrocitinib)\b', 'abro', text)
    text = re.sub(r'\bA100\b', 'abro 100 mg', text)
    text = re.sub(r'\bA200\b', 'abro 200 mg', text)
    text = re.sub(r'\bD\b', 'dupi', text)
    text = re.sub(r'\bA: pooled abro\b', 'pooled abro', text)
    text = text.replace('dupi: dupi', 'dupi')
    return text


def apply_display_labels(fig):
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            labels = [item.get_text() for item in axis.get_majorticklabels()]
            renamed = [display_text(value) for value in labels]
            if axis is ax.xaxis:
                renamed = [value.replace('abro ', 'abro\n', 1)
                           if value in ['abro 100 mg', 'abro 200 mg'] else value
                           for value in renamed]
            if renamed != labels:
                axis.set_ticks(axis.get_majorticklocs(), labels=renamed)
    for artist in fig.findobj(match=Text):
        artist.set_text(display_text(artist.get_text()))


def _entry(role):
    family, key = role.split('.', 1)
    return REGISTRY[family][key]


def color(role):
    return _entry(role)['hex']


def marker(role):
    return _entry(role).get('marker', 'o')


def role_style(role):
    return dict(_entry(role))


def sequence(name):
    return REGISTRY['sequences'][name]['hex']


def cmap(name):
    result = ListedColormap(REGISTRY['continuous'][name]['lut_hex'], name=name)
    result.set_bad(REGISTRY['neutral']['missing'])
    return result


def register(fig, role, actual_color=None):
    actual = to_hex(actual_color or color(role)).upper()
    expected = color(role).upper()
    if actual != expected:
        raise ValueError(f'{role}: expected {expected}, got {actual}')
    roles = getattr(fig, '_semantic_roles', {})
    if role in roles and roles[role] != actual:
        raise ValueError(f'Inconsistent colour for {role}')
    if any(other != role and rgb == actual for other, rgb in roles.items()):
        raise ValueError(f'Colour {actual} assigned to multiple categorical roles')
    roles[role] = actual
    fig._semantic_roles = roles
    return actual


def audit_figure(fig, figure_id):
    """Collect explicit categorical colours across all panels before export.

    Continuous images and scalar-mapped collections use their own numerical
    colour bars, so they are not treated as categorical colour assignments.
    """
    known = {}
    for family in ('treatment', 'visit', 'method'):
        for key, entry in REGISTRY[family].items():
            role = family + '.' + key
            rgb = entry['hex'].upper()
            if rgb in known:
                raise ValueError(f'{rgb}: duplicate roles {known[rgb]}, {role}')
            known[rgb] = role
    for name, entry in REGISTRY['sequences'].items():
        for i, value in enumerate(entry['hex']):
            rgb = value.upper()
            if rgb in known:
                raise ValueError(f'{rgb}: duplicate categorical role')
            known[rgb] = f'{name}.{i}'
    used = dict(getattr(fig, '_semantic_roles', {}))
    for artist in fig.findobj():
        samples = []
        if isinstance(artist, Line2D):
            samples = [artist.get_color(), artist.get_markerfacecolor(), artist.get_markeredgecolor()]
        elif isinstance(artist, Collection):
            if artist.get_array() is None:
                samples = list(artist.get_facecolors()) + list(artist.get_edgecolors())
        elif isinstance(artist, Patch):
            samples = [artist.get_facecolor(), artist.get_edgecolor()]
        for sample in samples:
            try:
                rgba = to_rgba(sample)
            except (ValueError, TypeError):
                continue
            if rgba[3] == 0:
                continue
            rgb = to_hex(rgba).upper()
            role = known.get(rgb)
            if role:
                if role in used and used[role] != rgb:
                    raise ValueError(f'{figure_id}: inconsistent colour for {role}')
                used[role] = rgb
    inverse = {}
    for role, rgb in used.items():
        if rgb in inverse and inverse[rgb] != role:
            raise ValueError(f'{figure_id}: {rgb} reused for different categorical roles')
        inverse[rgb] = role
    rule = REGISTRY['figure_rules'][figure_id]
    allowed = rule['categorical_families']
    unexpected = [role for role in used if role.split('.')[0] not in allowed]
    if unexpected:
        raise ValueError(f'{figure_id}: unexpected categorical roles {unexpected}')
    return {'status': 'PASS', 'figure': figure_id, 'roles': used,
            'continuous_scales': rule.get('continuous_scales', []),
            'scope': 'Categorical RGB consistency across the complete Figure; perceptual separation requires rendered review.'}
