"""Semantic colours shared by the manuscript and presentation figures."""
from pathlib import Path
import json
import re
import numpy as np
from matplotlib.colors import ListedColormap, to_hex, to_rgba
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.collections import Collection
from matplotlib.image import AxesImage
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
    # Keep abbreviation definitions informative after display-name conversion.
    text = re.sub(r'\bdupi( denotes |, )dupi\b', r'dupi\1dupilumab', text)
    text = re.sub(r'\babro( denotes |, )abro\b', r'abro\1abrocitinib', text)
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
    if family in REGISTRY.get('sequences', {}):
        return {'hex': REGISTRY['sequences'][family]['hex'][int(key)]}
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


def register(fig, role, actual_color=None, *, artists=None):
    """Declare a figure-wide colour meaning; optionally bind its actual artists."""
    actual = to_hex(color(role) if actual_color is None else actual_color).upper()
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
    if artists is not None:
        for artist in _flatten_artists(artists):
            if artist.figure is not fig:
                raise ValueError(f'{role}: bound artist belongs to another figure')
            previous = getattr(artist, '_semantic_color_role', role)
            if previous != role:
                raise ValueError(f'Artist assigned to both {previous} and {role}')
            artist._semantic_color_role = role
    return actual


def bind_artist(fig, role, artists):
    """Bind lines, patches, collections or an errorbar container to one role."""
    register(fig, role, artists=artists)


def _flatten_artists(items):
    if isinstance(items, Artist):
        yield items
    elif items is not None:
        for item in items:
            yield from _flatten_artists(item)


def categorical_entries():
    """Read semantic families, including future registry additions."""
    entries = {}
    for family, values in REGISTRY.items():
        if family in ('continuous', 'sequences', 'figure_rules', 'aesthetic') or not isinstance(values, dict):
            continue
        for key, entry in values.items():
            if isinstance(entry, dict) and isinstance(entry.get('hex'), str):
                entries[f'{family}.{key}'] = entry
    for name, entry in REGISTRY['sequences'].items():
        for i, value in enumerate(entry['hex']):
            entries[f'{name}.{i}'] = {'hex': value}
    return entries


def _visible(artist):
    if not artist.get_visible():
        return False
    alpha = artist.get_alpha()
    return alpha is None or not np.isscalar(alpha) or alpha != 0


def _artist_samples(artist):
    """Only sample painted channels, not disabled marker defaults."""
    samples = []
    if isinstance(artist, Line2D):
        n_points = len(artist.get_xdata())
        if n_points > 1 and artist.get_linestyle() not in ('None', 'none', '', ' ') and artist.get_linewidth() > 0:
            samples.append(('line', artist.get_color()))
        if n_points and artist.get_marker() not in (None, 'None', 'none', '', ' ') and artist.get_markersize() > 0:
            style = MarkerStyle(artist.get_marker(), fillstyle=artist.get_fillstyle())
            if style.is_filled():
                samples.append(('marker_face', artist.get_markerfacecolor()))
            if artist.get_markeredgewidth() > 0:
                samples.append(('marker_edge', artist.get_markeredgecolor()))
    elif isinstance(artist, Collection):
        if artist.get_array() is None:
            samples.extend(('face', value) for value in artist.get_facecolors())
            if any(np.asarray(artist.get_linewidths()) > 0):
                samples.extend(('edge', value) for value in artist.get_edgecolors())
    elif isinstance(artist, Patch):
        if artist.get_fill():
            samples.append(('face', artist.get_facecolor()))
        if artist.get_linewidth() > 0:
            samples.append(('edge', artist.get_edgecolor()))
    elif isinstance(artist, Text) and artist.get_text().strip():
        samples.append(('text', artist.get_color()))
    result = []
    for channel, sample in samples:
        try:
            rgba = to_rgba(sample)
        except (ValueError, TypeError):
            continue
        if rgba[3] == 0:
            continue
        result.append((channel, to_hex(rgba).upper(), float(rgba[3])))
    return result


def _neutral(rgb):
    values = np.asarray(to_rgba(rgb)[:3])
    return np.ptp(values) < .12


def _role_matches(rgb, entries, derived_roles=()):
    exact = []
    for role, entry in entries.items():
        for channel in ('hex', 'outline', 'area_fill'):
            if entry.get(channel) and to_hex(entry[channel]).upper() == rgb:
                exact.append((role, channel))
    if exact:
        return exact
    # A preblended fill has a semantic owner only when the artist was explicitly
    # bound. Similar pale RGBs alone do not establish a categorical meaning.
    target = np.asarray(to_rgba(rgb)[:3])
    matches = []
    for role, entry in entries.items():
        if role not in derived_roles:
            continue
        primary = np.asarray(to_rgba(entry['hex'])[:3])
        direction = primary - 1
        denominator = direction @ direction
        if denominator == 0:
            continue
        alpha = ((target - 1) @ direction) / denominator
        if .05 <= alpha <= .45 and np.max(np.abs(target - (1 + alpha * direction))) <= 1 / 255:
            matches.append((role, 'derived_fill'))
    return matches


def _aesthetic_colours():
    result = set()
    def collect(value):
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, str):
            try:
                result.add(to_hex(value).upper())
            except ValueError:
                pass
    collect(REGISTRY.get('neutral', {}))
    collect(REGISTRY.get('aesthetic', {}))
    return result


def _continuous_record(artist):
    cmap = artist.get_cmap()
    samples = np.asarray(cmap(np.linspace(0, 1, 257)))[:, :3]
    matched = []
    variant = 'complete'
    for name, entry in REGISTRY['continuous'].items():
        expected = ListedColormap(entry['lut_hex'])(np.linspace(0, 1, 257))[:, :3]
        if np.allclose(samples, expected, atol=1 / 255):
            matched.append(name)
            continue
        # Count maps deliberately omit the near-white first part of the LUT.
        if isinstance(cmap, ListedColormap) and 1 < cmap.N < len(entry['lut_hex']):
            actual = np.asarray(cmap(np.linspace(0, 1, cmap.N)))[:, :3]
            full = np.asarray([to_rgba(value)[:3] for value in entry['lut_hex']])
            for start in range(len(full) - len(actual) + 1):
                if np.allclose(actual, full[start:start + len(actual)], atol=1 / 255):
                    matched.append(name)
                    variant = f'contiguous subset {start}:{start + len(actual)}'
                    break
    return {'artist': type(artist).__name__, 'name': cmap.name,
            'registry_scale': matched[0] if len(matched) == 1 else None,
            'variant': variant,
            'vmin': None if artist.norm.vmin is None else float(artist.norm.vmin),
            'vmax': None if artist.norm.vmax is None else float(artist.norm.vmax)}


def audit_figure(fig, figure_id):
    """Check declared meanings against painted colours across the entire figure.

    Inferred colour matches are reported separately: RGB alone cannot establish
    what a reader or author means by a mark. Numerical colour maps are audited
    separately from categorical roles. The audit does not certify perception.
    """
    entries = categorical_entries()
    declared = dict(getattr(fig, '_semantic_roles', {}))
    rule = REGISTRY['figure_rules'][figure_id]
    allowed = rule['categorical_families']
    aesthetic = _aesthetic_colours()
    observed, unknown, continuous = {}, {}, []
    bound = []
    for artist in fig.findobj():
        if not _visible(artist):
            continue
        if isinstance(artist, (AxesImage, Collection)) and artist.get_array() is not None:
            continuous.append(_continuous_record(artist))
            continue
        samples = _artist_samples(artist)
        owner = getattr(artist, '_semantic_color_role', None)
        owner_seen = False
        for channel, rgb, alpha in samples:
            if rgb in aesthetic:
                continue
            matches = _role_matches(rgb, entries, (owner,) if owner else ())
            if owner:
                owned = [(role, variant) for role, variant in matches if role == owner]
                if owned:
                    matches = owned
                    owner_seen = True
                elif not _neutral(rgb):
                    raise ValueError(f'{figure_id}: artist bound to {owner} uses {rgb}')
            if len({role for role, _ in matches}) > 1:
                # A near-white tint can match more than one base by rounding.
                # It must not invent a categorical role without a binding.
                if all(variant == 'derived_fill' for _, variant in matches):
                    continue
                raise ValueError(f'{figure_id}: {rgb} has ambiguous categorical roles {matches}')
            if matches:
                role, variant = matches[0]
                record = observed.setdefault(role, {'hex': color(role).upper(), 'samples': 0,
                                                    'variants': set(), 'artist_types': set()})
                record['samples'] += 1
                record['variants'].add(variant)
                record['artist_types'].add(type(artist).__name__)
            elif not _neutral(rgb):
                record = unknown.setdefault(rgb, {'samples': 0, 'artist_types': set()})
                record['samples'] += 1
                record['artist_types'].add(type(artist).__name__)
        if owner:
            if not owner_seen:
                raise ValueError(f'{figure_id}: bound {owner} has no visible role-coloured mark')
            bound.append({'role': owner, 'artist': type(artist).__name__})
    for role, rgb in declared.items():
        if role not in entries or rgb != color(role).upper():
            raise ValueError(f'{figure_id}: invalid declaration {role}={rgb}')
        if role not in observed:
            raise ValueError(f'{figure_id}: registered {role} has no visible coloured artist')
    used = {role: record['hex'] for role, record in observed.items()}
    unexpected = [role for role in used if role.split('.')[0] not in allowed]
    if unexpected:
        raise ValueError(f'{figure_id}: unexpected categorical roles {unexpected}')
    scale_rules = rule.get('continuous_scales', [])
    for record in continuous:
        name = record['registry_scale']
        if name is not None and not any(value == name or value.startswith(name + ' ') for value in scale_rules):
            raise ValueError(f'{figure_id}: unexpected numerical scale {name}')
    warnings = []
    if unknown:
        warnings.append('Unregistered chromatic colours require source/visual review.')
    if any(record['registry_scale'] is None for record in continuous):
        warnings.append('Unregistered numerical colour map requires source/visual review.')
    for record in list(observed.values()) + list(unknown.values()):
        for key, value in list(record.items()):
            if isinstance(value, set):
                record[key] = sorted(value)
    inferred = {role: rgb for role, rgb in used.items() if role not in declared}
    return {'status': 'PASS' if not warnings else 'REVIEW', 'figure': figure_id,
            'roles': used, 'declared_roles': declared, 'inferred_roles': inferred,
            'observed_role_details': observed, 'bound_artists': bound,
            'unregistered_chromatic_colours': unknown,
            'continuous_scales': scale_rules, 'observed_continuous_scales': continuous,
            'warnings': warnings,
            'scope': ('Visible artist colours and numerical maps checked across the complete Figure. '
                      'Unbound semantic meanings and perceptual separation require source and rendered review.')}
