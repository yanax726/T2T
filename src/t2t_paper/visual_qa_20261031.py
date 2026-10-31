"""Read-only palette inventory and colour-vision review copies of exported figures.

Run after the authoritative figure renderer. All outputs are QA copies; no
source data, figure, statistical model or registry is edited by this module.
"""
from argparse import ArgumentParser
from contextlib import nullcontext
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np

from . import palette


# Machado, Oliveira & Fernandes, IEEE TVCG 2009, severity 1.0 matrices.
# Apply to linear RGB, then return to sRGB; these are simulations, not a
# certification that a particular reader can distinguish every category.
MATRICES = {
    'protanopia': [[.152286, 1.052583, -.204868], [.114503, .786281, .099216],
                   [-.003882, -.048116, 1.051998]],
    'deuteranopia': [[.367322, .860646, -.227968], [.280085, .672501, .047413],
                     [-.011820, .042940, .968881]],
    'tritanopia': [[1.255528, -.076749, -.178779], [-.078411, .930809, .147602],
                   [.004733, .691367, .303900]],
}
MODES = ['original', 'protanopia', 'deuteranopia', 'tritanopia', 'grayscale']
SOURCE = 'https://doi.org/10.1109/TVCG.2009.113'


def _unit_rgb(rgb):
    values = np.asarray(rgb, float)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError('Expected a final RGB channel dimension of three')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite RGB values cannot be simulated')
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError('Input RGB values must lie in [0, 1]')
    return values


def _mix_channels(values, matrix):
    # Explicit channel arithmetic avoids the platform BLAS path for large,
    # three-dimensional images and preserves the scalar matrix calculation.
    return np.stack([values[..., 0] * row[0] + values[..., 1] * row[1] +
                     values[..., 2] * row[2] for row in matrix], axis=-1)


def linear_rgb(rgb):
    rgb = _unit_rgb(rgb)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def srgb(linear):
    linear = np.asarray(linear, float)
    if not np.isfinite(linear).all():
        raise ValueError('Non-finite linear RGB values cannot be encoded')
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)


def simulate(rgb, mode):
    rgb = _unit_rgb(rgb)
    if mode == 'original':
        return rgb
    values = linear_rgb(rgb)
    if mode == 'grayscale':
        luminance = values[..., 0] * .2126 + values[..., 1] * .7152 + values[..., 2] * .0722
        result = srgb(np.repeat(luminance[..., None], 3, axis=-1))
    else:
        result = srgb(_mix_channels(values, MATRICES[mode]))
    return _unit_rgb(result)


def lab(rgb):
    xyz = _mix_channels(linear_rgb(rgb), [[.4124564, .3575761, .1804375],
                                         [.2126729, .7151522, .0721750],
                                         [.0193339, .1191920, .9503041]])
    xyz /= [.95047, 1., 1.08883]
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def image_rgb(path):
    image = np.asarray(plt.imread(path), float)
    if image.ndim == 2:
        return _unit_rgb(np.repeat(image[..., None], 3, axis=-1))
    if image.shape[-1] == 4:
        return _unit_rgb(image[..., :3] * image[..., 3:] + (1 - image[..., 3:]))
    return _unit_rgb(image[..., :3])


def swatch_sheet(output):
    entries = palette.categorical_entries()
    rows = list(entries)
    fig, ax = plt.subplots(figsize=(11.6, max(5., len(rows) * .27)))
    for i, role in enumerate(rows):
        rgb = np.asarray(to_rgb(entries[role]['hex']))
        for j, mode in enumerate(MODES):
            ax.add_patch(Rectangle((j, i - .36), .86, .72,
                                   facecolor=simulate(rgb, mode), linewidth=0))
        ax.text(-.1, i, role, ha='right', va='center', fontsize=8, color='#28313A')
    ax.set(xlim=(-2.5, 5), ylim=(len(rows) - .5, -1.1))
    for j, mode in enumerate(MODES):
        ax.text(j + .43, -.8, mode.capitalize(), ha='center', va='center', fontsize=9)
    ax.axis('off')
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.98)
    fig.savefig(output / 'palette_vision_review.png', dpi=180)
    fig.savefig(output / 'palette_vision_review.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def pair_diagnostics(roles):
    rows = []
    for (role_a, hex_a), (role_b, hex_b) in itertools.combinations(roles.items(), 2):
        rgb = np.array([to_rgb(hex_a), to_rgb(hex_b)])
        distances = {}
        for mode in MODES:
            points = lab(simulate(rgb, mode))
            distances[mode] = round(float(np.linalg.norm(points[0] - points[1])), 3)
        rows.append({'role_a': role_a, 'role_b': role_b, 'cie76_delta_e': distances})
    return rows


def run(root, output, figures=None, copies=True):
    root, output = Path(root), Path(output)
    source = root / 'manuscript/figures'
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('QA output must be separate from authoritative figures')
    catalogue = json.loads((root / 'manuscript/figure_catalog.json').read_text())
    ids = [row['id'] for row in catalogue if figures is None or row['id'] in figures]
    report = {'palette_version': palette.REGISTRY['version'], 'figures': [],
              'simulation_reference': SOURCE, 'page_order': MODES,
              'scope': ('QA simulations at full dichromacy severity on linear RGB; grayscale uses '
                        'relative luminance. Delta E76 is descriptive and is not an accessibility '
                        'pass/fail criterion. Inspect labels, overlap and actual physical size.'),
              'source_data_changed': False}
    thumbnails = []
    for figure_id in ids:
        png = source / f'{figure_id}.png'
        colour_report = root / 'results/audit/update_20261017/figures' / f'{figure_id}.colour.json'
        colours = json.loads(colour_report.read_text())
        rgb = image_rgb(png)
        row = {'id': figure_id, 'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
               'audit_status': colours['status'], 'declared_roles': colours.get('declared_roles', {}),
               'inferred_roles': colours.get('inferred_roles', {}),
               'roles': colours['roles'],
               'unregistered_chromatic_colours': colours.get('unregistered_chromatic_colours', {}),
               'continuous_scales': colours.get('observed_continuous_scales', []),
               'pairwise_separation': pair_diagnostics(colours['roles']),
               'source_rgb_shape': list(rgb.shape),
               'source_rgb_range': [float(rgb.min()), float(rgb.max())],
               'simulation_checks': {}}
        destination = output / f'{figure_id}_vision_review.pdf'
        context = PdfPages(destination, metadata={'CreationDate': None, 'ModDate': None}) if copies else nullcontext()
        with context as pdf:
            for mode in MODES:
                simulated = simulate(rgb, mode)
                row['simulation_checks'][mode] = {
                    'all_finite': bool(np.isfinite(simulated).all()),
                    'range': [float(simulated.min()), float(simulated.max())],
                    'pixels_changed': int(np.count_nonzero(np.any(np.abs(simulated - rgb) > 1e-7, axis=-1))),
                    'shape_preserved': simulated.shape == rgb.shape}
                if copies:
                    # Existing PNG exports are 300 dpi. Each review page keeps
                    # their original physical dimensions, without new labels.
                    fig = plt.figure(figsize=(rgb.shape[1] / 300, rgb.shape[0] / 300))
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.imshow(simulated, interpolation='none')
                    ax.axis('off')
                    pdf.savefig(fig, dpi=180)
                    plt.close(fig)
        if copies:
            row['review_pdf'] = destination.name
        # This montage is for whole-paper harmony; use individual review PDFs
        # for physical-size reading and never substitute the montage as evidence.
        thumbnails.append((figure_id, rgb[::3, ::3]))
        report['figures'].append(row)
    if thumbnails:
        rows = int(np.ceil(len(thumbnails) / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(16, rows * 3.5), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (figure_id, rgb) in zip(axes.flat, thumbnails):
            ax.imshow(rgb)
            ax.set_title(figure_id, fontsize=10, loc='left')
        fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.985, wspace=.06, hspace=.12)
        fig.savefig(output / 'whole_paper_contact_sheet.png', dpi=150)
        plt.close(fig)
    swatch_sheet(output)
    report['figure_count'] = len(report['figures'])
    report['figures_requiring_colour_review'] = [row['id'] for row in report['figures']
                                                 if row['audit_status'] != 'PASS']
    (output / 'colour_inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=palette.ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', nargs='+')
    parser.add_argument('--no-review-pdfs', action='store_true')
    args = parser.parse_args()
    report = run(args.root, args.output, args.figures, not args.no_review_pdfs)
    print(json.dumps({'figures': report['figure_count'],
                      'requires_review': report['figures_requiring_colour_review'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
