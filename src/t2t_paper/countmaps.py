"""Fixed-size cells show patient counts without varying symbol area."""
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm

from . import palette


def draw(ax, frame, x, y, width=2., height=2., maximum=None):
    """Draw occupied cells from lower-edge coordinates and integer counts."""
    if frame.empty:
        raise ValueError('A patient-count map needs observed cells.')
    if not frame['count'].gt(0).all() or not frame['count'].eq(frame['count'].astype(int)).all():
        raise ValueError('Patient-count cells must contain positive integers.')
    xmin, xmax = frame[x].min(), frame[x].max() + width
    ymin, ymax = frame[y].min(), frame[y].max() + height
    xe = np.arange(xmin, xmax + width / 2, width)
    ye = np.arange(ymin, ymax + height / 2, height)
    values = np.zeros((len(ye) - 1, len(xe) - 1), dtype=int)
    for row in frame[[x, y, 'count']].itertuples(index=False, name=None):
        i, j = int(round((row[1] - ymin) / height)), int(round((row[0] - xmin) / width))
        values[i, j] += int(row[2])
    assert values.sum() == frame['count'].sum()
    lut = palette.REGISTRY['continuous']['ocean_sequential']['lut_hex']
    cmap = ListedColormap(lut[45:], name='ocean_patient_counts')
    cmap.set_bad((1, 1, 1, 0))
    maximum = max(2, int(maximum or values.max()))
    image = ax.pcolormesh(xe, ye, np.ma.masked_equal(values, 0),
                         cmap=cmap, norm=LogNorm(vmin=1, vmax=maximum),
                         shading='flat', linewidth=0, rasterized=False)
    return image


def colorbar(fig, image, cax, maximum, label='Patients per cell'):
    ticks = [x for x in [1, 2, 5, 10, 20, 50, 100] if x <= maximum]
    if maximum not in ticks:
        ticks.append(int(maximum))
    bar = fig.colorbar(image, cax=cax, orientation='horizontal', ticks=ticks)
    bar.ax.set_xticklabels([str(x) for x in ticks])
    bar.ax.minorticks_off()
    bar.set_label(label, fontsize=6.5)
    return bar
