"""Shared typography and title-free manuscript figure presentation."""
import re

from matplotlib.text import Text
from matplotlib.axes import Axes


def apply(fig, source_id, caption):
    panel_keys = []
    removed = []
    for index, ax in enumerate(fig.findobj(match=Axes)):
        if ax.get_xlabel() == 'Absolute standardized difference':
            ax.set_xlabel('Absolute standardized\ndifference')
        if ax.get_xlabel() == 'Estimated median eosinophils (recorded units)':
            ax.set_xlabel('Estimated median eosinophils\n(recorded units)')
        if ax.get_ylabel() == 'Observed response':
            ax.set_ylabel('Observed response', labelpad=2)
        if ax.get_xlabel() == 'Percentage within observed-response row':
            ax.set_xlabel('Within observed-response row (%)')
        labels = [t.get_text() for t in ax.texts
                  if re.fullmatch('[a-z]', t.get_text())]
        letter = labels[0] if labels else chr(97 + index)
        if getattr(ax, '_panel_caption_title', ''):
            panel_keys.append((letter, ax._panel_caption_title))
        for loc in ['left', 'center', 'right']:
            title = ax.get_title(loc=loc)
            if title:
                ax.set_title('', loc=loc)
                removed.append(title)
                panel_keys.append((letter, re.sub(r'\s+', ' ', title).strip()))
        ax.xaxis.label.set_fontsize(8.5)
        ax.yaxis.label.set_fontsize(8.5)
        ax.tick_params(labelsize=7.5)
        ax.tick_params(which='major', width=.7, length=3)
    # Figure-level prose belongs in the caption, leaving graphical legends intact.
    figure_notes = []
    keep = {getattr(fig, '_supxlabel', None), getattr(fig, '_supylabel', None)}
    for artist in list(fig.texts):
        if artist not in keep:
            figure_notes.append(artist.get_text())
            artist.remove()
    for artist in fig.findobj(match=Text):
        if not artist.get_text():
            continue
        size = artist.get_fontsize()
        artist.set_fontsize(max(7.5, min(size, 9.5)))
        if re.fullmatch('[a-z]', artist.get_text()):
            artist.set_fontsize(9.5)
            artist.set_fontweight('bold')
    # Preserve identities formerly carried by titles after removing their graphics.
    if panel_keys and source_id != 'Figure2':
        key = '; '.join(f'{letter}, {text}' for letter, text in panel_keys)
        caption += ' Panel key: ' + key + '.'
    return caption, {
        'figure_source_id': source_id,
        'axes_titles_removed': removed,
        'figure_prose_removed_from_graphic': figure_notes,
        'axis_label_pt': 8.5,
        'tick_and_annotation_floor_pt': 7.5,
        'panel_letter_pt': 9.5,
        'data_values_changed': False,
    }
