"""Export the current appendix as editable Word tables and a labelled figure deck.

The LaTeX appendix and figure catalogue remain the content sources. Pandoc
converts the appendix's text, maths and native tables; no table is rasterized.
The review deck embeds the exact high-resolution figure previews and keeps
captions in notes. Vector publication files remain in manuscript/figures/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get('T2T_ARTIFACT_RUNTIME',
                              Path.home() / '.cache/codex-runtimes/codex-primary-runtime'))


def toolchain():
    dep = RUNTIME / 'dependencies'
    bundled_skills = RUNTIME / 'plugins/openai-primary-runtime/plugins'
    return {
        'python': dep / 'python/bin/python',
        'node': dep / 'node/bin/node',
        'modules': dep / 'node/node_modules',
        'bin': dep / 'bin/override',
        'documents': bundled_skills / 'documents/skills/documents',
        'presentations': bundled_skills / 'presentations/skills/presentations',
        'pandoc': shutil.which('pandoc') or '/opt/homebrew/bin/pandoc',
    }


def run(command, *, cwd=None, env=None, log=None):
    result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                            capture_output=True, text=True)
    if log:
        Path(log).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command[0]}\n'
                           + (result.stdout + result.stderr)[-8000:])
    return result.stdout


def replace_braced_command(text, name, callback):
    pattern = re.compile(r'\\' + name + r'(?:\[[^\]]*\])?\{')
    cursor = 0
    while match := pattern.search(text, cursor):
        depth, end = 1, match.end()
        while depth and end < len(text):
            if text[end] == '{' and text[end - 1] != '\\':
                depth += 1
            elif text[end] == '}' and text[end - 1] != '\\':
                depth -= 1
            end += 1
        if depth:
            raise ValueError(f'Unclosed {name} command')
        value = callback(text[match.end():end - 1])
        text = text[:match.start()] + value + text[end:]
        cursor = match.start() + len(value)
    return text


def flatten_tex(path, base):
    content = path.read_text()
    return re.sub(r'\\input\{([^}]+)\}',
                  lambda m: '\n' + flatten_tex(base / (m[1] + '.tex'), base) + '\n',
                  content)


def prepare_tex(root, build):
    manuscript = root / 'manuscript'
    text = flatten_tex(manuscript / 'appendix.tex', manuscript)
    aux = (manuscript / 'appendix.aux').read_text()
    labels = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', aux))
    bibnumbers = dict(re.findall(r'\\bibcite\{([^}]+)\}\{\{([^}]+)\}', aux))
    text = re.sub(r'\\(?:figref|ref)\{([^}]+)\}',
                  lambda m: labels.get(m[1], 'unresolved ' + m[1]), text)
    text = re.sub(r'\\cite[pt]?\{([^}]+)\}',
                  lambda m: '[' + ', '.join(bibnumbers.get(k, k)
                                           for k in m[1].split(',')) + ']', text)
    text = replace_braced_command(text, 'shortstack',
                                  lambda s: s.replace('\\\\', r'\newline '))
    text = text.replace('log$_2$', r'$\log_2$')
    text = re.sub(r'\b([A-Za-z])\$\^(\{[^}]+\}|\d)\$', r'$\1^\2$', text)
    text = re.sub(r'(\([^()\n]*\))\$\^(\{[^}]+\}|\d)\$',
                  r'$\1^\2$', text)
    # Word repeats its first table row; LaTeX's alternate continuation head
    # would otherwise become duplicate data rows in Pandoc's native table.
    text = re.sub(r'\\endfirsthead.*?\\endhead', lambda _: r'\endhead', text, flags=re.S)
    # Word supplies its own row spacing. Pandoc's LaTeX reader can otherwise
    # discard a following multicolumn label after this booktabs-only command.
    text = re.sub(r'\\addlinespace(?:\[[^\]]*\])?', '', text)
    text = re.sub(r'\\pdfbookmark\[[^\]]*\]\{[^}]*\}\{[^}]*\}', '', text)
    text = re.sub(r'\{\\large\\bfseries [^}]*\}\\par\\vspace\{[^}]*\}', '', text)
    text = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\.pdf\}',
                  lambda m: r'\includegraphics[width=160mm]{'
                  + str(manuscript / (m[1] + '.png')) + '}', text)
    # Preserve exact caption numbering rather than asking Word to renumber.
    table_number = [0]
    def caption(value):
        if value.startswith(r'\textbf{'):
            return r'\caption{' + value + '}'
        table_number[0] += 1
        return r'\caption{Table S' + str(table_number[0]) + '. ' + value + '}'
    # Table captions precede the figure block in the generated appendix.
    first_figure = text.find(r'\renewcommand{\thefigure}')
    before, figures = text[:first_figure], text[first_figure:]
    before = replace_braced_command(before, 'caption', caption)
    def figure_block(match):
        value = match[0]
        fig = re.search(r'\{figures/(FigureS\d+)\.pdf\}', value)
        if fig is None:
            fig = re.search(r'/(FigureS\d+)\.png\}', value)
        if fig:
            ident = fig[1].replace('Figure', 'Figure ')
            value = replace_braced_command(value, 'caption',
                    lambda s: r'\caption{' + ident + '. ' + s + '}')
        return value
    figures = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', figure_block,
                     figures, flags=re.S)
    text = before + figures
    bbl = (manuscript / 'appendix.bbl').read_text()
    entries = re.split(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', bbl)[1:]
    bibliography = [r'\section*{References}']
    for key, value in zip(entries[::2], entries[1::2]):
        value = value.split(r'\end{thebibliography}')[0]
        value = value.replace(r'\newblock', ' ').replace(r'\penalty0', '')
        value = replace_braced_command(value, 'doi',
                                        lambda x: r'\href{https://doi.org/' + x + '}{doi: ' + x + '}')
        bibliography.append(bibnumbers.get(key, '?') + '. ' + value.strip() + '\n\n')
    text = re.sub(r'\\bibliographystyle\{[^}]+\}', '', text)
    text = re.sub(r'\\bibliography\{[^}]+\}', lambda _: '\n'.join(bibliography), text)
    text = re.sub(r'\\title\{[^\n]+\}', r'\\title{Supplementary information}', text)
    text = re.sub(r'\\(?:clearpage|newpage)', '\n\n', text)
    target = build / 'appendix_flattened.tex'
    target.write_text(text)
    return target, table_number[0], len(entries) // 2


def word_style(path, numbered_tables):
    from docx import Document
    from docx.enum.section import WD_SECTION, WD_ORIENT
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Mm, Pt, RGBColor

    doc = Document(path)
    for sec in doc.sections:
        sec.page_width, sec.page_height = Mm(210), Mm(297)
        sec.top_margin, sec.bottom_margin = Mm(18), Mm(18)
        sec.left_margin, sec.right_margin = Mm(18), Mm(18)
    for name in ['Normal', 'Body Text', 'First Paragraph', 'Compact']:
        if name in doc.styles:
            style = doc.styles[name]
            style.font.name = 'Arial'
            style.font.size = Pt(10.5)
            style.paragraph_format.line_spacing = 1.08
            style.paragraph_format.space_after = Pt(6)
    for name in ['Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Heading 3',
                 'Heading 4', 'Header', 'Footer', 'Caption', 'Table Caption', 'Image Caption']:
        if name in doc.styles:
            style = doc.styles[name]
            style.font.name = 'Arial'
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.font.size = Pt(18 if name == 'Title' else 12 if 'Heading' in name else 9.5)
            style.paragraph_format.keep_with_next = ('Heading' in name or name in ['Title', 'Table Caption'])
    # Landscape only the supplementary table collection, keeping prose/figures portrait.
    starts = [p for p in doc.paragraphs if p.text.strip() == 'Supplementary tables']
    ends = [p for p in doc.paragraphs if p.text.strip() == 'Independent auxiliary target models'
            and p.style.name.startswith('Heading')]
    if starts:
        doc.add_section(WD_SECTION.NEW_PAGE)
        break_element = doc.element.body[-2]
        starts[0]._p.addprevious(break_element)
        last = doc.sections[-1]
        last.orientation = WD_ORIENT.LANDSCAPE
        last.page_width, last.page_height = Mm(297), Mm(210)
        last.top_margin, last.bottom_margin = Mm(16), Mm(16)
        last.left_margin, last.right_margin = Mm(17), Mm(17)
        later = next((p for p in ends if p._p.getprevious() is not None
                      and list(doc.element.body).index(p._p) > list(doc.element.body).index(starts[0]._p)), None)
        if later is not None:
            doc.add_section(WD_SECTION.NEW_PAGE)
            break_element = doc.element.body[-2]
            later._p.addprevious(break_element)
            last = doc.sections[-1]
            last.orientation = WD_ORIENT.PORTRAIT
            last.page_width, last.page_height = Mm(210), Mm(297)
            last.top_margin, last.bottom_margin = Mm(18), Mm(18)
            last.left_margin, last.right_margin = Mm(18), Mm(18)
    body_elements = list(doc.element.body)
    first_table_index = body_elements.index(starts[0]._p) if starts else len(body_elements)
    table_audit = []
    for n, table in enumerate(doc.tables, 1):
        table.autofit = False
        wide = body_elements.index(table._tbl) > first_table_index
        width_mm = 263 if wide else 174
        cols = len(table.columns)
        # Text labels need more room than numerical visit columns.
        if cols == 4:
            weights = [0.40, 0.20, 0.20, 0.20]
            if n >= len(doc.tables) - 4:
                weights = [0.17, 0.28, 0.29, 0.26]
        elif cols == 5:
            weights = [0.21, 0.22, 0.16, 0.18, 0.23]
        elif cols == 6:
            weights = [0.22, 0.26, 0.13, 0.13, 0.13, 0.13]
        else:
            weights = [1 / cols] * cols
        for col, weight in zip(table.columns, weights):
            col.width = Mm(width_mm * weight)
        pr = table._tbl.tblPr
        borders = pr.find(qn('w:tblBorders'))
        if borders is None:
            borders = OxmlElement('w:tblBorders'); pr.append(borders)
        for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            el = OxmlElement('w:' + edge)
            for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
                el.set(qn('w:' + key), value)
            borders.append(el)
        for row_no, row in enumerate(table.rows):
            trpr = row._tr.get_or_add_trPr()
            if row_no == 0:
                header = OxmlElement('w:tblHeader'); trpr.append(header)
            cant = OxmlElement('w:cantSplit'); trpr.append(cant)
            for column_no, cell in enumerate(row.cells):
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                tcpr = cell._tc.get_or_add_tcPr()
                margin = OxmlElement('w:tcMar')
                for edge, value in [('top', '65'), ('bottom', '65'), ('left', '80'), ('right', '80')]:
                    el = OxmlElement('w:' + edge); el.set(qn('w:w'), value); el.set(qn('w:type'), 'dxa'); margin.append(el)
                tcpr.append(margin)
                if row_no == 0:
                    shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E6EEF3'); tcpr.append(shade)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.line_spacing = 1.0
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if column_no == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run_ in p.runs:
                        run_.font.name = 'Arial'; run_.font.size = Pt(9)
                        run_.font.color.rgb = RGBColor(0, 0, 0)
        table_audit.append({'native_table': n, 'rows': len(table.rows), 'columns': cols,
                            'text_sha256': hashlib.sha256('\n'.join(c.text for row in table.rows for c in row.cells).encode()).hexdigest()})
    # Separate figure pages, retaining the complete editable caption with its image.
    for p in doc.paragraphs:
        following = p._p.getnext()
        if (following is not None and following.tag == qn('w:tbl')
                and (p._p.xpath('.//w:b') or 'Caption' in p.style.name)):
            p.paragraph_format.keep_with_next = True
        if p._p.xpath('.//w:drawing'):
            p.paragraph_format.page_break_before = True
            p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if numbered_tables and p.text.startswith(f'Table S{numbered_tables}.'):
            # Keep the final compact table and its explanatory notes together
            # before the portrait text section resumes.
            p.paragraph_format.page_break_before = True
    # Pandoc retains LaTeX alignment separators as visible runs in equation
    # arrays. Their removal preserves the native editable two-line equation.
    alignment_tokens = doc.element.xpath('.//m:eqArr//m:r[m:t="&"]')
    for token in alignment_tokens:
        token.getparent().remove(token)
    # A page field remains editable in Word.
    for sec in doc.sections:
        footer = sec.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if not footer._p.xpath('.//w:fldSimple'):
            field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    doc.save(path)
    return {'native_table_count': len(doc.tables), 'numbered_supplementary_tables': numbered_tables,
            'table_shapes': table_audit, 'inline_images': len(doc.inline_shapes),
            'removed_math_alignment_tokens': len(alignment_tokens)}


def make_docx(root, out, build, tc):
    flat, n_tables, n_refs = prepare_tex(root, build)
    raw = build / 'appendix_unstyled.docx'
    run([tc['pandoc'], flat, '-f', 'latex', '-t', 'docx', '-o', raw],
        cwd=root / 'manuscript', log=build / 'pandoc.log')
    final = out / 'appendix.docx'
    shutil.copy2(raw, final)
    audit = word_style(final, n_tables)
    from docx import Document
    def all_cells(document):
        return [[c.text for row in t.rows for c in row.cells] for t in document.tables]
    audit['native_table_text_unchanged_after_formatting'] = all_cells(Document(raw)) == all_cells(Document(final))
    if not audit['native_table_text_unchanged_after_formatting']:
        raise RuntimeError('A table value changed while applying Word formatting')
    labels = []
    source = flat.read_text()
    for match in re.finditer(r'\\multicolumn\{\d+\}\{l\}\{', source):
        depth, end = 1, match.end()
        while depth and end < len(source):
            if source[end] == '{' and source[end-1] != '\\':
                depth += 1
            elif source[end] == '}' and source[end-1] != '\\':
                depth -= 1
            end += 1
        label = source[match.end():end-1]
        label = replace_braced_command(label, 'textbf', lambda value: value)
        labels.append(label.replace(r'\%', '%'))
    normalize = lambda value: ''.join(c.lower() for c in value if c.isalnum())
    table_text = normalize(' '.join(value for table in all_cells(Document(final)) for value in table))
    missing = [label for label in labels if normalize(label) not in table_text]
    if missing:
        raise RuntimeError('A table population or metric label was lost during conversion: ' + ', '.join(missing))
    audit['table_group_labels_preserved'] = len(labels)
    audit['references'] = n_refs
    audit['tables_rasterized'] = 0
    return audit


def make_selection_docx(root, out, build, tc):
    source = root / 'reports/update_20261112/marker_selection/selection_process_zh.md'
    if not source.exists():
        return {'status': 'SOURCE_PENDING'}
    final = out / 'three_marker_selection.docx'
    run([tc['pandoc'], source, '-f', 'gfm', '-t', 'docx', '-o', final],
        log=build / 'selection_pandoc.log')
    audit = word_style(final, 0)
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt
    doc = Document(final)
    if doc.paragraphs:
        doc.paragraphs[0].text = '三分子候选组合的筛选过程'
        doc.paragraphs[0].style = doc.styles['Title']
    for style_name in ['Title', 'Heading 1', 'Heading 2', 'Heading 3']:
        if style_name in doc.styles:
            doc.styles[style_name].font.name = 'Arial'
    for style_name in ['Normal', 'Body Text', 'First Paragraph', 'Compact']:
        if style_name in doc.styles:
            doc.styles[style_name].font.name = 'Arial'
            doc.styles[style_name].font.size = Pt(10.5)
    cjk_font = os.environ.get('T2T_CJK_FONT', 'Heiti SC' if sys.platform == 'darwin' else 'Noto Sans CJK SC')
    for fonts in doc.element.xpath('.//w:rFonts'):
        fonts.set(qn('w:eastAsia'), cjk_font)
    for style in doc.styles:
        props = style.element.find(qn('w:rPr'))
        if props is None:
            props = OxmlElement('w:rPr'); style.element.append(props)
        fonts = props.find(qn('w:rFonts'))
        if fonts is None:
            fonts = OxmlElement('w:rFonts'); props.append(fonts)
        fonts.set(qn('w:eastAsia'), cjk_font)
    for run_ in doc.element.xpath('.//w:r'):
        props = run_.find(qn('w:rPr'))
        if props is None:
            props = OxmlElement('w:rPr'); run_.insert(0, props)
        fonts = props.find(qn('w:rFonts'))
        if fonts is None:
            fonts = OxmlElement('w:rFonts'); props.append(fonts)
        fonts.set(qn('w:eastAsia'), cjk_font)
    doc.save(final)
    audit.update(status='CREATED', source=str(source.relative_to(root)), cjk_font=cjk_font)
    return audit


def make_deck_pdf(out, build, tc, env):
    rendered = build / 'deck_render'
    rendered.mkdir(exist_ok=True)
    profile = build / 'libreoffice_deck_profile'
    command = [tc.get('soffice', tc['bin'] / 'soffice'), '--headless',
               '-env:UserInstallation=' + profile.as_uri(), '--convert-to', 'pdf',
               '--outdir', rendered, out / 'figure_deck.pptx']
    run(command, env=env, log=build / 'deck_pdf_render.log')
    pdf = rendered / 'figure_deck.pdf'
    if not pdf.exists():
        raise RuntimeError('Bundled LibreOffice did not render the deck PDF')
    shutil.copy2(pdf, out / 'figure_deck.pdf')


def cjk_render_environment(build, env):
    """Make installed CJK fonts visible to headless LibreOffice locally."""
    from xml.sax.saxutils import escape
    folders = [Path('/System/Library/Fonts'), Path('/Library/Fonts'),
               Path('/usr/share/fonts'), Path('/usr/local/share/fonts'),
               Path.home() / '.fonts', Path.home() / 'Library/Fonts',
               RUNTIME / 'dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype']
    cache = build / 'fontcache'; cache.mkdir(exist_ok=True)
    config = build / 'fonts.conf'
    config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig>'
                      + ''.join('<dir>' + escape(str(p)) + '</dir>' for p in folders if p.is_dir())
                      + '<cachedir>' + escape(str(cache)) + '</cachedir></fontconfig>')
    result = env.copy()
    result.update(FONTCONFIG_FILE=str(config), FONTCONFIG_PATH=str(build))
    return result


def portable_deck(root, out, build, tc, catalogue):
    """Rebuild the review deck with Pandoc and standard OOXML, without Codex."""
    from xml.etree import ElementTree as ET
    from PIL import Image
    ns = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
          'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for key, uri in ns.items():
        ET.register_namespace(key, uri)
    ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
    markdown = []
    for row in catalogue:
        kind = 'Supplementary' if row['id'].startswith('FigureS') else 'Main'
        markdown += ['## ' + row['id'].replace('Figure', 'Figure ') + '   ' + kind,
                     '', '![](' + str(root / 'manuscript/figures' / (row['id'] + '.png')) + ')',
                     '', '::: notes', row['caption'], '', 'Sources',
                     *row['sources'], ':::', '']
    source = build / 'portable_deck.md'
    source.write_text('\n'.join(markdown))
    candidate = build / 'portable_deck.pptx'
    run([tc['pandoc'], source, '-f', 'markdown', '-t', 'pptx', '--slide-level=2',
         '-o', candidate], log=build / 'portable_deck_pandoc.log')
    emu = lambda value: str(round(value * 9525))
    def transform(element, left, top, width, height):
        for child in list(element):
            if child.tag.endswith('}off') or child.tag.endswith('}ext'):
                element.remove(child)
        ET.SubElement(element, '{' + ns['a'] + '}off', {'x': emu(left), 'y': emu(top)})
        ET.SubElement(element, '{' + ns['a'] + '}ext', {'cx': emu(width), 'cy': emu(height)})
    with ZipFile(candidate) as saved, ZipFile(out / 'figure_deck.pptx', 'w') as result:
        for info in saved.infolist():
            value = saved.read(info.filename)
            if info.filename == 'ppt/presentation.xml':
                element = ET.fromstring(value)
                size = element.find('p:sldSz', ns)
                size.set('cx', emu(1120)); size.set('cy', emu(1280))
                size.set('type', 'custom')
                value = ET.tostring(element, encoding='utf-8', xml_declaration=True)
            match = re.fullmatch(r'ppt/slides/slide(\d+)\.xml', info.filename)
            if match:
                n = int(match[1]) - 1
                element = ET.fromstring(value)
                sw, sh = Image.open(root / 'manuscript/figures' / (catalogue[n]['id'] + '.png')).size
                scale = min(1072 / sw, 1205 / sh)
                for picture in element.findall('.//p:pic', ns):
                    props = picture.find('p:spPr', ns)
                    xfrm = props.find('a:xfrm', ns)
                    if xfrm is None:
                        xfrm = ET.SubElement(props, '{' + ns['a'] + '}xfrm')
                    transform(xfrm, 24 + (1072 - sw * scale) / 2,
                              51 + (1205 - sh * scale) / 2, sw * scale, sh * scale)
                    for crop in picture.findall('.//a:srcRect', ns):
                        picture.find('p:blipFill', ns).remove(crop)
                for shape in element.findall('.//p:sp', ns):
                    texts = shape.findall('.//a:t', ns)
                    if not texts:
                        continue
                    props = shape.find('p:spPr', ns)
                    xfrm = props.find('a:xfrm', ns)
                    if xfrm is None:
                        xfrm = ET.SubElement(props, '{' + ns['a'] + '}xfrm')
                    transform(xfrm, 28, 13, 950, 30)
                    body = shape.find('p:txBody/a:bodyPr', ns)
                    body.attrib.update({'anchor': 't', 'lIns': '0', 'rIns': '0', 'tIns': '0', 'bIns': '0'})
                    for paragraph in shape.findall('.//a:p', ns):
                        props_p = paragraph.find('a:pPr', ns)
                        if props_p is None:
                            props_p = ET.SubElement(paragraph, '{' + ns['a'] + '}pPr')
                        props_p.set('algn', 'l')
                    for run_ in shape.findall('.//a:r', ns):
                        props_r = run_.find('a:rPr', ns)
                        if props_r is None:
                            props_r = ET.Element('{' + ns['a'] + '}rPr'); run_.insert(0, props_r)
                        props_r.set('sz', '1425'); props_r.set('b', '0')
                        latin = props_r.find('a:latin', ns)
                        if latin is None:
                            latin = ET.SubElement(props_r, '{' + ns['a'] + '}latin')
                        latin.set('typeface', 'Arial')
                value = ET.tostring(element, encoding='utf-8', xml_declaration=True)
            result.writestr(info, value)


def write_deck_builder(root, out, build, tc, catalogue):
    (build / 'node_modules').symlink_to(tc['modules'], target_is_directory=True) if not (build / 'node_modules').exists() else None
    script = build / 'build_deck.mjs'
    values = {'root': str(root), 'out': str(out), 'build': str(build),
              'skill': str(tc['presentations']), 'python': str(tc['python']),
              'catalogue': catalogue,
              'finalPptx': str(build / 'validated_deck' / ('validated_' + uuid.uuid4().hex + '.pptx')),
              'receipt': str(build / 'receipts' / ('validation_' + uuid.uuid4().hex + '.json'))}
    script.write_text('const task = ' + json.dumps(values) + ';\n' + r'''
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const presentation = Presentation.create({slideSize:{width:1120,height:1280}});
for (const row of task.catalogue) {
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';
  const kind = row.id.startsWith('FigureS') ? 'Supplementary' : 'Main';
  const label = slide.shapes.add({geometry:'textbox', position:{left:28,top:13,width:950,height:30},fill:'none',line:{fill:'none',width:0}});
  label.text = `${row.id.replace('Figure','Figure ')}   ${kind}`;
  label.text.style = {typeface:'Arial',fontSize:19,color:'#20252A'};
  const bytes = await fs.readFile(path.join(task.root,'manuscript/figures',row.id+'.png'));
  slide.images.add({blob:new Uint8Array(bytes),contentType:'image/png',alt:row.caption,fit:'contain',position:{left:24,top:51,width:1072,height:1205}});
  slide.speakerNotes.text = row.caption + '\n\nSources\n' + row.sources.join('\n');
}
const candidatePath = path.join(task.build,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
await fs.mkdir(path.dirname(task.finalPptx),{recursive:true});
await fs.mkdir(path.dirname(task.receipt),{recursive:true});
const {finalizePresentation} = await import(pathToFileURL(path.join(task.skill,'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
 workspaceDir:task.root,candidatePath,finalPath:task.finalPptx,
 pythonExecutable:task.python,
 integrityValidatorPath:path.join(task.skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(task.skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','10668000,12192000','--validate-heading-fit'],
 explicitTotalSlideCount:task.catalogue.length,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Arial']},
 verifyArtifactToolImport:true,
 receiptPath:task.receipt
});
await fs.copyFile(task.finalPptx,path.join(task.out,'figure_deck.pptx'));
await fs.copyFile(task.receipt,path.join(task.build,'deck_validation.json'));
console.log(JSON.stringify({slides:task.catalogue.length,result}));
''')
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--only', choices=['docx', 'deck', 'all'], default='all')
    parser.add_argument('--render-word', action='store_true')
    parser.add_argument('--portable', action='store_true',
                        help='Use standard Pandoc/python-docx/LibreOffice; no Codex runtime')
    parser.add_argument('--soffice', type=Path, help='LibreOffice executable in portable mode')
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out_dir or root / 'manuscript').resolve()
    build = (args.build_dir or root / 'reports/update_20261112/exports/build').resolve()
    out.mkdir(parents=True, exist_ok=True); build.mkdir(parents=True, exist_ok=True)
    tc = toolchain()
    if args.portable:
        tc['python'] = Path(sys.executable)
        executable = args.soffice or shutil.which('soffice')
        if not executable:
            raise SystemExit('Portable export requires LibreOffice; pass --soffice /path/to/soffice')
        tc['soffice'] = Path(executable).resolve()
        tc['bin'] = tc['soffice'].parent
    elif Path(sys.executable).resolve() != tc['python'].resolve():
        raise SystemExit('Run with the bundled artifact Python: ' + str(tc['python']))
    env = os.environ.copy(); env['PATH'] = str(tc['bin']) + os.pathsep + env.get('PATH', '')
    env['RUNTIME_NODE_MODULES'] = str(tc['modules'])
    env['RUNTIME_NODE'] = str(tc['node'])
    env['RUNTIME_PYTHON'] = str(tc['python'])
    env['TMPDIR'] = '/private/tmp'
    catalogue = json.loads((root / 'manuscript/figure_catalog.json').read_text())
    inputs = [root / 'manuscript/figure_catalog.json',
              *[root / 'manuscript' / ('appendix.' + suffix) for suffix in ['tex', 'aux', 'bbl']],
              *sorted((root / 'manuscript/tables').glob('*.tex')),
              *[root / 'manuscript/figures' / (row['id'] + '.png') for row in catalogue]]
    selection_source = root / 'reports/update_20261112/marker_selection/selection_process_zh.md'
    if selection_source.exists():
        inputs.append(selection_source)
    input_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    receipt = build / 'export_verification.json'
    report = json.loads(receipt.read_text()) if args.only != 'all' and receipt.exists() else {}
    report.update(source='Current LaTeX appendix and figure catalogue', figure_count=len(catalogue))
    if args.only in ['docx', 'all']:
        report['word'] = make_docx(root, out, build, tc)
        report['selection_word'] = make_selection_docx(root, out, build, tc)
        if args.render_word and not args.portable:
            run([tc['python'], tc['documents'] / 'render_docx.py', out / 'appendix.docx',
                 '--output_dir', build / 'word_render', '--emit_pdf'],
                env=env, log=build / 'word_render.log')
            if (out / 'three_marker_selection.docx').exists():
                run([tc['python'], tc['documents'] / 'render_docx.py', out / 'three_marker_selection.docx',
                     '--output_dir', build / 'selection_render', '--emit_pdf'],
                    env=cjk_render_environment(build, env), log=build / 'selection_render.log')
        elif args.render_word:
            from pdf2image import convert_from_path
            rendered = build / 'word_render'; rendered.mkdir(exist_ok=True)
            run([tc['soffice'], '--headless',
                 '-env:UserInstallation=' + (build / 'word_profile').as_uri(),
                 '--convert-to', 'pdf', '--outdir', rendered, out / 'appendix.docx'],
                env=env, log=build / 'word_render.log')
            for number, page in enumerate(convert_from_path(rendered / 'appendix.pdf', dpi=100), 1):
                page.save(rendered / f'page-{number}.png')
    if args.only in ['deck', 'all']:
        if args.portable:
            portable_deck(root, out, build, tc, catalogue)
        else:
            builder = write_deck_builder(root, out, build, tc, catalogue)
            run([tc['node'], builder], cwd=build, env=env, log=build / 'deck_build.log')
        make_deck_pdf(out, build, tc, env)
        report['deck'] = {'slides': len(catalogue), 'embedded_figures': 'Exact high-resolution PNG previews; vector publication exports retained separately',
                          'figure_captions': 'PowerPoint speaker notes', 'native_labels': True}
        with ZipFile(out / 'figure_deck.pptx') as saved:
            image_hashes = {hashlib.sha256(saved.read(name)).hexdigest()
                            for name in saved.namelist() if name.startswith('ppt/media/')}
            unchanged = all(input_hashes['manuscript/figures/' + row['id'] + '.png'] in image_hashes
                            for row in catalogue)
            notes = sum(name.startswith('ppt/notesSlides/notesSlide') and name.endswith('.xml')
                        for name in saved.namelist())
        report['deck'].update(exact_source_image_bytes=unchanged, speaker_note_count=notes)
        if not unchanged or notes != len(catalogue):
            raise RuntimeError('A review figure or its caption is missing from the deck')
    changed = [path for path, checksum in input_hashes.items()
               if hashlib.sha256((root / path).read_bytes()).hexdigest() != checksum]
    report['input_sha256'] = input_hashes
    report['toolchain'] = 'Pandoc/python-docx/LibreOffice' if args.portable else 'Bundled Codex artifact runtime/Pandoc/LibreOffice'
    report['inputs_changed_during_export'] = changed
    if changed:
        raise RuntimeError('Source files changed during export; rebuild after the manuscript is stable: '
                           + ', '.join(changed))
    report['outputs'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in out.glob('*') if p.name in ['appendix.docx', 'three_marker_selection.docx', 'figure_deck.pptx', 'figure_deck.pdf']}
    (build / 'export_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
