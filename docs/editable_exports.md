# Editable appendix and figure review deck

The Word appendix and PowerPoint review deck are optional exports from the
current manuscript. They do not refit analyses. The core `run.sh rebuild`
command produces the publication figures, tables and PDFs independently of
these Office exports.

Run `./run.sh rebuild` first so that `manuscript/appendix.tex`, its auxiliary
reference files, the tables, figure catalogue and figure PNGs agree. The exporter
expands the LaTeX table inputs and converts them to native Word tables. Table
values are checked before and after formatting. Mathematical expressions are
retained as editable Office equations. Figures remain images; the Word file
does not turn a scientific plot into editable numerical chart data.

## Portable export

Install Pandoc and LibreOffice. Python dependencies are separate from the main
analysis environment:

```sh
python3 -m venv .exports-venv
.exports-venv/bin/python -m pip install -r requirements.exports.txt
.exports-venv/bin/python src/t2t_paper/editable_exports_20261112.py \
  --portable --soffice /path/to/soffice \
  --out-dir manuscript --build-dir results/export_build
```

If `soffice` is on `PATH`, omit `--soffice`. On macOS an installed LibreOffice
normally exposes it at `/Applications/LibreOffice.app/Contents/MacOS/soffice`.
Use `--only docx` or `--only deck` to export one format. `--render-word` additionally
renders the Word appendix for checking and requires Poppler (`pdftoppm` and
`pdfinfo`) on `PATH`.

The optional Chinese selection explanation also needs a Chinese font. The
exporter requests Heiti SC on macOS and Noto Sans CJK SC elsewhere; set
`T2T_CJK_FONT` to another installed Chinese family if needed. Its local
Fontconfig file makes installed font folders visible to headless LibreOffice
without changing system configuration. Always inspect that rendered document
for visible Chinese glyphs as well as extracting its text.

Outputs are `manuscript/appendix.docx`, `manuscript/figure_deck.pptx` and
`manuscript/figure_deck.pdf`. If the local Chinese marker-selection explanation
is available, `manuscript/three_marker_selection.docx` is also generated. Its
absence does not prevent exporting the appendix or figures.

The portable deck uses Pandoc's PowerPoint writer followed by explicit OOXML
placement on a portrait review canvas. Each slide contains one current figure,
its identifier and Main or Supplementary label. The complete caption and source
paths are in the PowerPoint speaker notes. LibreOffice renders the PDF from that
PowerPoint file. The review deck embeds the exact high-resolution PNG previews
to avoid SVG font substitution across Office applications. Publication PDF and
SVG figures remain available in `manuscript/figures/`.

## Checked boundaries

The 12 November export was tested in an isolated copy of the aggregate manuscript
inputs with Pandoc 3.9.0.2, LibreOffice 25.2, Python 3.12, python-docx 1.2.0,
Pillow 12.3.0 and pdf2image 1.17.0. No proprietary Artifact Tool import was used in
portable mode. The normal Codex authoring path uses its bundled Artifact Tool
and performs its presentation integrity and layout checks before copying the
final review deck.

Both modes verify that every figure image is embedded byte for byte and that
every slide has caption notes. They hash manuscript inputs before and after
export and stop if a source changes during the run. Execution details are saved
as `export_verification.json` under the chosen build directory. Word page breaks,
PDF rendering and Office file bytes may vary with fonts and software versions;
those differences do not change the source numerical results. Keep the editable
Word file as an editing copy and regenerate publication outputs from the source
analysis and LaTeX files.
