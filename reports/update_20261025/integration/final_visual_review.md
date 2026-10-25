# Final figure review — 25 October 2026

Reviewed the current five main figures and 21 supplementary figures from `manuscript/figure_catalog.json`. The main/appendix PDF exports were current when reviewed. The earlier presentation deck and speaking script were not altered.

The 26-figure visual review found no remaining clipping, unreadable panel labels or conflicting categorical colour assignments. Figure S5's ratio label now explicitly includes `(IgE + 1)`. This is a presentation review, not independent certification of the statistical analyses or clinical utility.

## Export checks

| Check | Result |
|---|---|
| Figure PDFs reviewed | 26 |
| PDF text-collision failures | 0 |
| PDF text-collision warnings | 0 |
| Outdated `D`, `A100`, `A200`, `pooled A` labels in figure PDF text | 0 |
| Smallest glyph in standalone figure PDFs | 6.3 pt |
| Smallest embedded glyph in main and appendix PDFs | Approximately 5.98 pt |
| Current panel palette | `palette-v5-clinical-20261025` |
| Figure changes since the earlier complete visual review | Figure 5 and Figure S5 only |

`final_visual_audit/summary.json` contains per-figure results and PDF hashes. `final_visual_audit/reviewed_manifest.json` identifies all reviewed PDF/SVG/PNG files and the main/appendix PDFs. The two new figures were also inspected at their final manuscript page placement: main page 11 and appendix page 23. Both figures and captions fit on their pages without clipping.

## Figure-level assessment

| Figure | Assessment |
|---|---|
| 1 | Framework stages and parallel research estimator remain separate; treatment decisions are not implied to use the model. |
| 2 | All 15 panels retain treatment colours. Median/IQR and proportion/95% CI displays are distinguished by the caption. Whiskers avoid overlapping shaded fills. Stage-specific composite targets are correctly unconnected. |
| 3 | Recorded IGA uses an ordered six-grade scale; reduction-pathway proportions have visible counts and explicit categories. |
| 4 | All 23 markers remain. Three coloured within-treatment panels and two neutral adjusted contrasts are visually separated; axis quantities and direction are explicit. Dense labels and significance marks remain legible. |
| 5 | Matched input comparisons share the same scales and row order; the clinical controls remain visible below the divider. All three visits retain the same colours in the error and response panels. The caption distinguishes matched-input denominators from retained primary-model denominators and defines the three markers. |
| S1 | Clinical and model inclusion flows retain separate counts and linked branches. |
| S2 | Balance and treatment-overlap panels have explicit method/treatment legends and do not reuse a colour for both roles. |
| S3–S4 | Observed and adjusted displays remain visibly separated; axes use recorded units, and no W4 IgE point is introduced. |
| S5 | Complete three-visit observations are separated from the CAMS missing-data comparisons. Treatment colours and log-axis ticks are consistent. The ratio label includes parentheses; the caption explains population differences, intervals and missingness assumptions. |
| S6–S9 | Treatment contrast names replace old shorthand. Observed/adjusted points are separated vertically, with common risk-difference scales and direction labels. |
| S10 | All 23 baseline marker rows remain; values and sequential colour scale are readable. |
| S11–S15 | Each time window has a complete title. Within-treatment change and adjusted contrasts have distinct axes. Sparse or unavailable cells are not presented as zero. |
| S16 | Displayed associations identify molecule, clinical measure, treatment, window and sample count; the q-value column is separate from effect intervals. |
| S17 | Region rows and treatment blocks are readable without repeated long labels. Numerical heatmap legends and pooled-treatment comparisons retain distinct roles. |
| S18 | The three visit colours are consistently used in each treatment facet. The baseline reference and later changes remain distinguishable. |
| S19–S20 | Fixed count cells avoid ambiguous bubble sizes. Visit/window panels share quantitative scales; count bars identify what colour means. |
| S21 | Classification matrices show counts and row percentages, with observed and model axes explicit. The first-repetition status is visible in the title. |

## Colour and document consistency

The active treatment palette is blue (`#245C8A`), amber (`#E69A3A`) and muted red (`#B84D59`); the visit palette is light, medium and dark sea-green (`#8ABDAE`, `#3C8F80`, `#1D514D`). Method comparisons use greys. No ordinary series requires a different point shape or dashed line to identify it. Continuous numerical colour bars are separately labelled.

The palette was also reviewed under the previously generated protanopia, deuteranopia, tritanopia and grayscale diagnostics in `../palette/palette_diagnostics.png`. These simulations show maintained lightness differences within the visit family and a distinctly light abro 100 mg series. They do not certify accessibility: treatment separation is weaker in grayscale, so titles, facets and legends remain important.

`docs/figure_style.md` now describes this palette, the actual current panels and the equation's placement in Results. It no longer claims that the archived deck was regenerated. Methods describes model development and validation; the equation is not inserted into the figure artwork.

Visual contacts are in `visual_contacts/contact_1.png` through `contact_5.png`. The final report contains no open visual-layout correction. Any later modification to an audited figure requires checking that figure again.
