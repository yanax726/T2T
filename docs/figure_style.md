# Figure style

The current manuscript contains five main figures and 21 supplementary figures. Their plotting sources share `config/palette_v4_registry.json`, version `palette-v5-clinical-20261025`; the filename is retained for compatibility. The earlier presentation deck and speaking script are preserved separately and were not regenerated in this manuscript revision.

## Colours and labels

| Role | Display label | Colour |
|---|---|---|
| Initial dupilumab | dupi | `#245C8A` |
| Initial abrocitinib 100 mg | abro 100 mg | `#E69A3A` |
| Initial abrocitinib 200 mg | abro 200 mg | `#B84D59` |
| Defined pooled abrocitinib estimate | pooled abro | `#9B6A42` |
| Baseline visit | W0 | `#8ABDAE` |
| Week 12 visit | W12 | `#3C8F80` |
| Week 24 visit | W24 | `#1D514D` |
| Observed or reference method | As labelled | `#9FA9B3` |
| Adjusted or current method | As labelled | `#515B66` |

Treatment and visit colours have separate meanings across the whole numbered figure. Labels identify which variable is coloured; a colour is never reassigned between panels. Treatment changes do not change the initial-treatment colour. Display abbreviations do not rename data fields or treatment codes.

Ordinary points are circles and trend lines are solid. Points, lines and intervals retain full opacity; uncertainty bands, where used, are derived from the same series colour. Titles and axis text are dark neutral. Dense displays use facets and direct labels rather than additional shapes or line styles.

## Main and supplementary displays

- **Figure 1:** the clinical target framework and the separate research estimator. The diagram does not imply that the estimator governed treatment decisions.
- **Figure 2:** treatment colours throughout. Panels a–c show observed medians with interquartile whiskers; d–n show threshold proportions with Wilson 95% confidence intervals. Slight horizontal offsets separate groups without changing their visit labels. Panel o shows stage-specific combined targets as unconnected points because the definitions change by visit.
- **Figure 3:** recorded IGA uses six ordered blue shades for grades 0–5. Treatment-reduction categories are explicitly labelled; counts and denominators accompany proportions.
- **Figure 4:** treatment colours identify within-group molecular changes in a–c. Neutral colours identify the adjusted contrasts in d–e. Their axes distinguish within-patient log₂ changes from between-treatment differences.
- **Figure 5:** visit colours identify W0, W12 and W24 in the matched input comparisons and retained primary-model performance panels. Clinical-only controls are separated by a divider. The matched comparison and primary validation populations have their own denominators in the caption.
- **Supplementary figures:** the same treatment, visit and method roles apply. S5 separates complete three-visit IgE observations from missing-data analyses, with population and assumptions stated in the caption. Molecular windows remain separately labelled. S19–S20 use fixed-size count cells rather than variable-size patient bubbles. All clinical, molecular and validation figures retain their stated units, uncertainty intervals and available-case denominators.

Signed heatmaps use an explicit 257-colour blue–near-white–brick lookup table, with negative values in blue and positive values in brick. Correlation limits are −1 to 1; other signed comparisons use a common symmetric range within each comparison family. Count and proportion heatmaps use sequential blues, with proportions fixed at 0–100%. The compatibility keys `ocean_coral_diverging` and `ocean_sequential` name these numerical lookup tables; they do not encode treatments or visits. Missing values, true zero and sparse estimates are distinct. Numerical colour bars carry their own quantity and units.

The fitted equation appears in **Results**, followed by its validation. **Methods** describes fitting, preprocessing and evaluation. Formula coefficients are not inserted into data panels.

## Export and review

PDF and SVG preserve editable vector text; PNG files are supplied for viewing. Shared export checks cover categorical-colour consistency and panel alignment. Final PDFs are checked for font sizes, text collisions, outdated treatment abbreviations and legibility; the minimum final glyph threshold is 5 pt. Actual-size viewing, grayscale and colour-vision simulations supplement these checks and are not accessibility certification.

The palette was informed by the supplied [scientific palette guide](https://www.simplifiedsciencepublishing.com/resources/best-color-palettes-for-scientific-figures-and-data-visualizations) and adapted to this figure set. The [Nature figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) support clear labels, accessible presentation and editable RGB artwork; they do not endorse a particular palette or certify these figures.
