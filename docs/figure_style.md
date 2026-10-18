# Figure colours and symbols

Treatment colours are a three-colour subset of the Okabe–Ito palette, discussed in [Nature Methods](https://doi.org/10.1038/nmeth.1618) and its [attribution correction](https://doi.org/10.1038/s41592-023-01974-0).

| Meaning | Colour | Additional cue |
|---|---|---|
| Initial dupilumab | Blue `#0072B2` | Circle |
| Initial abrocitinib 100 mg | Bluish green `#009E73` | Square |
| Initial abrocitinib 200 mg | Vermilion `#D55E00` | Triangle |
| W0 / W12 / W24 | Light / medium / dark grey | Triangle / circle / square; dotted / dashed / solid lines |
| Negative / zero / positive quantitative effect | Purple / near-white / gold | Labelled, symmetric numerical colour bar |
| IGA grades or proportions | Sequential grey | Explicit grade or percentage legend |
| Unavailable estimate | Pale grey | Explicit cross or not-estimable label |

Treatment colours are reserved for initial treatment. They never represent visit, prediction method, achievement status or effect direction within the same multipanel figure. A change in treatment is shown by directly labelled recorded regimens; it does not change the initial-group colour.

All figures read `config/paper_style.json`. Numerical source tables and captions are linked in `manuscript/figure_catalog.json`. Source PDF text is at least 6.3 pt; embedding is checked at the manuscript's final scale. PNG is supplied for viewing, with vector PDF and SVG for publication editing.
