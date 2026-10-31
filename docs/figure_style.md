# Figure style

All five main and 21 supplementary figures use one colour registry, `config/palette_v4_registry.json`. Its stable filename is retained for reproducibility; the current palette is **palette-v6-blue-apricot-coral-plum-20261031**. Earlier presentation files are unchanged.

## Colour roles

The four anchors are blue, apricot, coral and muted plum. Treatment identity uses blue, apricot and coral. Visit identity uses a separate light-to-dark plum sequence. Observed and adjusted estimates use taupe and muted plum. These are fixed roles across the complete numbered figure, not colours selected independently for each panel.

| Role | Colour |
|---|---|
| treatment.dupilumab | `#3B7197` |
| treatment.abrocitinib_100mg | `#C78557` |
| treatment.abrocitinib_200mg | `#B85060` |
| treatment.pooled_abrocitinib | `#A97865` |
| visit.W0 | `#A38ABB` |
| visit.W12 | `#7B5F9B` |
| visit.W24 | `#47305F` |
| method.reference | `#A18E7F` |
| method.current | `#775D78` |

Treatment changes do not reassign initial-group colour. Comparisons between treatments use an analysis colour rather than either treatment's colour. A single-series panel uses one colour; there is no colour code for significance. Circles and solid lines are the default. Open circles with explicit n mark descriptive molecular estimates based on fewer than five patients. Titles and labels remain dark neutral.

## Numerical scales and annotations

IGA grades 0–5 have their own six-step ordered plum scale. Treatment-pathway bars below them use treatment colours; the two colour meanings do not overlap. Signed heatmaps use a 257-entry blue–off-white–coral lookup table. Correlations retain −1 to 1; other signed quantities retain their original common symmetric limits. Sequential blue heatmaps show numeric concentration, proportion or count with separately labelled units and normalization. Proportion scales remain 0–100%. Missing values are not zero. Count cells retain all patients and use a logarithmic colour scale; single-patient cells remain visible.

Figure 2 keeps all specified outcomes and visits. Its full endpoint denominators are in Supplementary Table S1. IgE/EOS coverage is in Table S9. These repeated numeric strips are no longer placed below the plots. S10 retains all 207 baseline concentration cells, with exact values in the source table rather than printed in each cell. Counts stay in cohort flows, IGA denominators, sparse-data flags and confusion matrices, where they are essential to interpretation. Matched model MAE values also remain directly readable.

The selected-estimator figure retains molecular-only, combined and clinical-only input comparisons, the absolute-error distribution and EASI75 classification. The caption distinguishes sample sets, evaluation design, uncertainty and limited severe follow-up coverage. No numerical results or scientific conclusions change with this visual revision.

## Export and checking

PDF and SVG preserve editable text. PNG previews and the figure book show the complete collection. The shared exporter checks semantic colours and rendered panel alignment. Each PDF is checked for glyph size and collisions and inspected at final size. Colour-vision and grayscale views are diagnostic checks, not a claim of universal accessibility. Direct labels and panel positions remain necessary when colours converge in grayscale.

The [reference guide](https://www.simplifiedsciencepublishing.com/resources/best-color-palettes-for-scientific-figures-and-data-visualizations) informed restrained colour families and lightness contrast. The palette is adapted to this complete manuscript, not copied from a separate example for each chart. [Nature's figure guidance](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) does not certify a particular HEX palette.
