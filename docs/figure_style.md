# Figure style

All five main and 21 supplementary figures use one colour registry, `config/palette_v4_registry.json`. Its stable filename is retained for reproducibility; the current palette is **palette-v7-fixed-semantic-roles-20261107**. The current review deck contains the same complete figure set.

## Colour roles

The four anchors are blue, apricot, coral and muted plum. Treatment identity uses blue, apricot and coral. Visit identity uses light blue, teal and deep ocean blue. Observed and adjusted estimates use taupe and muted plum. These are fixed roles across the complete numbered figure, not colours selected independently for each panel.

| Role | Colour |
|---|---|
| treatment.dupilumab | `#3B7197` |
| treatment.abrocitinib_100mg | `#C78557` |
| treatment.abrocitinib_200mg | `#B85060` |
| treatment.pooled_abrocitinib | `#A97865` |
| visit.W0 | `#79B9CF` |
| visit.W12 | `#269DAB` |
| visit.W24 | `#245568` |
| method.reference | `#A18E7F` |
| method.current | `#775D78` |

Treatment changes do not reassign initial-group colour. Comparisons between treatments use an analysis colour rather than either treatment's colour. A single-series panel uses one colour; there is no colour code for significance. Circles and solid lines are the default. Open circles with explicit n mark descriptive molecular estimates based on fewer than five patients. Plot titles are removed. Axis labels, keys and captions preserve the comparison identities in dark neutral text.

## Numerical scales and annotations

IGA grades 0–5 have their own six-step ordered plum scale. The main IGA figure contains only the three grade-distribution panels; treatment pathways remain in aggregate tables. Signed heatmaps use a 257-entry blue–off-white–coral lookup table. Correlations retain −1 to 1; other signed quantities retain their original common symmetric limits. Sequential blue heatmaps show numeric concentration, proportion or count with separately labelled units and normalization. Proportion scales remain 0–100%. Missing values are not zero. Count cells retain all patients and use a logarithmic colour scale; single-patient cells remain visible.

Figure 2 keeps all specified outcomes and visits. Its full endpoint denominators are in Supplementary Table S1. IgE/EOS coverage is in Table S9. These repeated numeric strips are no longer placed below the plots. Baseline molecular concentrations are now a table (S12): original-scale median, quartiles and available numeric counts with recorded quantification flags. The retired baseline heatmap is not regenerated. Counts stay in cohort flows, IGA denominators, sparse-data flags and confusion matrices, where they are essential to interpretation. Matched model MAE values also remain directly readable.

The selected-estimator figure displays the four serum-containing input comparisons, the absolute-error distribution and EASI75 classification. Clinical-only controls remain in Table 2 and Tables S6–S7. The caption distinguishes sample sets, evaluation design, uncertainty and limited severe follow-up coverage. The frozen estimator is unchanged. New laboratory and molecular-only results have separate source tables and explicitly defined populations.

## Export and checking

PDF and SVG preserve editable text. PNG previews and the figure book show the complete collection. The shared exporter checks semantic colours and rendered panel alignment. Each PDF is checked for glyph size and collisions and inspected at final size. Colour-vision and grayscale views are diagnostic checks, not a claim of universal accessibility. Direct labels and panel positions remain necessary when colours converge in grayscale.

The [reference guide](https://www.simplifiedsciencepublishing.com/resources/best-color-palettes-for-scientific-figures-and-data-visualizations) informed restrained colour families and lightness contrast. The palette is adapted to this complete manuscript, not copied from a separate example for each chart. [Nature's figure guidance](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) does not certify a particular HEX palette.

## Three-visit molecular panel

Figure 4 uses W0/W12/W24 colours consistently within every treatment panel. Each marker-treatment row holds the same patients across visits; W0 is zero and follow-up values are within-person log2 changes from baseline. Stars refer to a 138-hypothesis family for this fixed-patient analysis, not the earlier all-available-pair tests. Drug identity is given by the panel key in the caption rather than borrowing a visit colour. Stars sit beside the marker row in the corresponding visit colour; duplicate W12/W24 star-column headings are removed. The unadjusted within-group trajectories and baseline-adjusted between-treatment comparisons remain distinct displays.

The English framework figure separates development of the local targets from clinical participant flow. Single-visit IEC-compatible states in Table S16 are not labelled sustained control, remission or full MDA. Estimated EOS values use explicit labels and uncertainty distinct from observed interquartile ranges.

## Typography and incomplete laboratory records

The shared exporter uses 8.5-pt axis labels, a 7.5-pt tick/annotation floor and 9.5-pt panel letters. Required mathematical subscripts can be smaller; the rendered PDF glyph floor is checked separately. Standalone plot titles and prose strips are removed, with necessary panel identities retained in captions. Final layout and collisions are checked after enlargement.

The revised S5 uses all available IgE responses, with pooled and CAMS-only standardized trajectories and own-baseline ratios. S4 keeps observed summaries separate from fitted trajectories. Hollow W4 points are conditional temporal interpolations, never measured data or test inputs. Their missing-cell support and centre assumptions are documented in the caption and Table S17.
