# Figure colours and symbols

All current manuscript and presentation figures use `config/palette_v4_registry.json`. The configuration assigns colours by meaning across the entire numbered figure, including every panel.

| Meaning | Colour | Additional cue |
|---|---|---|
| Initial dupilumab | Ocean blue `#2878B5` | Circle |
| Initial abrocitinib 100 mg | Peach `#E49A7C` | Circle with a darker edge |
| Initial abrocitinib 200 mg | Coral `#C84B4B` | Circle |
| W0 | Light blue `#79B9CF` | Circle with a darker edge; solid line |
| W12 | Teal blue `#269DAB` | Circle; solid line |
| W24 | Deep ocean blue `#245568` | Circle; solid line |
| Pooled abrocitinib | Muted warm brown `#A96856` | Circle; used only for a defined pooled estimate |

Observed/unadjusted estimates use muted blue `#91ADB6`; adjusted/current estimates use warm brown `#7F5D50` in explicitly labelled method comparisons. These are separate from treatment and visit roles.

Treatment colours do not also represent visits. In Figure 1, panels a, b and d use treatment colours; panel c uses W12/W24 colours. In Figure 4, the two windows ending at W24 share the endpoint colour but remain separate panels with full interval titles.

Signed heatmaps use the explicit 257-entry ocean–coral LUT, with cold blue for negative values and coral for positive values. Each estimand has its own labelled colour bar. Correlations retain −1 to 1; other families retain their original common symmetric limits. Proportion heatmaps use the ocean sequential LUT fixed at 0–100%. Recorded IGA grades use six ordered blues. Missing estimates, true zero and sparse cells remain distinct. An unknown recorded regimen is a counted category, not a missing cell.

Categorical points, lines and intervals have full opacity. Bands use alpha 0.16 and retain their stated statistical meaning. Dense forests use smaller marks to avoid collisions. Ordinary data points are circles and all trend lines are solid. Colour carries category identity; facet titles and direct labels separate pathways and intervals. Grayscale and colour-vision simulations are diagnostic checks, not accessibility certification.

The [Nature figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) support legible text, accessible presentation and editable RGB vector artwork; they do not prescribe these HEX values. PDF and SVG retain vector text; PNG is supplied for viewing. Source figures retain the project’s 6.3-pt text floor and are checked again at their final manuscript size.
