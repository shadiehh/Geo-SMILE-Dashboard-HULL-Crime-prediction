// Word report: Chapter-4 results for the Hull random-donor (absolute) Geo-SMILE design.
// Reads outputs/geo_random_donor_design/report/report_data.json (make_random_donor_report_data.py)
// and writes outputs/geo_random_donor_design/report/Hull_GeoSMILE_random_donor_chapter4_results.docx.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType, HeadingLevel,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber, TableOfContents,
} = require("docx");

const REP = path.join(__dirname, "outputs", "geo_random_donor_design", "report");
const D = JSON.parse(fs.readFileSync(path.join(REP, "report_data.json"), "utf8"));
const W = 9026; // A4 content width (DXA) with 1-inch margins
const F = D.feature, G = D.geo;
const ORDER = D.order, SHORT = { gp: "GP", transformer: "Trans.", catboost: "CatB.", lightgbm: "LGBM", xgboost: "XGB", ann: "ANN", svr: "SVR", cnn: "CNN", dl: "Keras", mlp: "MLP" };

const f = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const e = (v) => Number(v).toExponential(2).replace("e-", "×10⁻").replace("e+", "×10^").replace(/⁻0?(\d)/, (m, d1) => "⁻" + "⁰¹²³⁴⁵⁶⁷⁸⁹"[d1]);
const pf = (p) => (p < 0.001 ? (p < 1e-4 ? p.toExponential(1) : f(p, 4)) : f(p, 3));
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function P(text, opts = {}) {
  const runs = (Array.isArray(text) ? text : [text]).map((t) => (typeof t === "string" ? new TextRun(t) : new TextRun(t)));
  return new Paragraph({ children: runs, spacing: { after: 120 }, ...opts });
}
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: (Array.isArray(t) ? t : [t]).map((x) => new TextRun(x)), spacing: { after: 60 } });
const caption = (t) => new Paragraph({ children: [new TextRun({ text: t, italics: true, size: 18 })], spacing: { before: 60, after: 200 } });

function table(header, rows, widths, opts = {}) {
  const sz = opts.size || 18;
  const mk = (text, i, isHead, bold) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: isHead ? { fill: "E8EEF4", type: ShadingType.CLEAR } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text: String(text), bold: isHead || bold, size: sz })] })],
  });
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => mk(c, i, false, opts.boldRow && opts.boldRow(r))) }))],
  });
}
const even = (n, first) => { const rest = Math.floor((W - first) / (n - 1)); return [W - rest * (n - 1), ...Array(n - 1).fill(rest)]; };

function image(file, w, h) {
  return new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
    transformation: { width: w, height: h }, altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } })] });
}

// ---------- content ----------
const K = D.K.chosen_K, kt = D.K.table;
const geo4 = D.by_n.find((r) => r.n === 4000).geo, feat4 = D.by_n.find((r) => r.n === 4000).feature;
const acc = G.accuracy, fa = F.accuracy;
const children = [];

children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Empirical Validation of Geo-SMILE in Hull: Random-Donor Geo Design")] }));
children.push(P([{ text: "Chapter 4 results recomputed under the random-donor (absolute) Geo branch", italics: true }]));
children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 1 Summary
children.push(H1("1. Summary"));
children.push(P("This report recomputes every Chapter 4 result for Kingston upon Hull under a Geo branch aligned with the Seattle and Chicago designs. One XGBoost model is trained on the 30 predictors plus the British National Grid centroid easting and northing. The Feature branch masks only the 30 predictors and keeps every row's true coordinates; the Geo branch relocates the coordinates of masked LSOAs and keeps all 30 predictors. Instead of moving a masked LSOA to a single median point, each masked LSOA receives the centroid of a randomly drawn other Hull LSOA, and the response is averaged over K = " + K + " donor draws."));
children.push(bullet(`Fidelity is high in both branches: Feature R² = ${f(feat4.R2)}, Geo R² = ${f(geo4.R2)} at N = 4,000.`));
children.push(bullet(`The signed Geo coefficient reproduces the exact single-LSOA shift (ρ_s = ${f(G.exact_vs_signed_spearman, 3)}); ranking by |β| does not (ρ_s = ${f(G.exact_vs_abs_spearman, 2)}).`));
children.push(bullet(`Geo sensitivity concentrates on Old Town (029E) and the city-centre, Dairycoates and Hessle Road corridor, with significant clustering (Moran's I = ${f(G.moran.I, 3)}, p = ${f(G.moran.p_sim, 3)}).`));
children.push(bullet(`ATT accuracy: Feature ${fa.intersection}/10; Geo ${acc.intersection}/20 inside the expert-highlighted regions (lift ${f(acc.lift, 2)}, p = ${pf(G.accuracy_rule_robustness[Object.keys(G.accuracy_rule_robustness)[0]].p_hypergeom)}).`));
children.push(bullet(`Stability is complete (top-20 Jaccard 1.000 in all four directions) and consistency is good (rank ρ = ${f(G.consistency.rank_spearman_mean, 3)}).`));
children.push(bullet("ATT Faithfulness, the cross-model correlation between predictive R² and attribution quality, is not demonstrated in either branch."));
children.push(bullet(`Interpretive caveat: because the model uses a Poisson (log-link) objective, a relocation changes predictions roughly proportionally, so absolute shifts are larger where predicted crime is higher. The Geo ranking correlates with mean crime level at ρ = ${f(G.rho_with_crime_level, 2)}. The coefficient therefore measures where uncertainty about location would change predicted crime most in absolute terms, not where the model depends most on location in relative terms.`));

// 2 Design
children.push(H1("2. Study design"));
children.push(H2("2.1 Predictive model and branches"));
children.push(P("The data, LSOA-month panel, target (next-month recorded crime), chronological split (training to June 2025, validation July–December 2025, test from January 2026) and the 30 predictors are those of Chapter 4. The difference is that the predictive model includes the two centroid coordinates as inputs, as in the Seattle and Chicago experiments, and both branches explain that one model."));
children.push(table(["Design element", "Feature branch", "Geo branch"], [
  ["Predictive model", "XGBoost on 30 predictors + easting/northing", "The same model"],
  ["Explanation players", "30 predictors", "168 LSOAs"],
  ["Masking", "Masked predictor → training median; coordinates kept", "Masked LSOA's rows → centroid of a random other Hull LSOA; predictors kept"],
  ["Draws per perturbation", "–", `K = ${K} donor draws, mean response`],
  ["Perturbations", "4,000 binary masks", "4,000 coalitions of 5–10 LSOAs"],
  ["Response", "Global W1 between prediction distributions", "W1 over affected LSOA-month predictions (crimes/month)"],
  ["Surrogate", "Kernel-weighted Ridge, α = 1; 60/20/20 split", "Kernel-weighted Ridge, α = 1; 60/20/20 split"],
  ["Importance score", "Absolute coefficient", "Signed coefficient"],
  ["Expert reference", "Top 10 of 30", "Top 20 vs 58 LSOAs in expert regions (map 1)"],
], [2200, 3413, 3413]));
children.push(caption("Table 1. Final research design for the Feature and Geo branches."));
children.push(H2("2.2 Why random donors"));
children.push(P("Moving every masked LSOA to one median point makes the answer depend on that point: a design check showed that changing the reference point to a city-border location changed the ranking substantially (ρ = 0.25–0.53 with the median-point ranking), and that LSOAs far from the median point were favoured (ρ = 0.22 with the distance moved). Relocating to random real Hull locations averages over all of them, so no single reference point is privileged and the distance bias disappears (ρ = −0.01)."));
children.push(H2("2.3 Number of donor draws"));
children.push(P(`K was chosen on XGBoost at N = 4,000 as the smallest value with fidelity and consistency within 0.005 of the best value and a ranking Spearman of at least 0.99 with K = 50. No smaller K met all three criteria, so K = ${K} is used; consistency was still rising slightly, so the Geo ranking carries more sampling noise than the area-data design.`));
children.push(table(["K", "Fidelity R²", "Consistency rank ρ", "Mean SD", "ρ with K = 50"], kt.map((r) => [r.K, f(r.fidelity_R2), f(r.consistency_rank_rho, 3), f(r.consistency_mean_sd, 5), f(r.rho_with_K50, 3)]), even(5, 1400)));
children.push(caption("Table 2. Selection of the number of donor draws K (XGBoost, N = 4,000)."));

// 3 Model selection
children.push(H1("3. Selection of the predictive model"));
children.push(H2("3.1 Stage 1: predictive adequacy"));
const pr = ORDER.map((m) => D.predictive[m]).sort((a, b) => b.r2 - a.r2);
children.push(table(["Model", "Family", "MAE", "RMSE", "R²"], pr.map((r) => [r.name, r.family, f(r.mae), f(r.rmse), f(r.r2)]), [2300, 1900, 1600, 1600, 1626]));
children.push(caption("Table 3. Held-out predictive performance with the 30 predictors and the two coordinates."));
children.push(P(`The Gaussian process (R² = ${f(D.predictive.gp.r2)}), XGBoost (${f(D.predictive.xgboost.r2)}), CatBoost (${f(D.predictive.catboost.r2)}) and LightGBM (${f(D.predictive.lightgbm.r2)}) again form a top tier within 0.006 of each other. XGBoost is retained as the reference model.`));
const menuTable = (menu, cap) => {
  const rows = Object.keys(menu).map((s) => [s.replace("$\\alpha=0$", "α = 0"), ...ORDER.map((m) => f(menu[s][m].r2))]);
  children.push(table(["Surrogate", ...ORDER.map((m) => SHORT[m])], rows, [1726, ...Array(10).fill(730)], { size: 14 }));
  children.push(caption(cap));
};
children.push(H2("3.2 Stage 2: surrogate fidelity"));
menuTable(D.feature_menu, "Table 4. Feature-branch surrogate fidelity (held-out R², N = 4,000). Null, SD is the standard deviation of the response.");
menuTable(D.geo_menu, "Table 5. Geo-branch surrogate fidelity (held-out R², N = 4,000, random-donor design).");
children.push(P(`Weighted Ridge is again effectively indistinguishable from the other linear estimators and is retained. On the Geo branch it reaches R² = ${f(D.geo_menu["Weighted Ridge"].xgboost.r2)} for XGBoost, with values between ${f(Math.min(...ORDER.map((m) => D.geo_menu["Weighted Ridge"][m].r2)))} and ${f(Math.max(...ORDER.map((m) => D.geo_menu["Weighted Ridge"][m].r2)))} across the ten models.`));
const wr = ORDER.map((m) => { const a = D.feature_menu["Weighted Ridge"][m], b = D.geo_menu["Weighted Ridge"][m];
  return [D.long[m], f(a.r2), f(a.mae, 3), f(a.wrmse, 3), f(b.r2), f(b.mae, 3), f(b.wrmse, 3)]; });
children.push(table(["Model", "Feature R²", "MAE", "WRMSE", "Geo R²", "MAE", "WRMSE"], wr, [2000, 1171, 1171, 1171, 1171, 1171, 1171], { size: 16 }));
children.push(caption("Table 6. Weighted-Ridge fidelity for all ten models, both branches (N = 4,000)."));

// 4 Results
children.push(H1("4. Results"));
children.push(H2("4.1 Global Feature importance"));
children.push(image(D.figures.feature_global, 520, 325));
children.push(caption("Figure 1. Leading global Feature-SMILE drivers (absolute weighted-Ridge coefficient, N = 4,000)."));
children.push(P("Recent crime history dominates: " + D.feature_importance.slice(0, 6).map((x) => `${x.feature} (${f(x.coef)})`).join(", ") + ". All ten leading coefficients are negative; because the response is a Wasserstein distance, the sign is not a direction of prediction change and importance is assessed by magnitude."));
children.push(H2("4.2 Geographic sensitivity"));
children.push(image(path.join(REP, "geo_map_random_donor.png"), 620, 441));
children.push(caption("Figure 2. (a) Geo-SMILE geographic sensitivity coefficient, top 10 numbered and named; (b) single-LSOA Wasserstein prediction shift (mean over all 167 donor locations), with the expert regions outlined."));
children.push(image(path.join(REP, "geo_sign_random_donor.png"), 500, 516));
children.push(caption("Figure 3. Signed Geo-SMILE coefficient on a symmetric diverging scale centred at zero (capped at the 98th percentile of |β|), with the expert regions (map 1) outlined. Dark circles: ten most positive LSOAs; blue circles: five most negative. A positive coefficient means relocating the LSOA raises the coalition W1 above the surrogate average; a negative one means it dilutes it. The sign is not a direction of predicted crime change."));
children.push(P(`${G.n_negative} of 168 coefficients are negative, but all are small (minimum −0.072) and spread across the city; the five most negative lie in Bransholme, Garden Village and the Avenues, where relocating the LSOA changes its predictions by only about 0.1 crimes per month. The positive coefficients cluster in Old Town and the Dairycoates–Hessle Road corridor, largely inside the expert regions.`));
children.push(image(path.join(REP, "geo_clusters_random_donor.png"), 620, 492));
children.push(caption("Figure 4. Annotated signed Geo-SMILE map. Outlines are significant local Moran's I clusters (queen contiguity, 999 permutations, p < 0.05) plus the top-10 corridor: A, the Old Town–Stepney–Dairycoates peak corridor; B, the eastern secondary pocket (Marfleet, Greatfield, Summergangs); C, an isolated high Sculcoates LSOA; D, the northern Low–Low cluster (Bransholme, Kingswood); E, Low–Low pockets beside the high cores (Garden Village, the Avenues)."));
children.push(table(["Rank", "LSOA", "Name", "Coefficient", "Single-LSOA shift"], G.top10.map((r) => [r.geo_rank, r.lsoa21cd, r.lsoa21nm.replace("Kingston upon Hull ", ""), f(r.geo_coefficient), f(r.exact_w1, 3)]), [900, 1900, 2000, 2000, 2226]));
children.push(caption("Table 7. The ten LSOAs with the highest Geo-SMILE coefficients."));
children.push(P(`Sensitivity is concentrated in Old Town (029E, coefficient ${f(G.top10[0].geo_coefficient, 3)}, single-LSOA shift ${f(G.top10[0].exact_w1, 2)} crimes per month) and along the city-centre, Dairycoates and Hessle Road corridor west of the River Hull. The signed coefficient ranking agrees with the single-LSOA shift (ρ_s = ${f(G.exact_vs_signed_spearman, 3)}), whereas the absolute-coefficient ranking does not (ρ_s = ${f(G.exact_vs_abs_spearman, 2)}). The coefficients are positively spatially autocorrelated (global Moran's I = ${f(G.moran.I, 3)}, p = ${f(G.moran.p_sim, 3)}, queen contiguity, 999 permutations), with ${G.moran.lisa_counts.HH || 0} High–High LSOAs. The median single-LSOA shift is ${f(G.exact_w1_median, 2)} crimes per month and the 90th percentile ${f(G.exact_w1_p90, 2)}; ${G.n_negative} of 168 coefficients are negative.`));
children.push(H2("4.3 Local Feature importance"));
children.push(image(D.figures.feature_local, 560, 373));
children.push(caption("Figure 5. Signed local coefficients of the six leading global Feature drivers."));
children.push(H2("4.4 Surrogate fidelity and perturbation budget"));
children.push(table(["N", "Feature R²", "Feature wR²", "Geo R²", "Geo wR²"], D.by_n.map((r) => [r.n.toLocaleString("en-GB"), f(r.feature.R2), f(r.feature.wR2), f(r.geo.R2), f(r.geo.wR2)]), even(5, 1400), { boldRow: (r) => r[0] === "4,000" }));
children.push(caption("Table 8. Held-out weighted-Ridge fidelity by perturbation count."));

// 5 Evaluation
children.push(H1("5. Evaluation metrics"));
children.push(H2("5.1 Expert references"));
children.push(P("Three expert references are reported. Map 1 is the Chapter 4 reference (58 LSOAs in the Sculcoates–City Centre–Dairycoates corridor, Bransholme–Sutton–Stoneferry and Southcoates–Marfleet). Map 2 is a second, schematic expert map with seven regions (57 LSOAs), georeferenced with an affine fit on five control points (residuals 60–290 m). Map 3 is map 2 with the expert's hand-drawn extensions of Orchard Park, the city-centre corridor and a joint Preston Road–Greatfield region (77 LSOAs)."));
children.push(image(D.figures.expert_map1, 430, 330));
children.push(caption("Figure 6. Expert map 1 (Chapter 4 reference)."));
children.push(image(D.figures.expert_map2, 400, 320));
children.push(caption("Figure 7. Expert map 2, georeferenced (57 LSOAs)."));
children.push(image(D.figures.expert_map3, 400, 320));
children.push(caption("Figure 8. Expert map 3, map 2 with hand-drawn extensions (77 LSOAs)."));
children.push(H2("5.2 ATT fidelity"));
const ff = F.fidelity, gf = G.fidelity;
children.push(table(["Branch", "R²", "Weighted R²", "MAE", "WMAE", "RMSE", "WRMSE"], [
  ["Feature", f(ff.R2), f(ff["Weighted R2"]), f(ff.MAE), f(ff.WMAE), f(ff.RMSE), f(ff.WRMSE)],
  ["Geo", f(gf.R2), f(gf["Weighted R2"]), f(gf.MAE), f(gf.WMAE), f(gf.RMSE), f(gf.WRMSE)]], even(7, 1400)));
children.push(caption("Table 9. ATT fidelity at N = 4,000 (XGBoost, seed 42)."));
children.push(H2("5.3 ATT accuracy"));
children.push(table(["Metric", "Feature", "Geo"], [
  ["n / reference size / k", "30 / 10 / 10", `168 / ${acc.reference_size} / 20`],
  ["Intersection", fa.intersection, acc.intersection],
  ["Accuracy (chance)", `${f(fa.intersection / 10, 3)} (0.333)`, `${f(acc.accuracy, 3)} (${f(acc.chance, 3)})`],
  ["Lift", f(fa.intersection / 10 / (1 / 3), 2), f(acc.lift, 2)],
  ["ATT ACC", f(fa.accuracy), f(acc.att_acc)], ["ATT F1", f(fa.f1), f(acc.att_f1)], ["ATT AUROC", f(fa.auroc), f(acc.att_auroc)],
  ["Precision@k / Recall@k", `${f(fa.precision, 3)} / ${f(fa.recall, 3)}`, `${f(acc.precision, 3)} / ${f(acc.recall, 3)}`],
  ["Jaccard@k", f(fa.jaccard_at_k), f(acc.jaccard)]], [3226, 2900, 2900]));
children.push(caption("Table 10. Primary ATT accuracy at N = 4,000. Feature: top 10 against the 10-predictor reference. Geo: top 20 against expert map 1."));
children.push(P(`The Feature branch recovers ${fa.intersection} of the 10 reference predictors. ${acc.intersection} of the 20 most sensitive LSOAs lie inside the expert-highlighted regions; the four outside are ${acc.top_k_outside_reference.join(", ")}.`));
const rr = Object.entries(G.accuracy_rule_robustness).map(([k, v]) => [k.replace(">=", "≥").replace(">=", "≥").replace("km2", "km²"), v.reference_size, v.intersection, f(v.chance, 3), f(v.lift, 2), pf(v.p_hypergeom)]);
children.push(table(["Inclusion rule (map 1)", "Reference", "Top 20 inside", "Chance", "Lift", "p"], rr, [3426, 1100, 1200, 1100, 1000, 1200]));
children.push(caption("Table 11. Robustness of Geo accuracy to the rule assigning LSOAs to the expert regions (one-sided hypergeometric p)."));
const km = (m, k) => (m === "map1" ? (k === 20 ? acc : G.accuracy_k[`map1_k${k}`]) : G.accuracy_other_maps[`${m}_k${k}`]);
const krow = (m, label) => [label, ...[20, 30, 42].map((k) => { const a = km(m, k); return `${a.intersection}/${k} (p = ${pf(a.p_hypergeom)})`; })];
children.push(table(["Expert reference", "Top 20", "Top 30", "Top 42"], [krow("map1", "Map 1 (58 LSOAs)"), krow("map2", "Map 2 (57 LSOAs)"), krow("map3", "Map 3 (77 LSOAs)")], [2426, 2200, 2200, 2200]));
children.push(caption("Table 12. Geo accuracy at k = 20, 30 and 42 against the three expert maps."));
children.push(P("Agreement is strong against maps 1 and 3 and not significant against map 2, which adds the northern and eastern estates (Orchard Park, Bransholme, Longhill, Greatfield, Preston Road). A crime-level ranking alone places 16 of 20 inside map 1 and 17 of 20 inside map 3, so much of the agreement reflects that both the expert and the absolute random-donor sensitivity concentrate on high-crime areas."));
children.push(H2("5.4 ATT Faithfulness"));
const fr = D.faithfulness.filter((r) => r.subset === "all");
children.push(table(["Attribution measure", "M", "Pearson r", "Two-sided p", "Spearman ρ"], fr.map((r) => [r.measure, r.M, f(r.pearson_r, 2), f(r.p_two_sided, 2), f(r.spearman_rho, 2)]), [3826, 800, 1400, 1500, 1500]));
children.push(caption("Table 13. ATT Faithfulness: correlation, across the ten models, between predictive R² and attribution quality."));
children.push(table(["Model", "R²", "Geo Ridge R²", "Geo top 20 (map 1)", "Geo AUROC", "Feature top 10"], D.models.map((r) => [D.long[r.model], f(r.r2, 3), f(r.geo_ridge_r2, 3), r.geo_overlap_map1, f(r.geo_auc_map1, 3), r.feat_overlap]), [2126, 1100, 1400, 1600, 1400, 1400]));
children.push(caption("Table 14. Model-level results underlying ATT Faithfulness."));
children.push(P("No correlation is statistically significant in either branch, including without the two non-reproducible Keras models. As in Chapter 4, ATT Faithfulness under the performance–attribution criterion is not demonstrated; with ten models only strong correlations would be detectable."));
children.push(H2("5.5 ATT stability"));
const sd = G.stability.directions, fsd = F.stability.directions, DIRS = ["north", "east", "south", "west"];
children.push(table(["Branch", ...DIRS.flatMap((d) => [`${d[0].toUpperCase() + d.slice(1)} J`, "ρ"])], [
  ["Feature", ...DIRS.flatMap((d) => [f(fsd[d].jaccard, 3), f(fsd[d].spearman_abs, 4)])],
  ["Geo", ...DIRS.flatMap((d) => [f(sd[d].jaccard, 3), f(sd[d].spearman, 4)])]], [1226, ...Array(8).fill(975)], { size: 16 }));
children.push(caption("Table 15. ATT stability at N = 4,000 under one neighbouring-LSOA addition per cardinal direction."));
children.push(table(["Addition", "LSOA", "Rank", "Coefficient", "In top 20"], DIRS.map((d) => [d[0].toUpperCase() + d.slice(1), sd[d].lsoa21cd, sd[d].added_rank, f(sd[d].added_coefficient), sd[d].added_in_top20 ? "Yes" : "No"]), even(5, 1800)));
children.push(caption("Table 16. Geo ranks of the added LSOAs."));
const wst = G.fixed_kernel_west;
children.push(P(`Both top sets are fully preserved (Jaccard 1.000). Full-vector Spearman is ${f(F.stability.spearman_mean, 4)} for Feature and ${f(G.stability.spearman_mean, 4)} for Geo; the Geo value is lower than in the area-data design because a neighbour addition also changes the donor pool. Under a fixed inverse kernel, the west addition retains all 20 baseline LSOAs (Jaccard ${f(wst.jaccard, 3)}, ρ = ${f(wst.spearman, 4)}, added rank ${wst.added_rank}).`));
children.push(H2("5.6 ATT consistency"));
const fc = F.consistency, gc = G.consistency;
children.push(table(["Branch", "Mean var.", "Median var.", "Mean SD", "Median SD", "Max SD"], [
  ["Feature", f(fc.mean_variance, 6), f(fc.median_variance, 6), f(fc.mean_sd, 5), f(fc.median_sd, 5), f(fc.max_sd, 5)],
  ["Geo", e(gc.mean_variance), e(gc.median_variance), f(gc.mean_sd, 5), f(gc.median_sd, 5), f(gc.max_sd, 5)]], even(6, 1426)));
children.push(caption("Table 17. ATT consistency across five independently seeded perturbation runs (seeds 100–104)."));
children.push(P(`The largest Geo SD occurs for ${gc.max_sd_lsoa_name.replace("Kingston upon Hull ", "")} (Old Town), ${f(gc.max_sd_pct_of_mean, 1)}% of its mean coefficient of ${f(gc.max_sd_lsoa_mean_coefficient, 3)}. Supplementary measures: rank Spearman ${f(fc.rank_spearman_mean, 3)} (Feature) and ${f(gc.rank_spearman_mean, 3)} (Geo); score Pearson ${f(fc.score_pearson_mean, 4)} and ${f(gc.score_pearson_mean, 4)}; top-20% Jaccard ${f(fc.top20pct_jaccard_mean, 3)} and ${f(gc.top20pct_jaccard_mean, 3)}.`));
children.push(H2("5.7 Additional diagnostics"));
const fcn = F.concentration, gcn = G.concentration_exact_shift;
children.push(P(`Attribution concentration. Feature Hoyer sparsity ${f(fcn.hoyer, 3)}, Gini ${f(fcn.gini, 3)}, entropy ${f(fcn.entropy, 3)}; Geo (on the single-LSOA shift) Hoyer ${f(gcn.hoyer, 3)}, Gini ${f(gcn.gini, 3)}, entropy ${f(gcn.entropy, 3)}. Feature attribution is concentrated on a few crime-history predictors; geographic sensitivity is spread more widely, with Old Town as a pronounced outlier.`));
const fi = F.intervention, gi = G.intervention;
children.push(table(["Metric", "Feature", "Geo"], [
  ["k", fi.k, gi.k], ["Top response", f(fi.top_response), f(gi.top_response)], ["Bottom response", f(fi.bottom_response), f(gi.bottom_response)],
  ["Random mean", f(fi.random_mean), f(gi.random_mean)], ["Random 95% interval", `${f(fi.random_ci95_low, 3)}–${f(fi.random_ci95_high, 3)}`, `${f(gi.random_ci95_low, 3)}–${f(gi.random_ci95_high, 3)}`],
  ["Top − random", f(fi.top_minus_random), f(gi.top_minus_random)], ["Top − bottom", f(fi.top_minus_bottom), f(gi.top_minus_bottom)],
  ["Empirical p", f(fi.empirical_p_top_vs_random, 3), f(gi.empirical_p_top_vs_random, 3)]], [3226, 2900, 2900]));
children.push(caption("Table 18. Intervention-based attribution impact (1,000 random sets; Geo random sets matched on population and area)."));
children.push(P("In both branches the highest-ranked players change the model's predictions more than random sets (Geo p = 0.001, the smallest attainable), so the rankings identify players with genuinely larger model impact."));
children.push(table(["Branch", "N", "Fidelity R²", "Accuracy", "ΔW1", "Stability J", "Mean var.", "Mean SD"],
  D.by_n.flatMap((r) => []).concat(
    D.by_n.map((r) => ["Feature", r.n.toLocaleString("en-GB"), f(r.feature.R2), `${r.feature.acc}/10`, f(r.feature.dw, 3), f(r.feature.stab, 3), f(r.feature.var, 6), f(r.feature.sd, 5)]),
    D.by_n.map((r) => ["Geo", r.n.toLocaleString("en-GB"), f(r.geo.R2), `${r.geo.acc}/20`, f(r.geo.dw, 3), f(r.geo.stab, 3), e(r.geo.var), f(r.geo.sd, 5)])),
  [1026, 900, 1150, 1050, 1000, 1100, 1500, 1300], { size: 16, boldRow: (r) => r[1] === "4,000" }));
children.push(caption("Table 19. Evaluation diagnostics across perturbation budgets."));
children.push(image(D.figures.metrics_by_n, 620, 139));
children.push(caption("Figure 9. Fidelity, intervention impact, stability and consistency across perturbation budgets."));
children.push(P("Geo accuracy is 16/20 at every budget and the intervention effect is stable at about 1.2–1.3. Geo fidelity rises from 0.910 at N = 1,000 to about 0.93 from N = 4,000; Geo stability is complete from N = 3,000 (one top-20 change per addition at N = 1,000 and 2,000); Geo rank consistency rises steadily from 0.775 to 0.948 at N = 10,000, reflecting the additional sampling noise of random donors."));

// 6 Conclusion
children.push(H1("6. Conclusion"));
children.push(P("Under the random-donor design, both branches explain one XGBoost model that includes location, as in Seattle and Chicago. The Geo branch is well represented by the weighted-Ridge surrogate (R² ≈ 0.93), reproduces the exact single-LSOA ranking, is stable under neighbouring-LSOA additions and identifies LSOAs whose relocation changes predictions far more than random sets. Its ranking concentrates on Old Town and the city-centre to Hessle Road corridor and agrees with expert maps 1 and 3."));
children.push(P("Three qualifications should accompany these results. First, because the model is Poisson, the absolute response grows with an area's predicted crime level; the Geo ranking correlates with crime level at ρ = " + f(G.rho_with_crime_level, 2) + ", and a crime-level ranking alone achieves similar expert agreement. The design therefore identifies where location uncertainty changes predicted crime most in absolute terms. Second, the effect of location is small: relocating an area changes its predictions by about " + f(G.exact_w1_median, 2) + " crimes per month at the median. Third, ATT Faithfulness is not demonstrated, and agreement with the expert depends on which expert map is used (significant for maps 1 and 3, not for map 2)."));
children.push(P([{ text: "Sources: ", bold: true }, "outputs/geo_random_donor_design/ (Geo branch; geo_random_donor_eval.py) and outputs/geo_aligned_design/ (Feature branch and predictive models; hull_aligned_design_eval.py). All results are model sensitivity, not causal effects."]));

const doc = new Document({
  creator: "GeoSMILE analysis", title: "Hull Geo-SMILE random-donor design: Chapter 4 results",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 36, bold: true, color: "1F3864" }, paragraph: { spacing: { after: 120 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, color: "2E5597" }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  features: { updateFields: true },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(REP, "Hull_GeoSMILE_random_donor_chapter4_results.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
