// Builds Hull_Geo_Area_Data_Evaluation.docx from data.json (assembled from the project outputs).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType,
  HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat, PageOrientation, Footer,
  PageNumber, TableOfContents, PageBreak,
} = require("docx");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const FONT = "Calibri";
const INK = "12171C", MUTED = "57646D", HEAD_FILL = "E8ECF0", ACCENT = "B0003A", RULE = "BFC7CE";
const PORTRAIT_W = 9026, LAND_W = 13958;

// ---------- helpers ----------
const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pfmt = (p) => (p < 0.001 ? (p < 1e-4 ? "<0.0001" : p.toFixed(4)) : p.toFixed(3));
const sgn = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

function para(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun({ text, ...(opts.run || {}) })];
  return new Paragraph({ children: runs, spacing: { before: 100, after: 120, line: 276 }, ...opts.p });
}
function rich(parts) {
  // parts: array of strings or [text, {bold, italics}]
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    children: parts.map((p) => (typeof p === "string" ? new TextRun(p) : new TextRun({ text: p[0], ...p[1] }))),
  });
}
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (parts) => new Paragraph({
  numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 264 },
  children: (Array.isArray(parts) ? parts : [parts]).map((p) => (typeof p === "string" ? new TextRun(p) : new TextRun({ text: p[0], ...p[1] }))),
});
const numbered = (text) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 80 }, children: [new TextRun(text)] });

let tableNo = 0, figNo = 0;
function caption(text, kind = "Table") {
  const n = kind === "Table" ? ++tableNo : ++figNo;
  return new Paragraph({
    spacing: { before: kind === "Table" ? 200 : 80, after: kind === "Table" ? 80 : 240 },
    keepNext: kind === "Table",
    children: [new TextRun({ text: `${kind} ${n}. `, bold: true, size: 19 }), new TextRun({ text, size: 19, color: MUTED })],
  });
}
function note(text) {
  return new Paragraph({ spacing: { before: 60, after: 200 }, children: [new TextRun({ text, size: 17, italics: true, color: MUTED })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const borders = { top: border, bottom: border, left: border, right: border };

// cols: [{h, w, align}] ; rows: array of arrays of strings or {t, bold, color, fill}
function table(cols, rows, { width = PORTRAIT_W, size = 18 } = {}) {
  const total = cols.reduce((a, c) => a + c.w, 0);
  const scale = width / total;
  const widths = cols.map((c) => Math.round(c.w * scale));
  widths[widths.length - 1] += width - widths.reduce((a, b) => a + b, 0);
  const cell = (content, i, header) => {
    const c = typeof content === "object" && content !== null ? content : { t: String(content) };
    const align = header ? (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER) : (cols[i].align || (i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT));
    return new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: header ? { fill: HEAD_FILL, type: ShadingType.CLEAR, color: "auto" } : c.fill ? { fill: c.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      children: [new Paragraph({ alignment: align, children: [new TextRun({ text: c.t, bold: header || c.bold, size, color: c.color || INK, italics: c.italics })] })],
    });
  };
  return new Table({
    width: { size: width, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: cols.map((c, i) => cell(c.h, i, true)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((v, i) => cell(v, i, false)) })),
    ],
  });
}
function image(file, wPx, hPx, alt) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({
      type: "jpg", data: fs.readFileSync(path.join(__dirname, file)), transformation: { width: wPx, height: hPx },
      altText: { title: alt, description: alt, name: file },
    })],
  });
}
const scaleImg = (w, h, target) => [target, Math.round((h * target) / w)];

// ---------- data shortcuts ----------
const M = D.accuracy.map((a) => a.model);
const acc = Object.fromEntries(D.accuracy.map((a) => [a.model, a]));
const fid = Object.fromEntries(D.fidelity.map((r) => [r.model, r]));
const faithRow = (set, m) => D.faith.find((r) => r.model_set === set && r.measure === m);
const X = D.xgb;
const n4000 = D.nsweep.find((r) => r.N === 4000);
const minmax = (arr) => [Math.min(...arr), Math.max(...arr)];
const [a42lo, a42hi] = minmax(M.map((m) => acc[m].k[42].accuracy));
const [a20lo, a20hi] = minmax(M.map((m) => acc[m].k[20].accuracy));
const [frLo, frHi] = minmax(D.fidelity.map((r) => r.r2));
const fAll = faithRow("all models", "att_auroc"), fExcl = faithRow("excl. dl, cnn", "att_auroc");
const RM = D.ref_meta, CH = f(RM.chance, 3);
const sig = (r) => r.pearson_p < 0.05;
const rp = (r) => `r = ${sgn(r.pearson_r)}, p = ${f(r.pearson_p, 3)}`;
const regionName = (r) => r.replace(/-/g, " – ");
const imgSize = (file, target) => scaleImg(D.images[file][0], D.images[file][1], target);

// ---------- content ----------
const titleBlock = [
  new Paragraph({ spacing: { before: 1200, after: 120 }, children: [new TextRun({ text: "GEOSMILE · KINGSTON UPON HULL", size: 20, bold: true, color: MUTED, characterSpacing: 40 })] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "Evaluation of the Area-Data Geo Branch", size: 52, bold: true, color: INK })] }),
  new Paragraph({ spacing: { after: 480 }, children: [new TextRun({ text: "ATT fidelity, ATT accuracy against expert-highlighted regions, ATT faithfulness, ATT stability and ATT consistency for ten candidate models", size: 26, color: MUTED })] }),
  table(
    [{ h: "Item", w: 3000 }, { h: "Setting", w: 6000, align: AlignmentType.LEFT }],
    [
      ["Prediction task", "Next-month crime count per LSOA (Kingston upon Hull)"],
      ["Spatial units", "168 LSOAs (ONS 2021 boundaries)"],
      ["Candidate models", "XGBoost, LightGBM, CatBoost, SVR, GP, MLP, DL, ANN, CNN, Transformer"],
      ["Geo perturbations", "N = 4,000, seed 42 (5–10 LSOAs masked per perturbation)"],
      ["Expert reference", `${RM.n} LSOAs inside ${D.ref_summary.length} expert-highlighted regions`],
      ["Test period", "January–March 2026"],
    ],
  ),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Contents", bold: true, size: 32 })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const summary = [
  h1("1. Executive summary"),
  para("This report evaluates the Geo branch of GeoSMILE under the area-data design, in which masking an LSOA replaces all of the data the 30-feature model receives from that location. Attribution accuracy is measured against the regions an expert highlighted as the areas whose crime events inform their assessment of Hull's crime situation. All five attribution-evaluation criteria used for SMILE are reported."),
  bullet([["ATT fidelity. ", { bold: true }], `The weighted-Ridge surrogate reproduces the perturbation responses closely for every model: held-out R² ${f(frLo)}–${f(frHi)}.`]),
  bullet([["ATT accuracy. ", { bold: true }], `Every model's most sensitive LSOAs fall in the expert regions far more often than chance (${CH}): accuracy@20 ${f(a20lo, 2)}–${f(a20hi, 2)} and accuracy@42 ${f(a42lo, 3)}–${f(a42hi, 3)}, all with hypergeometric p < 0.001.`]),
  bullet([["ATT faithfulness. ", { bold: true }], `Across the ten models, agreement with the expert regions does not rise significantly with predictive R² (ATT-AUC: ${rp(fAll)}). Without the two non-reproducible Keras models, ATT-AUC gives ${rp(fExcl)}${sig(fExcl) ? "" : " (not significant)"} and top-20 overlap gives ${rp(faithRow("excl. dl, cnn", "hits@20"))}; this is a sensitivity result only.`]),
  bullet([["ATT stability (XGBoost). ", { bold: true }], `Adding a neighbouring LSOA outside Hull in any compass direction leaves the top 20 unchanged (Jaccard ${f(X.stability.jaccard_mean, 2)}) and the ranking almost identical (Spearman ${f(X.stability.spearman_mean, 4)}).`]),
  bullet([["ATT consistency (XGBoost). ", { bold: true }], `Across five seeded runs the coefficients vary little: mean variance ${f(X.consistency.mean_variance, 4)}, mean SD ${f(X.consistency.mean_sd, 3)}, median SD ${f(X.consistency.median_sd, 3)}. The largest SD (${f(X.consistency.max_sd, 3)}) belongs to the LSOA with by far the largest coefficient and is only ${f(X.consistency.max_sd_pct_of_mean, 1)}% of it.`]),
  caption("Summary of the five evaluation criteria under the area-data Geo design."),
  table(
    [{ h: "Criterion", w: 1900 }, { h: "Measure", w: 3100, align: AlignmentType.LEFT }, { h: "Models", w: 1300, align: AlignmentType.CENTER }, { h: "Result", w: 2700, align: AlignmentType.LEFT }],
    [
      ["ATT fidelity", "Held-out R² of the weighted-Ridge surrogate", "All 10", `${f(frLo)}–${f(frHi)}`],
      ["ATT accuracy", `Share of top-k LSOAs in expert regions (chance ${CH})`, "All 10", `@20: ${f(a20lo, 2)}–${f(a20hi, 2)}; @42: ${f(a42lo, 3)}–${f(a42hi, 3)}`],
      ["ATT faithfulness", "Pearson r (predictive R², ATT-AUC) across models", "All 10", `r = ${sgn(fAll.pearson_r)} (p = ${f(fAll.pearson_p, 2)}); not demonstrated`],
      ["ATT stability", "Top-20 Jaccard / rank Spearman after adding a neighbour LSOA", "XGBoost", `${f(X.stability.jaccard_mean, 2)} / ${f(X.stability.spearman_mean, 4)}`],
      ["ATT consistency", "Run-to-run coefficient variance and SD over 5 seeds", "XGBoost", `Mean var ${f(X.consistency.mean_variance, 4)}; mean SD ${f(X.consistency.mean_sd, 3)}; max SD ${f(X.consistency.max_sd, 3)}`],
    ].map((r) => r.map((v, i) => (i === 3 ? { t: v } : v))),
  ),
  note("Stability and consistency were run for the final XGBoost explanation only (as in Chapter 4); they do not depend on the expert reference."),
];

const design = [
  h1("2. The area-data Geo design"),
  para("In the earlier Geo branch, a coordinates-only model was explained and masking an LSOA replaced only its easting and northing. Under the area-data design the Geo branch explains the same fitted 30-feature model as the Feature branch, and masking an LSOA replaces all of its model inputs with the training median. Each Geo coefficient therefore measures how sensitive the model is to the data it takes from that location."),
  para("Because every masked row receives the same inputs, the masked prediction is one typical-area value, and the per-LSOA response orders LSOAs by how far their predictions sit from it. The signed coefficient preserves that ordering, while its absolute value would fold it, so all rankings use the signed coefficient. Coefficients describe model sensitivity, not causal effects of places on crime."),
  caption("Settings of the area-data Geo branch."),
  table(
    [{ h: "Component", w: 2600 }, { h: "Setting", w: 6400, align: AlignmentType.LEFT }],
    [
      ["Explained model", "The same 30-predictor model as the Feature branch, one per candidate model (seed 42)"],
      ["Players", "168 Hull LSOAs"],
      ["Perturbation", "5–10 randomly chosen LSOAs; all of their inputs replaced by the training median"],
      ["Response", "First Wasserstein distance between original and masked predictions on the affected LSOA-month rows"],
      ["Surrogate", "Kernel-weighted Ridge (α = 1); kernel (RBF, Laplace, inverse) chosen on a validation split"],
      ["Pool size", "N = 4,000 perturbations; 60/20/20 train/validation/test split of the pool"],
      ["Ranking", "Signed Geo-SMILE coefficient"],
    ],
  ),
];

const refRows = D.ref_summary.map((r) => [regionName(r.region), String(r.n), String(r.exp), String(r.all10)]);
const refTot = D.ref_summary.reduce((a, r) => [a[0] + r.n, a[1] + r.exp, a[2] + r.all10], [0, 0, 0]);
const [rw, rh] = imgSize("regions.jpg", 560);
const reference = [
  h1("3. Expert reference regions"),
  para("The expert marked on street maps the areas whose crime events inform their assessment of Hull's crime situation. The drawings were georeferenced on street landmarks (The Deep, MKM Stadium, the Hedon Road roundabouts and Marfleet Lane, the Bransholme estate, East Park and the 1 km map scale) and intersected with the ONS 2021 LSOA boundaries. There are three regions: the Sculcoates – City Centre – Dairycoates corridor; a Southcoates – Marfleet strip along Preston Road; and a northern region formed from two overlapping drawings, a loop from Raich Carter Way round Bransholme and down the River Hull to Clough Road, and a block between Stoneferry Road, Sutton Road and East Park."),
  para(`Inclusion rule: an LSOA belongs to the corridor or the Southcoates – Marfleet region if at least 20% of its area, or at least 0.1 km² of it, lies inside the drawn region. For the northern region the expert asked that nothing extending beyond the boundary be included, so an LSOA counts only if the majority of its area (at least 50%) lies inside. The reference contains ${RM.n} of the ${RM.n_all} LSOAs, so a randomly chosen LSOA is a reference LSOA with probability ${CH}. It includes ${RM.expert_top20_inside} of the 20 LSOAs in the earlier expert top-20 list. The full list is in Appendix A.`),
  caption("Expert reference LSOAs by region."),
  table(
    [{ h: "Region", w: 3900 }, { h: "LSOAs", w: 1300 }, { h: "In earlier expert top 20", w: 1900 }, { h: "In top 50 of all 10 models", w: 1900 }],
    [...refRows, [{ t: "Total", bold: true }, { t: String(refTot[0]), bold: true }, { t: String(refTot[1]), bold: true }, { t: String(refTot[2]), bold: true }]],
  ),
  image("regions.jpg", rw, rh, "Map of the expert reference regions over Hull LSOAs"),
  caption("Reference LSOAs by region (shaded), traced expert outlines (dashed), and LSOAs touching a region but below the inclusion rule (hatched).", "Figure"),
];

// ---------- 4. Fidelity ----------
const fidelity = [
  h1("4. ATT fidelity"),
  para("ATT fidelity measures how well the kernel-weighted Ridge surrogate reproduces the model's perturbation responses on held-out perturbations. It is reported as R², weighted R², and the mean absolute and root-mean-square errors, unweighted and kernel-weighted (MAE, WMAE, RMSE, WRMSE; in crimes per month)."),
  caption("Geo fidelity of the weighted-Ridge surrogate for all ten models (N = 4,000, held-out 20% of the pool)."),
  table(
    [{ h: "Model", w: 1300 }, { h: "Family", w: 1100, align: AlignmentType.LEFT }, { h: "Predictive R²", w: 1100 }, { h: "Fidelity R²", w: 1000 }, { h: "Weighted R²", w: 1000 }, { h: "MAE", w: 800 }, { h: "WMAE", w: 800 }, { h: "RMSE", w: 800 }, { h: "WRMSE", w: 800 }, { h: "Kernel", w: 900, align: AlignmentType.CENTER }],
    D.fidelity.map((r) => [r.model, r.family, f(r.pred_r2), f(r.r2), f(r.wr2), f(r.mae), f(r.wmae), f(r.rmse), f(r.wrmse), r.kernel]),
    { size: 17 },
  ),
  note("Fidelity is uniformly high (R² 0.945–0.949) and barely differs between models, so differences in accuracy or faithfulness below are not explained by surrogate quality. Error scales differ because the models' predictions and responses have different ranges."),
];

const menu = [
  h2("4.1 Surrogate comparison"),
  para("The full Chapter-4 surrogate menu was fitted to each model's pool and scored on the same held-out split. Linear surrogates (weighted and unweighted Ridge, OLS, Bayesian ridge) all reach R² ≈ 0.95; tree ensembles are lower and the GLMs and kernel ridge fit poorly. The permutation null row gives the standard deviation of R² when responses are shuffled."),
  caption("Held-out R² of each surrogate for each model (N = 4,000)."),
  table(
    [{ h: "Surrogate", w: 2300 }, ...D.menu.models.map((m) => ({ h: m, w: 1160 }))],
    D.menu.rows.map((r) => [r.surrogate === "Weighted Ridge" ? { t: r.surrogate, bold: true } : r.surrogate,
      ...r.vals.map((v) => (r.surrogate === "Weighted Ridge" ? { t: f(v), bold: true } : f(v)))]),
    { width: LAND_W, size: 17 },
  ),
];

const nsw = D.nsweep;
const fidN = [
  h2("4.2 Fidelity across perturbation budgets (XGBoost)"),
  caption("XGBoost Geo fidelity by number of perturbations N."),
  table(
    [{ h: "N", w: 1500 }, { h: "Fidelity R²", w: 2500 }, { h: "Weighted R²", w: 2500 }],
    nsw.map((r) => [r.N.toLocaleString("en-GB"), f(r.fidelity_r2), f(r.fidelity_weighted_r2)]),
  ),
  note("Fidelity stabilises at about 0.94–0.95 from N = 3,000; N = 4,000 gives the highest value of the sweep."),
];

// ---------- 5. Accuracy ----------
const KS = RM.ks, KR = KS[3];  // KR = reference size, where accuracy equals recall
const sorted42 = [...M].sort((a, b) => acc[b].k[42].accuracy - acc[a].k[42].accuracy || acc[b].auroc - acc[a].auroc);
const hi = (v, best) => (Math.abs(v - best) < 1e-9 ? { t: f(v, 3), bold: true, color: ACCENT } : f(v, 3));
const best = (k) => Math.max(...M.map((m) => acc[m].k[k].accuracy));
const bestAuc = Math.max(...M.map((m) => acc[m].auroc));
const [pw, ph] = imgSize("top50_panel.jpg", 600);
const [xw, xh] = imgSize("xgb_top50.jpg", 400);
// overlap tables: columns are model, one per region "(of n)", total, share, in top 20, mean rank
const regionCols = D.overlap50.cols.slice(1, -4);
const regionSize = (c) => Number(c.match(/of (\d+)/)[1]);
const shortName = (c) => c.replace(/ \(of \d+\)$/, "").replace("Sculcoates-City Centre-Dairycoates", "Corridor").replace(/-/g, "–");
const ovCols = () => [{ h: "Model", w: 1300 }, ...regionCols.map((c) => ({ h: `${shortName(c)} (${regionSize(c)})`, w: 1500 })), { h: `Total (${RM.n})`, w: 1000 }, { h: "Share of top k", w: 1100 }, { h: "Also in top 20", w: 1000 }];
const ovRows = (o) => o.rows.map((r) => [r[0], ...regionCols.map((_, i) => String(r[1 + i])), { t: String(r[1 + regionCols.length]), bold: true }, `${f(r[2 + regionCols.length], 1)}%`, String(r[3 + regionCols.length])]);
const colRange = (o, i) => minmax(o.rows.map((r) => r[1 + i]));
const xa = acc.xgboost, XF = D.xgb_facts;
const maxP = RM.max_p;
const regionPhrase = (counts) => Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([r, n]) => `${n} in ${regionName(r)}`).join(", ");
const regionRangeText = regionCols.map((c, i) => { const [lo, hi2] = colRange(D.overlap50, i); return `${lo}–${hi2} of the ${regionSize(c)} ${shortName(c) === "Corridor" ? "corridor" : shortName(c)} LSOAs`; }).join("; ");

const accuracy = [
  h1("5. ATT accuracy"),
  para(`ATT accuracy measures how many of a model's most sensitive LSOAs fall in the expert-highlighted regions. Accuracy@k is the share of the model's top-k LSOAs (by signed coefficient) that are reference LSOAs. A random ranking would score ${RM.n}/${RM.n_all} = ${CH} at every k. Lift is accuracy divided by that chance rate, and p is the one-sided hypergeometric probability that a random set of k LSOAs contains at least as many reference LSOAs. ATT-AUC scores the whole signed ranking of all ${RM.n_all} LSOAs against reference membership and needs no cut-off; its chance value is 0.5.`),
  para(`Accuracy is reported at k = 20 (the size of the earlier expert list), 42 and 50 (the cut-offs requested for this study) and ${KR} (the size of the reference, where accuracy equals recall).`),
  caption("ATT accuracy of all ten models against the expert regions, sorted by accuracy@42 (best values in colour)."),
  table(
    [{ h: "Model", w: 1300 }, { h: "Predictive R²", w: 1100 }, { h: "Hits@20", w: 850 }, { h: "Acc@20", w: 850 }, { h: "Hits@42", w: 850 }, { h: "Acc@42", w: 850 }, { h: "Hits@50", w: 850 }, { h: "Acc@50", w: 850 }, { h: `Acc@${KR}`, w: 850 }, { h: "ATT-AUC", w: 900 }],
    [
      ...sorted42.map((m) => [m, f(acc[m].pred_r2), String(acc[m].k[20].hits), hi(acc[m].k[20].accuracy, best(20)), String(acc[m].k[42].hits), hi(acc[m].k[42].accuracy, best(42)), String(acc[m].k[50].hits), hi(acc[m].k[50].accuracy, best(50)), hi(acc[m].k[KR].accuracy, best(KR)), hi(acc[m].auroc, bestAuc)]),
      [{ t: "Chance", italics: true }, "–", { t: f(RM.expected_hits[20], 1), italics: true }, { t: CH, italics: true }, { t: f(RM.expected_hits[42], 1), italics: true }, { t: CH, italics: true }, { t: f(RM.expected_hits[50], 1), italics: true }, { t: CH, italics: true }, { t: CH, italics: true }, { t: "0.500", italics: true }],
    ],
    { size: 17 },
  ),
  caption("Lift over chance and hypergeometric significance of the top-k overlap."),
  table(
    [{ h: "Model", w: 1400 }, ...KS.flatMap((k) => [{ h: `Lift@${k}`, w: 950 }, { h: `p@${k}`, w: 1000 }])],
    sorted42.map((m) => [m, ...KS.flatMap((k) => [`${f(acc[m].k[k].lift, 2)}×`, pfmt(acc[m].k[k].p_hypergeom)])]),
    { size: 17 },
  ),
  note(`Every model is significantly better than chance at every cut-off (largest p = ${f(maxP[0], 4)}, ${maxP[1]} at k = ${maxP[2]}).`),
  h2("5.1 Where the agreement comes from"),
  para(`Breaking the hits down by region shows where each model's sensitive LSOAs lie. Across the ten models, the top 50 contains ${regionRangeText}. The city-centre corridor carries the most consistent agreement; the northern region varies most between models.`),
  caption("Reference LSOAs in each model's top 50, by region."),
  table(ovCols(), ovRows(D.overlap50), { size: 17 }),
  caption("Reference LSOAs in each model's top 42, by region."),
  table(ovCols(), ovRows(D.overlap42), { size: 17 }),
  note("“Also in top 20” counts reference LSOAs within the model's top 20. Moving from k = 50 to k = 42 removes hits mainly outside the corridor, so the share of the top k that lies in the expert regions rises."),
  image("top50_panel.jpg", pw, ph, "Top 50 LSOAs of each of the ten models"),
  caption("Each model's top 50 LSOAs coloured by rank (darker = more sensitive). Black outlines mark the expert-highlighted regions; each title gives the number of the model's top 50 inside them.", "Figure"),
  h2("5.2 XGBoost in detail"),
  caption("XGBoost ATT accuracy metrics against the expert regions (N = 4,000)."),
  table(
    [{ h: "k", w: 700 }, { h: "Hits", w: 800 }, { h: "Accuracy", w: 1100 }, { h: "Lift", w: 900 }, { h: "p", w: 1000 }, { h: "Recall", w: 1000 }, { h: "F1", w: 900 }, { h: "ATT-Acc", w: 1000 }, { h: "Jaccard", w: 1000 }],
    KS.map((k) => { const r = xa.k[k]; return [String(k), String(r.hits), f(r.accuracy), `${f(r.lift, 2)}×`, pfmt(r.p_hypergeom), f(r.recall), f(r.att_f1), f(r.att_acc), f(r.jaccard)]; }),
  ),
  para(`XGBoost (predictive R² ${f(xa.pred_r2)}, ATT-AUC ${f(xa.auroc)}) places ${xa.k[20].hits} of its top 20 in the expert regions (${regionPhrase(XF.by_region[20])}) and ${xa.k[42].hits} of its top 42 (${regionPhrase(XF.by_region[42])}). Its top-20 LSOAs outside the regions are ${XF.top20_misses.map(([r, c]) => `${c} (rank ${r})`).join(" and ")}; E01012805 lies just outside the Southcoates strip, with under 10% of its area inside. Several reference LSOAs sit near the bottom of its ranking (ranks ${XF.lowest_ref_ranks[0]}–${XF.lowest_ref_ranks[XF.lowest_ref_ranks.length - 1]}), which is why the threshold-free ATT-AUC is only moderate even though the top of the ranking agrees well.`),
  image("xgb_top50.jpg", xw, xh, "XGBoost top 50 LSOAs"),
  caption("XGBoost top 50 LSOAs by signed area-data Geo-SMILE coefficient, with rank labels. Black outlines mark the expert-highlighted regions.", "Figure"),
];

// ---------- 6. Faithfulness ----------
const fr = (set, m, lab) => { const r = faithRow(set, m); return [set === "all models" ? "All models" : "Excl. DL, CNN", lab, String(r.M), sgn(r.pearson_r), f(r.pearson_p, 3), f(r.pearson_p_one_sided, 3), sgn(r.spearman_rho), f(r.spearman_p, 3)]; };
const measures = [["att_auroc", "ATT-AUC (primary)"], ...KS.map((k) => [`hits@${k}`, `Hits@${k}`])];
const [fw, fh] = imgSize("faithfulness.jpg", 600);
const F = X.faithfulness;
const mlabel = (m) => (m === "att_auroc" ? "ATT-AUC" : m.replace("hits@", "top-") + " overlap");
const sigList = (set) => measures.map(([m]) => [m, faithRow(set, m)]).filter(([, r]) => sig(r));
const allSig = sigList("all models"), exclSig = sigList("excl. dl, cnn");
const closest = measures.map(([m]) => [m, faithRow("all models", m)]).sort((a, b) => a[1].pearson_p - b[1].pearson_p)[0];
const faithResult = `Across all ten models, ATT faithfulness is not demonstrated: the primary ATT-AUC correlation is ${rp(fAll)}` +
  (allSig.length ? `, although ${allSig.map(([m, r]) => `${mlabel(m)} (${rp(r)})`).join(" and ")} reaches significance.` :
    `, and none of the correlations reaches significance (the strongest is ${mlabel(closest[0])}: ${rp(closest[1])}, one-sided p = ${f(closest[1].pearson_p_one_sided, 3)}).`) +
  " Against the earlier expert top-20 list the same area-data Geo branch gave r = −0.00 for ATT-AUC and r = −0.48 for top-20 overlap." +
  " DL (R² 0.666) and CNN (R² 0.635) predict worst yet match the other models on attribution quality, which flattens the correlation." +
  ` Without them, ATT-AUC gives ${rp(fExcl)}${sig(fExcl) ? "" : ", which is not significant"}` +
  (exclSig.length ? `; ${exclSig.map(([m, r]) => `${mlabel(m)} (${rp(r)})`).join(" and ")} ${exclSig.length > 1 ? "correlate" : "correlates"} significantly with R².` : "; no measure correlates significantly with R².");
const faithfulness = [
  h1("6. ATT faithfulness"),
  para("ATT faithfulness follows the performance–attribution correspondence defined for the Hull study, adapted from KG-SMILE: the Pearson correlation, across the candidate models, between predictive competence (held-out predictive R²) and attribution quality. A strong positive correlation means that better predictors also produce explanations that agree more with the expert. ATT-AUC against the expert regions is the primary attribution-quality measure and hits@k the secondary one. Because the Geo branch explains the 30-feature model, each model's Geo score is paired with that model's own predictive R²."),
  para("The two-sided p of the Pearson correlation is the primary test; the one-sided p (H₁: r > 0) and Spearman's ρ with its p are reported beside it. A sensitivity analysis excludes the two Keras models (DL and CNN), whose training is not reproducible under a fixed seed and whose predictive R² values are therefore unstable."),
  caption("ATT faithfulness: correlation between predictive R² and Geo attribution quality against the expert regions."),
  table(
    [{ h: "Model set", w: 1350 }, { h: "Attribution measure", w: 1800, align: AlignmentType.LEFT }, { h: "M", w: 500 }, { h: "Pearson r", w: 1000 }, { h: "p (two-sided)", w: 1100 }, { h: "p (one-sided)", w: 1100 }, { h: "Spearman ρ", w: 1050 }, { h: "p (Spearman)", w: 1100 }],
    [...measures.map(([m, l]) => fr("all models", m, l)), ...measures.map(([m, l]) => fr("excl. dl, cnn", m, l))],
    { size: 17 },
  ),
  para([new TextRun({ text: "Result. ", bold: true }), new TextRun(faithResult)]),
  para("The ten-model result should be reported as the primary finding and the eight-model result only as a sensitivity analysis: the exclusion is motivated, but it was examined after seeing the data, and the remaining eight models span a narrow R² range (0.853–0.899). The eight-model ATT-AUC result also changed noticeably between earlier versions of the expert regions, so it is sensitive to the exact boundary."),
  image("faithfulness.jpg", fw, fh, "Scatter plots of predictive R2 against attribution quality"),
  caption("Predictive R² against ATT-AUC, hits@20 and hits@42 for all ten models, with Pearson r and Spearman ρ in each panel title.", "Figure"),
  h2("6.1 Intervention impact (XGBoost, supplementary)"),
  para("ATT faithfulness is a cross-model statistic and has no value for a single model. For the final XGBoost explanation, an intervention analysis tests a related but different property: whether masking the highest-ranked LSOAs changes the model's predictions more than masking random LSOAs. It uses the model's own ranking and does not involve the expert regions."),
  caption("Intervention impact for the XGBoost area-data Geo explanation (N = 4,000, k = 20)."),
  table(
    [{ h: "Measure", w: 4500 }, { h: "Value", w: 2500 }],
    [
      ["Top-20 response (W₁)", f(F.top_response, 2)],
      ["Bottom-20 response (W₁)", f(F.bottom_response, 2)],
      ["Random-set mean (W₁)", f(F.random_mean, 2)],
      ["Random-set 95% interval", `${f(F.random_ci95_low, 2)} – ${f(F.random_ci95_high, 2)}`],
      ["Top − random", f(F.top_minus_random, 2)],
      ["Top − bottom", f(F.top_minus_bottom, 2)],
      ["Empirical p (top vs random)", f(F.empirical_p_top_vs_random, 4)],
      ["Random sets", String(F.n_random)],
    ],
  ),
  note("The top-20 response lies far above the random-set 95% interval and p = 0.001 is the smallest value attainable with 1,000 random sets: the Geo ranking reflects the model's real dependence on each location."),
];

// ---------- 7. Stability ----------
const S = X.stability;
const dirs = ["north", "east", "south", "west"];
const stability = [
  h1("7. ATT stability"),
  para("ATT stability tests whether the explanation changes when the spatial context is slightly extended. For each compass direction, the nearest LSOA outside the Hull boundary is added to the panel as an extra player, the Geo pool is rebuilt, and the new ranking of the 168 Hull LSOAs is compared with the base ranking: the Jaccard index of the two top-20 sets, the Spearman correlation of the rankings, and where the added LSOA itself ranks. Stability was run for XGBoost at N = 4,000."),
  caption("XGBoost ATT stability after adding one neighbouring LSOA in each direction."),
  table(
    [{ h: "Direction", w: 1200 }, { h: "Added LSOA", w: 1500, align: AlignmentType.CENTER }, { h: "Top-20 Jaccard", w: 1300 }, { h: "Top-20 retained", w: 1300 }, { h: "Rank Spearman", w: 1400 }, { h: "Rank of added LSOA", w: 1400 }, { h: "Added coefficient", w: 1400 }],
    [
      ...dirs.map((d) => { const r = S.directions[d]; return [d[0].toUpperCase() + d.slice(1), r.lsoa21cd, f(r.jaccard, 2), `${r.top20_retained}/20`, f(r.spearman, 4), `${r.added_rank} of 169`, f(r.added_coefficient, 3)]; }),
      [{ t: "Mean", bold: true }, "–", { t: f(S.jaccard_mean, 2), bold: true }, "–", { t: f(S.spearman_mean, 4), bold: true }, "–", "–"],
    ],
    { size: 17 },
  ),
  note("The top 20 is identical in all four directions, rank correlations exceed 0.9997, and no added LSOA enters the top 20 (their coefficients are small and negative): the explanation is robust to extending the study area."),
  caption("XGBoost ATT stability by number of perturbations N."),
  table(
    [{ h: "N", w: 1500 }, { h: "Top-20 Jaccard (mean)", w: 2600 }, { h: "Rank Spearman (mean)", w: 2600 }],
    nsw.map((r) => [r.N.toLocaleString("en-GB"), f(r.stability_jaccard, 2), f(r.stability_spearman, 4)]),
  ),
];

// ---------- 8. Consistency ----------
const C = X.consistency;
const consistency = [
  h1("8. ATT consistency"),
  para("ATT consistency evaluates whether Geo-SMILE produces reproducible attribution coefficients when the same fitted model is explained repeatedly. The model, data, surrogate, perturbation budget (N = 4,000) and all other settings were held fixed, and only the random seed that generates the perturbation sample was changed (seeds 100–104, R = 5 runs). Variation between runs therefore reflects the stochastic variation of the explanation procedure itself."),
  para("Following Dehghani, consistency is quantified by the run-to-run variance and standard deviation of each attribution coefficient. For LSOA j with coefficient βⱼ⁽ʳ⁾ in run r, the variance is sⱼ² = Σᵣ (βⱼ⁽ʳ⁾ − β̄ⱼ)² / (R − 1) and the SD is sⱼ = √sⱼ². Lower variance and SD mean greater consistency. Consistency was run for XGBoost at N = 4,000."),
  caption("XGBoost ATT consistency: coefficient variance and SD across five seeded runs (N = 4,000)."),
  table(
    [{ h: "Branch", w: 1300 }, { h: "Mean variance", w: 1500 }, { h: "Median variance", w: 1500 }, { h: "Mean SD", w: 1500 }, { h: "Median SD", w: 1500 }, { h: "Maximum SD", w: 1500 }],
    [["Geo", f(C.mean_variance, 5), f(C.median_variance, 5), f(C.mean_sd, 5), f(C.median_sd, 5), f(C.max_sd, 5)]],
  ),
  para(`Most Geo coefficients vary little between runs: mean SD ${f(C.mean_sd, 4)} and median SD ${f(C.median_sd, 4)}. The largest SD, ${f(C.max_sd, 4)}, occurs for ${C.max_sd_lsoa}, which also has by far the largest mean coefficient (${f(C.max_sd_lsoa_mean_coefficient, 2)}); relative to that coefficient the SD is ${f(C.max_sd_pct_of_mean, 2)}%, so the dominant attribution is highly reproducible.`),
  caption("Coefficient SD of the ten LSOAs with the largest mean coefficient over the five runs."),
  table(
    [{ h: "Rank", w: 700 }, { h: "LSOA", w: 1400, align: AlignmentType.CENTER }, { h: "Mean coefficient", w: 1500 }, { h: "Variance", w: 1300 }, { h: "SD", w: 1200 }, { h: "SD / |mean|", w: 1300 }, { h: "Expert region", w: 1200, align: AlignmentType.CENTER }],
    D.cons_top10.map((r) => [String(r.rank_by_mean), r.lsoa21cd, f(r.mean_coefficient, 3), f(r.variance, 5), f(r.sd, 4), `${f(r.sd_pct_of_abs_mean, 1)}%`, r.expert_region ? "Yes" : "–"]),
    { size: 17 },
  ),
  caption("Coefficient variance and SD by group of LSOAs."),
  table(
    [{ h: "Group", w: 2600, align: AlignmentType.LEFT }, { h: "LSOAs", w: 800 }, { h: "Mean variance", w: 1300 }, { h: "Mean SD", w: 1100 }, { h: "Median SD", w: 1100 }, { h: "Max SD", w: 1100 }, { h: "Median SD / |mean|", w: 1300 }],
    D.cons_groups.map((g) => [g.group, String(g.n), f(g.mean_var, 4), f(g.mean_sd, 4), f(g.median_sd, 4), f(g.max_sd, 4), `${f(g.med_rel, 1)}%`]),
    { size: 17 },
  ),
  note("The absolute SD is about 0.1 for every group, inside and outside the expert regions. The highly ranked LSOAs have large coefficients, so their relative SD is small (median 3.6% in the top 20); many lower-ranked LSOAs have coefficients close to zero, so the same absolute noise is large relative to them."),
  caption("XGBoost ATT consistency by number of perturbations N."),
  table(
    [{ h: "N", w: 1500 }, { h: "Mean variance", w: 2500 }, { h: "Mean SD", w: 2500 }],
    nsw.map((r) => [r.N.toLocaleString("en-GB"), f(r.consistency_mean_variance, 5), f(r.consistency_mean_sd, 4)]),
  ),
  note("Coefficient variability falls steadily as N grows: mean SD 0.226 at N = 1,000, 0.110 at N = 4,000 and 0.070 at N = 10,000."),
  h2("8.1 Supplementary consistency diagnostics"),
  para("Rank- and set-based agreement between runs is reported as a supplementary diagnostic only and does not replace the variance and SD definition. Every pair of the five runs was compared by the Spearman correlation of the rankings, the Pearson correlation of the coefficients, and the Jaccard index of their top-20% sets (the 34 highest-scoring LSOAs)."),
  caption("Supplementary rank-based consistency diagnostics for XGBoost."),
  table(
    [{ h: "Diagnostic", w: 5200 }, { h: "N = 4,000", w: 1500 }],
    [
      ["Mean pairwise rank Spearman", f(C.rank_spearman_mean, 4)],
      ["Mean pairwise coefficient Pearson", f(C.score_pearson_mean, 4)],
      ["Mean pairwise top-20% Jaccard", f(C.top20pct_jaccard_mean, 4)],
    ],
  ),
  note("Rank agreement is lower than coefficient agreement because many mid-ranked LSOAs have near-identical coefficients close to zero, whose order swaps between runs. The rank Spearman rises from 0.877 at N = 1,000 to 0.983 at N = 10,000."),
];

// ---------- 9–10 ----------
const closing = [
  h1("9. Limitations"),
  bullet([["Tracing uncertainty. ", { bold: true }], "The regions were traced from hand-drawn screenshots with a positional error of about 100–300 m. LSOAs with small overlaps, and the large dock and industrial LSOAs admitted by the 0.1 km² rule in the two southern regions, are borderline and could change a model's hit count by one or two. The faithfulness results in particular moved between successive versions of the regions."]),
  bullet([["Choice of k. ", { bold: true }], `Accuracy depends on the cut-off. Results at k = ${KS.join(", ")} all lead to the same conclusions.`]),
  bullet([["Small number of models. ", { bold: true }], "Faithfulness rests on ten models (eight in the sensitivity analysis), so the correlations have wide uncertainty."]),
  bullet([["Scope of stability and consistency. ", { bold: true }], "Both were run for XGBoost only; the other nine models were explained with one pool each (seed 42)."]),
  bullet([["Keras reproducibility. ", { bold: true }], "The predictive R² of DL and CNN varies between runs under a fixed seed, which affects the faithfulness correlation directly."]),
  bullet([["Sensitivity, not causation. ", { bold: true }], "Coefficients measure how much the fitted model depends on each location's data, not the causal effect of places on crime."]),
  h1("10. Reproduction"),
  numbered("hull_geo_area_data_models.py — fits the ten models and builds each area-data Geo pool (N = 4,000); fidelity and surrogate menu."),
  numbered("hull_geo_area_data_eval.py — XGBoost N sweep: fidelity, intervention impact, stability and consistency."),
  numbered("geo_consistency_per_lsoa.py — per-LSOA coefficient variance and SD over the five consistency seeds."),
  numbered("map_area_data_top50.py — signed rankings and per-model top-50 maps."),
  numbered("select_highlighted_region_lsoas.py — traced expert regions, inclusion and exclusion rules, reference list and region overlap tables."),
  numbered("eval_expert_region_accuracy.py — accuracy@k, lift, hypergeometric p, ATT-AUC and ATT faithfulness."),
  para("Outputs are in outputs/geo_area_data_models/ (top50_maps/, expert_region_accuracy/) and outputs/geo_area_data_eval/."),
];

// ---------- Appendix ----------
const refSorted = [...D.ref].sort((a, b) => a.region.localeCompare(b.region) || b.overlap_fraction - a.overlap_fraction);
const appendix = [
  h1("Appendix A. Expert reference LSOAs"),
  para(`The ${RM.n} LSOAs of the expert reference, with the share of each LSOA's area inside its region, the number of models that rank it in their top 50, and whether it is in the earlier expert top-20 list.`),
  caption("Expert reference LSOAs."),
  table(
    [{ h: "Region", w: 3300, align: AlignmentType.LEFT }, { h: "LSOA", w: 1150, align: AlignmentType.CENTER }, { h: "Name", w: 1950, align: AlignmentType.LEFT }, { h: "Area inside", w: 850 }, { h: "Models (top 50)", w: 900 }, { h: "Earlier top 20", w: 850, align: AlignmentType.CENTER }],
    refSorted.map((r) => [r.region.replace(/-/g, "–"), r.lsoa21cd, r.lsoa21nm, `${Math.round(r.overlap_fraction * 100)}%`, String(r.models_with_lsoa_in_top50), r.expert_top20 ? "Yes" : "–"]),
    { size: 16 },
  ),
];

// ---------- document ----------
const footer = new Footer({
  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "Hull GeoSMILE · area-data Geo branch evaluation · page ", size: 16, color: MUTED }),
    new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED }),
  ] })],
});
const portrait = { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } };
const landscape = { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } };

const doc = new Document({
  creator: "Hull GeoSMILE study",
  title: "Evaluation of the Area-Data Geo Branch",
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: INK }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: ACCENT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  features: { updateFields: true },
  sections: [
    { properties: portrait, footers: { default: footer }, children: [...titleBlock, ...summary, ...design, ...reference, ...fidelity] },
    { properties: landscape, footers: { default: footer }, children: menu },
    { properties: portrait, footers: { default: footer }, children: [...fidN, ...accuracy, ...faithfulness, ...stability, ...consistency, ...closing, ...appendix] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, process.argv[2] || "Hull_Geo_Area_Data_Evaluation.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out);
});
