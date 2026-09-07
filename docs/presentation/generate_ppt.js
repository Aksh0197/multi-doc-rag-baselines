const PptxGenJS = require("pptxgenjs");
const pptx = new PptxGenJS();

// ── Global theme ──────────────────────────────────────────────
pptx.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in

const C = {
  bg:     "080818",
  bg2:    "0F0F2A",
  blue:   "4FC3F7",
  purple: "CE93D8",
  green:  "69F0AE",
  orange: "FFAB40",
  red:    "EF5350",
  gold:   "FFD54F",
  teal:   "4DD0E1",
  white:  "E8E8F4",
  muted:  "7070A0",
  card:   "0F0F28",
  border: "252545",
  darkblue: "0D1B3E",
};

// Shared slide background
const SLIDE_BG = { fill: { color: C.bg } };

// ─────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────
function addBg(slide) {
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: "100%", h: "100%", fill: { color: C.bg }, line: { color: C.bg } });
}

function slideHeader(slide, title, subtitle) {
  // accent bar
  slide.addShape(pptx.ShapeType.rect, { x: 0.4, y: 0.22, w: 12.53, h: 0.03, fill: { color: C.border }, line: { color: C.border } });
  slide.addText(title, {
    x: 0.4, y: 0.1, w: 12.53, h: 0.45,
    fontSize: 22, bold: true, color: C.blue,
    fontFace: "Segoe UI", valign: "middle",
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.4, y: 0.52, w: 12.53, h: 0.28,
      fontSize: 11, color: C.muted, fontFace: "Segoe UI", italic: true,
    });
  }
}

function card(slide, x, y, w, h, opts = {}) {
  const lineObj = { color: opts.borderColor || C.border, width: 1 };
  if (opts.borderTransparency != null) lineObj.transparency = opts.borderTransparency;
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.1,
    fill: { color: opts.fillColor || C.card },
    line: lineObj,
  });
  if (opts.topBarColor) {
    slide.addShape(pptx.ShapeType.rect, { x, y, w, h: 0.06, fill: { color: opts.topBarColor }, line: { color: opts.topBarColor } });
  }
  if (opts.leftBarColor) {
    slide.addShape(pptx.ShapeType.rect, { x, y, w: 0.06, h, fill: { color: opts.leftBarColor }, line: { color: opts.leftBarColor } });
  }
}

function pill(slide, x, y, w, h, text, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.18,
    fill: { color: color, transparency: 80 },
    line: { color: color, width: 1 },
  });
  slide.addText(text, { x: x + 0.05, y, w: w - 0.1, h, fontSize: 9, color, fontFace: "Segoe UI", bold: true, align: "center", valign: "middle" });
}

function barH(slide, x, y, w, h, fillPct, fillColor, label, valText) {
  // track
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: "1A1A35" }, line: { color: C.border } });
  // fill
  const fw = w * fillPct;
  if (fw > 0.05) {
    slide.addShape(pptx.ShapeType.rect, { x, y, w: fw, h, fill: { color: fillColor }, line: { color: fillColor } });
  }
  // label left
  slide.addText(label, { x: x - 1.5, y: y - 0.01, w: 1.45, h: h + 0.02, fontSize: 9, color: C.white, fontFace: "Segoe UI", align: "right", valign: "middle" });
  // val right
  slide.addText(valText, { x: x + w + 0.06, y: y - 0.01, w: 0.5, h: h + 0.02, fontSize: 9, color: C.muted, fontFace: "Segoe UI", valign: "middle" });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 1 — TITLE
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);

  // gradient overlay shape (simulate)
  sl.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: 6, h: 7.5,
    fill: { color: "1a0a3a", transparency: 60 }, line: { color: "1a0a3a", transparency: 60 },
  });

  // label
  sl.addText("RESEARCH PROJECT", {
    x: 1, y: 0.9, w: 11.33, h: 0.3,
    fontSize: 10, color: C.muted, fontFace: "Segoe UI",
    charSpacing: 4, align: "center",
  });

  // title lines
  sl.addText("Multi-Document\nRAG Evaluation\nFramework", {
    x: 1, y: 1.2, w: 11.33, h: 2.6,
    fontSize: 44, bold: true, color: C.white, fontFace: "Segoe UI",
    align: "center", lineSpacingMultiple: 1.15,
  });

  // pill row
  const papers = [
    { text: "RAG", color: C.blue },
    { text: "RankRAG", color: C.purple },
    { text: "Self-RAG", color: C.green },
    { text: "MADAM-RAG", color: C.orange },
    { text: "RAGChecker", color: C.teal },
  ];
  const pw = 1.6, gap = 0.25, totalW = papers.length * pw + (papers.length - 1) * gap;
  let px = (13.33 - totalW) / 2;
  papers.forEach(p => {
    pill(sl, px, 3.95, pw, 0.38, p.text, p.color);
    px += pw + gap;
  });

  sl.addText("End-to-end benchmark pipeline grounded in 5 research papers", {
    x: 1, y: 4.5, w: 11.33, h: 0.3,
    fontSize: 12, color: C.muted, fontFace: "Segoe UI", italic: true, align: "center",
  });
  sl.addText("HotpotQA  ·  RAMDocs  ·  3 LLM Backends  ·  Faithfulness Analysis", {
    x: 1, y: 4.85, w: 11.33, h: 0.3,
    fontSize: 11, color: C.muted, fontFace: "Segoe UI", align: "center",
  });

  // bottom stats strip
  sl.addShape(pptx.ShapeType.rect, { x: 0, y: 6.7, w: 13.33, h: 0.8, fill: { color: "0D0D25" }, line: { color: C.border } });
  const stats = ["5 Research Papers", "9 Pipeline Steps", "400+ Test Queries", "3 LLM Backends", "EM 0.800 ★"];
  const sw = 13.33 / stats.length;
  stats.forEach((s, i) => {
    sl.addText(s, {
      x: i * sw, y: 6.7, w: sw, h: 0.8,
      fontSize: 11, color: i === stats.length - 1 ? C.gold : C.muted,
      fontFace: "Segoe UI", bold: i === stats.length - 1, align: "center", valign: "middle",
    });
    if (i < stats.length - 1) {
      sl.addShape(pptx.ShapeType.rect, { x: (i + 1) * sw - 0.01, y: 6.78, w: 0.02, h: 0.64, fill: { color: C.border }, line: { color: C.border } });
    }
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 2 — THE PROBLEM
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "The Problem", "Multi-document question answering is fundamentally harder than single-passage retrieval");

  // example box
  sl.addShape(pptx.ShapeType.roundRect, {
    x: 0.4, y: 0.88, w: 12.53, h: 1.15, rectRadius: 0.1,
    fill: { color: "0A1B30" }, line: { color: C.blue, width: 1 },
  });
  sl.addShape(pptx.ShapeType.rect, { x: 0.4, y: 0.88, w: 0.06, h: 1.15, fill: { color: C.blue }, line: { color: C.blue } });
  sl.addText("EXAMPLE MULTI-HOP QUESTION", { x: 0.6, y: 0.92, w: 4, h: 0.22, fontSize: 8, color: C.blue, fontFace: "Segoe UI", bold: true, charSpacing: 2 });
  sl.addText('"What is the nationality of the director of Spectre?"', { x: 0.6, y: 1.12, w: 12.1, h: 0.3, fontSize: 13, color: C.gold, fontFace: "Segoe UI", bold: true });

  // doc chain
  const docItems = [
    { label: "Doc 1: Spectre (film)", sub: "director = Sam Mendes", color: C.blue },
    { label: "+", sub: "", color: C.muted, isOp: true },
    { label: "Doc 2: Sam Mendes", sub: "nationality = British", color: C.purple },
    { label: "→", sub: "", color: C.muted, isOp: true },
    { label: "Answer: British", sub: "", color: C.green, isAnswer: true },
  ];
  let dx = 0.6;
  docItems.forEach(d => {
    if (d.isOp) {
      sl.addText(d.label, { x: dx, y: 1.47, w: 0.3, h: 0.4, fontSize: 14, color: d.color, fontFace: "Segoe UI", align: "center", valign: "middle" });
      dx += 0.35;
    } else if (d.isAnswer) {
      sl.addShape(pptx.ShapeType.roundRect, { x: dx, y: 1.45, w: 1.6, h: 0.42, rectRadius: 0.08, fill: { color: "003322" }, line: { color: C.green } });
      sl.addText(d.label, { x: dx + 0.05, y: 1.45, w: 1.5, h: 0.42, fontSize: 11, color: C.green, fontFace: "Segoe UI", bold: true, align: "center", valign: "middle" });
      dx += 1.65;
    } else {
      sl.addShape(pptx.ShapeType.roundRect, { x: dx, y: 1.45, w: 2.4, h: 0.42, rectRadius: 0.08, fill: { color: "0A1428" }, line: { color: d.color } });
      sl.addText(d.label, { x: dx + 0.1, y: 1.45, w: 2.2, h: 0.22, fontSize: 10, color: d.color, fontFace: "Segoe UI", bold: true, valign: "middle" });
      sl.addText(d.sub, { x: dx + 0.1, y: 1.65, w: 2.2, h: 0.2, fontSize: 9, color: C.muted, fontFace: "Segoe UI" });
      dx += 2.5;
    }
  });

  // 3 challenge cards
  const challenges = [
    { icon: "🔗", title: "Multi-hop Reasoning", body: "Answer requires chaining evidence across 2+ documents — no single doc contains the full answer", color: C.blue },
    { icon: "🎭", title: "Misinformation in Corpus", body: "Real corpora contain adversarial documents designed to mislead — model must actively suppress them", color: C.red },
    { icon: "🌀", title: "Hallucination Risk", body: "LLMs generate plausible but unsupported text — hard to distinguish from grounded answers", color: C.orange },
  ];
  const cw = 4.0, cgap = 0.26, cy = 2.18;
  challenges.forEach((c, i) => {
    const cx = 0.4 + i * (cw + cgap);
    card(sl, cx, cy, cw, 2.38, { topBarColor: c.color });
    sl.addText(c.icon, { x: cx + 0.15, y: cy + 0.22, w: 0.6, h: 0.55, fontSize: 22 });
    sl.addText(c.title, { x: cx + 0.15, y: cy + 0.82, w: cw - 0.3, h: 0.36, fontSize: 12, bold: true, color: C.white, fontFace: "Segoe UI" });
    sl.addText(c.body, { x: cx + 0.15, y: cy + 1.18, w: cw - 0.3, h: 1.0, fontSize: 10, color: C.muted, fontFace: "Segoe UI", wrap: true });
  });

  // bottom note
  card(sl, 0.4, 4.72, 12.53, 0.48, { fillColor: "0A0A20" });
  sl.addText("Existing baselines evaluate retrieval or generation — rarely both together with end-to-end failure attribution", {
    x: 0.6, y: 4.76, w: 12.2, h: 0.4, fontSize: 11, color: C.muted, fontFace: "Segoe UI", italic: true, valign: "middle",
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 3 — WHAT WE BUILT
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "What We Built", "A unified end-to-end benchmark — retrieval through faithfulness, all in one pipeline");

  // 3 pillar cards
  const pillars = [
    { icon: "📊", title: "Benchmark Suite", color: C.blue, items: ["2 datasets (HotpotQA, RAMDocs)", "5 retrieval methods", "3 LLM backends", "400+ test queries"] },
    { icon: "⚙️", title: "Production Pipeline", color: C.purple, items: ["Retrieval → Reranking", "Multi-agent debate", "Rule-based + LLM generation", "Faithfulness checking"] },
    { icon: "🔍", title: "Failure Analysis", color: C.green, items: ["Retrieval vs generation failures", "Hallucination detection", "Misinformation suppression", "Joint F1 tracking"] },
  ];
  const pw2 = 4.0, pgap = 0.26;
  pillars.forEach((p, i) => {
    const px = 0.4 + i * (pw2 + pgap);
    card(sl, px, 0.88, pw2, 2.55, { topBarColor: p.color });
    sl.addText(p.icon + "  " + p.title, { x: px + 0.18, y: 1.05, w: pw2 - 0.3, h: 0.38, fontSize: 13, bold: true, color: C.white, fontFace: "Segoe UI" });
    p.items.forEach((item, j) => {
      sl.addText("▸  " + item, { x: px + 0.18, y: 1.5 + j * 0.34, w: pw2 - 0.3, h: 0.32, fontSize: 10, color: C.muted, fontFace: "Segoe UI" });
    });
  });

  // datasets
  sl.addText("Datasets", { x: 0.4, y: 3.6, w: 3, h: 0.32, fontSize: 12, bold: true, color: C.gold, fontFace: "Segoe UI" });
  const datasets = [
    { name: "HotpotQA", tag: "multi-hop reasoning", desc: "10 candidate docs/query · 200 test examples · bridge-entity questions", color: C.blue },
    { name: "RAMDocs", tag: "adversarial misinformation", desc: "Gold / Misinfo / Noise docs · 200 test examples · ambiguous entities", color: C.orange },
  ];
  datasets.forEach((d, i) => {
    const dy = 3.95 + i * 0.9;
    card(sl, 0.4, dy, 5.8, 0.78, { leftBarColor: d.color });
    sl.addText(d.name, { x: 0.7, y: dy + 0.06, w: 2.5, h: 0.3, fontSize: 12, bold: true, color: C.white, fontFace: "Segoe UI" });
    pill(sl, 3.2, dy + 0.08, 2.7, 0.26, d.tag, d.color);
    sl.addText(d.desc, { x: 0.7, y: dy + 0.38, w: 5.3, h: 0.32, fontSize: 9, color: C.muted, fontFace: "Segoe UI" });
  });

  // metrics cloud
  sl.addText("Metrics Tracked", { x: 6.7, y: 3.6, w: 6, h: 0.32, fontSize: 12, bold: true, color: C.gold, fontFace: "Segoe UI" });
  const metrics = ["Recall@K", "MRR", "Multi-doc Hit Rate", "Answer EM", "Answer F1", "Joint F1", "Faithfulness Score", "Hallucination Rate", "Failure Mode", "Misinfo Suppressed"];
  const mc = [C.blue, C.purple, C.blue, C.green, C.green, C.teal, C.orange, C.red, C.orange, C.purple];
  const positions = [
    [6.7, 4.0], [7.9, 4.0], [9.0, 4.0], [10.85, 4.0],
    [6.7, 4.48], [7.85, 4.48], [9.2, 4.48], [10.55, 4.48],
    [6.7, 4.96], [8.05, 4.96],
  ];
  metrics.forEach((m, i) => {
    const [mx, my] = positions[i];
    const mw = m.length * 0.092 + 0.3;
    pill(sl, mx, my, mw, 0.32, m, mc[i]);
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 4 — RESEARCH FOUNDATION
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "Research Foundation", "Each pipeline component is directly grounded in a peer-reviewed paper");

  const papers = [
    { name: "RAG", author: "Lewis et al., 2020", contrib: "Retrieval-Augmented Generation — the foundational pattern", tag: "Architecture", color: C.blue, component: "Base scaffold + schema" },
    { name: "RankRAG", author: "Yu et al., 2024", contrib: "LLM as pointwise reranker — score each document independently", tag: "Reranking", color: C.purple, component: "LLM reranker module" },
    { name: "Self-RAG", author: "Asai et al., 2023", contrib: "Iterative retrieval with reflection tokens for adaptive queries", tag: "Retrieval", color: C.green, component: "Iterative Hybrid retrieval" },
    { name: "MADAM-RAG", author: "Yoran et al., 2024", contrib: "Multi-agent debate — one agent per doc, arbitrator decides", tag: "Debate", color: C.orange, component: "Debate + misinfo suppression" },
    { name: "RAGChecker", author: "Ru et al., 2024", contrib: "Claim-level faithfulness checking + failure mode taxonomy", tag: "Evaluation", color: C.teal, component: "Faithfulness + failure modes" },
  ];

  const cw = 2.35, cgap = 0.13;
  papers.forEach((p, i) => {
    const cx = 0.4 + i * (cw + cgap);
    card(sl, cx, 0.88, cw, 3.35, { topBarColor: p.color });

    // Paper name
    sl.addText(p.name, { x: cx + 0.12, y: 1.08, w: cw - 0.24, h: 0.38, fontSize: 15, bold: true, color: p.color, fontFace: "Segoe UI" });
    sl.addText(p.author, { x: cx + 0.12, y: 1.44, w: cw - 0.24, h: 0.26, fontSize: 9, color: C.muted, fontFace: "Segoe UI", italic: true });
    // divider
    sl.addShape(pptx.ShapeType.rect, { x: cx + 0.12, y: 1.72, w: cw - 0.24, h: 0.02, fill: { color: C.border }, line: { color: C.border } });
    // contribution
    sl.addText(p.contrib, { x: cx + 0.12, y: 1.78, w: cw - 0.24, h: 1.0, fontSize: 9.5, color: C.white, fontFace: "Segoe UI", wrap: true });

    // tag pill
    pill(sl, cx + 0.12, 2.88, cw - 0.24, 0.26, p.tag, p.color);

    // component label
    sl.addText(p.component, { x: cx + 0.12, y: 3.2, w: cw - 0.24, h: 0.9, fontSize: 8.5, color: C.muted, fontFace: "Segoe UI", wrap: true, italic: true });
  });

  // bottom integration note
  card(sl, 0.4, 4.38, 12.53, 0.55, { fillColor: "080820", leftBarColor: C.gold });
  sl.addText("Novel contribution: ", { x: 0.65, y: 4.48, w: 1.8, h: 0.35, fontSize: 11, bold: true, color: C.gold, fontFace: "Segoe UI", valign: "middle" });
  sl.addText("All 5 papers integrated into a single runnable pipeline with shared schema, unified metrics, and graceful LLM fallbacks", {
    x: 2.4, y: 4.48, w: 10.3, h: 0.35, fontSize: 11, color: C.white, fontFace: "Segoe UI", valign: "middle",
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 5 — PIPELINE ARCHITECTURE
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "System Architecture", "9-step pipeline from raw question to verified answer");

  // Pipeline phases with boxes
  const phases = [
    { label: "INPUT",       name: "Question\n+ Corpus",       color: C.blue,   x: 0.3 },
    { label: "RETRIEVAL",   name: "Retrieval\nBM25/Hybrid/Iter", color: "818CF8", x: 2.05 },
    { label: "RERANKING",   name: "LLM\nReranker",            color: C.purple, x: 4.35 },
    { label: "DEBATE",      name: "Multi-Agent\nDebate",       color: C.orange, x: 6.35 },
    { label: "GENERATION",  name: "Answer\nExtraction",        color: C.green,  x: 8.65 },
    { label: "FAITHFULNESS",name: "Faithfulness\nCheck",       color: C.teal,   x: 10.7 },
  ];

  const bw = 1.6, bh = 1.0, by = 0.88;
  phases.forEach((ph, i) => {
    // phase label
    sl.addText(ph.label, { x: ph.x, y: by - 0.24, w: bw, h: 0.22, fontSize: 7, color: ph.color, fontFace: "Segoe UI", bold: true, align: "center", charSpacing: 1 });
    // main box
    sl.addShape(pptx.ShapeType.roundRect, {
      x: ph.x, y: by, w: bw, h: bh, rectRadius: 0.1,
      fill: { color: "0D0D25" }, line: { color: ph.color, width: 1.5 },
    });
    sl.addText(ph.name, { x: ph.x + 0.05, y: by + 0.08, w: bw - 0.1, h: bh - 0.16, fontSize: 10, bold: true, color: ph.color, fontFace: "Segoe UI", align: "center", valign: "middle" });

    // arrow (except last)
    if (i < phases.length - 1) {
      const ax = ph.x + bw + 0.04;
      sl.addShape(pptx.ShapeType.rect, { x: ax, y: by + bh / 2 - 0.02, w: 0.15, h: 0.04, fill: { color: "4A4A6A" }, line: { color: "4A4A6A" } });
      // arrowhead triangle approximated
      sl.addShape(pptx.ShapeType.triangle, { x: ax + 0.12, y: by + bh / 2 - 0.1, w: 0.12, h: 0.2, fill: { color: "4A4A6A" }, line: { color: "4A4A6A" } });
    }
  });

  // Detail cards below each phase
  const details = [
    { x: 0.3, items: ["Query input", "Corpus documents", "10 docs/query (HotpotQA)", "Gold/Misinfo/Noise (RAMDocs)"] },
    { x: 2.05, items: ["BM25 sparse scoring", "Hybrid: BM25+Dense+RRF", "Iterative: 2-pass bridge", "Top-K → reranker"] },
    { x: 4.35, items: ["LLM scores each doc 1-10", "Re-orders by relevance", "Fallback: original order", "Circuit breaker on error"] },
    { x: 6.35, items: ["1 agent per document", "Local answer + confidence", "Label weights for misinfo", "Arbitrator picks winner"] },
    { x: 8.65, items: ["Rule-based: regex/match", "LLM: full debate synthesis", "Heuristic fallback", "EM + F1 evaluation"] },
    { x: 10.7, items: ["Is answer grounded?", "Detect hallucinations", "Failure mode: correct /", "  retrieval / generation"] },
  ];

  const detailColors = [C.blue, "818CF8", C.purple, C.orange, C.green, C.teal];
  details.forEach((d, i) => {
    card(sl, d.x, 2.06, bw, 2.1, { borderColor: detailColors[i], borderTransparency: 73 });
    d.items.forEach((item, j) => {
      sl.addText("▸ " + item, { x: d.x + 0.1, y: 2.16 + j * 0.46, w: bw - 0.2, h: 0.44, fontSize: 8.5, color: C.muted, fontFace: "Segoe UI" });
    });
  });

  // QueryTrace bar
  sl.addShape(pptx.ShapeType.roundRect, {
    x: 2.05, y: 4.28, w: 9.25, h: 0.4, rectRadius: 0.06,
    fill: { color: "0A0A1E" }, line: { color: C.border, width: 1, dashType: "dash" },
  });
  sl.addText("QueryTrace — unified schema capturing every intermediate result for reproducibility and audit", {
    x: 2.1, y: 4.32, w: 9.15, h: 0.32, fontSize: 9, color: C.muted, fontFace: "Segoe UI", align: "center", italic: true,
  });

  // Datasets on left
  card(sl, 0.3, 4.82, 1.6, 0.45, { borderColor: C.blue, borderTransparency: 73 });
  sl.addText("HotpotQA", { x: 0.4, y: 4.88, w: 1.4, h: 0.32, fontSize: 9, color: C.blue, fontFace: "Segoe UI", bold: true, align: "center", valign: "middle" });
  card(sl, 0.3, 5.32, 1.6, 0.45, { borderColor: C.orange, borderTransparency: 73 });
  sl.addText("RAMDocs", { x: 0.4, y: 5.38, w: 1.4, h: 0.32, fontSize: 9, color: C.orange, fontFace: "Segoe UI", bold: true, align: "center", valign: "middle" });

  // Final answer box
  card(sl, 10.7, 4.82, 1.93, 0.95, { borderColor: C.gold, topBarColor: C.gold });
  sl.addText("Final Answer", { x: 10.8, y: 5.0, w: 1.73, h: 0.32, fontSize: 11, bold: true, color: C.gold, fontFace: "Segoe UI", align: "center" });
  sl.addText("+ failure mode\n+ evidence", { x: 10.8, y: 5.32, w: 1.73, h: 0.4, fontSize: 8, color: C.muted, fontFace: "Segoe UI", align: "center" });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 6 — RETRIEVAL METHODS
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "Three Retrieval Strategies", "Progressive complexity — each addresses a limitation of the previous");

  const methods = [
    {
      badge: "Baseline", name: "BM25", color: "818CF8",
      desc: "Sparse keyword matching using TF-IDF weights. Fast, interpretable, strong baseline.",
      steps: ["① Tokenize query", "② Score docs by term overlap", "③ Return Top-K by BM25 score"],
      limit: "Misses semantic synonyms & paraphrases",
      metric: "Recall@5: 0.650",
    },
    {
      badge: "Improved", name: "Hybrid\n(BM25 + Dense)", color: C.blue,
      desc: "Fuses sparse BM25 with dense semantic embeddings via Reciprocal Rank Fusion.",
      steps: ["① BM25 score (sparse)", "② Sentence-transformer embeddings", "③ RRF fusion → re-ranked list"],
      limit: "Gains on paraphrases, still single-pass",
      metric: "Recall@5: 0.768",
    },
    {
      badge: "Best ★", name: "Iterative\nHybrid", color: C.green,
      desc: "Two-pass retrieval inspired by Self-RAG: first pass extracts bridge entities for the second query.",
      steps: ["① Pass 1: Hybrid on original Q", "② Extract bridge entities from results", "③ Pass 2: Hybrid on Q + entities", "④ Merge both result sets"],
      limit: "Solves multi-hop retrieval chains",
      metric: "Recall@5: 0.770 ★",
    },
  ];

  const mw = 4.0, mgap = 0.26;
  methods.forEach((m, i) => {
    const mx = 0.4 + i * (mw + mgap);
    card(sl, mx, 0.88, mw, 4.35, { topBarColor: m.color });

    pill(sl, mx + 0.15, 1.05, 1.1, 0.28, m.badge, m.color);
    sl.addText(m.name, { x: mx + 0.15, y: 1.38, w: mw - 0.3, h: 0.6, fontSize: 14, bold: true, color: C.white, fontFace: "Segoe UI" });
    sl.addText(m.desc, { x: mx + 0.15, y: 1.98, w: mw - 0.3, h: 0.7, fontSize: 9.5, color: C.muted, fontFace: "Segoe UI", wrap: true });

    sl.addShape(pptx.ShapeType.rect, { x: mx + 0.15, y: 2.7, w: mw - 0.3, h: 0.02, fill: { color: C.border }, line: { color: C.border } });

    m.steps.forEach((s, j) => {
      sl.addText(s, { x: mx + 0.15, y: 2.76 + j * 0.34, w: mw - 0.3, h: 0.32, fontSize: 9.5, color: C.white, fontFace: "Segoe UI" });
    });

    // bottom metric & limit
    const noteY = mx === 0.4 ? 4.34 : 4.34; // same for all
    sl.addShape(pptx.ShapeType.roundRect, {
      x: mx + 0.1, y: 4.68, w: mw - 0.2, h: 0.44, rectRadius: 0.07,
      fill: { color: i === 2 ? "003320" : "0A0A20" }, line: { color: m.color, transparency: 60 },
    });
    sl.addText(m.metric, { x: mx + 0.2, y: 4.72, w: mw - 0.4, h: 0.35, fontSize: 10, bold: true, color: m.color, fontFace: "Segoe UI", valign: "middle" });
  });

  // Example strip
  card(sl, 0.4, 5.28, 12.53, 0.52, { fillColor: "0A1520", leftBarColor: C.blue });
  sl.addText("Iterative Hybrid example:  ", { x: 0.65, y: 5.35, w: 2.2, h: 0.38, fontSize: 10, bold: true, color: C.blue, fontFace: "Segoe UI", valign: "middle" });
  sl.addText('"Spectre director nationality?" → Pass 1 finds Spectre article → extracts bridge entity "Sam Mendes" → Pass 2 finds Sam Mendes bio → both gold docs retrieved ✓', {
    x: 2.8, y: 5.35, w: 10, h: 0.38, fontSize: 10, color: C.white, fontFace: "Segoe UI", valign: "middle",
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 7 — RERANKING + DEBATE
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "Reranking & Multi-Agent Debate", "RankRAG-style LLM scoring feeds directly into MADAM-RAG debate");

  // LEFT: Reranker
  sl.addText("① LLM Reranker  (RankRAG)", { x: 0.4, y: 0.82, w: 5.9, h: 0.38, fontSize: 13, bold: true, color: C.purple, fontFace: "Segoe UI" });
  card(sl, 0.4, 1.22, 5.9, 2.18);
  sl.addText("For each retrieved document, the LLM is asked:", { x: 0.6, y: 1.36, w: 5.5, h: 0.3, fontSize: 10, color: C.muted, fontFace: "Segoe UI" });
  sl.addShape(pptx.ShapeType.roundRect, {
    x: 0.6, y: 1.68, w: 5.5, h: 0.6, rectRadius: 0.08,
    fill: { color: "060614" }, line: { color: C.border },
  });
  sl.addText('"On a scale 1–10, how relevant is this document to the question?"', {
    x: 0.75, y: 1.72, w: 5.2, h: 0.52, fontSize: 10, color: "AACCBB", fontFace: "Courier New", italic: true, valign: "middle",
  });
  const rItems = ["Assigns relevance score per document", "Re-orders documents before debate", "Deterministic fallback: original BM25 order"];
  rItems.forEach((r, j) => {
    sl.addText("▸ " + r, { x: 0.6, y: 2.36 + j * 0.3, w: 5.5, h: 0.28, fontSize: 10, color: C.white, fontFace: "Segoe UI" });
  });

  card(sl, 0.4, 3.5, 5.9, 0.52, { fillColor: "120820", leftBarColor: C.purple });
  sl.addText("Impact:", { x: 0.65, y: 3.58, w: 0.8, h: 0.35, fontSize: 10, bold: true, color: C.purple, fontFace: "Segoe UI", valign: "middle" });
  sl.addText("EM jumps from  0.595 → 0.74  with BM25 + LLM reranker  (+24%)", {
    x: 1.45, y: 3.58, w: 4.7, h: 0.35, fontSize: 10, color: C.white, fontFace: "Segoe UI", valign: "middle",
  });

  // RIGHT: Debate
  sl.addText("② Multi-Agent Debate  (MADAM-RAG)", { x: 6.9, y: 0.82, w: 6.0, h: 0.38, fontSize: 13, bold: true, color: C.orange, fontFace: "Segoe UI" });
  card(sl, 6.9, 1.22, 6.0, 2.18);

  // Agent boxes
  const agents = [
    { label: "Agent₁", sub: "gold doc", color: C.green },
    { label: "Agent₂", sub: "misinfo", color: C.red },
    { label: "Agent₃", sub: "noise", color: C.muted },
    { label: "Agent₄", sub: "gold doc", color: C.green },
  ];
  const aw = 1.22, agap = 0.12;
  agents.forEach((a, i) => {
    const ax = 7.0 + i * (aw + agap);
    sl.addShape(pptx.ShapeType.roundRect, { x: ax, y: 1.36, w: aw, h: 0.72, rectRadius: 0.08, fill: { color: "0A0A20" }, line: { color: a.color, transparency: 47 } });
    sl.addText(a.label, { x: ax + 0.05, y: 1.4, w: aw - 0.1, h: 0.3, fontSize: 11, bold: true, color: a.color, fontFace: "Segoe UI", align: "center" });
    sl.addText(a.sub, { x: ax + 0.05, y: 1.7, w: aw - 0.1, h: 0.3, fontSize: 8.5, color: a.color, fontFace: "Segoe UI", align: "center" });
  });
  // arrow down
  sl.addText("↓  weighted votes  ↓", { x: 7.0, y: 2.1, w: 5.72, h: 0.3, fontSize: 10, color: C.muted, fontFace: "Segoe UI", italic: true, align: "center" });
  // arbitrator
  sl.addShape(pptx.ShapeType.roundRect, { x: 8.1, y: 2.42, w: 3.5, h: 0.62, rectRadius: 0.1, fill: { color: "1A0E00" }, line: { color: C.orange, width: 1.5 } });
  sl.addText("⚖️  Arbitrator — final answer", { x: 8.1, y: 2.42, w: 3.5, h: 0.62, fontSize: 11, bold: true, color: C.orange, fontFace: "Segoe UI", align: "center", valign: "middle" });
  // weights
  const weights = [
    { label: "gold doc", weight: "× +2.0", color: C.green },
    { label: "misinformation", weight: "× −1.5", color: C.red },
    { label: "noise doc", weight: "× −0.5", color: C.muted },
  ];
  weights.forEach((w2, j) => {
    sl.addShape(pptx.ShapeType.roundRect, { x: 7.0 + j * 1.95, y: 3.12, w: 1.82, h: 0.52, rectRadius: 0.07, fill: { color: "080818" }, line: { color: w2.color, transparency: 60 } });
    sl.addText(w2.label, { x: 7.05 + j * 1.95, y: 3.14, w: 1.72, h: 0.26, fontSize: 9, color: w2.color, fontFace: "Segoe UI", align: "center" });
    sl.addText(w2.weight, { x: 7.05 + j * 1.95, y: 3.38, w: 1.72, h: 0.22, fontSize: 9, bold: true, color: w2.color, fontFace: "Segoe UI", align: "center" });
  });

  card(sl, 6.9, 3.5, 6.0, 0.52, { fillColor: "12100A", leftBarColor: C.orange });
  sl.addText("Misinfo suppressed:", { x: 7.15, y: 3.58, w: 2, h: 0.35, fontSize: 10, bold: true, color: C.orange, fontFace: "Segoe UI", valign: "middle" });
  sl.addText("96.5% on RAMDocs   |   Active suppression, not passive observation", {
    x: 9.1, y: 3.58, w: 3.6, h: 0.35, fontSize: 10, color: C.white, fontFace: "Segoe UI", valign: "middle",
  });

  // divider
  sl.addShape(pptx.ShapeType.rect, { x: 6.55, y: 0.82, w: 0.02, h: 3.3, fill: { color: C.border }, line: { color: C.border } });

  // bottom shared note
  card(sl, 0.4, 4.18, 12.53, 1.38, { topBarColor: C.gold });
  sl.addText("RAMDocs — Misinformation Tracing", { x: 0.6, y: 4.3, w: 5, h: 0.3, fontSize: 11, bold: true, color: C.gold, fontFace: "Segoe UI" });
  sl.addText("Every document carries a label (gold / misinfo / noise) loaded at dataset parse time → passed to each debate agent → used as multiplier weight → arbitrator picks answer with highest weighted score → final flag indicates whether the winning answer matches any known wrong answer", {
    x: 0.6, y: 4.62, w: 12.1, h: 0.85, fontSize: 9.5, color: C.muted, fontFace: "Segoe UI", wrap: true,
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 8 — FAITHFULNESS CHECKING
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "Faithfulness & Failure Analysis", "RAGChecker-style — two paths, always produces a classification");

  // Path A
  sl.addText("Path A — LLM Judge  (when available)", { x: 0.4, y: 0.82, w: 5.9, h: 0.35, fontSize: 13, bold: true, color: C.teal, fontFace: "Segoe UI" });
  card(sl, 0.4, 1.2, 5.9, 2.55);
  sl.addText("LLM receives: question · predicted answer · gold answer · retrieved docs (first 600 chars each)", {
    x: 0.6, y: 1.32, w: 5.5, h: 0.4, fontSize: 9.5, color: C.muted, fontFace: "Segoe UI", wrap: true,
  });
  sl.addText("Returns structured JSON:", { x: 0.6, y: 1.72, w: 5.5, h: 0.25, fontSize: 9.5, color: C.muted, fontFace: "Segoe UI" });
  sl.addShape(pptx.ShapeType.roundRect, { x: 0.6, y: 1.98, w: 5.5, h: 1.0, rectRadius: 0.08, fill: { color: "040410" }, line: { color: C.border } });
  sl.addText(
    '{\n  "faithfulness_score": 7,          // 0–10\n  "hallucinated_claims": ["..."],\n  "failure_mode": "generation_failure"\n}',
    { x: 0.75, y: 2.04, w: 5.2, h: 0.9, fontSize: 9.5, color: "88DDBB", fontFace: "Courier New" }
  );
  sl.addText("Score 0–10 normalised to 0.0–1.0", { x: 0.6, y: 3.02, w: 5.5, h: 0.28, fontSize: 9, color: C.muted, fontFace: "Segoe UI", italic: true });
  sl.addText("Circuit breaker: if LLM returns 'unknown' → fallback to Path B", { x: 0.6, y: 3.28, w: 5.5, h: 0.28, fontSize: 9, color: C.muted, fontFace: "Segoe UI", italic: true });

  // Path B
  sl.addText("Path B — Deterministic Fallback  (no LLM needed)", { x: 6.9, y: 0.82, w: 6.0, h: 0.35, fontSize: 13, bold: true, color: C.gold, fontFace: "Segoe UI" });
  card(sl, 6.9, 1.2, 6.0, 2.55);

  const modes = [
    { cond: "EM = 1", result: "correct", color: C.green, bg: "003322" },
    { cond: "EM = 0, gold docs retrieved", result: "generation_failure", color: C.orange, bg: "1A1000" },
    { cond: "EM = 0, gold docs missing", result: "retrieval_failure", color: C.red, bg: "200000" },
  ];
  sl.addText("Pure logic — no LLM required:", { x: 7.1, y: 1.32, w: 5.6, h: 0.3, fontSize: 10, color: C.muted, fontFace: "Segoe UI" });
  modes.forEach((m, j) => {
    const my = 1.68 + j * 0.76;
    sl.addShape(pptx.ShapeType.roundRect, { x: 7.05, y: my, w: 5.7, h: 0.62, rectRadius: 0.08, fill: { color: m.bg }, line: { color: m.color, transparency: 60 } });
    sl.addText(m.cond, { x: 7.2, y: my + 0.05, w: 2.6, h: 0.52, fontSize: 10, color: C.white, fontFace: "Segoe UI", valign: "middle" });
    sl.addText("→", { x: 9.8, y: my + 0.05, w: 0.4, h: 0.52, fontSize: 12, color: C.muted, fontFace: "Segoe UI", align: "center", valign: "middle" });
    sl.addText(m.result, { x: 10.2, y: my + 0.05, w: 2.4, h: 0.52, fontSize: 10, bold: true, color: m.color, fontFace: "Segoe UI", valign: "middle" });
  });

  // divider
  sl.addShape(pptx.ShapeType.rect, { x: 6.55, y: 0.82, w: 0.02, h: 3.0, fill: { color: C.border }, line: { color: C.border } });

  // 3 big stat cards
  const fmStats = [
    { num: "~90%", label: "of all failures are", sub: "retrieval failures", color: C.red },
    { num: "~10%", label: "of all failures are", sub: "generation failures", color: C.orange },
    { num: "9:1", label: "retrieval : generation", sub: "failure ratio", color: C.green },
  ];
  const fsw = 4.0, fsgap = 0.26;
  fmStats.forEach((f, i) => {
    const fx = 0.4 + i * (fsw + fsgap);
    card(sl, fx, 3.88, fsw, 1.62, { leftBarColor: f.color });
    sl.addText(f.num, { x: fx + 0.18, y: 3.98, w: fsw - 0.3, h: 0.72, fontSize: 36, bold: true, color: f.color, fontFace: "Segoe UI" });
    sl.addText(f.label, { x: fx + 0.18, y: 4.7, w: fsw - 0.3, h: 0.28, fontSize: 10, color: C.muted, fontFace: "Segoe UI" });
    sl.addText(f.sub, { x: fx + 0.18, y: 4.96, w: fsw - 0.3, h: 0.28, fontSize: 10, bold: true, color: f.color, fontFace: "Segoe UI" });
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 9 — EXPERIMENTAL SETUP
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "Experimental Setup", "Four experimental runs — each isolating a different pipeline stage");

  // Table
  const headers = ["Run", "Retrieval", "Reranker", "Generation", "Faithfulness", "Purpose"];
  const rows = [
    ["Baseline", "BM25 / Hybrid", "None", "Rule-based", "Deterministic", "No-reranker floor"],
    ["LLM Rerank", "BM25 / Hybrid / Iterative", "Groq 8B", "Rule-based", "Deterministic", "Reranker impact"],
    ["Haiku Best ★", "BM25", "Claude Haiku", "Rule-based", "LLM judge", "Best-quality LLM"],
    ["LLM Generation", "Hybrid / Iterative", "Groq 8B", "LLM debate", "LLM judge", "Full LLM pipeline"],
  ];
  const colW = [1.7, 2.3, 1.7, 1.5, 1.5, 2.53];
  const tStartX = 0.4, tStartY = 0.88;
  const rowH = 0.55, headerH = 0.45;

  // header row
  let cx = tStartX;
  headers.forEach((h, i) => {
    sl.addShape(pptx.ShapeType.rect, { x: cx, y: tStartY, w: colW[i], h: headerH, fill: { color: "0D1B30" }, line: { color: C.border } });
    sl.addText(h, { x: cx + 0.08, y: tStartY, w: colW[i] - 0.16, h: headerH, fontSize: 10, bold: true, color: C.blue, fontFace: "Segoe UI", valign: "middle" });
    cx += colW[i];
  });

  rows.forEach((row, ri) => {
    cx = tStartX;
    const ry = tStartY + headerH + ri * rowH;
    const isHighlight = ri === 2;
    row.forEach((cell, ci) => {
      sl.addShape(pptx.ShapeType.rect, {
        x: cx, y: ry, w: colW[ci], h: rowH,
        fill: { color: isHighlight ? "0A1A0F" : (ri % 2 ? "0A0A1A" : "080818") },
        line: { color: C.border },
      });
      const cellColor = ci === 0 && isHighlight ? C.green : ci === 0 && ri === 0 ? C.muted : C.white;
      sl.addText(cell, { x: cx + 0.08, y: ry, w: colW[ci] - 0.16, h: rowH, fontSize: 9.5, color: cellColor, fontFace: "Segoe UI", valign: "middle", bold: isHighlight && ci === 0 });
      cx += colW[ci];
    });
    if (isHighlight) {
      // left green bar
      sl.addShape(pptx.ShapeType.rect, { x: tStartX, y: ry, w: 0.06, h: rowH, fill: { color: C.green }, line: { color: C.green } });
    }
  });

  // 3 info cards
  const infoCards = [
    { title: "LLM Backends", items: ["🖥️ Ollama Llama 3.2 3B (local)", "☁️ Groq Llama 3.1 8B (free cloud)", "✨ Anthropic Claude Haiku (paid)"], color: C.blue },
    { title: "Scale", items: ["200 queries × HotpotQA", "200 queries × RAMDocs", "~3,000+ LLM calls per full run"], color: C.purple },
    { title: "K Values", items: ["Recall@2, @3, @5, @10", "Multi-doc hit rate @K", "MRR (mean reciprocal rank)"], color: C.teal },
  ];
  const icw = 3.95, icgap = 0.26;
  infoCards.forEach((ic, i) => {
    const icx = 0.4 + i * (icw + icgap);
    const icy = tStartY + headerH + rows.length * rowH + 0.26;
    card(sl, icx, icy, icw, 1.6, { topBarColor: ic.color });
    sl.addText(ic.title, { x: icx + 0.15, y: icy + 0.15, w: icw - 0.3, h: 0.3, fontSize: 11, bold: true, color: ic.color, fontFace: "Segoe UI" });
    ic.items.forEach((item, j) => {
      sl.addText(item, { x: icx + 0.15, y: icy + 0.5 + j * 0.32, w: icw - 0.3, h: 0.3, fontSize: 9.5, color: C.muted, fontFace: "Segoe UI" });
    });
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 10 — RESULTS: RETRIEVAL QUALITY
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "Results — Retrieval Quality", "HotpotQA · LLM Rerank run · higher = better");

  // Table
  const headers = ["Method", "Recall@5", "MRR", "Multi-doc Hit Rate"];
  const rows = [
    { cells: ["BM25 (baseline)", "0.650", "0.850", "0.245"], highlight: false },
    { cells: ["BM25 + LLM Rerank", "0.750", "0.875", "0.385"], highlight: false },
    { cells: ["Hybrid + LLM Rerank", "0.768", "0.926", "0.425"], highlight: false },
    { cells: ["Iterative Hybrid ★", "0.770", "0.926", "0.435"], highlight: true },
  ];
  const colW = [3.2, 2.0, 2.0, 2.5];
  const tX = 0.4, tY = 0.88, rH = 0.52, hH = 0.45;

  let cx = tX;
  headers.forEach((h, i) => {
    sl.addShape(pptx.ShapeType.rect, { x: cx, y: tY, w: colW[i], h: hH, fill: { color: "0D1B30" }, line: { color: C.border } });
    sl.addText(h, { x: cx + 0.1, y: tY, w: colW[i] - 0.2, h: hH, fontSize: 11, bold: true, color: C.blue, fontFace: "Segoe UI", valign: "middle" });
    cx += colW[i];
  });
  rows.forEach((row, ri) => {
    cx = tX;
    const ry = tY + hH + ri * rH;
    row.cells.forEach((cell, ci) => {
      sl.addShape(pptx.ShapeType.rect, {
        x: cx, y: ry, w: colW[ci], h: rH,
        fill: { color: row.highlight ? "0A1A0F" : (ri % 2 ? "0A0A1A" : "080818") },
        line: { color: C.border },
      });
      const isGreen = row.highlight && ci > 0;
      sl.addText(cell, { x: cx + 0.1, y: ry, w: colW[ci] - 0.2, h: rH, fontSize: row.highlight ? 11 : 10, color: isGreen ? C.green : (row.highlight ? C.white : C.white), fontFace: "Segoe UI", valign: "middle", bold: row.highlight });
      cx += colW[ci];
    });
    if (row.highlight) {
      sl.addShape(pptx.ShapeType.rect, { x: tX, y: ry, w: 0.06, h: rH, fill: { color: C.green }, line: { color: C.green } });
    }
  });

  // Improvement note
  card(sl, 0.4, 3.22, 9.7, 0.45, { fillColor: "0A100A", leftBarColor: C.green });
  sl.addText("LLM reranker alone adds +10% Recall@5 over baseline BM25  (0.650 → 0.750)", { x: 0.65, y: 3.28, w: 9.3, h: 0.33, fontSize: 10, color: C.muted, fontFace: "Segoe UI", valign: "middle" });

  // Bar charts (right side)
  sl.addText("Recall@5 comparison", { x: 10.0, y: 0.88, w: 3.1, h: 0.3, fontSize: 10, color: C.muted, fontFace: "Segoe UI", italic: true });

  const bars = [
    { label: "BM25 baseline", val: 0.650, pct: 0.844, color: "555580" },
    { label: "BM25 + Rerank", val: 0.750, pct: 0.974, color: "4466CC" },
    { label: "Hybrid + Rerank", val: 0.768, pct: 0.997, color: "2288CC" },
    { label: "Iterative ★", val: 0.770, pct: 1.0, color: C.green },
  ];
  const barX = 10.0, barW = 3.0, barH2 = 0.28, barGap = 0.52;
  bars.forEach((b, i) => {
    const by2 = 1.26 + i * barGap;
    sl.addText(b.label, { x: barX, y: by2, w: barW, h: 0.22, fontSize: 8.5, color: b.color === C.green ? C.green : C.muted, fontFace: "Segoe UI" });
    sl.addShape(pptx.ShapeType.rect, { x: barX, y: by2 + 0.22, w: barW, h: barH2, fill: { color: "1A1A35" }, line: { color: C.border } });
    sl.addShape(pptx.ShapeType.rect, { x: barX, y: by2 + 0.22, w: barW * b.pct, h: barH2, fill: { color: b.color }, line: { color: b.color } });
    sl.addText(b.val.toFixed(3), { x: barX + barW + 0.06, y: by2 + 0.2, w: 0.4, h: 0.3, fontSize: 9, color: C.muted, fontFace: "Segoe UI" });
  });

  // Key insight
  card(sl, 0.4, 3.82, 12.53, 2.28);
  sl.addText("Why Iterative Hybrid wins", { x: 0.6, y: 3.96, w: 5, h: 0.32, fontSize: 12, bold: true, color: C.gold, fontFace: "Segoe UI" });
  const wps = [
    ["BM25 (no rerank)", "Recall@5 = 0.650", "MRR = 0.850"],
    ["BM25 + LLM Rerank", "Recall@5 = 0.750  (+15%)", "MRR = 0.875"],
    ["Hybrid + LLM Rerank", "Recall@5 = 0.768", "MRR = 0.926"],
    ["Iterative Hybrid ★", "Recall@5 = 0.770 (best)", "MRR = 0.926  (best)"],
  ];
  const wpColors = [C.muted, "818CF8", C.blue, C.green];
  wps.forEach((wp, i) => {
    const wx = 0.55 + i * 3.08;
    sl.addShape(pptx.ShapeType.roundRect, { x: wx, y: 4.35, w: 2.9, h: 1.55, rectRadius: 0.08, fill: { color: i === 3 ? "0A1A0A" : "0A0A1A" }, line: { color: wpColors[i], transparency: 60 } });
    sl.addText(wp[0], { x: wx + 0.1, y: 4.45, w: 2.7, h: 0.3, fontSize: 9.5, bold: true, color: wpColors[i], fontFace: "Segoe UI" });
    sl.addText(wp[1], { x: wx + 0.1, y: 4.78, w: 2.7, h: 0.28, fontSize: 9, color: C.white, fontFace: "Segoe UI" });
    sl.addText(wp[2], { x: wx + 0.1, y: 5.06, w: 2.7, h: 0.28, fontSize: 9, color: C.white, fontFace: "Segoe UI" });
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 11 — RESULTS: ANSWER QUALITY
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "Results — Answer Quality", "HotpotQA Exact Match across all pipeline configurations");

  // Table
  const headers = ["Configuration", "EM (HotpotQA)", "Answer F1", "Faithfulness", "Halluc. Rate"];
  const rows = [
    { cells: ["Baseline (BM25)", "0.595", "0.602", "—", "—"], hl: false, color: null },
    { cells: ["BM25 + LLM Rerank", "0.740", "0.745", "—", "—"], hl: false, color: null },
    { cells: ["Hybrid + LLM Rerank", "0.710", "0.716", "—", "—"], hl: false, color: null },
    { cells: ["Haiku + BM25 (BEST) ★", "0.800", "0.800", "—", "0%"], hl: true, color: C.green },
    { cells: ["LLM Gen · BM25", "0.220", "0.359", "0.482", "36%"], hl: false, color: C.red },
    { cells: ["LLM Gen · Hybrid", "0.180", "0.347", "0.490", "28%"], hl: false, color: C.orange },
    { cells: ["LLM Gen · Iterative", "0.260", "0.422", "0.574", "18%"], hl: false, color: C.orange },
  ];
  const colW = [3.2, 1.85, 1.85, 1.85, 1.68];
  const tX = 0.4, tY = 0.88, rH = 0.46, hH = 0.42;
  let cx = tX;
  headers.forEach((h, i) => {
    sl.addShape(pptx.ShapeType.rect, { x: cx, y: tY, w: colW[i], h: hH, fill: { color: "0D1B30" }, line: { color: C.border } });
    sl.addText(h, { x: cx + 0.1, y: tY, w: colW[i] - 0.2, h: hH, fontSize: 10, bold: true, color: C.blue, fontFace: "Segoe UI", valign: "middle" });
    cx += colW[i];
  });
  rows.forEach((row, ri) => {
    cx = tX;
    const ry = tY + hH + ri * rH;
    row.cells.forEach((cell, ci) => {
      sl.addShape(pptx.ShapeType.rect, {
        x: cx, y: ry, w: colW[ci], h: rH,
        fill: { color: row.hl ? "0A1A0F" : row.color === C.red ? "150505" : row.color === C.orange ? "140B00" : (ri % 2 ? "0A0A1A" : "080818") },
        line: { color: C.border },
      });
      let cellColor = C.white;
      if (row.hl && ci > 0) cellColor = C.green;
      if (!row.hl && row.color === C.red && ci > 0) cellColor = C.red;
      if (!row.hl && row.color === C.orange && ci > 0) cellColor = C.orange;
      sl.addText(cell, { x: cx + 0.1, y: ry, w: colW[ci] - 0.2, h: rH, fontSize: row.hl ? 10.5 : 9.5, color: cellColor, fontFace: "Segoe UI", valign: "middle", bold: row.hl });
      cx += colW[ci];
    });
    if (row.hl) {
      sl.addShape(pptx.ShapeType.rect, { x: tX, y: ry, w: 0.06, h: rH, fill: { color: C.green }, line: { color: C.green } });
    }
    if (row.color === C.red || row.color === C.orange) {
      sl.addShape(pptx.ShapeType.rect, { x: tX, y: ry, w: 0.06, h: rH, fill: { color: row.color }, line: { color: row.color } });
    }
  });

  // RAMDocs note
  card(sl, 0.4, 4.28, 10.23, 0.4, { fillColor: "0A1A0F", leftBarColor: C.green });
  sl.addText("RAMDocs:  all methods ≈ 0.96 EM  ·  Misinformation suppressed 96.5%", {
    x: 0.65, y: 4.32, w: 9.8, h: 0.32, fontSize: 10, color: C.green, fontFace: "Segoe UI", valign: "middle",
  });

  // Key surprise box
  card(sl, 0.4, 4.84, 12.53, 0.76, { fillColor: "180808", leftBarColor: C.red });
  sl.addText("Key Surprise", { x: 0.65, y: 4.92, w: 2, h: 0.28, fontSize: 11, bold: true, color: C.red, fontFace: "Segoe UI" });
  sl.addText("Full LLM debate generation EM dropped from 0.80 → 0.22.  LLMs paraphrase correct answers in ways that break exact-match — they say \"British\" as \"British nationality\" or \"United Kingdom\".  Rule-based extraction wins for factoid QA.", {
    x: 2.55, y: 4.92, w: 10.2, h: 0.6, fontSize: 9.5, color: C.white, fontFace: "Segoe UI", wrap: true,
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 12 — LLM BACKEND COMPARISON
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "LLM Backend Comparison", "Same pipeline, three backends — quality vs cost trade-off");

  // Table
  const headers = ["Configuration", "Backend", "EM (HotpotQA)", "Faithfulness", "Halluc. Rate", "Cost"];
  const rows = [
    { cells: ["BM25 + Ollama rerank", "Llama 3.2 3B · local", "0.650", "—", "—", "Free"], hl: false },
    { cells: ["BM25 + Groq rerank", "Llama 3.1 8B · cloud", "0.740", "0.482", "36%", "Free tier"], hl: false },
    { cells: ["Iterative + Groq", "Llama 3.1 8B · cloud", "0.705", "0.574", "18%", "Free tier"], hl: false },
    { cells: ["BM25 + Haiku rerank ★", "Claude Haiku · API", "0.800", "—", "0%", "Paid"], hl: true },
  ];
  const colW = [2.8, 2.5, 1.9, 1.7, 1.7, 1.13];
  const tX = 0.4, tY = 0.88, rH = 0.56, hH = 0.45;
  let cx2 = tX;
  headers.forEach((h, i) => {
    sl.addShape(pptx.ShapeType.rect, { x: cx2, y: tY, w: colW[i], h: hH, fill: { color: "0D1B30" }, line: { color: C.border } });
    sl.addText(h, { x: cx2 + 0.1, y: tY, w: colW[i] - 0.2, h: hH, fontSize: 10, bold: true, color: C.blue, fontFace: "Segoe UI", valign: "middle" });
    cx2 += colW[i];
  });
  rows.forEach((row, ri) => {
    cx2 = tX;
    const ry = tY + hH + ri * rH;
    row.cells.forEach((cell, ci) => {
      sl.addShape(pptx.ShapeType.rect, {
        x: cx2, y: ry, w: colW[ci], h: rH,
        fill: { color: row.hl ? "0A1A0F" : (ri % 2 ? "0A0A1A" : "080818") },
        line: { color: C.border },
      });
      const isGreen = row.hl && ci >= 2;
      sl.addText(cell, {
        x: cx2 + 0.1, y: ry, w: colW[ci] - 0.2, h: rH,
        fontSize: row.hl ? 10.5 : 10, color: isGreen ? C.green : C.white,
        fontFace: "Segoe UI", valign: "middle", bold: row.hl,
      });
      cx2 += colW[ci];
    });
    if (row.hl) {
      sl.addShape(pptx.ShapeType.rect, { x: tX, y: ry, w: 0.06, h: rH, fill: { color: C.green }, line: { color: C.green } });
    }
  });

  // 3 backend cards
  const backends = [
    {
      icon: "🖥️", name: "Ollama 3B", sub: "Local · Free",
      items: ["Runs fully offline, no API keys", "Weakest reranking quality", "EM near baseline (0.650)", "Good for private/air-gapped use"],
      color: C.muted,
    },
    {
      icon: "☁️", name: "Groq 8B", sub: "Free cloud tier",
      items: ["Strong reranker, good F1 scores", "Best faithfulness: 0.574 (Iterative)", "Rate-limited: 30 RPM → 3.5 hr run", "Free but slow at scale"],
      color: C.blue,
    },
    {
      icon: "✨", name: "Claude Haiku", sub: "Paid API",
      items: ["Highest answer quality", "0.800 EM — best overall result", "0% hallucination (rule-based)", "Fast, accurate, consistent"],
      color: C.purple,
    },
  ];
  const bkw = 4.0, bkgap = 0.26;
  backends.forEach((bk, i) => {
    const bkx = 0.4 + i * (bkw + bkgap);
    const bky = 3.56;
    card(sl, bkx, bky, bkw, 2.1, { topBarColor: bk.color });
    sl.addText(bk.icon + "  " + bk.name, { x: bkx + 0.15, y: bky + 0.15, w: bkw - 0.3, h: 0.35, fontSize: 13, bold: true, color: C.white, fontFace: "Segoe UI" });
    pill(sl, bkx + 0.15, bky + 0.54, 1.5, 0.26, bk.sub, bk.color);
    bk.items.forEach((item, j) => {
      sl.addText("▸ " + item, { x: bkx + 0.15, y: bky + 0.88 + j * 0.29, w: bkw - 0.3, h: 0.27, fontSize: 9, color: C.muted, fontFace: "Segoe UI" });
    });
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 13 — KEY INSIGHTS
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  slideHeader(sl, "Key Insights", "What the numbers tell us");

  const insights = [
    {
      num: "9:1", label: "Retrieval dominates failure",
      desc: "90% of wrong answers occur because the right documents were never retrieved — not hallucination. Invest in retrieval first.",
      color: C.red,
    },
    {
      num: "+20%", label: "LLM reranker = highest impact",
      desc: "Adding a high-quality LLM reranker (Haiku) was the single biggest gain — EM from 0.595 → 0.800. One component, biggest jump.",
      color: C.green,
    },
    {
      num: "↓ 64%", label: "LLM generation hurts EM",
      desc: "Full debate generation dropped EM from 0.80 → 0.22. LLMs paraphrase correct answers in ways that break exact match. Rule-based wins for factoid QA.",
      color: C.orange,
    },
    {
      num: "96.5%", label: "Misinformation suppressed",
      desc: "MADAM-RAG label-weighted voting actively blocks adversarial documents on RAMDocs. Not observed — actively suppressed.",
      color: C.purple,
    },
  ];

  const iw = 6.0, ih = 2.4, igap = 0.26;
  insights.forEach((ins, i) => {
    const ix = i % 2 === 0 ? 0.4 : 6.7;
    const iy = i < 2 ? 0.88 : 3.46;
    card(sl, ix, iy, iw, ih, { leftBarColor: ins.color });
    sl.addText(ins.num, { x: ix + 0.22, y: iy + 0.18, w: 2.5, h: 0.85, fontSize: 40, bold: true, color: ins.color, fontFace: "Segoe UI" });
    sl.addText(ins.label, { x: ix + 0.22, y: iy + 1.02, w: iw - 0.4, h: 0.32, fontSize: 12, bold: true, color: C.white, fontFace: "Segoe UI" });
    sl.addText(ins.desc, { x: ix + 0.22, y: iy + 1.36, w: iw - 0.4, h: 0.88, fontSize: 9.5, color: C.muted, fontFace: "Segoe UI", wrap: true });
  });
}

// ─────────────────────────────────────────────────────────────────
// SLIDE 14 — CONCLUSION
// ─────────────────────────────────────────────────────────────────
{
  const sl = pptx.addSlide();
  addBg(sl);
  // gradient shape
  sl.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 6, h: 7.5, fill: { color: "0A1020", transparency: 50 }, line: { color: "0A1020", transparency: 50 } });
  slideHeader(sl, "Conclusion", "");

  // What we built
  sl.addText("What We Built", { x: 0.4, y: 0.82, w: 5.9, h: 0.35, fontSize: 14, bold: true, color: C.gold, fontFace: "Segoe UI" });
  const built = [
    { dot: C.blue, text: "End-to-end RAG pipeline combining 5 research papers into one unified, runnable system" },
    { dot: C.purple, text: "Three retrieval methods, LLM reranker, multi-agent debate with misinformation suppression" },
    { dot: C.green, text: "Faithfulness checking with deterministic fallback — graceful degradation at every step" },
    { dot: C.orange, text: "Benchmarked across 400 queries, 3 LLM backends, full failure-mode taxonomy" },
  ];
  built.forEach((b, i) => {
    sl.addShape(pptx.ShapeType.ellipse, { x: 0.4, y: 1.3 + i * 0.58, w: 0.14, h: 0.14, fill: { color: b.dot }, line: { color: b.dot } });
    sl.addText(b.text, { x: 0.65, y: 1.24 + i * 0.58, w: 5.5, h: 0.46, fontSize: 10, color: C.white, fontFace: "Segoe UI", wrap: true, valign: "middle" });
  });

  // divider
  sl.addShape(pptx.ShapeType.rect, { x: 6.55, y: 0.82, w: 0.02, h: 3.8, fill: { color: C.border }, line: { color: C.border } });

  // What we proved
  sl.addText("What We Proved", { x: 6.9, y: 0.82, w: 6.0, h: 0.35, fontSize: 14, bold: true, color: C.gold, fontFace: "Segoe UI" });
  const proved = [
    { val: "EM 0.800", desc: "on HotpotQA — competitive with published baselines", color: C.green },
    { val: "96.5%", desc: "misinformation suppression on adversarial RAMDocs", color: C.purple },
    { val: "9:1", desc: "retrieval : generation failure ratio — fix retrieval first", color: C.red },
    { val: "Iterative ★", desc: "uniquely solves multi-hop retrieval via bridge-entity chaining", color: C.blue },
  ];
  proved.forEach((p, i) => {
    card(sl, 6.9, 1.28 + i * 0.78, 6.0, 0.66, { leftBarColor: p.color });
    sl.addText(p.val, { x: 7.1, y: 1.35 + i * 0.78, w: 1.6, h: 0.52, fontSize: 16, bold: true, color: p.color, fontFace: "Segoe UI", valign: "middle" });
    sl.addText(p.desc, { x: 8.7, y: 1.35 + i * 0.78, w: 4.0, h: 0.52, fontSize: 10, color: C.white, fontFace: "Segoe UI", valign: "middle", wrap: true });
  });

  // Bottom stats strip
  sl.addShape(pptx.ShapeType.rect, { x: 0, y: 6.28, w: 13.33, h: 1.22, fill: { color: "0A0A20" }, line: { color: C.border } });
  const stats = [
    { val: "5", label: "Research Papers" },
    { val: "9", label: "Pipeline Steps" },
    { val: "400+", label: "Test Queries" },
    { val: "3", label: "LLM Backends" },
    { val: "EM 0.800", label: "Best Result ★" },
  ];
  const sw = 13.33 / stats.length;
  stats.forEach((s, i) => {
    sl.addText(s.val, { x: i * sw, y: 6.35, w: sw, h: 0.5, fontSize: i === stats.length - 1 ? 18 : 20, bold: true, color: i === stats.length - 1 ? C.gold : C.blue, fontFace: "Segoe UI", align: "center" });
    sl.addText(s.label, { x: i * sw, y: 6.82, w: sw, h: 0.3, fontSize: 9, color: C.muted, fontFace: "Segoe UI", align: "center" });
    if (i < stats.length - 1) {
      sl.addShape(pptx.ShapeType.rect, { x: (i + 1) * sw - 0.01, y: 6.36, w: 0.02, h: 0.72, fill: { color: C.border }, line: { color: C.border } });
    }
  });
}

// ─────────────────────────────────────────────────────────────────
// WRITE FILE
// ─────────────────────────────────────────────────────────────────
pptx.writeFile({ fileName: "./presentation.pptx" })
  .then(() => console.log("✅  presentation.pptx created"))
  .catch(err => console.error("Error:", err));
