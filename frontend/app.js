const API_BASE = window.TOX21_API_BASE || `${window.location.protocol}//${window.location.hostname}:8000/api`;
const smilesInput = document.querySelector("#smiles-input");
const analyzeButton = document.querySelector("#analyze-button");
const serviceStatus = document.querySelector("#service-status");
const toast = document.querySelector("#toast");
const batchFile = document.querySelector("#batch-file");
const batchRunButton = document.querySelector("#batch-run-button");
const batchDownload = document.querySelector("#batch-download");
const explainSelect = document.querySelector("#explain-target");
let assayChart;
let shapChart;
let toastTimer;
let explanationData;
let targets = [];
let biology;
let analyzedSmiles = "";
let mapRequestId = 0;

const RISK_COLORS = { Low: "#78ae83", Medium: "#d4b459", High: "#e58b70" };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 4500);
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    if (contentType.includes("application/json")) {
      const error = await response.json();
      message = error.detail || error.error || message;
    }
    throw new Error(message);
  }
  return contentType.includes("image/svg+xml") ? response.text() : response.json();
}

function setServiceStatus(kind, label) {
  serviceStatus.className = `service-status ${kind}`;
  serviceStatus.lastElementChild.textContent = label;
}

async function checkHealth() {
  try {
    const health = await request("/health");
    setServiceStatus(health.models_loaded ? "ready" : "error", health.models_loaded ? "Models ready" : "Training required");
  } catch {
    setServiceStatus("error", "API unavailable");
  }
}

function updateInputCount() {
  document.querySelector("#char-count").textContent = `${smilesInput.value.length} / 1000`;
}

function svgImage(svg, alt) {
  const image = document.createElement("img");
  image.alt = alt;
  image.src = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  return image;
}

function replaceImage(container, image) {
  const previous = container.querySelector("img");
  if (previous?.src.startsWith("blob:")) URL.revokeObjectURL(previous.src);
  container.replaceChildren(image);
}

/* ---------- endpoint catalogue ---------- */

async function loadTargets() {
  try {
    targets = (await request("/targets")).targets;
  } catch {
    return;
  }
  const readouts = document.querySelector("#assay-readouts");
  readouts.replaceChildren();
  explainSelect.replaceChildren();
  for (const target of targets) {
    const row = el("div", "assay-readout");
    row.dataset.target = target.name;
    row.setAttribute("role", "listitem");
    row.title = `${target.description}. Base rate in training data: ${target.base_rate ? (target.base_rate * 100).toFixed(1) : "?"}%.`;
    const left = el("div");
    left.append(el("span", `assay-symbol ${target.group === "Nuclear receptor" ? "" : "sr"}`, target.group === "Nuclear receptor" ? "NR" : "SR"), el("span", "assay-name", target.name));
    const value = el("strong", "", "—");
    row.append(left, value);
    readouts.append(row);
    explainSelect.append(new Option(target.name, target.name));
  }
  explainSelect.value = targets.find((t) => t.name === "NR-AhR") ? "NR-AhR" : targets[0]?.name;
}

/* ---------- protein and pathway layer ---------- */

async function loadBiology() {
  try {
    biology = await request("/biology");
  } catch {
    return;
  }
  const table = document.querySelector("#protein-table");
  table.replaceChildren();
  const head = el("tr");
  for (const name of ["Assay", "Protein or process", "Gene", "Pathway", "What it does"]) head.append(el("th", "", name));
  table.append(head);
  for (const row of biology.endpoints) {
    const tr = el("tr");
    tr.append(el("td", "", row.assay), el("td", "", row.protein), el("td", "", row.gene || "process"), el("td", "", row.pathway));
    const detail = el("td", "wrap", row.assay_note ? `${row.function} ${row.assay_note}` : row.function);
    tr.append(detail);
    table.append(tr);
  }
}

function renderPathways(result) {
  const table = document.querySelector("#pathway-table");
  const caption = table.querySelector("caption");
  table.replaceChildren(caption);
  const head = el("tr");
  for (const name of ["Pathway", "Proteins / genes", "Score", "Band", "Driven by", "Held-out AUC"]) head.append(el("th", "", name));
  table.append(head);
  const rows = [...result.pathways].sort((a, b) => b.score - a.score);
  for (const pathway of rows) {
    const tr = el("tr");
    tr.append(el("td", "", pathway.name), el("td", "", pathway.genes.length ? pathway.genes.join(", ") : "mitochondria"));
    const scoreCell = el("td");
    const wrap = el("span", "path-score");
    const track = el("span", "bar-track");
    const fill = el("span", "bar-fill");
    fill.style.width = `${(pathway.score * 100).toFixed(1)}%`;
    fill.style.background = RISK_COLORS[pathway.risk_level];
    track.append(fill);
    wrap.append(track, el("span", "", `${(pathway.score * 100).toFixed(0)}%`));
    scoreCell.append(wrap);
    const band = el("td", "band", pathway.risk_level.toUpperCase());
    band.style.color = RISK_COLORS[pathway.risk_level];
    const driver = el("td", "", pathway.members.length > 1 ? `${pathway.driving_assay} (${pathway.n_high_assays}/${pathway.members.length} high)` : pathway.driving_assay);
    const auc = el("td", "", pathway.held_out_roc_auc == null ? "n/a" : pathway.held_out_roc_auc.toFixed(2));
    auc.title = "ROC-AUC of this pathway score on held-out scaffold test molecules";
    tr.append(scoreCell, band, driver, auc);
    table.append(tr);
  }
  document.querySelector("#pathway-state").textContent = "7 PATHWAYS / 12 ASSAYS";
}

/* ---------- structure, predictions, properties ---------- */

async function renderStructure(smiles, result) {
  const svg = await request("/structure", { method: "POST", body: JSON.stringify({ smiles }) });
  replaceImage(document.querySelector("#structure-visual"), svgImage(svg, `2D structure for ${result.canonical_smiles}`));
  document.querySelector("#structure-state").textContent = "VALID STRUCTURE";
  document.querySelector("#canonical-smiles").textContent = result.canonical_smiles;
  document.querySelector("#molecular-formula").textContent = result.molecular_formula;
  const status = document.querySelector("#domain-status");
  const detail = document.querySelector("#domain-detail");
  if (result.domain) {
    const kind = result.domain.status === "In domain" ? "in" : result.domain.status === "Borderline" ? "borderline" : "out";
    status.className = `domain-status ${kind}`;
    status.textContent = result.domain.status;
    detail.textContent = `Max Tanimoto ${result.domain.max_tanimoto.toFixed(2)} to training set. ${kind === "out" ? "Treat scores as extrapolation." : kind === "borderline" ? "Scores are less reliable." : "Resembles training data."}`;
  }
}

function renderPredictions(result) {
  const entries = Object.entries(result.targets);
  const labels = entries.map(([name]) => name);
  const values = entries.map(([, value]) => value.probability * 100);
  const colors = entries.map(([, value]) => RISK_COLORS[value.risk_level]);
  document.querySelector("#chart-empty").hidden = true;
  document.querySelector("#result-state").textContent = "PREDICTION COMPLETE";
  if (assayChart) assayChart.destroy();
  assayChart = new Chart(document.querySelector("#assay-chart"), {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 1, barThickness: 15 }] },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.raw.toFixed(1)}% calibrated probability` } } },
      scales: {
        x: { min: 0, max: 100, grid: { color: "#36403a" }, border: { display: false }, ticks: { stepSize: 50, color: "#9aa69e", callback: (v) => `${v}%`, font: { size: 9 } } },
        y: { grid: { display: false }, border: { display: false }, ticks: { color: "#c7d0c9", font: { size: 9, family: "DM Mono" } } },
      },
    },
  });
  for (const [name, prediction] of entries) {
    const row = document.querySelector(`.assay-readout[data-target="${name}"]`);
    if (!row) continue;
    row.className = `assay-readout ${prediction.risk_level.toLowerCase()}`;
    const strong = row.querySelector("strong");
    strong.replaceChildren(`${(prediction.probability * 100).toFixed(1)}%`);
    const small = el("small", "", prediction.risk_level.toUpperCase());
    const enrichment = el("span", "enrichment", prediction.enrichment ? `${prediction.enrichment.toFixed(1)}× base` : "");
    strong.append(small, enrichment);
  }
}

function renderProperties(result) {
  const properties = result.properties;
  document.querySelector("#property-mw").textContent = properties.molecular_weight_g_mol.toFixed(2);
  document.querySelector("#property-logp").textContent = properties.log_p.toFixed(2);
  document.querySelector("#property-hbd").textContent = properties.hydrogen_bond_donors;
  document.querySelector("#property-hba").textContent = properties.hydrogen_bond_acceptors;
  document.querySelector("#property-tpsa").textContent = properties.tpsa_angstrom2.toFixed(2);
  const badge = document.querySelector("#lipinski-badge");
  const violationCount = properties.lipinski_violations.length;
  badge.textContent = properties.lipinski_pass ? "ROF / PASS" : `ROF / ${violationCount} ${violationCount === 1 ? "VIOLATION" : "VIOLATIONS"}`;
  badge.classList.toggle("pass", properties.lipinski_pass);
  badge.classList.toggle("fail", !properties.lipinski_pass);
  document.querySelector("#lipinski-detail").textContent = violationCount
    ? `Rule of Five thresholds exceeded: ${properties.lipinski_violations.map((n) => n.replaceAll("_", " ")).join(", ")}.`
    : "Lipinski Rule of Five passed. This is a drug-likeness heuristic, not a toxicity rule.";
}

/* ---------- explanations ---------- */

function renderExplanations(result) {
  explanationData = result;
  renderSelectedExplanation();
}

function featureLabel(feature) {
  return feature.kind === "fingerprint_bit" ? `Bit ${feature.feature_index}` : feature.name;
}

function renderSelectedExplanation() {
  if (!explanationData) return;
  const targetName = explainSelect.value;
  const target = explanationData.targets[targetName];
  const features = [...target.positive_features, ...target.negative_features]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)).slice(0, 10);
  document.querySelector("#shap-empty").hidden = true;
  document.querySelector("#explain-state").textContent = `TOP SIGNAL / ${targetName}`;
  if (shapChart) shapChart.destroy();
  shapChart = new Chart(document.querySelector("#shap-chart"), {
    type: "bar",
    data: {
      labels: features.map(featureLabel),
      datasets: [{ data: features.map((f) => f.shap_value), backgroundColor: features.map((f) => (f.shap_value >= 0 ? "#e58b70" : "#82a9ce")), borderRadius: 1, barThickness: 14 }],
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` SHAP ${c.raw.toFixed(4)}` } } },
      scales: {
        x: { grid: { color: "#36403a" }, border: { display: false }, ticks: { color: "#9aa69e", maxTicksLimit: 5, font: { size: 9 } } },
        y: { grid: { display: false }, border: { display: false }, ticks: { color: "#c7d0c9", font: { size: 9, family: "DM Mono" } } },
      },
    },
  });
  const list = document.querySelector("#fragment-list");
  list.replaceChildren();
  for (const feature of features) {
    const row = el("div", "fragment-row");
    const kindLabel = { fingerprint_bit: "BIT", maccs_key: "MACCS", descriptor: "DESC" }[feature.kind];
    const text = feature.kind === "fingerprint_bit"
      ? (feature.fragments.length ? feature.fragments.join(" / ") : "No fragment mapped")
      : `${feature.name} = ${Number.isInteger(feature.value) ? feature.value : feature.value.toFixed(2)}`;
    const code = el("code", "", text);
    code.title = text;
    row.append(code, el("span", "", `${kindLabel} · ${feature.shap_value > 0 ? "+" : ""}${feature.shap_value.toFixed(3)}`));
    list.append(row);
  }
  loadAtomMap(targetName, target.fingerprint_share);
}

async function loadAtomMap(targetName, fingerprintShare) {
  const requestId = ++mapRequestId;
  const container = document.querySelector("#atom-map-visual");
  const caption = document.querySelector("#atom-map-caption");
  try {
    const svg = await request("/explain/map", { method: "POST", body: JSON.stringify({ smiles: analyzedSmiles, target: targetName }) });
    if (requestId !== mapRequestId) return;
    replaceImage(container, svgImage(svg, `Atom-level SHAP map for ${targetName}`));
    const share = Math.round((fingerprintShare ?? 0) * 100);
    caption.textContent = `${targetName}: red atoms raise the score, blue lower it. Fingerprint bits carry ${share}% of this prediction's total SHAP weight; the rest is descriptors and MACCS keys, which cannot be drawn on atoms.`;
  } catch (error) {
    if (requestId === mapRequestId) caption.textContent = error.message;
  }
}

/* ---------- batch ---------- */

function updateBatchFile() {
  const file = batchFile.files[0];
  batchDownload.hidden = true;
  document.querySelector("#batch-summary").hidden = true;
  document.querySelector("#batch-file-name").textContent = file ? file.name : "No file selected";
  if (!file) { batchRunButton.disabled = true; return; }
  if (!file.name.toLowerCase().endsWith(".csv") || file.size > 5 * 1024 * 1024) {
    batchRunButton.disabled = true;
    document.querySelector("#batch-status").textContent = "Choose a CSV file no larger than 5 MB.";
    return;
  }
  batchRunButton.disabled = false;
  document.querySelector("#batch-status").textContent = "Ready to screen. The CSV must contain a column named smiles.";
}

function renderBatchSummary(summary) {
  const box = document.querySelector("#batch-summary");
  box.replaceChildren();
  const stats = el("div", "stat-row");
  const addStat = (value, label, cls = "") => { const s = el("div", `stat ${cls}`); s.append(el("strong", "", String(value)), el("span", "", label)); stats.append(s); };
  addStat(summary.total, "ROWS");
  addStat(summary.valid, "VALID");
  addStat(summary.invalid, "INVALID");
  addStat(summary.compounds_with_alert, "WITH HIGH-RISK ALERT", summary.compounds_with_alert ? "alert" : "");
  addStat(summary.domain["Out of domain"] + summary.domain["Borderline"], "BORDERLINE / OUT OF DOMAIN");
  box.append(stats);
  if (summary.top_alerts.length) {
    const table = el("table", "alert-table");
    const head = el("tr");
    for (const name of ["Row", "SMILES", "Top endpoint", "Probability", "High-risk endpoints", "Domain"]) head.append(el("th", "", name));
    table.append(head);
    for (const alert of summary.top_alerts) {
      const row = el("tr");
      const cells = [alert.row_number, alert.smiles, alert.top_endpoint, `${(alert.top_probability * 100).toFixed(1)}%`, alert.high_risk_endpoints.length, alert.domain_status || "—"];
      for (const value of cells) { const cell = el("td", "", String(value)); cell.title = String(value); row.append(cell); }
      table.append(row);
    }
    box.append(table);
  } else {
    box.append(el("p", "summary-note", "No compound reached the High band (calibrated probability ≥ 50%) on any endpoint."));
  }
  box.hidden = false;
}

async function runBatch() {
  const file = batchFile.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  batchRunButton.disabled = true;
  batchRunButton.textContent = "Screening...";
  batchDownload.hidden = true;
  document.querySelector("#batch-status").textContent = "Validating structures and scoring 12 endpoints...";
  try {
    const response = await fetch(`${API_BASE}/predict/batch?format=json`, { method: "POST", body: formData });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Batch request failed (${response.status}).`);
    }
    const { summary, csv } = await response.json();
    if (batchDownload.href.startsWith("blob:")) URL.revokeObjectURL(batchDownload.href);
    batchDownload.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    batchDownload.hidden = false;
    renderBatchSummary(summary);
    document.querySelector("#batch-status").textContent = `Complete: ${summary.valid} valid, ${summary.invalid} invalid, ${summary.total} total. Download the full results CSV below.`;
  } catch (error) {
    document.querySelector("#batch-status").textContent = error.message;
  } finally {
    batchRunButton.disabled = false;
    batchRunButton.textContent = "Run batch";
  }
}

/* ---------- main flow ---------- */

async function analyze() {
  const smiles = smilesInput.value.trim();
  if (!smiles) {
    showToast("Enter a SMILES string or choose a quick-test compound.");
    smilesInput.focus();
    return;
  }
  analyzeButton.disabled = true;
  analyzeButton.firstElementChild.textContent = "Analyzing molecule...";
  try {
    const result = await request("/predict", { method: "POST", body: JSON.stringify({ smiles }) });
    analyzedSmiles = smiles;
    renderPredictions(result);
    renderPathways(result);
    renderProperties(result);
    await renderStructure(smiles, result);
    const explanation = await request("/explain", { method: "POST", body: JSON.stringify({ smiles }) });
    renderExplanations(explanation);
    setServiceStatus("ready", "Models ready");
  } catch (error) {
    showToast(error.message);
    if (error.message.includes("Models are not trained")) setServiceStatus("error", "Training required");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.firstElementChild.textContent = "Analyze compound";
  }
}

for (const preset of document.querySelectorAll(".preset")) {
  preset.addEventListener("click", () => {
    smilesInput.value = preset.dataset.smiles;
    updateInputCount();
    smilesInput.focus();
  });
}
smilesInput.addEventListener("input", updateInputCount);
analyzeButton.addEventListener("click", analyze);
batchFile.addEventListener("change", updateBatchFile);
batchRunButton.addEventListener("click", runBatch);
explainSelect.addEventListener("change", renderSelectedExplanation);
smilesInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyze();
});

updateInputCount();
checkHealth();
loadTargets();
loadBiology();
