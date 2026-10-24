"""Protein and pathway layer: what each Tox21 assay measures, and pathway-level roll-ups.

The mapping is curated from the Tox21 Data Challenge assay descriptions (Huang et al. 2016, Front. Environ.
Sci.) and standard gene/pathway biology. It is knowledge, not something learned from data, and it describes
assay biology only: it says nothing about 3-D binding or about whether a molecule reaches the protein.

A pathway score is the HIGHEST calibrated probability among the pathway's assays ("any-hit" logic). Assays
inside a pathway are correlated (for example full-length and ligand-binding-domain assays of one receptor),
so combining them as independent evidence would overstate risk; the maximum does not.
"""

from app.core.config import RISK_BANDS, TARGETS

# assay -> (protein or process, gene symbol or None, what it does, assay note or None, pathway slug)
ENDPOINT_BIOLOGY: dict[str, dict[str, object]] = {
    "NR-AR": {"protein": "Androgen receptor (full length)", "gene": "AR", "pathway": "endocrine",
              "function": "Nuclear receptor that mediates male sex-hormone (androgen) signalling.",
              "assay_note": "Reporter assay in the MDA cell line."},
    "NR-AR-LBD": {"protein": "Androgen receptor, ligand-binding domain", "gene": "AR", "pathway": "endocrine",
                  "function": "The part of the androgen receptor that binds the hormone.",
                  "assay_note": "Isolated ligand-binding domain."},
    "NR-ER": {"protein": "Estrogen receptor alpha (full length)", "gene": "ESR1", "pathway": "endocrine",
              "function": "Nuclear receptor that mediates estrogen signalling.",
              "assay_note": "Reporter assay in the BG1 cell line."},
    "NR-ER-LBD": {"protein": "Estrogen receptor alpha, ligand-binding domain", "gene": "ESR1", "pathway": "endocrine",
                  "function": "The part of estrogen receptor alpha that binds the hormone.",
                  "assay_note": "Isolated ligand-binding domain."},
    "NR-Aromatase": {"protein": "Aromatase (enzyme)", "gene": "CYP19A1", "pathway": "endocrine",
                     "function": "Enzyme that converts androgens into estrogens.",
                     "assay_note": "Flags disruption of aromatase activity."},
    "NR-AhR": {"protein": "Aryl hydrocarbon receptor", "gene": "AHR", "pathway": "xenobiotic",
               "function": "Ligand-activated transcription factor that senses aromatic hydrocarbons such as dioxins and switches on foreign-compound metabolising enzymes.",
               "assay_note": None},
    "NR-PPAR-gamma": {"protein": "Peroxisome proliferator-activated receptor gamma", "gene": "PPARG", "pathway": "metabolic",
                      "function": "Nuclear receptor regulating lipid and glucose metabolism and fat-cell differentiation.",
                      "assay_note": None},
    "SR-ARE": {"protein": "Nrf2 / antioxidant response element", "gene": "NFE2L2", "pathway": "oxidative",
               "function": "Master regulator of the cell's antioxidant defence.",
               "assay_note": None},
    "SR-ATAD5": {"protein": "ATAD5", "gene": "ATAD5", "pathway": "genotoxic",
                 "function": "Protein involved in DNA replication stress and DNA damage response.",
                 "assay_note": "Luciferase-tagged ATAD5 in human embryonic kidney cells."},
    "SR-p53": {"protein": "p53 tumour suppressor", "gene": "TP53", "pathway": "genotoxic",
               "function": "Responds to DNA damage and cell stress by arresting the cell cycle, repairing DNA or triggering cell death.",
               "assay_note": None},
    "SR-HSE": {"protein": "Heat shock factor 1 response element", "gene": "HSF1", "pathway": "proteotoxic",
               "function": "Drives the heat-shock response to protein misfolding and cellular stress.",
               "assay_note": None},
    "SR-MMP": {"protein": "Mitochondrial membrane potential", "gene": None, "pathway": "mitochondrial",
               "function": "Indicator of mitochondrial health; a collapse signals mitochondrial dysfunction. A cellular process, not a single protein.",
               "assay_note": None},
}

PATHWAYS: dict[str, dict[str, str]] = {
    "endocrine": {"name": "Endocrine (sex-hormone) signalling", "description": "Androgen and estrogen receptor signalling and estrogen synthesis."},
    "xenobiotic": {"name": "Xenobiotic sensing (AhR)", "description": "Detection of foreign aromatic compounds and induction of metabolising enzymes."},
    "metabolic": {"name": "Lipid and glucose metabolism (PPAR)", "description": "PPAR-gamma control of metabolism and cell differentiation."},
    "oxidative": {"name": "Oxidative stress response (Nrf2/ARE)", "description": "Antioxidant defence triggered by reactive or electrophilic chemicals."},
    "genotoxic": {"name": "DNA damage and genotoxic stress", "description": "p53 and ATAD5 responses to DNA damage and replication stress."},
    "proteotoxic": {"name": "Heat-shock / protein-folding stress", "description": "Cellular response to misfolded proteins."},
    "mitochondrial": {"name": "Mitochondrial function", "description": "Loss of mitochondrial membrane potential."},
}


def _check() -> None:
    assert set(ENDPOINT_BIOLOGY) == set(TARGETS), "ENDPOINT_BIOLOGY must cover every Tox21 target exactly once"
    assert {v["pathway"] for v in ENDPOINT_BIOLOGY.values()} == set(PATHWAYS), "every pathway needs at least one assay"


_check()


def pathway_members(slug: str) -> list[str]:
    """Assays that belong to a pathway, in Tox21 order."""
    return [t for t in TARGETS if ENDPOINT_BIOLOGY[t]["pathway"] == slug]


def pathway_genes(slug: str) -> list[str]:
    genes: list[str] = []
    for target in pathway_members(slug):
        gene = ENDPOINT_BIOLOGY[target]["gene"]
        if gene and gene not in genes:
            genes.append(str(gene))
    return genes


def risk_band(probability: float) -> str:
    if probability < RISK_BANDS[0]:
        return "Low"
    if probability < RISK_BANDS[1]:
        return "Medium"
    return "High"


def pathway_scores(target_results: dict[str, dict], metrics: dict[str, dict] | None = None) -> list[dict[str, object]]:
    """Roll the 12 calibrated assay probabilities up into pathway scores (maximum over member assays)."""
    metrics = metrics or {}
    rows = []
    for slug, info in PATHWAYS.items():
        members = pathway_members(slug)
        probabilities = {t: float(target_results[t]["probability"]) for t in members}
        driver = max(probabilities, key=probabilities.get)
        score = probabilities[driver]
        stored = metrics.get(slug, {})
        base = stored.get("base_rate")
        rows.append({
            "slug": slug, "name": info["name"], "genes": pathway_genes(slug), "score": score, "risk_level": risk_band(score),
            "driving_assay": driver, "n_high_assays": sum(1 for p in probabilities.values() if risk_band(p) == "High"),
            "members": [{"assay": t, "probability": p} for t, p in probabilities.items()],
            "base_rate": base, "enrichment": (score / base) if base else None,
            "held_out_roc_auc": stored.get("roc_auc"),
        })
    return rows


def biology_catalogue(metrics: dict[str, dict] | None = None) -> dict[str, object]:
    """Static knowledge table for the API: every assay with its protein, and every pathway with its members."""
    metrics = metrics or {}
    endpoints = [{"assay": t, "pathway": PATHWAYS[ENDPOINT_BIOLOGY[t]["pathway"]]["name"], "pathway_slug": ENDPOINT_BIOLOGY[t]["pathway"],
                  **{k: ENDPOINT_BIOLOGY[t][k] for k in ("protein", "gene", "function", "assay_note")}} for t in TARGETS]
    pathways = [{"slug": s, "name": i["name"], "description": i["description"], "assays": pathway_members(s), "genes": pathway_genes(s), "metrics": metrics.get(s)}
                for s, i in PATHWAYS.items()]
    return {"endpoints": endpoints, "pathways": pathways,
            "method": "Pathway score = highest calibrated probability among the pathway's assays. Mapping is curated from Tox21 assay descriptions, not learned from data."}
