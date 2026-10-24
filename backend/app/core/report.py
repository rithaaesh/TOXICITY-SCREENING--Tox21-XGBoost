"""Batch screening output: spreadsheet-safe CSV and a risk-alert summary."""

import csv
import io

from app.core.config import TARGETS

BASE_COLUMNS = [
    "row_number", "smiles", "valid", "error", "molecular_formula", "molecular_weight_g_mol", "log_p",
    "hydrogen_bond_donors", "hydrogen_bond_acceptors", "tpsa_angstrom2", "lipinski_pass",
    "max_tanimoto", "domain_status", "risk_alert", "high_risk_endpoints", "top_pathway", "high_risk_pathways",
]
TARGET_COLUMNS = [column for slug in TARGETS.values() for column in (f"{slug}_probability", f"{slug}_risk_level")]
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(value: object) -> object:
    """Neutralize spreadsheet formula injection in user-supplied text."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def high_risk_endpoints(result: dict) -> list[str]:
    return [t for t, p in result["targets"].items() if p["risk_level"] == "High"]


def build_csv(smiles_values: list[str], predictions: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=BASE_COLUMNS + TARGET_COLUMNS)
    writer.writeheader()
    for row_number, (smiles, result) in enumerate(zip(smiles_values, predictions), start=1):
        row: dict[str, object] = {
            "row_number": row_number, "smiles": csv_safe(smiles), "valid": result["valid"],
            "error": csv_safe(result.get("error", "")),
        }
        if result["valid"]:
            properties, flagged = result["properties"], high_risk_endpoints(result)
            row.update({
                "molecular_formula": result["molecular_formula"],
                "molecular_weight_g_mol": round(properties["molecular_weight_g_mol"], 3),
                "log_p": round(properties["log_p"], 3),
                "hydrogen_bond_donors": properties["hydrogen_bond_donors"],
                "hydrogen_bond_acceptors": properties["hydrogen_bond_acceptors"],
                "tpsa_angstrom2": round(properties["tpsa_angstrom2"], 3),
                "lipinski_pass": properties["lipinski_pass"],
                "max_tanimoto": round(result["domain"]["max_tanimoto"], 3) if result["domain"] else "",
                "domain_status": result["domain"]["status"] if result["domain"] else "",
                "risk_alert": "ALERT" if flagged else "",
                "high_risk_endpoints": ";".join(flagged),
            })
            pathways = result.get("pathways", [])
            if pathways:
                row["top_pathway"] = max(pathways, key=lambda p: p["score"])["name"]
                row["high_risk_pathways"] = ";".join(p["name"] for p in pathways if p["risk_level"] == "High")
            for target, slug in TARGETS.items():
                row[f"{slug}_probability"] = round(result["targets"][target]["probability"], 4)
                row[f"{slug}_risk_level"] = result["targets"][target]["risk_level"]
        writer.writerow(row)
    return output.getvalue()


def summarize(smiles_values: list[str], predictions: list[dict], top: int = 10) -> dict:
    """Counts per endpoint and risk band, domain coverage, and the highest-scoring compounds."""
    valid = [(i, r) for i, r in enumerate(predictions, start=1) if r["valid"]]
    endpoints = {t: {"Low": 0, "Medium": 0, "High": 0} for t in TARGETS}
    domain = {"In domain": 0, "Borderline": 0, "Out of domain": 0}
    alerts = []
    for row_number, result in valid:
        for target, prediction in result["targets"].items():
            endpoints[target][prediction["risk_level"]] += 1
        if result["domain"]:
            domain[result["domain"]["status"]] += 1
        flagged = high_risk_endpoints(result)
        if flagged:
            best = max(flagged, key=lambda t: result["targets"][t]["probability"])
            alerts.append({
                "row_number": row_number, "smiles": smiles_values[row_number - 1], "high_risk_endpoints": flagged,
                "top_endpoint": best, "top_probability": result["targets"][best]["probability"],
                "domain_status": result["domain"]["status"] if result["domain"] else None,
            })
    pathway_counts: dict[str, dict[str, int]] = {}
    for _, result in valid:
        for pathway in result.get("pathways", []):
            pathway_counts.setdefault(pathway["name"], {"Low": 0, "Medium": 0, "High": 0})[pathway["risk_level"]] += 1
    alerts.sort(key=lambda a: a["top_probability"], reverse=True)
    return {
        "total": len(predictions), "valid": len(valid), "invalid": len(predictions) - len(valid),
        "compounds_with_alert": len(alerts), "domain": domain, "endpoints": endpoints, "pathways": pathway_counts, "top_alerts": alerts[:top],
    }
