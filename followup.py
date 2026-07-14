import os

import pandas as pd

from data_query import _DISEASE_ALIASES, _normalize
from fibrotic_release import load_fibrotic_release


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DISEASE_DISPLAY_NAMES = {
    "CKD": "Chronic Kidney Disease (CKD)",
    "Cardiac_Fibrosis": "Cardiac Fibrosis",
    "Crohns_Disease": "Crohn's Disease",
    "Fibrosis_of_Skin": "Skin Fibrosis",
    "IPF": "Idiopathic Pulmonary Fibrosis (IPF)",
    "NASH": "Metabolic Dysfunction-Associated Steatohepatitis (MASH)",
    "Pulmonary Fibrosis": "Pulmonary Fibrosis",
    "SSc_Connective_Tissue": "Systemic Sclerosis (SSc)",
}
EMBED_DISEASE_MAP = {
    "CKD": "CKD",
    "NASH": "MASH",
    "Pulmonary Fibrosis": "Pulmonary_fibrosis",
    "SSc_Connective_Tissue": "SSc_Connective_Tissue",
    "Crohns_Disease": "Crohns_Disease",
    "Fibrosis_of_Skin": "Fibrosis_of_Skin",
}

_pathway_df = None


def _load_pathways():
    global _pathway_df
    if _pathway_df is None:
        path = os.path.join(DATA_DIR, "pathway_enrichment.csv")
        _pathway_df = pd.read_csv(path)
    return _pathway_df


def _resolve_disease(query, diseases):
    normalized = _normalize(query)
    for disease in diseases:
        if _normalize(disease) == normalized:
            return disease
    for alias, canonical in _DISEASE_ALIASES.items():
        if alias == normalized and canonical in diseases:
            return canonical
    return None


def get_pathway_enrichment(disease_query, top_n=10):
    df = _load_pathways()
    canonical = _resolve_disease(disease_query, set(df["disease"].unique()))
    if canonical is None:
        return None

    cohort = df[df["disease"] == canonical].sort_values("rank").head(top_n)
    if cohort.empty:
        return None

    pathways = []
    for _, row in cohort.iterrows():
        pathways.append(
            {
                "pathway": row["pathway"],
                "source": row["source"],
                "gene_count": int(row["gene_count"]),
                "enrichment_ratio": round(float(row["enrichment_ratio"]), 2),
                "p_adjusted": f"{row['p_adjusted']:.2e}",
            }
        )

    return {
        "disease": canonical,
        "disease_label": DISEASE_DISPLAY_NAMES.get(canonical, canonical),
        "pathways": pathways,
    }


def _centroid(points):
    return {
        "x": round(sum(float(point["tsne_x"]) for point in points) / len(points), 2),
        "y": round(sum(float(point["tsne_y"]) for point in points) / len(points), 2),
    }


def describe_embedding_context(disease_query):
    embedding, _ = load_fibrotic_release()
    points = embedding["points"]
    display_diseases = {point["disease"] for point in points}
    canonical = _resolve_disease(disease_query, set(EMBED_DISEASE_MAP))
    if canonical is None:
        return None

    embed_disease = EMBED_DISEASE_MAP.get(canonical)
    if embed_disease not in display_diseases:
        return None
    cohort = [point for point in points if point["disease"] == embed_disease]
    total = len(cohort)

    groups = {}
    for group in ("pure", "intermediate", "overlap"):
        count = sum(point["group"] == group for point in cohort)
        groups[group] = {"count": count, "pct": round(100.0 * count / total, 1)}

    centroid = _centroid(cohort)
    neighbors = []
    for other in sorted(display_diseases - {embed_disease}):
        other_centroid = _centroid(
            [point for point in points if point["disease"] == other]
        )
        distance = (
            (centroid["x"] - other_centroid["x"]) ** 2
            + (centroid["y"] - other_centroid["y"]) ** 2
        ) ** 0.5
        neighbors.append({"disease": other, "distance": round(distance, 2)})
    neighbors.sort(key=lambda item: item["distance"])

    return {
        "disease": canonical,
        "disease_label": DISEASE_DISPLAY_NAMES.get(canonical, canonical),
        "embed_disease": embed_disease,
        "dataset_version": embedding["dataset_version"],
        "total_patients": total,
        "centroid": centroid,
        "groups": groups,
        "nearest_clusters": neighbors[:3],
    }
