"""
task13_cross_lineage_fptree.py
==============================
Dr. Ng Action Item 3: Cross-lineage Joint Analysis

Merges all 4 lineage SAE annotations into a unified Functional Programme Tree:
  Root
    Ubiquitous       (programmes spanning >=3 lineages)
    B-lineage
    T/NK-lineage
    DC-lineage
    Mono-lineage

INPUTS (loaded from sae_code/):
  sae_annotations_hybrid.json         B-cell, 170 unique programmes (task9)
  sae_annotations_tcnk_routeB.json    T/NK,  109 unique programmes (Session 24)
  sae_annotations_dc_routeB.json      DC,    122 unique programmes (Session 26)
  sae_annotations_mono_routeB.json    Mono,  ?   unique programmes (this session)

METHOD
  1. Load all annotations, skip failed/LOW confidence entries
  2. For each programme, build a text document =
       programme_name (3x weight) + functional_description (1x) + top genes (2x)
  3. TF-IDF vectorise (1- and 2-grams, English stopwords)
  4. Cosine similarity -> Ward hierarchical clustering
  5. Cut tree at multiple k values (10, 15, 20, 30, 50); use k=30 as primary
  6. Layer = "Ubiquitous" if cluster spans >=3 lineages, else "{dominant}-lineage"
  7. Bonus: direct programme_name matches across lineages -> cross_lineage_matches.csv

OUTPUTS (under sae_code/action_item_3_outputs/):
  fp_tree_assignments.csv              full table: lineage, feature_id, name, layer
  fp_tree_layer_summary.csv            counts per layer
  fp_tree_cluster_breakdown.csv        per-cluster lineage distribution
  cross_lineage_matches.csv            programmes with identical names across lineages
  fp_tree_donut.png                    layer distribution visual
  fp_tree_dendrogram.png               truncated dendrogram

GRACEFUL DEGRADATION
  If a lineage JSON is missing, it's skipped with a warning; analysis runs on
  whatever is available (useful while Mono Cell 6 is still annotating).
"""

import os
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "action_item_3_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

LINEAGE_FILES = {
    "B-cell":  "sae_annotations_hybrid.json",
    "T/NK":    "sae_annotations_tcnk_routeB.json",
    "DC":      "sae_annotations_dc_routeB.json",
    "Mono":    "sae_annotations_mono_routeB.json",
}

PRIMARY_K          = 30   # main cluster cut
K_VALUES           = [10, 15, 20, 30, 50]
UBIQ_THRESHOLD     = 3    # cluster spans >=N lineages -> Ubiquitous candidate
UBIQ_DOMINANCE_MAX = 0.70 # v2: if any single lineage > 70% of cluster, override to lineage-specific
NAME_WEIGHT        = 3    # programme_name repetition count in TF-IDF doc
GENE_WEIGHT        = 2    # gene list repetition count in TF-IDF doc

# -------------------------------------------------------------------------
# Step 1: Load all lineage annotations
# -------------------------------------------------------------------------
def normalise_key_evidence(ev_list):
    """Extract gene symbols from key_evidence (handles str and dict formats)."""
    genes = []
    for ev in ev_list[:5]:
        if isinstance(ev, str):
            genes.append(ev.split(":")[0].strip())
        elif isinstance(ev, dict):
            g = ev.get("gene", "") or ev.get("symbol", "")
            if g:
                genes.append(g)
    return [g for g in genes if g]

print("=" * 60)
print("Step 1: Loading annotations from all lineages")
print("=" * 60)

failure_names = {"api call failed", "annotation parse error", ""}
all_records   = []

for lineage, fname in LINEAGE_FILES.items():
    path = os.path.join(SCRIPT_DIR, fname)
    if not os.path.exists(path):
        print(f"  SKIP {lineage:<8}: file not found ({fname})")
        continue

    with open(path) as fh:
        records = json.load(fh)

    # Handle dict-vs-list format variation
    if isinstance(records, dict):
        records = list(records.values())

    n_kept = 0
    n_dropped = 0
    for r in records:
        name = r.get("programme_name", "")
        conf = r.get("confidence", "LOW")
        if name in failure_names or conf == "LOW":
            n_dropped += 1
            continue

        all_records.append({
            "lineage":                lineage,
            "feature_id":             r.get("feature_id"),
            "programme_name":         name,
            "functional_description": r.get("functional_description", ""),
            "key_evidence":           r.get("key_evidence", []),
            "top_genes_extracted":    normalise_key_evidence(r.get("key_evidence", [])),
            "confidence":             conf,
            "celltype_composition":   r.get("celltype_composition", {}),
            "anova_rank":             r.get("anova_rank"),
            "f_stat":                 r.get("f_stat"),
        })
        n_kept += 1

    print(f"  {lineage:<8}: {n_kept:>4} kept, {n_dropped:>3} dropped (failed/LOW)")

print(f"\nTotal annotations across {len(set(r['lineage'] for r in all_records))} lineages: {len(all_records)}")

if len(all_records) < 50:
    raise RuntimeError(
        f"Only {len(all_records)} programmes loaded; need >=50 for meaningful clustering. "
        "Check that the annotation JSON files exist and are populated."
    )

lineage_counts = Counter(r["lineage"] for r in all_records)
for lineage, n in lineage_counts.most_common():
    print(f"  {lineage:<8}: {n}")

# -------------------------------------------------------------------------
# Step 2: Build TF-IDF text documents
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Building TF-IDF text documents")
print("=" * 60)

def make_text(r):
    """Programme name (3x) + description (1x) + top gene symbols (2x)."""
    name_repeated  = (r["programme_name"] + " ") * NAME_WEIGHT
    desc           = r["functional_description"]
    genes_repeated = (" ".join(r["top_genes_extracted"]) + " ") * GENE_WEIGHT
    return f"{name_repeated}{desc} {genes_repeated}"

texts = [make_text(r) for r in all_records]
print(f"  Sample text [0]: {texts[0][:200]}...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.7,
    stop_words="english",
)
tfidf_matrix = vectorizer.fit_transform(texts)
print(f"  TF-IDF shape: {tfidf_matrix.shape}  ({tfidf_matrix.shape[1]} unique terms)")

# -------------------------------------------------------------------------
# Step 3: Hierarchical clustering
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Hierarchical clustering (Ward linkage on cosine distance)")
print("=" * 60)

sim_matrix  = cosine_similarity(tfidf_matrix).astype(np.float32)
dist_matrix = 1.0 - sim_matrix
np.fill_diagonal(dist_matrix, 0.0)
condensed   = squareform(dist_matrix, checks=False)
Z           = linkage(condensed, method="ward")
print(f"  Linkage matrix shape: {Z.shape}")

# Multi-k clustering
for k in K_VALUES:
    labels = fcluster(Z, k, criterion="maxclust")
    for r, l in zip(all_records, labels):
        r[f"cluster_k{k}"] = int(l)
    print(f"  k={k:>2}: {len(set(labels))} clusters")

# -------------------------------------------------------------------------
# Step 4: Assign FP-Tree layer (Ubiquitous vs lineage-specific)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Step 4: Layer assignment (PRIMARY_K={PRIMARY_K}, UBIQ_THRESHOLD>={UBIQ_THRESHOLD} lineages)")
print("=" * 60)

cluster_layers = {}
cluster_dominant_lineage = {}
cluster_dominance_pct = {}
for cid in set(r[f"cluster_k{PRIMARY_K}"] for r in all_records):
    cluster_records  = [r for r in all_records if r[f"cluster_k{PRIMARY_K}"] == cid]
    lineages_in      = set(r["lineage"] for r in cluster_records)
    lineage_counter  = Counter(r["lineage"] for r in cluster_records)
    dominant, dom_n  = lineage_counter.most_common(1)[0]
    dom_pct          = dom_n / len(cluster_records)

    # v2 layer assignment:
    # 1) Need >= UBIQ_THRESHOLD lineages to be Ubiquitous candidate
    # 2) If dominant lineage > UBIQ_DOMINANCE_MAX (70%), override to lineage-specific
    if len(lineages_in) >= UBIQ_THRESHOLD and dom_pct <= UBIQ_DOMINANCE_MAX:
        cluster_layers[cid] = "Ubiquitous"
    else:
        cluster_layers[cid] = f"{dominant}-lineage"
    cluster_dominant_lineage[cid] = dominant
    cluster_dominance_pct[cid] = dom_pct

for r in all_records:
    cid = r[f"cluster_k{PRIMARY_K}"]
    r["fp_tree_layer"] = cluster_layers[cid]
    r["fp_tree_cluster_id"] = cid

# -------------------------------------------------------------------------
# Step 5: Direct cross-lineage name matches (bonus signal)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Direct programme_name matches across lineages")
print("=" * 60)

name_to_lineages = {}
for r in all_records:
    name = r["programme_name"]
    name_to_lineages.setdefault(name, []).append(r["lineage"])

cross_matches = [
    {"programme_name": name, "lineages": sorted(set(lineages)),
     "occurrences": len(lineages), "n_lineages": len(set(lineages))}
    for name, lineages in name_to_lineages.items()
    if len(set(lineages)) >= 2
]
cross_matches.sort(key=lambda x: -x["n_lineages"])
print(f"  Programme names appearing in >=2 lineages: {len(cross_matches)}")
for cm in cross_matches[:10]:
    print(f"    {cm['n_lineages']} lineages | occ={cm['occurrences']:>2} | {cm['programme_name']}")

# -------------------------------------------------------------------------
# Step 6: Save CSV outputs
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Saving outputs")
print("=" * 60)

# Main assignment CSV
df = pd.DataFrame(all_records)
main_cols = ["lineage", "feature_id", "programme_name", "confidence",
             "fp_tree_layer", "fp_tree_cluster_id"]
for k in K_VALUES:
    if k != PRIMARY_K:
        main_cols.append(f"cluster_k{k}")
df_main = df[main_cols].sort_values(["fp_tree_layer", "fp_tree_cluster_id", "lineage"])
main_csv = os.path.join(OUT_DIR, "fp_tree_assignments.csv")
df_main.to_csv(main_csv, index=False)
print(f"  {main_csv}")

# Layer summary
layer_counts = Counter(r["fp_tree_layer"] for r in all_records)
df_layer = pd.DataFrame([
    {"layer": l, "n_programmes": n}
    for l, n in sorted(layer_counts.items(), key=lambda x: -x[1])
])
layer_csv = os.path.join(OUT_DIR, "fp_tree_layer_summary.csv")
df_layer.to_csv(layer_csv, index=False)
print(f"  {layer_csv}")

# Per-cluster breakdown
breakdown_rows = []
for cid in sorted(set(r[f"cluster_k{PRIMARY_K}"] for r in all_records)):
    cluster_records = [r for r in all_records if r[f"cluster_k{PRIMARY_K}"] == cid]
    lineage_counter = Counter(r["lineage"] for r in cluster_records)
    sample_names    = list({r["programme_name"] for r in cluster_records})[:3]
    breakdown_rows.append({
        "cluster_id":         cid,
        "layer":              cluster_layers[cid],
        "n_programmes":       len(cluster_records),
        "n_lineages":         len(lineage_counter),
        "dominant_pct":       round(cluster_dominance_pct[cid] * 100, 1),
        "lineage_distribution": "; ".join(f"{l}={n}" for l, n in lineage_counter.most_common()),
        "sample_programmes":  " | ".join(sample_names),
    })
df_break = pd.DataFrame(breakdown_rows).sort_values(["layer", "n_programmes"], ascending=[True, False])
break_csv = os.path.join(OUT_DIR, "fp_tree_cluster_breakdown.csv")
df_break.to_csv(break_csv, index=False)
print(f"  {break_csv}")

# Cross-lineage matches
df_cross = pd.DataFrame([
    {"programme_name": cm["programme_name"],
     "n_lineages":     cm["n_lineages"],
     "occurrences":    cm["occurrences"],
     "lineages":       ", ".join(cm["lineages"])}
    for cm in cross_matches
])
cross_csv = os.path.join(OUT_DIR, "cross_lineage_matches.csv")
df_cross.to_csv(cross_csv, index=False)
print(f"  {cross_csv}")

# -------------------------------------------------------------------------
# Step 7: Visualizations
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 7: Visualizations")
print("=" * 60)

# Donut chart
fig, ax = plt.subplots(figsize=(8, 8))
layers_sorted = sorted(layer_counts.keys())
sizes  = [layer_counts[l] for l in layers_sorted]
labels = [f"{l}\n(n={layer_counts[l]})" for l in layers_sorted]
colors = plt.cm.Set2(np.linspace(0, 0.95, len(layers_sorted)))
ax.pie(sizes, labels=labels, colors=colors, wedgeprops={"width": 0.4},
       startangle=90, textprops={"fontsize": 11})
ax.set_title(
    f"Cross-lineage FP-Tree: layer distribution (k={PRIMARY_K}, total={len(all_records)} programmes)",
    fontsize=13, pad=20,
)
donut_path = os.path.join(OUT_DIR, "fp_tree_donut.png")
plt.savefig(donut_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  {donut_path}")

# Dendrogram
fig, ax = plt.subplots(figsize=(22, 9))
dendrogram(Z, truncate_mode="lastp", p=PRIMARY_K, ax=ax,
           leaf_rotation=90, leaf_font_size=8, color_threshold=0)
ax.set_title(
    f"Cross-lineage FP-Tree dendrogram (truncated to {PRIMARY_K} cluster leaves)",
    fontsize=14,
)
ax.set_xlabel("Cluster leaf (# programmes shown in parentheses)")
ax.set_ylabel("Ward linkage distance")
dend_path = os.path.join(OUT_DIR, "fp_tree_dendrogram.png")
plt.savefig(dend_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  {dend_path}")

# -------------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nTotal programmes analysed: {len(all_records)}")
print(f"Per-lineage:")
for l, n in lineage_counts.most_common():
    print(f"  {l:<8} {n:>4}")
print(f"\nFP-Tree layer distribution (k={PRIMARY_K}):")
for l, n in sorted(layer_counts.items(), key=lambda x: -x[1]):
    pct = 100 * n / len(all_records)
    print(f"  {l:<25} {n:>4}  ({pct:.1f}%)")
print(f"\nCross-lineage name matches (>=2 lineages): {len(cross_matches)}")
print(f"\nOutputs in: {OUT_DIR}")
print("Ready for Notion / Dr. Ng presentation.")
