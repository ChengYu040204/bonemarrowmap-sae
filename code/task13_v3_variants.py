"""
task13_v3_variants.py
=====================
v3: Test stricter Ubiquitous definitions per Dr. Ng's request (next meeting feedback).

Reuses the SAME TF-IDF clustering as v2 (k=30), but applies 4 different layer
assignment rules and prints a comparison table. Saves Ubiquitous cluster details
for each variant so you can inspect which version keeps the cleanest set.

VARIANTS COMPARED
  v2_baseline : >=3 lineages AND dominance <= 70%   (current v2, 16.1% Ubiquitous)
  v3a_dom60   : >=3 lineages AND dominance <= 60%   (tighter dominance only)
  v3b_thresh4 : >=4 lineages AND dominance <= 70%   (must span all 4 lineages)
  v3c_strict  : >=4 lineages AND dominance <= 60%   (combination, strictest)

OUTPUTS (under sae_code/action_item_3_outputs/)
  v3_variant_comparison.csv        Side-by-side layer distribution for all 4 variants
  v3_ubiquitous_clusters.csv       For each variant, the Ubiquitous clusters with details

USAGE
  cd sae_code/
  /opt/miniconda3/envs/scanpy-env/bin/python task13_v3_variants.py
"""

import os
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# -------------------------------------------------------------------------
# Config (mirrors task13 v2)
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

PRIMARY_K   = 30
NAME_WEIGHT = 3
GENE_WEIGHT = 2

VARIANTS = [
    {"name": "v2_baseline",  "threshold": 3, "dominance": 0.70},
    {"name": "v3a_dom60",    "threshold": 3, "dominance": 0.60},
    {"name": "v3b_thresh4",  "threshold": 4, "dominance": 0.70},
    {"name": "v3c_strict",   "threshold": 4, "dominance": 0.60},
]

# -------------------------------------------------------------------------
# Step 1: Load all lineage annotations (same logic as task13)
# -------------------------------------------------------------------------
def normalise_key_evidence(ev_list):
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
print("Step 1: Loading annotations from all 4 lineages")
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

    if isinstance(records, dict):
        records = list(records.values())

    n_kept = 0
    for r in records:
        name = r.get("programme_name", "")
        conf = r.get("confidence", "LOW")
        if name in failure_names or conf == "LOW":
            continue
        all_records.append({
            "lineage":                lineage,
            "feature_id":             r.get("feature_id"),
            "programme_name":         name,
            "functional_description": r.get("functional_description", ""),
            "top_genes_extracted":    normalise_key_evidence(r.get("key_evidence", [])),
        })
        n_kept += 1
    print(f"  {lineage:<8}: {n_kept:>4} kept")

print(f"\nTotal: {len(all_records)} programmes")

# -------------------------------------------------------------------------
# Step 2: TF-IDF + Ward hierarchical clustering (same as task13 v2)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Step 2: TF-IDF clustering (k={PRIMARY_K})")
print("=" * 60)

def make_text(r):
    name_repeated  = (r["programme_name"] + " ") * NAME_WEIGHT
    desc           = r["functional_description"]
    genes_repeated = (" ".join(r["top_genes_extracted"]) + " ") * GENE_WEIGHT
    return f"{name_repeated}{desc} {genes_repeated}"

texts = [make_text(r) for r in all_records]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.7, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)

sim_matrix  = cosine_similarity(tfidf_matrix).astype(np.float32)
dist_matrix = 1.0 - sim_matrix
np.fill_diagonal(dist_matrix, 0.0)
condensed   = squareform(dist_matrix, checks=False)
Z           = linkage(condensed, method="ward")
labels      = fcluster(Z, PRIMARY_K, criterion="maxclust")

for r, l in zip(all_records, labels):
    r["cluster"] = int(l)

print(f"  TF-IDF: {tfidf_matrix.shape}")
print(f"  Generated {PRIMARY_K} clusters")

# -------------------------------------------------------------------------
# Step 3: Apply each variant's layer assignment rule
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Layer assignment under each variant")
print("=" * 60)

cluster_ids = sorted(set(r["cluster"] for r in all_records))

# Precompute cluster stats once
cluster_stats = {}
for cid in cluster_ids:
    cluster_records = [r for r in all_records if r["cluster"] == cid]
    lineage_counter = Counter(r["lineage"] for r in cluster_records)
    dominant, dom_n = lineage_counter.most_common(1)[0]
    dom_pct = dom_n / len(cluster_records)
    sample_names = list({r["programme_name"] for r in cluster_records})[:3]
    cluster_stats[cid] = {
        "n_programmes":         len(cluster_records),
        "n_lineages":           len(lineage_counter),
        "dominant_lineage":     dominant,
        "dominant_pct":         dom_pct,
        "lineage_distribution": "; ".join(f"{l}={n}" for l, n in lineage_counter.most_common()),
        "sample_programmes":    " | ".join(sample_names),
    }

# Apply each variant
variant_assignments = {}
for v in VARIANTS:
    cluster_layers = {}
    for cid in cluster_ids:
        st = cluster_stats[cid]
        if st["n_lineages"] >= v["threshold"] and st["dominant_pct"] <= v["dominance"]:
            cluster_layers[cid] = "Ubiquitous"
        else:
            cluster_layers[cid] = f"{st['dominant_lineage']}-lineage"
    variant_assignments[v["name"]] = cluster_layers

# -------------------------------------------------------------------------
# Step 4: Build comparison table
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Variant comparison")
print("=" * 60)

n_total = len(all_records)
comparison_rows = []
for v in VARIANTS:
    layers      = variant_assignments[v["name"]]
    layer_counts = Counter(layers[r["cluster"]] for r in all_records)
    cluster_layer_counts = Counter(layers.values())  # by cluster, not programme

    ubiq_prog = layer_counts.get("Ubiquitous", 0)
    row = {
        "variant":           v["name"],
        "threshold (>=N)":   v["threshold"],
        "dominance (<=)":    f"{v['dominance']*100:.0f}%",
        "Ubiq programmes":   ubiq_prog,
        "Ubiq %":            round(100 * ubiq_prog / n_total, 1),
        "Ubiq clusters":     cluster_layer_counts.get("Ubiquitous", 0),
        "Mono-lineage":      layer_counts.get("Mono-lineage", 0),
        "B-cell-lineage":    layer_counts.get("B-cell-lineage", 0),
        "T/NK-lineage":      layer_counts.get("T/NK-lineage", 0),
        "DC-lineage":        layer_counts.get("DC-lineage", 0),
    }
    comparison_rows.append(row)

df_compare = pd.DataFrame(comparison_rows)
print("\n" + df_compare.to_string(index=False))

comp_path = os.path.join(OUT_DIR, "v3_variant_comparison.csv")
df_compare.to_csv(comp_path, index=False)
print(f"\nSaved: {comp_path}")

# -------------------------------------------------------------------------
# Step 5: Save Ubiquitous cluster details per variant
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Ubiquitous cluster details per variant")
print("=" * 60)

detail_rows = []
for v in VARIANTS:
    layers = variant_assignments[v["name"]]
    for cid in cluster_ids:
        if layers[cid] == "Ubiquitous":
            st = cluster_stats[cid]
            detail_rows.append({
                "variant":              v["name"],
                "cluster_id":           cid,
                "n_programmes":         st["n_programmes"],
                "n_lineages":           st["n_lineages"],
                "dominant_pct":         round(st["dominant_pct"] * 100, 1),
                "lineage_distribution": st["lineage_distribution"],
                "sample_programmes":    st["sample_programmes"],
            })

df_detail = pd.DataFrame(detail_rows)
detail_path = os.path.join(OUT_DIR, "v3_ubiquitous_clusters.csv")
df_detail.to_csv(detail_path, index=False)
print(f"  Saved: {detail_path}")

# Quick inline preview: print each variant's Ubiquitous clusters
for v in VARIANTS:
    rows = [r for r in detail_rows if r["variant"] == v["name"]]
    print(f"\n  --- {v['name']} ({len(rows)} Ubiquitous clusters) ---")
    for r in rows:
        print(f"    cluster {r['cluster_id']:>2}: n={r['n_programmes']:>3}, "
              f"lineages={r['n_lineages']}, dom={r['dominant_pct']:>4.1f}% "
              f"| {r['lineage_distribution']}")
        print(f"      e.g.: {r['sample_programmes'][:90]}")

print(f"\nDone. Inspect {comp_path} and {detail_path} to pick the best variant for Dr. Ng.")
