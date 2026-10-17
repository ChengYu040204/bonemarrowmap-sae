#!/usr/bin/env python3
"""
Post-run analysis: compare t-test RAG annotations vs Pearson RAG annotations.
Run this immediately after task7c_ttest_annotation.py completes.

Usage (no special env needed, just numpy/pandas/json):
  python3 compare_ttest_results.py
"""

import json, numpy as np, pandas as pd
from collections import Counter

# ============================================================
# Load all annotation sources
# ============================================================
with open("sae_annotations.json") as f:
    ann_zeroshot = {a["feature_id"]: a for a in json.load(f)}

with open("sae_annotations_rag.json") as f:
    ann_pearson = {a["feature_id"]: a for a in json.load(f)}

with open("sae_annotations_rag_ttest.json") as f:
    ann_ttest_list = json.load(f)
    ann_ttest = {a["feature_id"]: a for a in ann_ttest_list}

with open("gene_signatures.json") as f:
    raw_p = json.load(f)
    pearson_sigs = {int(k): set(v["genes"]) for k, v in raw_p.items()}

with open("gene_signatures_ttest.json") as f:
    raw_t = json.load(f)
    ttest_sigs = {int(k): set(v["genes"]) for k, v in raw_t.items()}

df_clusters = pd.read_csv("task6_cluster_assignments.csv")

print("=" * 65)
print("T-TEST vs PEARSON RAG ANNOTATION COMPARISON")
print("=" * 65)

# ============================================================
# 1. Overall uniqueness: how many distinct programme names?
# ============================================================
print("\n[1] UNIQUE PROGRAMME NAMES (all 209 active features)")
print(f"  Zero-shot (Pearson genes): {len(set(a['programme_name'] for a in ann_zeroshot.values()))}")
print(f"  RAG Pearson:               {len(set(a['programme_name'] for a in ann_pearson.values()))}")
print(f"  RAG t-test (new):          {len(set(a['programme_name'] for a in ann_ttest.values()))}")

# ============================================================
# 2. Cluster 2 deep-dive: was the plasma cell collapse fixed?
# ============================================================
c2_fids = df_clusters[df_clusters["cluster"] == 2]["feature_id"].tolist()

c2_names_pearson = [ann_pearson.get(fid, {}).get("programme_name", "") for fid in c2_fids]
c2_names_ttest   = [ann_ttest.get(fid,   {}).get("programme_name", "") for fid in c2_fids]

print(f"\n[2] CLUSTER 2 (Plasma Cell, n={len(c2_fids)}) — Name Diversity")
print(f"  RAG Pearson unique names:  {len(set(c2_names_pearson))}")
print(f"  RAG t-test  unique names:  {len(set(c2_names_ttest))}")

print("\n  t-test name distribution (Cluster 2):")
for name, cnt in Counter(c2_names_ttest).most_common():
    print(f"    [{cnt:2d}x] {name}")

# ============================================================
# 3. Gene list diversity: did t-test reduce Jaccard overlap?
# ============================================================
def mean_jaccard(sigs, fids):
    pairs, vals = 0, []
    fids = [f for f in fids if f in sigs]
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            a, b = sigs[fids[i]], sigs[fids[j]]
            if a or b:
                vals.append(len(a & b) / len(a | b))
                pairs += 1
    return np.mean(vals) if vals else 0, pairs

p_jac, p_n = mean_jaccard(pearson_sigs, c2_fids)
t_jac, t_n = mean_jaccard(ttest_sigs,   c2_fids)

print(f"\n[3] GENE LIST DIVERSITY — Cluster 2 (pairwise Jaccard)")
print(f"  Pearson mean Jaccard: {p_jac:.3f}  ({p_n} pairs)")
print(f"  t-test  mean Jaccard: {t_jac:.3f}  ({t_n} pairs)")
print(f"  Reduction: {(p_jac - t_jac) / p_jac * 100:.1f}% lower overlap (higher = better diversity)")

# ============================================================
# 4. ANOVA top 10 side-by-side
# ============================================================
anova_fids = sorted(ann_zeroshot.keys(),
                    key=lambda f: ann_zeroshot[f].get("anova_rank") or 999)[:10]

print("\n[4] ANOVA TOP 10 — Three-Way Comparison")
print(f"  {'Rank':>4} {'F#':>5}  {'Zero-shot':<35} {'RAG Pearson':<35} {'RAG t-test':<35}")
print("  " + "-" * 115)
for rank, fid in enumerate(anova_fids, 1):
    z = ann_zeroshot.get(fid, {}).get("programme_name", "")[:33]
    p = ann_pearson.get(fid,   {}).get("programme_name", "")[:33]
    t = ann_ttest.get(fid,     {}).get("programme_name", "")[:33]
    print(f"  {rank:>4} F#{fid:<4}  {z:<35} {p:<35} {t:<35}")

# ============================================================
# 5. Cluster-level summary
# ============================================================
print("\n[5] UNIQUE NAMES PER CLUSTER (t-test vs Pearson)")
print(f"  {'Cluster':>8} {'n':>4}  {'Pearson unique':>14}  {'t-test unique':>13}  {'Change':>8}")
for c in sorted(df_clusters["cluster"].unique()):
    fids = df_clusters[df_clusters["cluster"] == c]["feature_id"].tolist()
    n_p = len(set(ann_pearson.get(f, {}).get("programme_name", "") for f in fids))
    n_t = len(set(ann_ttest.get(f,   {}).get("programme_name", "") for f in fids))
    diff = n_t - n_p
    arrow = f"+{diff}" if diff > 0 else str(diff)
    print(f"  {'C'+str(c):>8} {len(fids):>4}  {n_p:>14}  {n_t:>13}  {arrow:>8}")

print("\nDone.")
