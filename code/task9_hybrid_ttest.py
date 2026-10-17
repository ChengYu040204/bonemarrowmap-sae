#!/usr/bin/env python3
"""
Task 9: Hybrid Background t-test Annotation
============================================

Problem diagnosed across Tasks 7-8:
  No single background strategy works optimally for all 8 clusters.

  Global background (task7c):
    Good for erythroid clusters 3/4/5: +18 unique names vs Pearson
    Bad for Cluster 2: Jaccard 0.592 (all plasma cell features identical)

  B-cell restricted background (task8):
    Good for Cluster 2: 34/41 unique names (vs 25 with global)
    Bad for other clusters: total unique names drop 172 -> 140

Solution: Cluster-specific background selection

  Cluster 2 (plasma cell, 41 features):
    Background: B-cell lineage only (~17,800 cells)
    Gene sigs: reused from gene_signatures_bcell_ttest.json (task8 cache)

  All other clusters (1, 3, 4, 5, 6, 7, 8):
    Background: 2000 random cells from full 161K
    Gene sigs: computed fresh (global t-test)

Expected result:
  Total unique names: ~185+ (best of both strategies combined)
  Cluster 2 unique names: 34/41 (preserved from task8)
  Erythroid clusters: restored to task7c quality

Usage:
  cd "Papers & Datas/SAE training/sae_code/"
  export ANTHROPIC_API_KEY="sk-ant-..."
  /opt/miniconda3/envs/scanpy-env/bin/python task9_hybrid_ttest.py

Outputs:
  gene_signatures_hybrid.json     per-feature gene signatures (merged cache)
  gene_function_cache.json        MyGene.info cache (extended in-place)
  sae_annotations_hybrid.json     final RAG annotations

Author: SAE x Haematopoiesis project, UCL Gurdon Studentship 2026
"""

import os
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import mygene
import anthropic
import scanpy as sc

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =============================================================================
# CONFIGURATION
# =============================================================================

ADATA_PATH       = "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad"
SAE_PKL          = "sae_results_all.pkl"
ANN_IN           = "sae_annotations.json"
CLUSTER_CSV      = "task6_cluster_assignments.csv"

# Cluster 2 (plasma cell) signatures already in task8 cache: reuse directly
BCELL_SIG_CACHE  = "gene_signatures_bcell_ttest.json"

# This script's own merged signature cache
HYBRID_SIG_FILE  = "gene_signatures_hybrid.json"

GENE_CACHE       = "gene_function_cache.json"
OUTPUT           = "sae_annotations_hybrid.json"

API_KEY          = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL            = "claude-haiku-4-5-20251001"

TOP_CELLS        = 200     # foreground cells per feature
TOP_GENES        = 15      # genes sent to LLM per feature
N_GLOBAL_BG      = 2000    # random background size for non-Cluster-2 features

PLASMA_CLUSTER   = 2       # cluster ID that gets B-cell restricted background

B_CELL_TYPES = [
    "HSC", "LMPP", "MLP", "MLP-II", "CLP",
    "Pro-B Cycling", "Pro-B VDJ", "Large Pre-B",
    "Small Pre-B", "Immature B", "Mature B", "Plasma Cell",
]

MAX_FEATURES     = None    # None = all 209; set integer for quick test
SLEEP_API        = 1.2
SLEEP_MYGENE     = 1.0


# =============================================================================
# STEP 1: Load SAE feature matrix
# =============================================================================
print("\n[1/7] Loading SAE features from pkl ...", flush=True)

with open(SAE_PKL, "rb") as fh:
    d = pickle.load(fh)

raw = d[1]
if hasattr(raw, "detach"):
    X_sae = raw.detach().cpu().numpy().astype(np.float32)
else:
    X_sae = np.array(raw, dtype=np.float32)

N_CELLS_TOTAL, N_FEATURES = X_sae.shape
print(f"      X_sae shape: {X_sae.shape}", flush=True)


# =============================================================================
# STEP 2: Load cluster assignments and prior annotations
# =============================================================================
print("[2/7] Loading cluster assignments ...", flush=True)

df_clusters = pd.read_csv(CLUSTER_CSV)
# df_clusters columns: feature_id, programme_name, cluster

if MAX_FEATURES is not None:
    df_clusters = df_clusters.head(MAX_FEATURES)
    print(f"      TEST MODE: limiting to {MAX_FEATURES} features", flush=True)

active_fids    = df_clusters["feature_id"].tolist()
fid_to_cluster = dict(zip(df_clusters["feature_id"], df_clusters["cluster"]))

c2_fids    = [f for f in active_fids if fid_to_cluster[f] == PLASMA_CLUSTER]
other_fids = [f for f in active_fids if fid_to_cluster[f] != PLASMA_CLUSTER]

print(f"      Total active features: {len(active_fids)}", flush=True)
print(f"      Cluster {PLASMA_CLUSTER} (B-cell restricted bg): {len(c2_fids)} features", flush=True)
print(f"      Other clusters (global bg):                      {len(other_fids)} features", flush=True)

with open(ANN_IN) as fh:
    old_annotations = json.load(fh)
fid_to_prior_name = {a["feature_id"]: a.get("programme_name", "unknown") for a in old_annotations}


# =============================================================================
# STEP 3: Load Cluster 2 signatures from task8 cache
# =============================================================================
print("[3/7] Loading Cluster 2 signatures from task8 cache ...", flush=True)

gene_signatures  = {}   # fid -> list of gene symbols
ct_compositions  = {}   # fid -> dict {celltype: pct}
sig_method       = {}   # fid -> "bcell_restricted" or "global"

if os.path.exists(BCELL_SIG_CACHE):
    with open(BCELL_SIG_CACHE) as fh:
        bcell_raw = json.load(fh)

    loaded_c2 = 0
    for fid in c2_fids:
        key = str(fid)
        if key in bcell_raw:
            gene_signatures[fid] = bcell_raw[key]["genes"]
            ct_compositions[fid] = bcell_raw[key].get("celltype_composition", {})
            sig_method[fid]      = "bcell_restricted"
            loaded_c2 += 1

    print(f"      Loaded {loaded_c2}/{len(c2_fids)} Cluster 2 signatures from cache", flush=True)
    missing_c2 = [f for f in c2_fids if f not in gene_signatures]
    if missing_c2:
        print(f"      WARNING: {len(missing_c2)} Cluster 2 features missing from cache: {missing_c2}", flush=True)
else:
    print(f"      WARNING: {BCELL_SIG_CACHE} not found. Cluster 2 will use global bg.", flush=True)
    missing_c2 = c2_fids


# =============================================================================
# STEP 4: Check hybrid cache for already-computed non-C2 signatures
# =============================================================================
print("[4/7] Checking hybrid signature cache for non-Cluster-2 features ...", flush=True)

if os.path.exists(HYBRID_SIG_FILE):
    with open(HYBRID_SIG_FILE) as fh:
        hybrid_raw = json.load(fh)
    for fid in other_fids:
        key = str(fid)
        if key in hybrid_raw:
            gene_signatures[fid] = hybrid_raw[key]["genes"]
            ct_compositions[fid] = hybrid_raw[key].get("celltype_composition", {})
            sig_method[fid]      = hybrid_raw[key].get("method", "global")
    print(f"      Loaded {len(hybrid_raw)} entries from {HYBRID_SIG_FILE}", flush=True)
else:
    hybrid_raw = {}

fids_to_compute = [
    f for f in (other_fids + (missing_c2 if 'missing_c2' in dir() else []))
    if f not in gene_signatures
]
print(f"      Non-Cluster-2 features needing global t-test: {len(fids_to_compute)}", flush=True)


# =============================================================================
# STEP 5: Compute global t-test signatures for non-Cluster-2 features
#
# For each feature f not in Cluster 2:
#   fg = top TOP_CELLS cells from ALL 161K by X_sae[:, f] activation
#   bg = N_GLOBAL_BG random cells from the remaining 161K (excluding fg)
#   t-stat per gene = Welch t, NaN -> -inf
#   Top TOP_GENES genes => feature's signature
#
# Why global background is better for non-Cluster-2:
#   Erythroid features (clusters 3/4/5): fg enriched for erythroid cells.
#   Global bg includes non-erythroid cells, making erythroid genes (ALAS2, HBB)
#   highly significant in t-test.
#   B-cell restricted bg would miss these entirely.
# =============================================================================

if fids_to_compute:
    print("[5/7] Computing global t-test signatures for non-Cluster-2 features ...", flush=True)
    print(f"      Loading AnnData (may take 30-60 seconds) ...", flush=True)

    adata       = sc.read_h5ad(ADATA_PATH)
    gene_names  = np.array(adata.var_names)
    cell_types  = adata.obs["CellType"].values.astype(str)
    X_raw       = adata.X
    del adata

    if hasattr(X_raw, "toarray"):
        X_genes = X_raw.toarray().astype(np.float32)
    else:
        X_genes = np.array(X_raw, dtype=np.float32)
    del X_raw

    assert X_genes.shape[0] == N_CELLS_TOTAL, (
        f"Cell count mismatch: AnnData {X_genes.shape[0]} vs SAE pkl {N_CELLS_TOTAL}"
    )
    print(f"      Gene matrix: {X_genes.shape}", flush=True)

    all_cell_indices = np.arange(N_CELLS_TOTAL)

    for i, fid in enumerate(fids_to_compute):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"      Global t-test: feature {i+1}/{len(fids_to_compute)} (fid={fid}) ...", flush=True)

        acts_f = X_sae[:, fid]

        # Foreground: top TOP_CELLS cells globally
        fg_idx = np.argsort(acts_f)[::-1][:TOP_CELLS]

        # Background: random N_GLOBAL_BG cells from the rest
        fg_set   = set(fg_idx.tolist())
        non_fg   = np.array([idx for idx in all_cell_indices if idx not in fg_set])
        bg_size  = min(N_GLOBAL_BG, len(non_fg))
        bg_idx   = np.random.choice(non_fg, size=bg_size, replace=False)

        fg_expr = X_genes[fg_idx]
        bg_expr = X_genes[bg_idx]

        n_fg = fg_expr.shape[0]
        n_bg = bg_expr.shape[0]

        m1 = fg_expr.mean(axis=0)
        m2 = bg_expr.mean(axis=0)
        v1 = fg_expr.var(axis=0) / n_fg
        v2 = bg_expr.var(axis=0) / n_bg

        t_stats = (m1 - m2) / np.sqrt(v1 + v2 + 1e-10)
        t_stats = np.where(np.isnan(t_stats) | np.isinf(t_stats), -np.inf, t_stats)

        top_gene_idx        = np.argsort(t_stats)[::-1][:TOP_GENES]
        gene_signatures[fid] = gene_names[top_gene_idx].tolist()
        sig_method[fid]      = "global"

        cts_fg = cell_types[fg_idx]
        ct_counts = pd.Series(cts_fg).value_counts(normalize=True)
        ct_compositions[fid] = {
            k: round(float(v) * 100, 1)
            for k, v in ct_counts.items()
            if float(v) >= 0.01
        }

    del X_genes
    print(f"      Global t-test complete for {len(fids_to_compute)} features", flush=True)

    # Save merged hybrid signatures to disk
    for fid in fids_to_compute:
        hybrid_raw[str(fid)] = {
            "genes":               gene_signatures[fid],
            "celltype_composition": ct_compositions[fid],
            "method":              sig_method[fid],
        }
    with open(HYBRID_SIG_FILE, "w") as fh:
        json.dump(hybrid_raw, fh, indent=2)
    print(f"      Saved updated hybrid signatures to {HYBRID_SIG_FILE}", flush=True)

else:
    print("[5/7] All signatures already cached. Skipping AnnData load.", flush=True)


# =============================================================================
# STEP 6: MyGene.info lookup for all gene functions
# =============================================================================
print("[6/7] Querying MyGene.info for gene functions ...", flush=True)

if os.path.exists(GENE_CACHE):
    with open(GENE_CACHE) as fh:
        gene_cache = json.load(fh)
    print(f"      Loaded {len(gene_cache)} cached entries", flush=True)
else:
    gene_cache = {}

all_genes_needed = set()
for fid in active_fids:
    if fid in gene_signatures:
        all_genes_needed.update(gene_signatures[fid])

uncached = sorted(all_genes_needed - set(gene_cache.keys()))
print(f"      Unique genes needed: {len(all_genes_needed)} | New lookups: {len(uncached)}", flush=True)

if uncached:
    mg         = mygene.MyGeneInfo()
    chunk_size = 500
    n_batches  = (len(uncached) - 1) // chunk_size + 1

    for batch_i in range(n_batches):
        chunk = uncached[batch_i * chunk_size : (batch_i + 1) * chunk_size]
        print(f"      MyGene batch {batch_i+1}/{n_batches} ({len(chunk)} genes) ...", flush=True)
        results = mg.querymany(
            chunk, scopes="symbol", fields="summary,name,symbol",
            species="human", returnall=True,
        )
        for r in results["out"]:
            sym = r.get("query", "")
            gene_cache[sym] = {
                "name":    r.get("name", sym),
                "summary": "" if r.get("notfound") else r.get("summary", ""),
            }
        time.sleep(SLEEP_MYGENE)

    with open(GENE_CACHE, "w") as fh:
        json.dump(gene_cache, fh, indent=2)
    print(f"      Cache updated: {len(gene_cache)} genes total", flush=True)


# =============================================================================
# STEP 7: RAG-enhanced LLM annotation
#
# System prompt signals to Claude WHICH background was used per feature,
# so it can calibrate its interpretation appropriately:
#   - B-cell restricted: "genes distinguish this B-cell subtype from other B cells"
#   - Global: "genes distinguish this cell state from the full haematopoietic atlas"
# =============================================================================
print("[7/7] Running RAG-enhanced LLM annotation ...", flush=True)

if not API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not set.\n"
        "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
        "  /opt/miniconda3/envs/scanpy-env/bin/python task9_hybrid_ttest.py"
    )

client = anthropic.Anthropic(api_key=API_KEY)

SYSTEM_PROMPT_BCELL = (
    "You are an expert haematologist and computational biologist specialising in "
    "haematopoietic differentiation and B-cell development. "
    "You are annotating features from a Sparse Autoencoder (SAE) trained on scVI latent "
    "representations of the BoneMarrowMap human bone marrow atlas (~160K cells). "
    "Each feature represents a candidate Functional Programme (FP).\n\n"
    "BACKGROUND FOR THIS FEATURE: B-cell restricted (top-200 B cells vs ~17,800 B-cell background). "
    "The genes below are specifically upregulated within B-cell subtypes relative to other B cells. "
    "For plasma cell features: distinguish IgKappa vs IgLambda vs IgG vs IgA isotypes precisely. "
    "Distinguish UPR initiation vs ER chaperone folding vs ERAD vs mTORC1-driven translation "
    "vs bone marrow survival signalling. "
    "Do NOT use generic labels like 'Plasma cell immunoglobulin secretion programme'. "
    "Name the specific mechanism (e.g. 'IgKappa light chain ER quality control', "
    "'XBP1-driven ER chaperone induction', 'ERAD-mediated misfolded Ig clearance').\n\n"
    "Respond ONLY as valid JSON:\n"
    "{\n"
    '  "programme_name": "concise mechanistic name maximum 7 words",\n'
    '  "functional_description": "2-3 sentences on the specific molecular mechanism",\n'
    '  "key_evidence": ["GENE1: why this gene is diagnostic", "GENE2: ...", "GENE3: ..."],\n'
    '  "confidence": "HIGH | MEDIUM | LOW",\n'
    '  "suggested_validation": "one specific experiment"\n'
    "}"
)

SYSTEM_PROMPT_GLOBAL = (
    "You are an expert haematologist and computational biologist specialising in "
    "haematopoietic differentiation. "
    "You are annotating features from a Sparse Autoencoder (SAE) trained on scVI latent "
    "representations of the BoneMarrowMap human bone marrow atlas (~160K cells). "
    "Each feature represents a candidate Functional Programme (FP).\n\n"
    "BACKGROUND FOR THIS FEATURE: Global (top-200 activating cells vs 2000 random cells "
    "from the full 160K haematopoietic atlas). "
    "The genes below are specifically upregulated in cells that activate this feature "
    "relative to the full haematopoietic spectrum. "
    "Use the cell-type composition and gene functions to identify the precise programme: "
    "erythroid differentiation stage, proliferation subtype, lineage identity, "
    "or cross-lineage metabolic programme. Be specific about the differentiation stage.\n\n"
    "Respond ONLY as valid JSON:\n"
    "{\n"
    '  "programme_name": "concise mechanistic name maximum 7 words",\n'
    '  "functional_description": "2-3 sentences on the specific molecular mechanism",\n'
    '  "key_evidence": ["GENE1: why this gene is diagnostic", "GENE2: ...", "GENE3: ..."],\n'
    '  "confidence": "HIGH | MEDIUM | LOW",\n'
    '  "suggested_validation": "one specific experiment"\n'
    "}"
)


def build_prompt(fid, cluster, prior_name, ct_comp, genes_with_context):
    celltype_lines = "\n".join(
        f"  {pct:5.1f}%  {ct}" for ct, pct in sorted(ct_comp.items(), key=lambda x: -x[1])
    ) or "  (no cell-type breakdown available)"
    gene_lines = "\n".join(
        f"  {j+1:2d}. {entry}" for j, entry in enumerate(genes_with_context)
    )
    method_label = sig_method.get(fid, "unknown")
    return (
        f"Feature ID: {fid}\n"
        f"Cluster: {cluster} | Background method: {method_label}\n"
        f"Prior zero-shot name: \"{prior_name}\"\n\n"
        f"Cell-type composition of top {TOP_CELLS} activating cells:\n"
        f"{celltype_lines}\n\n"
        f"Top {len(genes_with_context)} genes (t-test ranked, with NCBI Gene context):\n"
        f"{gene_lines}\n"
    )


def parse_json_response(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        content = parts[1]
        if content.startswith("json"):
            content = content[4:]
        text = content.strip()
    return json.loads(text)


# Resume logic: skip features already successfully annotated
if os.path.exists(OUTPUT):
    with open(OUTPUT) as fh:
        results_list = json.load(fh)
    # Only skip features that were annotated successfully (not api call failures)
    done_fids = {
        r["feature_id"] for r in results_list
        if r.get("confidence", "LOW") != "LOW" or
           r.get("programme_name", "") not in ("api call failed", "annotation parse error", "")
    }
    print(f"      Resuming: {len(done_fids)} features already successfully annotated", flush=True)
    # Keep only successfully annotated records
    results_list = [r for r in results_list if r["feature_id"] in done_fids]
else:
    results_list = []
    done_fids    = set()

n_total   = len(active_fids)
n_skipped = 0
n_errors  = 0

for i, fid in enumerate(active_fids):
    if fid in done_fids:
        n_skipped += 1
        continue

    if fid not in gene_signatures:
        print(f"      WARNING: no signature for fid={fid}, skipping", flush=True)
        n_errors += 1
        continue

    genes      = gene_signatures[fid]
    ct_comp    = ct_compositions.get(fid, {})
    cluster    = fid_to_cluster.get(fid, "?")
    prior_name = fid_to_prior_name.get(fid, "unknown")
    method     = sig_method.get(fid, "global")

    # Select system prompt based on which background was used
    system_p = SYSTEM_PROMPT_BCELL if method == "bcell_restricted" else SYSTEM_PROMPT_GLOBAL

    genes_with_context = []
    for g in genes:
        entry   = gene_cache.get(g, {})
        summary = entry.get("summary", "").strip()
        if len(summary) > 120:
            summary = summary[:117] + "..."
        genes_with_context.append(
            f"{g}: {summary}" if summary else f"{g}: (no functional summary available)"
        )

    user_msg = build_prompt(fid, cluster, prior_name, ct_comp, genes_with_context)

    annotation = None
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=900,
                system=system_p,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text   = response.content[0].text
            annotation = parse_json_response(raw_text)
            break
        except json.JSONDecodeError as e:
            if attempt < 2:
                time.sleep(2)
            else:
                annotation = {
                    "programme_name":         "annotation parse error",
                    "functional_description": f"Raw: {raw_text[:200]}",
                    "key_evidence":           genes[:3],
                    "confidence":             "LOW",
                    "suggested_validation":   "",
                }
                n_errors += 1
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"      API failed fid={fid}: {e}", flush=True)
                annotation = {
                    "programme_name":         "api call failed",
                    "functional_description": str(e),
                    "key_evidence":           [],
                    "confidence":             "LOW",
                    "suggested_validation":   "",
                }
                n_errors += 1

    record = {
        "feature_id":             fid,
        "cluster":                cluster,
        "prior_zero_shot_name":   prior_name,
        "top_genes":              genes,
        "celltype_composition":   ct_comp,
        "programme_name":         annotation.get("programme_name", ""),
        "functional_description": annotation.get("functional_description", ""),
        "key_evidence":           annotation.get("key_evidence", []),
        "confidence":             annotation.get("confidence", "LOW"),
        "suggested_validation":   annotation.get("suggested_validation", ""),
        "method":                 method,
        "background":             (
            "B-cell lineage only (~17800 cells)" if method == "bcell_restricted"
            else f"Global random {N_GLOBAL_BG} from 161K"
        ),
    }
    results_list.append(record)
    done_fids.add(fid)

    features_done = i + 1 - n_skipped
    if features_done % 20 == 0 or features_done == 1:
        print(
            f"      [{features_done}/{n_total - n_skipped}] "
            f"fid={fid} cluster={cluster} [{method}] | "
            f"{annotation.get('programme_name', '')}",
            flush=True,
        )

    if features_done % 10 == 0:
        with open(OUTPUT, "w") as fh:
            json.dump(results_list, fh, indent=2)

    time.sleep(SLEEP_API)

with open(OUTPUT, "w") as fh:
    json.dump(results_list, fh, indent=2)

print(
    f"\nDone.\n"
    f"  Total features:     {n_total}\n"
    f"  Skipped (cached):   {n_skipped}\n"
    f"  Newly annotated:    {n_total - n_skipped - n_errors}\n"
    f"  Errors:             {n_errors}\n"
    f"  Output:             {OUTPUT}",
    flush=True,
)
