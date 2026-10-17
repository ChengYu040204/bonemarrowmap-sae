#!/usr/bin/env python3
"""
Task 8: B-cell-Restricted Background t-test Annotation
=======================================================

Fix for the plasma-cell sub-programme collapse problem.

Problem with task7c_ttest_annotation.py:
  Background = 2000 random cells from all 161K.
  All 41 plasma cell features have top-200 composed entirely of plasma cells.
  Contrast: plasma cell vs everything else => identical gene lists for every feature.
  Result: Pairwise Jaccard = 0.592, LLM outputs same name for all 41 features.

Fix (confirmed in Dr. Ng meeting 2026-05-14):
  Background = remaining B-cell lineage cells ONLY (~17,800 B cells).
  Top-200: highest-activating B cells for THIS feature.
  Background: other B cells not in top-200.
  Now: IgKappa feature top-200 is IGKC-heavy plasma cells.
      IgLambda feature top-200 is IGLC2/IGLC3-heavy plasma cells.
      Background for BOTH is the same general B-cell pool.
  Contrast reveals isotype-level and sub-programme differences.

Usage:
  cd "Papers & Datas/SAE training/sae_code/"
  export ANTHROPIC_API_KEY="sk-ant-..."
  /opt/miniconda3/envs/scanpy-env/bin/python task8_bcell_restricted_ttest.py

Outputs:
  gene_signatures_bcell_ttest.json     per-feature gene signatures (new cache)
  gene_function_cache.json             MyGene.info cache (shared, extended in-place)
  sae_annotations_bcell_ttest.json     final RAG annotations

Author: SAE x Haematopoiesis project (UCL Gurdon Studentship 2026)
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


# =============================================================================
# CONFIGURATION
# =============================================================================

ADATA_PATH    = "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad"
SAE_PKL       = "sae_results_all.pkl"
ANN_IN        = "sae_annotations.json"       # zero-shot prior (for prior_name field)
CLUSTER_CSV   = "task6_cluster_assignments.csv"

GENE_SIG_FILE = "gene_signatures_bcell_ttest.json"  # new output cache
GENE_CACHE    = "gene_function_cache.json"           # shared with task7, extended here
OUTPUT        = "sae_annotations_bcell_ttest.json"

API_KEY       = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL         = "claude-haiku-4-5-20251001"

TOP_CELLS     = 200    # high-activation B cells per feature
TOP_GENES     = 15     # genes passed to LLM per feature

# B-cell lineage types: these are the ONLY cells used for both fg and bg
# Matches the list in extract_scvi_activations_new.py exactly
B_CELL_TYPES = [
    "HSC", "LMPP", "MLP", "MLP-II", "CLP",
    "Pro-B Cycling", "Pro-B VDJ", "Large Pre-B",
    "Small Pre-B", "Immature B", "Mature B", "Plasma Cell",
]

MAX_FEATURES  = None   # None = all active features; set integer for quick test run

SLEEP_API     = 1.2    # seconds between Claude API calls (rate limit safety)
SLEEP_MYGENE  = 1.0    # seconds between MyGene.info batch calls


# =============================================================================
# STEP 1: Load SAE feature matrix
# =============================================================================
print("\n[1/6] Loading SAE features from pkl ...", flush=True)

with open(SAE_PKL, "rb") as fh:
    d = pickle.load(fh)

# sae_results_all.pkl is a tuple: (reconstructions, features)
# features shape: (N_CELLS, N_FEATURES) = (161604, 2048)
raw = d[1]
if hasattr(raw, "detach"):
    X_sae = raw.detach().cpu().numpy().astype(np.float32)
else:
    X_sae = np.array(raw, dtype=np.float32)

print(f"      X_sae shape: {X_sae.shape}", flush=True)
N_CELLS_TOTAL, N_FEATURES = X_sae.shape


# =============================================================================
# STEP 2: Load AnnData and extract B-cell mask
# =============================================================================
print("[2/6] Loading AnnData and extracting B-cell indices ...", flush=True)

adata = sc.read_h5ad(ADATA_PATH)
gene_names  = np.array(adata.var_names)          # (N_GENES,)
cell_types  = adata.obs["CellType"].values.astype(str)   # (N_CELLS,)
X_raw       = adata.X                            # sparse or dense (N_CELLS, N_GENES)
del adata   # free memory before densifying

# Densify gene expression matrix once
if hasattr(X_raw, "toarray"):
    X_genes = X_raw.toarray().astype(np.float32)
else:
    X_genes = np.array(X_raw, dtype=np.float32)
del X_raw

N_CELLS_CHECK, N_GENES = X_genes.shape
assert N_CELLS_CHECK == N_CELLS_TOTAL, (
    f"Cell count mismatch: AnnData {N_CELLS_CHECK} vs SAE pkl {N_CELLS_TOTAL}"
)

# Build B-cell global index array (indices into the full 161K cell space)
bcell_mask    = np.array([ct in B_CELL_TYPES for ct in cell_types])
bcell_indices = np.where(bcell_mask)[0]          # global indices, shape (N_BCELLS,)
N_BCELLS      = len(bcell_indices)

print(f"      Gene matrix: {X_genes.shape}", flush=True)
print(f"      B-cell types: {B_CELL_TYPES}", flush=True)
print(f"      B-cell count: {N_BCELLS} / {N_CELLS_TOTAL} total cells", flush=True)

if N_BCELLS < TOP_CELLS + 50:
    raise RuntimeError(
        f"Only {N_BCELLS} B cells found. Check CellType column spelling against B_CELL_TYPES."
    )


# =============================================================================
# STEP 3: Load cluster assignments and prior zero-shot annotations
# =============================================================================
print("[3/6] Loading metadata ...", flush=True)

df_clusters = pd.read_csv(CLUSTER_CSV)
active_fids = df_clusters["feature_id"].tolist()    # list of int feature IDs

if MAX_FEATURES is not None:
    active_fids = active_fids[:MAX_FEATURES]
    print(f"      TEST MODE: limiting to first {MAX_FEATURES} features", flush=True)

print(f"      Active features to annotate: {len(active_fids)}", flush=True)

# Map feature_id => cluster label for prompt context
fid_to_cluster = dict(zip(df_clusters["feature_id"], df_clusters["cluster"]))

# Prior zero-shot programme names (used as context in prompt)
with open(ANN_IN) as fh:
    old_annotations = json.load(fh)
fid_to_prior_name = {a["feature_id"]: a.get("programme_name", "unknown") for a in old_annotations}


# =============================================================================
# STEP 4: Compute B-cell-restricted gene signatures via Welch t-test
#
# For each active feature f:
#   fg = top TOP_CELLS B cells by X_sae[:, f] activation (their global indices)
#   bg = all other B cells (global indices in bcell_indices but not in fg)
#   t-stat per gene = (mean_fg - mean_bg) / sqrt(var_fg/n_fg + var_bg/n_bg + eps)
#   Top TOP_GENES genes by t-stat => feature's signature
#
# Why this fixes the plasma-cell collapse:
#   Every feature's bg is the same general B-cell pool (~17,800 cells).
#   An IgKappa feature's fg is enriched for IGKC-high plasma cells;
#   an IgLambda feature's fg is enriched for IGLC2/IGLC3-high plasma cells.
#   The contrast exposes isotype- and sub-programme-level differences
#   that were invisible when bg was non-B cells (where ALL plasma cell features
#   looked the same relative to HSCs, erythroid cells, etc.)
# =============================================================================
print("[4/6] Computing B-cell-restricted t-test gene signatures ...", flush=True)

if os.path.exists(GENE_SIG_FILE):
    print(f"      Cache found: {GENE_SIG_FILE}. Loading ...", flush=True)
    with open(GENE_SIG_FILE) as fh:
        raw_sigs = json.load(fh)
    gene_signatures  = {int(k): v["genes"]               for k, v in raw_sigs.items()}
    ct_compositions  = {int(k): v["celltype_composition"] for k, v in raw_sigs.items()}
    print(f"      Loaded {len(gene_signatures)} cached signatures", flush=True)

else:
    print(
        f"      Background pool: {N_BCELLS} B cells total\n"
        f"      Foreground: top {TOP_CELLS} activating B cells per feature\n"
        f"      Background: remaining B cells (~{N_BCELLS - TOP_CELLS})",
        flush=True,
    )

    # Pre-extract B-cell sub-matrix once (avoids repeated indexing into 161K matrix)
    # Shape: (N_BCELLS, N_GENES)
    X_bcell = X_genes[bcell_indices]        # (N_BCELLS, N_GENES)
    acts_bcell = X_sae[bcell_indices, :]    # (N_BCELLS, N_FEATURES)

    gene_signatures = {}
    ct_compositions = {}

    for i, fid in enumerate(active_fids):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"      Feature {i+1}/{len(active_fids)} (fid={fid}) ...", flush=True)

        # Activation values for this feature, restricted to B cells
        acts_f = acts_bcell[:, fid]           # (N_BCELLS,)

        # Foreground: top TOP_CELLS B cells by activation
        fg_local = np.argsort(acts_f)[::-1][:TOP_CELLS]   # local indices into B-cell arrays

        # Background: all remaining B cells
        all_local = np.arange(N_BCELLS)
        fg_mask   = np.zeros(N_BCELLS, dtype=bool)
        fg_mask[fg_local] = True
        bg_local  = all_local[~fg_mask]       # shape: (N_BCELLS - TOP_CELLS,)

        # Gene expression sub-matrices
        fg_expr = X_bcell[fg_local]           # (TOP_CELLS, N_GENES)
        bg_expr = X_bcell[bg_local]           # (N_BG, N_GENES)

        n_fg = fg_expr.shape[0]
        n_bg = bg_expr.shape[0]

        # Welch t-test vectorised over all genes
        m1 = fg_expr.mean(axis=0)             # (N_GENES,)
        m2 = bg_expr.mean(axis=0)
        v1 = fg_expr.var(axis=0) / n_fg
        v2 = bg_expr.var(axis=0) / n_bg

        t_stats = (m1 - m2) / np.sqrt(v1 + v2 + 1e-10)

        # Replace NaN and Inf (genes with zero variance in both groups)
        t_stats = np.where(np.isnan(t_stats) | np.isinf(t_stats), -np.inf, t_stats)

        # Top genes by t-statistic
        top_gene_local = np.argsort(t_stats)[::-1][:TOP_GENES]
        gene_signatures[fid] = gene_names[top_gene_local].tolist()

        # Cell-type composition of foreground cells (using global cell_types array)
        global_fg_idx = bcell_indices[fg_local]
        cts_fg = cell_types[global_fg_idx]
        ct_counts = pd.Series(cts_fg).value_counts(normalize=True)
        ct_compositions[fid] = {
            k: round(float(v) * 100, 1)
            for k, v in ct_counts.items()
            if float(v) >= 0.01
        }

    # Persist to disk (avoids recomputation if script is rerun)
    sig_cache = {
        str(fid): {
            "genes": gene_signatures[fid],
            "celltype_composition": ct_compositions[fid],
        }
        for fid in active_fids
        if fid in gene_signatures
    }
    with open(GENE_SIG_FILE, "w") as fh:
        json.dump(sig_cache, fh, indent=2)
    print(f"      Saved {len(sig_cache)} signatures to {GENE_SIG_FILE}", flush=True)

    # Free large B-cell sub-matrices now that signatures are computed
    del X_bcell, acts_bcell


# =============================================================================
# STEP 5: Query MyGene.info for gene functional summaries
#         Reuses the shared gene_function_cache.json from task7 runs.
#         Only queries genes not already in cache.
# =============================================================================
print("[5/6] Querying MyGene.info for gene functions ...", flush=True)

if os.path.exists(GENE_CACHE):
    with open(GENE_CACHE) as fh:
        gene_cache = json.load(fh)
    print(f"      Loaded {len(gene_cache)} cached gene entries from {GENE_CACHE}", flush=True)
else:
    gene_cache = {}

# Collect all unique genes across new signatures
all_genes_needed = set()
for fid in active_fids:
    if fid in gene_signatures:
        all_genes_needed.update(gene_signatures[fid])

uncached_genes = sorted(all_genes_needed - set(gene_cache.keys()))
print(
    f"      Unique genes needed: {len(all_genes_needed)} | New lookups: {len(uncached_genes)}",
    flush=True,
)

if uncached_genes:
    mg = mygene.MyGeneInfo()
    chunk_size = 500
    n_batches  = (len(uncached_genes) - 1) // chunk_size + 1

    for batch_i in range(n_batches):
        chunk = uncached_genes[batch_i * chunk_size : (batch_i + 1) * chunk_size]
        print(f"      MyGene batch {batch_i + 1}/{n_batches} ({len(chunk)} genes) ...", flush=True)

        results = mg.querymany(
            chunk,
            scopes="symbol",
            fields="summary,name,symbol",
            species="human",
            returnall=True,
        )
        for r in results["out"]:
            sym = r.get("query", "")
            if r.get("notfound"):
                gene_cache[sym] = {"name": sym, "summary": ""}
            else:
                gene_cache[sym] = {
                    "name": r.get("name", sym),
                    "summary": r.get("summary", ""),
                }
        time.sleep(SLEEP_MYGENE)

    with open(GENE_CACHE, "w") as fh:
        json.dump(gene_cache, fh, indent=2)
    print(f"      Cache updated: {len(gene_cache)} genes total", flush=True)


# =============================================================================
# STEP 6: RAG-enhanced LLM annotation
#
# Each prompt receives:
#   - Feature ID, cluster, prior zero-shot name
#   - Cell-type composition of top-200 B cells (foreground)
#   - Top 15 genes with NCBI Gene functional summaries (the RAG component)
#
# The system prompt instructs Claude to use gene function specifics to
# distinguish sub-programmes within the same cell type (e.g. IgKappa vs IgLambda).
# =============================================================================
print("[6/6] Running RAG-enhanced LLM annotation ...", flush=True)

if not API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not set. Export it before running:\n"
        "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
        "  /opt/miniconda3/envs/scanpy-env/bin/python task8_bcell_restricted_ttest.py"
    )

client = anthropic.Anthropic(api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are an expert haematologist and computational biologist specialising in "
    "haematopoietic differentiation and B-cell development. "
    "You are annotating features from a Sparse Autoencoder (SAE) trained on scVI latent "
    "representations of the BoneMarrowMap human bone marrow atlas (~160K cells). "
    "Each feature represents a candidate Functional Programme (FP): a coordinated gene "
    "expression programme active in a specific cell state or differentiation stage.\n\n"
    "IMPORTANT CONTEXT FOR THIS TASK:\n"
    "The gene lists below were computed by B-cell-restricted Welch t-test: "
    "top-200 activating B cells vs all remaining B cells (~17,800) as background. "
    "This means the genes listed are specifically upregulated within B-cell subtypes "
    "that activate this feature, relative to other B cells. "
    "Use the gene functional summaries to identify the precise sub-programme mechanism. "
    "For plasma cell features: distinguish IgKappa vs IgLambda vs IgG isotypes, "
    "UPR initiation vs chaperone-mediated folding vs ERAD, mTORC1-driven translation "
    "vs secretory pathway expansion vs bone marrow survival signalling. "
    "Do NOT default to generic labels like 'Plasma cell immunoglobulin secretion programme'. "
    "Instead name the specific mechanism (e.g. 'IgKappa light chain secretion programme', "
    "'XBP1-driven ER chaperone induction', 'ERAD-mediated misfolded Ig clearance').\n\n"
    "Respond ONLY as valid JSON with exactly these fields:\n"
    "{\n"
    '  "programme_name": "concise mechanistic name maximum 7 words",\n'
    '  "functional_description": "2-3 sentences on the specific molecular mechanism",\n'
    '  "key_evidence": ["GENE1: why this gene is diagnostic", "GENE2: ...", "GENE3: ..."],\n'
    '  "confidence": "HIGH | MEDIUM | LOW",\n'
    '  "suggested_validation": "one specific experiment to validate this programme"\n'
    "}"
)


def build_prompt(fid, cluster, prior_name, ct_comp, genes_with_context, n_fg):
    """Assemble the user message for one feature."""
    celltype_lines = "\n".join(
        f"  {pct:5.1f}%  {ct}" for ct, pct in sorted(ct_comp.items(), key=lambda x: -x[1])
    ) or "  (no cell-type breakdown available)"

    gene_lines = "\n".join(
        f"  {j+1:2d}. {entry}" for j, entry in enumerate(genes_with_context)
    )

    return (
        f"Feature ID: {fid}\n"
        f"Cluster: {cluster} | Prior zero-shot name: \"{prior_name}\"\n\n"
        f"Cell-type composition of top {n_fg} activating B cells:\n"
        f"{celltype_lines}\n\n"
        f"Top {len(genes_with_context)} genes (B-cell-restricted t-test, with gene function context):\n"
        f"{gene_lines}\n"
    )


def parse_json_response(raw_text):
    """Strip markdown code fences from Claude response, then parse JSON."""
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the content between first ``` and second ```
        content = parts[1]
        if content.startswith("json"):
            content = content[4:]
        text = content.strip()
    return json.loads(text)


# Load existing output if present (enables resume after interruption)
if os.path.exists(OUTPUT):
    with open(OUTPUT) as fh:
        results_list = json.load(fh)
    done_fids = {r["feature_id"] for r in results_list}
    print(f"      Resuming: {len(done_fids)} features already annotated", flush=True)
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
        print(f"      WARNING: no gene signature for fid={fid}, skipping", flush=True)
        n_errors += 1
        continue

    # Build gene context string (gene symbol + functional summary)
    genes       = gene_signatures[fid]
    ct_comp     = ct_compositions.get(fid, {})
    cluster     = fid_to_cluster.get(fid, "?")
    prior_name  = fid_to_prior_name.get(fid, "unknown")

    genes_with_context = []
    for g in genes:
        entry = gene_cache.get(g, {})
        summary = entry.get("summary", "").strip()
        # Truncate long summaries to keep prompt concise
        if len(summary) > 120:
            summary = summary[:117] + "..."
        if summary:
            genes_with_context.append(f"{g}: {summary}")
        else:
            genes_with_context.append(f"{g}: (no functional summary available)")

    user_msg = build_prompt(fid, cluster, prior_name, ct_comp, genes_with_context, TOP_CELLS)

    # Claude API call with retry on transient errors
    annotation = None
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=900,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text   = response.content[0].text
            annotation = parse_json_response(raw_text)
            break
        except json.JSONDecodeError as e:
            if attempt < 2:
                print(f"      JSON parse error fid={fid} attempt {attempt+1}: {e}", flush=True)
                time.sleep(2)
            else:
                print(f"      JSON parse failed fid={fid} after 3 attempts, using fallback", flush=True)
                annotation = {
                    "programme_name":         "annotation parse error",
                    "functional_description": f"Raw response: {raw_text[:200]}",
                    "key_evidence":           genes[:3],
                    "confidence":             "LOW",
                    "suggested_validation":   "",
                }
                n_errors += 1
        except Exception as e:
            if attempt < 2:
                print(f"      API error fid={fid} attempt {attempt+1}: {e}", flush=True)
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
        "feature_id":            fid,
        "cluster":               cluster,
        "prior_zero_shot_name":  prior_name,
        "top_genes":             genes,
        "celltype_composition":  ct_comp,
        "programme_name":        annotation.get("programme_name", ""),
        "functional_description": annotation.get("functional_description", ""),
        "key_evidence":          annotation.get("key_evidence", []),
        "confidence":            annotation.get("confidence", "LOW"),
        "suggested_validation":  annotation.get("suggested_validation", ""),
        "method":                "bcell_restricted_ttest_rag",
        "background":            "B-cell lineage only (~17800 cells)",
    }
    results_list.append(record)
    done_fids.add(fid)

    features_done = i + 1 - n_skipped
    if features_done % 20 == 0 or features_done == 1:
        print(
            f"      [{features_done}/{n_total - len(done_fids) + features_done}] "
            f"fid={fid} cluster={cluster} | {annotation.get('programme_name', '')}",
            flush=True,
        )

    # Checkpoint: save after every 10 annotations
    if features_done % 10 == 0:
        with open(OUTPUT, "w") as fh:
            json.dump(results_list, fh, indent=2)

    time.sleep(SLEEP_API)

# Final save
with open(OUTPUT, "w") as fh:
    json.dump(results_list, fh, indent=2)

print(
    f"\nDone.\n"
    f"  Total features:   {n_total}\n"
    f"  Skipped (cached): {n_skipped}\n"
    f"  Newly annotated:  {n_total - n_skipped - n_errors}\n"
    f"  Errors:           {n_errors}\n"
    f"  Output:           {OUTPUT}",
    flush=True,
)
