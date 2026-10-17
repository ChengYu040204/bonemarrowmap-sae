#!/usr/bin/env python3
"""
Task 7C: RAG Annotation with t-test Gene Ranking
=================================================
Replaces Pearson correlation (Step 4 of task7_rag_annotation.py) with
per-feature Welch t-test (top activating cells vs background sample).

Why: Pearson gives identical top genes for all plasma cell features because
     activation is globally correlated. t-test finds genes specifically
     upregulated in cells where *this* feature is highest, exposing sub-programme
     differences even within the same cluster.

Usage:
  cd sae_code/
  export ANTHROPIC_API_KEY="sk-ant-..."
  /opt/miniconda3/envs/scanpy-env/bin/python task7c_ttest_annotation.py

Outputs:
  gene_signatures_ttest.json      -- t-test gene signatures (new cache)
  gene_function_cache.json        -- shared MyGene.info cache (reused from task7)
  sae_annotations_rag_ttest.json  -- final annotations with t-test gene input
"""

import os, json, pickle, time, warnings
import numpy as np
import pandas as pd
import mygene
import anthropic
import scanpy as sc

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
ADATA_PATH  = "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad"
SAE_PKL     = "sae_results_all.pkl"
ANN_IN      = "sae_annotations.json"
CLUSTER_CSV = "task6_cluster_assignments.csv"

GENE_SIG_FILE  = "gene_signatures_ttest.json"   # new cache (t-test)
GENE_CACHE     = "gene_function_cache.json"      # shared with task7
OUTPUT         = "sae_annotations_rag_ttest.json"

API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL      = "claude-haiku-4-5-20251001"
TOP_CELLS  = 200    # high-activation cells per feature
BG_CELLS   = 2000   # random background sample size (fast approximation)
TOP_GENES  = 15     # genes sent to LLM per feature
MAX_FEATURES = None  # None = all 209; set integer for quick test

# ============================================================
# STEP 1: Load SAE features
# ============================================================
print("\n[1/6] Loading SAE features from pkl...", flush=True)
with open(SAE_PKL, "rb") as f:
    d = pickle.load(f)
raw = d[1]
X_sae = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.array(raw)
print(f"      X_sae: {X_sae.shape}", flush=True)

# ============================================================
# STEP 2: Load AnnData
# ============================================================
print("[2/6] Loading AnnData...", flush=True)
adata = sc.read_h5ad(ADATA_PATH)
gene_names = np.array(adata.var_names)
cell_types  = adata.obs["CellType"].values.astype(str)
X_raw       = adata.X
del adata

if hasattr(X_raw, "toarray"):
    X_genes = X_raw.toarray().astype(np.float32)
else:
    X_genes = np.array(X_raw, dtype=np.float32)
del X_raw
print(f"      X_genes: {X_genes.shape} | genes: {len(gene_names)}", flush=True)

N_CELLS, N_GENES = X_genes.shape

# ============================================================
# STEP 3: Load metadata
# ============================================================
print("[3/6] Loading cluster assignments and prior annotations...", flush=True)
with open(ANN_IN) as f:
    old_annotations = json.load(f)
old_ann_dict = {a["feature_id"]: a for a in old_annotations}

df_clusters = pd.read_csv(CLUSTER_CSV)
active_fids = df_clusters["feature_id"].tolist()
if MAX_FEATURES:
    active_fids = active_fids[:MAX_FEATURES]
print(f"      Features to annotate: {len(active_fids)}", flush=True)

# ============================================================
# STEP 4: Compute gene signatures via Welch t-test
# ============================================================
if os.path.exists(GENE_SIG_FILE):
    print("[4/6] Loading cached t-test gene signatures...", flush=True)
    with open(GENE_SIG_FILE) as f:
        raw_sigs = json.load(f)
    gene_signatures = {int(k): v["genes"] for k, v in raw_sigs.items()}
    ct_compositions = {int(k): v["celltype_composition"] for k, v in raw_sigs.items()}
    print(f"      Loaded signatures for {len(gene_signatures)} features", flush=True)
else:
    print("[4/6] Computing t-test gene signatures...", flush=True)
    print(f"      Method: top {TOP_CELLS} cells vs {BG_CELLS} random background cells", flush=True)

    rng = np.random.default_rng(42)
    gene_signatures = {}
    ct_compositions = {}

    for i, fid in enumerate(active_fids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"      Feature {i+1}/{len(active_fids)}...", flush=True)

        acts     = X_sae[:, fid]

        # Top activating cells (the cells that define this feature)
        top_idx  = np.argsort(acts)[::-1][:TOP_CELLS]

        # Random background: exclude top cells, sample BG_CELLS
        all_idx  = np.arange(N_CELLS)
        mask_top = np.zeros(N_CELLS, dtype=bool)
        mask_top[top_idx] = True
        bg_pool  = all_idx[~mask_top]
        bg_idx   = rng.choice(bg_pool, size=min(BG_CELLS, len(bg_pool)), replace=False)

        top_expr = X_genes[top_idx]    # (TOP_CELLS, N_GENES)
        bg_expr  = X_genes[bg_idx]     # (BG_CELLS,  N_GENES)

        # Welch t-test (unequal variance), vectorised over all genes
        m1 = top_expr.mean(axis=0)     # (N_GENES,)
        m2 = bg_expr.mean(axis=0)
        v1 = top_expr.var(axis=0) / TOP_CELLS
        v2 = bg_expr.var(axis=0) / len(bg_idx)

        denom   = np.sqrt(v1 + v2 + 1e-10)
        t_stats = (m1 - m2) / denom

        # Handle NaN (both groups zero) — send to bottom
        t_stats = np.where(np.isnan(t_stats) | np.isinf(t_stats), -np.inf, t_stats)

        top_gene_idx     = np.argsort(t_stats)[::-1][:TOP_GENES]
        gene_signatures[fid] = gene_names[top_gene_idx].tolist()

        # Cell type composition of top cells
        cts      = cell_types[top_idx]
        ct_counts = pd.Series(cts).value_counts(normalize=True)
        ct_compositions[fid] = {
            k: round(float(v) * 100, 1)
            for k, v in ct_counts.items() if float(v) >= 0.01
        }

    # Cache to disk
    sig_cache = {
        str(fid): {
            "genes": gene_signatures[fid],
            "celltype_composition": ct_compositions[fid]
        }
        for fid in active_fids
    }
    with open(GENE_SIG_FILE, "w") as f:
        json.dump(sig_cache, f, indent=2)
    print(f"      t-test signatures saved to {GENE_SIG_FILE}", flush=True)

# ============================================================
# STEP 5: Query MyGene.info (reuse shared cache)
# ============================================================
print("[5/6] Querying MyGene.info for gene functions...", flush=True)

if os.path.exists(GENE_CACHE):
    with open(GENE_CACHE) as f:
        gene_cache = json.load(f)
    print(f"      Loaded {len(gene_cache)} cached gene entries", flush=True)
else:
    gene_cache = {}

all_genes = set()
for fid in active_fids:
    all_genes.update(gene_signatures[fid])

uncached = [g for g in all_genes if g not in gene_cache]
print(f"      Unique genes: {len(all_genes)} | New queries needed: {len(uncached)}", flush=True)

if uncached:
    mg = mygene.MyGeneInfo()
    chunk_size = 500
    for i in range(0, len(uncached), chunk_size):
        chunk = uncached[i: i + chunk_size]
        batch_num = i // chunk_size + 1
        total_batches = (len(uncached) - 1) // chunk_size + 1
        print(f"      Batch {batch_num}/{total_batches} ({len(chunk)} genes)...", flush=True)
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
        time.sleep(1)

    with open(GENE_CACHE, "w") as f:
        json.dump(gene_cache, f, indent=2)
    print(f"      Cache updated: {len(gene_cache)} genes", flush=True)

# ============================================================
# STEP 6: RAG-enhanced LLM annotation
# ============================================================
print("[6/6] Running RAG-enhanced annotation (t-test gene input)...", flush=True)
assert API_KEY, "ERROR: Set ANTHROPIC_API_KEY environment variable before running"
client = anthropic.Anthropic(api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert haematologist and computational biologist specialising in
haematopoietic differentiation. You are annotating features from a Sparse Autoencoder (SAE)
trained on the BoneMarrowMap human bone marrow atlas. Each feature represents a candidate
Functional Programme (FP) — a coordinated gene expression programme active in a specific
cell state or differentiation stage.

The gene list provided comes from a t-test comparing top-activating cells vs background:
these are genes SPECIFICALLY upregulated in cells where this feature fires most strongly.
Use this specificity to distinguish mechanistically distinct sub-programmes even within
the same cell type (e.g. UPR initiation vs ERAD vs mTORC1 within plasma cells).

Respond ONLY as valid JSON with exactly these fields:
{
  "programme_name": "concise mechanistic name (maximum 6 words)",
  "functional_description": "2-3 sentences describing the specific biological mechanism",
  "key_evidence": ["GENE1: mechanistic reason", "GENE2: mechanistic reason", "GENE3: mechanistic reason"],
  "confidence": "HIGH | MEDIUM | LOW",
  "suggested_validation": "one specific experimental approach to validate this programme"
}"""

USER_TEMPLATE = """Feature ID: {feature_id}
Cluster: {cluster} | Prior annotation: "{prior_name}"

Cell-type composition of top {top_cells} activating cells:
{celltype_str}

Top {n_genes} genes by t-test (specifically upregulated in high-activation cells vs background):
{gene_context}

Identify the specific Functional Programme this SAE feature represents.
These genes are enriched by t-test, not global correlation — treat them as
specific markers of this feature's unique activation context."""

rag_annotations = []
n_total = len(active_fids)

for i, fid in enumerate(active_fids):
    genes = gene_signatures.get(fid, [])
    gene_lines = []
    for g in genes:
        info    = gene_cache.get(g, {})
        name    = info.get("name", g)
        summary = info.get("summary", "")
        if summary:
            short = summary[:130] + "..." if len(summary) > 130 else summary
            gene_lines.append(f"  {g} ({name}): {short}")
        else:
            gene_lines.append(f"  {g} ({name})")

    ct      = ct_compositions.get(fid, {})
    ct_str  = "\n".join(
        f"  {ct_name}: {pct}%"
        for ct_name, pct in sorted(ct.items(), key=lambda x: -x[1])
    ) or "  (no dominant cell type identified)"

    row        = df_clusters[df_clusters["feature_id"] == fid]
    cluster_id = int(row["cluster"].values[0]) if len(row) else "?"
    prior_name = old_ann_dict.get(fid, {}).get("programme_name", "unknown")

    user_msg = USER_TEMPLATE.format(
        feature_id   = fid,
        cluster      = cluster_id,
        prior_name   = prior_name,
        top_cells    = TOP_CELLS,
        celltype_str = ct_str,
        n_genes      = len(genes),
        gene_context = "\n".join(gene_lines),
    )

    try:
        response = client.messages.create(
            model      = MODEL,
            max_tokens = 512,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        result = json.loads(raw_text.strip())
    except Exception as e:
        print(f"  ERROR F#{fid}: {e}", flush=True)
        result = {
            "programme_name"       : prior_name,
            "functional_description": "",
            "key_evidence"         : [],
            "confidence"           : "LOW",
            "suggested_validation" : "",
        }

    new_name = result.get("programme_name", prior_name)
    changed  = "UPDATED" if new_name != prior_name else "same   "
    print(f"  [{i+1:3d}/{n_total}] F#{fid:4d} | {changed} | {new_name}", flush=True)

    rag_annotations.append({
        "feature_id"            : fid,
        "cluster"               : cluster_id,
        "anova_rank"            : old_ann_dict.get(fid, {}).get("anova_rank"),
        "top_genes_ttest"       : genes,
        "programme_name"        : new_name,
        "functional_description": result.get("functional_description", ""),
        "key_evidence"          : result.get("key_evidence", []),
        "confidence"            : result.get("confidence", "LOW"),
        "suggested_validation"  : result.get("suggested_validation", ""),
        "prior_programme_name"  : prior_name,
    })

    time.sleep(1.5)

# ============================================================
# Save
# ============================================================
with open(OUTPUT, "w") as f:
    json.dump(rag_annotations, f, indent=2)

n_updated = sum(1 for a in rag_annotations if a["programme_name"] != a["prior_programme_name"])
print(f"\nDone. {len(rag_annotations)} features annotated.", flush=True)
print(f"Updated: {n_updated}/{len(rag_annotations)} changed from prior zero-shot.", flush=True)
print(f"Saved to: {OUTPUT}", flush=True)
