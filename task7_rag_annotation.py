#!/usr/bin/env python3
"""
Task 7: RAG-Enhanced SAE Feature Annotation
================================================
Pipeline:
  SAE features -> gene signatures (correlation) -> MyGene.info RAG -> Claude API -> enriched annotations

Usage:
  cd sae_code/
  export ANTHROPIC_API_KEY="sk-ant-..."
  python3 task7_rag_annotation.py

Outputs:
  gene_signatures.json        -- top genes per active feature (cached, reusable)
  gene_function_cache.json    -- MyGene.info lookup cache (persists across runs)
  sae_annotations_rag.json    -- final RAG-enhanced annotations
"""

import os, json, pickle, time, warnings
import numpy as np
import pandas as pd
import mygene
import anthropic
import scanpy as sc

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — adjust if needed
# ============================================================
ADATA_PATH  = "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad"
SAE_PKL     = "sae_results_all.pkl"
ANN_IN      = "sae_annotations.json"
CLUSTER_CSV = "task6_cluster_assignments.csv"

GENE_SIG_FILE  = "gene_signatures.json"
GENE_CACHE     = "gene_function_cache.json"
OUTPUT         = "sae_annotations_rag.json"

API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL      = "claude-haiku-4-5-20251001"   # fast + cheap; swap for claude-opus-4-6 for higher quality
TOP_CELLS  = 200   # cells used to define "high activating"
TOP_GENES  = 15    # genes per feature sent to LLM (fewer = tighter prompt)

# Set to a number (e.g. 10) to test on subset; None = annotate all 209 features
MAX_FEATURES = None

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
# STEP 2: Load AnnData -> gene names, cell types, expression
# ============================================================
print("[2/6] Loading AnnData...", flush=True)
adata = sc.read_h5ad(ADATA_PATH)
gene_names  = np.array(adata.var_names)
cell_types  = adata.obs["CellType"].values.astype(str)
X_raw       = adata.X
del adata

if hasattr(X_raw, "toarray"):
    X_genes = X_raw.toarray().astype(np.float32)
else:
    X_genes = np.array(X_raw, dtype=np.float32)
del X_raw

print(f"      X_genes: {X_genes.shape} | genes: {len(gene_names)}", flush=True)

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
# STEP 4: Compute gene signatures (vectorized Pearson correlation)
# ============================================================
if os.path.exists(GENE_SIG_FILE):
    print("[4/6] Loading cached gene signatures...", flush=True)
    with open(GENE_SIG_FILE) as f:
        raw_sigs = json.load(f)
    gene_signatures = {int(k): v["genes"] for k, v in raw_sigs.items()}
    ct_compositions = {int(k): v["celltype_composition"] for k, v in raw_sigs.items()}
    print(f"      Loaded signatures for {len(gene_signatures)} features", flush=True)
else:
    print("[4/6] Computing gene signatures (vectorized correlation)...", flush=True)

    # Extract active feature columns
    X_sae_active = X_sae[:, active_fids].astype(np.float32)   # (n_cells, n_active)

    # Center both matrices for Pearson correlation
    feat_mean = X_sae_active.mean(0)
    feat_std  = X_sae_active.std(0) + 1e-8
    X_sae_c   = (X_sae_active - feat_mean)  # (n_cells, n_active)

    gene_mean  = X_genes.mean(0)
    gene_std   = X_genes.std(0) + 1e-8
    X_genes   -= gene_mean   # center in-place to save RAM

    print("      Matrix multiply (may take ~30s)...", flush=True)
    # corr_matrix: (n_active, n_genes)
    corr_unnorm = X_sae_c.T @ X_genes             # (n_active, n_genes)
    corr_matrix = corr_unnorm / (X_sae_c.shape[0] * feat_std[:, None] * gene_std[None, :])
    del X_sae_c, corr_unnorm
    print(f"      corr_matrix: {corr_matrix.shape}", flush=True)

    # Extract top genes and cell type composition per feature
    gene_signatures = {}
    ct_compositions = {}

    for i, fid in enumerate(active_fids):
        # Top genes by correlation
        top_idx = np.argsort(corr_matrix[i])[::-1][:TOP_GENES]
        gene_signatures[fid] = gene_names[top_idx].tolist()

        # Cell type composition among top 200 activating cells
        acts      = X_sae[:, fid]
        top_cells = np.argsort(acts)[::-1][:TOP_CELLS]
        cts       = cell_types[top_cells]
        ct_counts = pd.Series(cts).value_counts(normalize=True)
        ct_compositions[fid] = {
            k: round(float(v) * 100, 1)
            for k, v in ct_counts.items() if float(v) >= 0.01
        }

    # Cache gene signatures to disk
    sig_cache = {
        str(fid): {"genes": gene_signatures[fid], "celltype_composition": ct_compositions[fid]}
        for fid in active_fids
    }
    with open(GENE_SIG_FILE, "w") as f:
        json.dump(sig_cache, f, indent=2)
    print(f"      Gene signatures saved to {GENE_SIG_FILE}", flush=True)

# ============================================================
# STEP 5: Query MyGene.info (with persistent cache)
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
print(f"      Unique genes: {len(all_genes)} | To query: {len(uncached)}", flush=True)

if uncached:
    mg = mygene.MyGeneInfo()
    chunk_size = 500
    for i in range(0, len(uncached), chunk_size):
        chunk = uncached[i : i + chunk_size]
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
    print(f"      Cache updated: {len(gene_cache)} genes saved to {GENE_CACHE}", flush=True)

# ============================================================
# STEP 6: RAG-enhanced LLM annotation
# ============================================================
print("[6/6] Running RAG-enhanced annotation...", flush=True)
assert API_KEY, "ERROR: Set ANTHROPIC_API_KEY environment variable before running"
client = anthropic.Anthropic(api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert haematologist and computational biologist specialising in
haematopoietic differentiation. You are annotating features from a Sparse Autoencoder (SAE)
trained on the BoneMarrowMap human bone marrow atlas. Each feature represents a candidate
Functional Programme (FP) — a coordinated gene expression programme active in a specific
cell state or differentiation stage.

Use the gene function information provided to give mechanistically specific annotations.
Distinguish sub-programmes within broad categories (e.g. UPR vs mTORC1 within plasma cells,
or heme biosynthesis vs globin translation within erythroid differentiation).

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

Top {n_genes} genes with functional context from gene databases:
{gene_context}

Based on the gene functions above, identify the specific Functional Programme this SAE feature
represents. Be mechanistically specific — if the prior annotation was too generic, provide a
more precise sub-programme name."""

rag_annotations = []
n_total = len(active_fids)

for i, fid in enumerate(active_fids):
    # Build gene context block with MyGene.info summaries
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

    # Cell type composition
    ct  = ct_compositions.get(fid, {})
    ct_str = "\n".join(
        f"  {ct_name}: {pct}%"
        for ct_name, pct in sorted(ct.items(), key=lambda x: -x[1])
    ) or "  (no dominant cell type identified)"

    # Cluster and prior annotation
    row        = df_clusters[df_clusters["feature_id"] == fid]
    cluster_id = int(row["cluster"].values[0]) if len(row) else "?"
    prior_name = old_ann_dict.get(fid, {}).get("programme_name", "unknown")

    user_msg = USER_TEMPLATE.format(
        feature_id  = fid,
        cluster     = cluster_id,
        prior_name  = prior_name,
        top_cells   = TOP_CELLS,
        celltype_str= ct_str,
        n_genes     = len(genes),
        gene_context= "\n".join(gene_lines),
    )

    # API call with error handling
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
            "programme_name"      : prior_name,
            "functional_description": "",
            "key_evidence"        : [],
            "confidence"          : "LOW",
            "suggested_validation": "",
        }

    new_name = result.get("programme_name", prior_name)
    changed  = "UPDATED" if new_name != prior_name else "same   "
    print(f"  [{i+1:3d}/{n_total}] F#{fid:4d} | {changed} | {new_name}", flush=True)

    rag_annotations.append({
        "feature_id"           : fid,
        "cluster"              : cluster_id,
        "anova_rank"           : old_ann_dict.get(fid, {}).get("anova_rank"),
        "top_genes"            : genes,
        "programme_name"       : new_name,
        "functional_description": result.get("functional_description", ""),
        "key_evidence"         : result.get("key_evidence", []),
        "confidence"           : result.get("confidence", "LOW"),
        "suggested_validation" : result.get("suggested_validation", ""),
        "prior_programme_name" : prior_name,
    })

    time.sleep(1.5)  # rate limit

# ============================================================
# Save and summarise
# ============================================================
with open(OUTPUT, "w") as f:
    json.dump(rag_annotations, f, indent=2)

n_updated = sum(1 for a in rag_annotations if a["programme_name"] != a["prior_programme_name"])
print(f"\nDone! {len(rag_annotations)} features annotated.", flush=True)
print(f"Updated: {n_updated}/{len(rag_annotations)} annotations changed from prior.", flush=True)
print(f"Results saved to: {OUTPUT}", flush=True)
