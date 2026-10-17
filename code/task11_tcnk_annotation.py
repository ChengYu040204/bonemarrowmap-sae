"""
task11_tcnk_annotation.py
==========================
Route B: ANOVA + t-test + RAG-LLM annotation for T/NK-specific SAE.

This script assumes:
  - sae_tcnk_results_all.pkl  : (recons, features) tuple from train_sae_tcnk.py
  - tcnk_global_indices.pkl   : global positional indices of T/NK cells in adata_unsorted
  - BoneMarrowMap_Unsorted_scVI.h5ad available at relative path ../../Papers' Dataset/

Pipeline:
  Step 1: Load T/NK SAE feature matrix (70714, 2048)
  Step 2: Load AnnData gene expression for T/NK cells
  Step 3: ANOVA within T/NK (T cells vs NK cells as the two groups)
          Filter: active features (std > 0) + F-stat >= MIN_F_STAT
          Direction filter: mean_tcell > mean_nk OR mean_nk > mean_tcell
          (all active features are T/NK-specific by construction)
  Step 4: Within-group t-test gene signatures (T/NK-restricted background)
          Cluster-conditional background:
            T-dominant features: foreground = top200 T cells,
                                 background = remaining T/NK cells
            NK-dominant features: foreground = top200 NK cells,
                                  background = remaining T/NK cells
  Step 5: MyGene.info RAG (extends existing gene_function_cache.json)
  Step 6: Claude API annotation loop
          Prompt selection: T_DOMINANT / NK_DOMINANT / MIXED based on
          cell-type composition in top-200 activating cells

Output files:
  tcnk_anova_routeB.csv            ANOVA results for T/NK SAE features
  gene_signatures_tcnk_routeB.json t-test gene signatures
  sae_annotations_tcnk_routeB.json RAG-LLM annotations

Usage:
  cd "Papers & Datas/SAE training/sae_code/"
  export ANTHROPIC_API_KEY="sk-ant-..."
  /opt/miniconda3/envs/scanpy-env/bin/python task11_tcnk_annotation.py

Author: SAE x Haematopoiesis project (FP-Decode), UCL Gurdon Studentship 2026
"""

import os
import json
import pickle
import time
import warnings
import csv as csv_module

import numpy as np
import anthropic

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
H5AD_PATH    = os.path.join(SCRIPT_DIR, "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad")

MIN_F_STAT   = 50        # minimum ANOVA F-stat (T cells vs NK cells) to include
TOP_CELLS    = 200       # foreground cells per feature in t-test
TOP_GENES    = 15        # genes sent to LLM per feature
SLEEP_API    = 1.2       # seconds between API calls
MAX_TOKENS   = 900

NK_DOMINANT_THRESH = 0.40
T_DOMINANT_THRESH  = 0.40

T_CELL_TYPES = [
    "CD4 Naive", "CD4 Central Memory", "CD4 Effector Memory", "CD4 Regulatory",
    "CD8 Naive", "CD8 Central Memory", "CD8 Effector Memory 1", "CD8 Effector Memory 2",
    "T Proliferating",
]
NK_CELL_TYPES = ["NK", "NK CD56high"]
T_NK_TYPES    = T_CELL_TYPES + NK_CELL_TYPES

# -------------------------------------------------------------------------------
# LLM prompts
# -------------------------------------------------------------------------------
SYSTEM_PROMPT_TCELL = """You are a computational biology expert specialising in T cell biology and human immunology.
You will be shown a gene signature extracted by a Sparse Autoencoder trained on scVI latent representations
of T cells from the BoneMarrowMap atlas (bone marrow T/NK cells).
Your task is to identify the specific T cell sub-programme this SAE feature represents.

Focus on distinguishing:
- CD4 naive vs central memory vs effector memory vs regulatory (Treg)
- CD8 naive vs central memory vs effector memory vs terminally exhausted
- Cytokine secretion programmes (IFN-gamma, IL-2, TNF)
- TCR signalling vs homeostatic proliferation vs antigen-driven activation
- Exhaustion programmes (TOX, PDCD1, LAG3, TIGIT)
- Tissue-resident vs circulating T cell identity

Respond ONLY as valid JSON with fields:
  programme_name       (6 words max, specific sub-programme, no generic "T cell activation")
  functional_description  (2-3 sentences)
  key_evidence         (list of up to 5 strings: "GENE: one-sentence rationale")
  confidence           (HIGH / MEDIUM / LOW)
  suggested_validation (one sentence)
"""

SYSTEM_PROMPT_NK = """You are a computational biology expert specialising in NK cell biology and innate immunity.
You will be shown a gene signature extracted by a Sparse Autoencoder trained on scVI latent representations
of NK cells from the BoneMarrowMap atlas.
Your task is to identify the specific NK cell sub-programme this SAE feature represents.

Focus on distinguishing:
- CD56bright (cytokine-producing, bone marrow resident) vs CD56dim (cytotoxic, peripheral)
- KIR repertoire (KIR2DL1/2DL2/3DL1/3DL2) and HLA class I recognition
- Cytotoxicity machinery (PRF1, GZMB/H/K, GNLY)
- Cytokine secretion (IFNG, TNF, IL-12 responsiveness)
- NK maturation markers (CD57/B3GAT1, KLRG1, NKG2A vs NKG2C)
- Activating vs inhibitory receptor balance

Respond ONLY as valid JSON with fields:
  programme_name       (6 words max, specific sub-programme)
  functional_description  (2-3 sentences)
  key_evidence         (list of up to 5 strings: "GENE: one-sentence rationale")
  confidence           (HIGH / MEDIUM / LOW)
  suggested_validation (one sentence)
"""

SYSTEM_PROMPT_MIXED = """You are a computational biology expert specialising in lymphocyte biology.
You will be shown a gene signature extracted by a Sparse Autoencoder trained on scVI latent
representations of T and NK cells from the BoneMarrowMap atlas.
This feature activates across both T cells and NK cells; identify the shared lymphocyte programme.

Focus on:
- Shared cytotoxic machinery (PRF1, GZMB, GNLY)
- Shared proliferation signals (MKI67, TOP2A, E2F targets)
- Common metabolic programmes
- Shared inhibitory receptor co-expression (PDCD1, TIGIT, LAG3)
- Bone marrow residency vs circulation signals

Respond ONLY as valid JSON with fields:
  programme_name       (6 words max)
  functional_description  (2-3 sentences)
  key_evidence         (list of up to 5 strings: "GENE: one-sentence rationale")
  confidence           (HIGH / MEDIUM / LOW)
  suggested_validation (one sentence)
"""

USER_TEMPLATE = """Feature ID: {fid}
ANOVA rank (T vs NK): {rank}
ANOVA F-stat: {fstat:.1f}
Dominant group: {dominant}

Top {n_genes} differentially expressed genes (within-T/NK t-test):
{gene_list}

Cell-type composition of top-200 activating cells:
{celltype_comp}

Activation fold-enrichment (top-200 mean / T/NK-wide mean): {fold:.2f}x

Gene function context (from NCBI Gene):
{gene_functions}

Classify this T/NK SAE feature as a specific functional sub-programme."""


# -------------------------------------------------------------------------------
# Step 1: Load T/NK SAE features
# -------------------------------------------------------------------------------
print("[1/6] Loading T/NK SAE feature matrix...", flush=True)

with open(os.path.join(SCRIPT_DIR, "sae_tcnk_results_all.pkl"), "rb") as f:
    sae_data = pickle.load(f)

if isinstance(sae_data, tuple):
    raw_features = sae_data[1]
else:
    raw_features = sae_data

if hasattr(raw_features, "detach"):
    X_sae = raw_features.detach().cpu().numpy()
else:
    X_sae = np.array(raw_features)

N_TCNK, N_SAE = X_sae.shape
print(f"  T/NK SAE features: {X_sae.shape}  (expect ~70714 x 2048)", flush=True)

# Load global indices mapping T/NK rows back to adata_unsorted
with open(os.path.join(SCRIPT_DIR, "tcnk_global_indices.pkl"), "rb") as f:
    tcnk_global_idx = pickle.load(f)
tcnk_global_idx = np.array(tcnk_global_idx)
assert len(tcnk_global_idx) == N_TCNK, "Index count mismatch"


# -------------------------------------------------------------------------------
# Step 2: Load AnnData and extract T/NK gene expression
# -------------------------------------------------------------------------------
print("[2/6] Loading AnnData...", flush=True)
import scanpy as sc
adata = sc.read_h5ad(H5AD_PATH)
print(f"  adata shape: {adata.shape}", flush=True)

# Cell type labels for T/NK cells (local indices within T/NK subset)
tcnk_celltypes = adata.obs["CellType"].values[tcnk_global_idx]
t_mask_local   = np.array([ct in T_CELL_TYPES for ct in tcnk_celltypes])
nk_mask_local  = np.array([ct in NK_CELL_TYPES for ct in tcnk_celltypes])

print(f"  T cells:  {t_mask_local.sum()}", flush=True)
print(f"  NK cells: {nk_mask_local.sum()}", flush=True)

# Gene expression matrix for T/NK cells
import scipy.sparse
X_raw = adata.X[tcnk_global_idx]
if scipy.sparse.issparse(X_raw):
    X_genes = X_raw.toarray().astype(np.float32)
else:
    X_genes = np.array(X_raw, dtype=np.float32)

gene_names = np.array(adata.var_names)
print(f"  Gene matrix: {X_genes.shape}", flush=True)

del adata  # free memory after extraction


# -------------------------------------------------------------------------------
# Step 3: ANOVA (T cells vs NK cells as two groups)
# -------------------------------------------------------------------------------
print("[3/6] Running ANOVA (T cells vs NK cells)...", flush=True)

anova_cache = os.path.join(SCRIPT_DIR, "tcnk_anova_routeB.csv")

if os.path.exists(anova_cache):
    print(f"  Loading cached ANOVA from {anova_cache}", flush=True)
    import pandas as pd
    anova_df = pd.read_csv(anova_cache)
else:
    from scipy import stats as sp_stats

    acts_t  = X_sae[t_mask_local]   # (N_T, 2048)
    acts_nk = X_sae[nk_mask_local]  # (N_NK, 2048)

    print(f"  Computing F-stats for {N_SAE} features...", flush=True)
    f_stats    = np.zeros(N_SAE)
    mean_t_arr = np.zeros(N_SAE)
    mean_nk_arr = np.zeros(N_SAE)

    for fid in range(N_SAE):
        g1 = acts_t[:, fid]
        g2 = acts_nk[:, fid]
        if g1.std() < 1e-9 and g2.std() < 1e-9:
            f_stats[fid] = 0.0
        else:
            f, _ = sp_stats.f_oneway(g1, g2)
            f_stats[fid] = f if np.isfinite(f) else 0.0
        mean_t_arr[fid]  = g1.mean()
        mean_nk_arr[fid] = g2.mean()

    ranks = (-f_stats).argsort().argsort() + 1
    import pandas as pd
    anova_df = pd.DataFrame({
        "feature_id": np.arange(N_SAE),
        "f_stat":     f_stats,
        "mean_t":     mean_t_arr,
        "mean_nk":    mean_nk_arr,
        "rank":       ranks,
    }).sort_values("rank")

    anova_df.to_csv(anova_cache, index=False)
    print(f"  ANOVA saved to {anova_cache}", flush=True)

# Select features: F >= MIN_F_STAT AND active
selected = anova_df[anova_df["f_stat"] >= MIN_F_STAT].copy()
print(f"  Features passing F >= {MIN_F_STAT}: {len(selected)} / {N_SAE}", flush=True)
top_fids = selected["feature_id"].astype(int).tolist()
fid_to_rank  = {int(r["feature_id"]): int(r["rank"])  for _, r in selected.iterrows()}
fid_to_fstat = {int(r["feature_id"]): float(r["f_stat"]) for _, r in selected.iterrows()}


# -------------------------------------------------------------------------------
# Step 4: Within-T/NK t-test gene signatures
# -------------------------------------------------------------------------------
print("[4/6] Computing t-test gene signatures (T/NK-restricted background)...", flush=True)

sig_cache = os.path.join(SCRIPT_DIR, "gene_signatures_tcnk_routeB.json")

if os.path.exists(sig_cache):
    with open(sig_cache) as f:
        gene_sigs = json.load(f)
    print(f"  Loaded {len(gene_sigs)} cached signatures", flush=True)
    cached_fids = set(int(k) for k in gene_sigs.keys())
    remaining = [fid for fid in top_fids if fid not in cached_fids]
else:
    gene_sigs = {}
    remaining = top_fids

print(f"  Features to compute: {len(remaining)}", flush=True)

for i, fid in enumerate(remaining):
    acts_f = X_sae[:, fid]   # (N_TCNK,)
    sorted_local = np.argsort(acts_f)[::-1]
    fg_local = sorted_local[:TOP_CELLS]
    fg_mask  = np.zeros(N_TCNK, dtype=bool)
    fg_mask[fg_local] = True
    bg_local = np.where(~fg_mask)[0]

    if len(bg_local) < 50:
        continue

    fg_expr = X_genes[fg_local]      # (TOP_CELLS, N_GENES)
    bg_expr = X_genes[bg_local]      # (N_BG, N_GENES)

    m1 = fg_expr.mean(axis=0)
    m2 = bg_expr.mean(axis=0)
    v1 = fg_expr.var(axis=0) / (len(fg_local) + 1e-10)
    v2 = bg_expr.var(axis=0) / (len(bg_local) + 1e-10)
    t  = (m1 - m2) / np.sqrt(v1 + v2 + 1e-10)
    t  = np.where(np.isnan(t) | np.isinf(t), -np.inf, t)

    top_gene_idx = np.argsort(t)[::-1][:TOP_GENES]
    top_genes    = gene_names[top_gene_idx].tolist()

    # Cell type composition in top-200 cells
    fg_celltypes = tcnk_celltypes[fg_local]
    unique_cts, counts = np.unique(fg_celltypes, return_counts=True)
    comp = {ct: round(float(c) / TOP_CELLS * 100, 1)
            for ct, c in zip(unique_cts, counts) if c / TOP_CELLS >= 0.01}

    fold = float(acts_f[fg_local].mean() / (acts_f.mean() + 1e-10))

    gene_sigs[str(fid)] = {
        "genes": top_genes,
        "celltype_composition": comp,
        "fold_enrichment": fold,
    }

    if (i + 1) % 25 == 0 or (i + 1) == len(remaining):
        with open(sig_cache, "w") as f:
            json.dump(gene_sigs, f, indent=2)
        print(f"  Progress: {i+1}/{len(remaining)} signatures", flush=True)

print(f"  Total signatures: {len(gene_sigs)}", flush=True)


# -------------------------------------------------------------------------------
# Step 5: MyGene.info RAG
# -------------------------------------------------------------------------------
print("[5/6] Fetching gene functions (MyGene.info)...", flush=True)

fn_cache_path = os.path.join(SCRIPT_DIR, "gene_function_cache.json")
if os.path.exists(fn_cache_path):
    with open(fn_cache_path) as f:
        gene_fn_cache = json.load(f)
else:
    gene_fn_cache = {}

all_genes_needed = set()
for fid in top_fids:
    sig = gene_sigs.get(str(fid), {})
    for g in sig.get("genes", []):
        all_genes_needed.add(g)

new_genes = [g for g in all_genes_needed if g not in gene_fn_cache]
print(f"  New genes to query: {len(new_genes)}", flush=True)

if new_genes:
    import mygene
    mg = mygene.MyGeneInfo()
    BATCH = 100
    for start in range(0, len(new_genes), BATCH):
        batch = new_genes[start:start+BATCH]
        try:
            results = mg.querymany(batch, scopes="symbol", fields="summary,name", species="human", verbose=False)
            for r in results:
                sym = r.get("query", "")
                summary = r.get("summary", "")
                name    = r.get("name", "")
                if summary:
                    gene_fn_cache[sym] = f"{name}: {summary}"
                elif name:
                    gene_fn_cache[sym] = name
                else:
                    gene_fn_cache[sym] = ""
            time.sleep(0.5)
        except Exception as e:
            print(f"  MyGene.info error for batch starting {start}: {e}", flush=True)

    with open(fn_cache_path, "w") as f:
        json.dump(gene_fn_cache, f, indent=2)
    print(f"  Gene function cache updated ({len(gene_fn_cache)} total genes)", flush=True)


# -------------------------------------------------------------------------------
# Step 6: Claude API annotation loop
# -------------------------------------------------------------------------------
print("[6/6] Running RAG-LLM annotation...", flush=True)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
ann_path = os.path.join(SCRIPT_DIR, "sae_annotations_tcnk_routeB.json")

if os.path.exists(ann_path):
    with open(ann_path) as f:
        results_list = json.load(f)
else:
    results_list = []

done_fids = {
    r["feature_id"] for r in results_list
    if r.get("confidence", "LOW") != "LOW"
    and r.get("programme_name", "") not in ("api call failed", "annotation parse error", "")
}
todo_fids = [fid for fid in top_fids if fid not in done_fids]
print(f"  Already done: {len(done_fids)}  Remaining: {len(todo_fids)}", flush=True)


def select_prompt(comp_dict):
    t_pct  = sum(v for k, v in comp_dict.items() if k in T_CELL_TYPES)
    nk_pct = sum(v for k, v in comp_dict.items() if k in NK_CELL_TYPES)
    if t_pct >= T_DOMINANT_THRESH * 100:
        return SYSTEM_PROMPT_TCELL, "T-dominant"
    elif nk_pct >= NK_DOMINANT_THRESH * 100:
        return SYSTEM_PROMPT_NK, "NK-dominant"
    else:
        return SYSTEM_PROMPT_MIXED, "Mixed"


def strip_json_markdown(text):
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


for i, fid in enumerate(todo_fids):
    sig  = gene_sigs.get(str(fid), {})
    genes = sig.get("genes", [])
    comp  = sig.get("celltype_composition", {})
    fold  = sig.get("fold_enrichment", 1.0)

    gene_functions_str = "\n".join(
        f"  {g}: {gene_fn_cache.get(g, 'function unknown')}"
        for g in genes
    )
    celltype_str = "\n".join(
        f"  {ct}: {pct:.1f}%" for ct, pct in sorted(comp.items(), key=lambda x: -x[1])
    )
    t_pct  = sum(v for k, v in comp.items() if k in T_CELL_TYPES)
    nk_pct = sum(v for k, v in comp.items() if k in NK_CELL_TYPES)
    dominant = f"T={t_pct:.0f}%  NK={nk_pct:.0f}%"

    system_prompt, prompt_type = select_prompt(comp)
    user_msg = USER_TEMPLATE.format(
        fid=fid,
        rank=fid_to_rank.get(fid, "?"),
        fstat=fid_to_fstat.get(fid, 0.0),
        dominant=dominant,
        n_genes=len(genes),
        gene_list="\n".join(f"  {j+1}. {g}" for j, g in enumerate(genes)),
        celltype_comp=celltype_str if celltype_str else "  (no composition data)",
        fold=fold,
        gene_functions=gene_functions_str if gene_functions_str else "  (no function data)",
    )

    record = {
        "feature_id": fid,
        "anova_rank": fid_to_rank.get(fid),
        "f_stat": fid_to_fstat.get(fid),
        "top_genes": genes,
        "celltype_composition": comp,
        "prompt_type": prompt_type,
    }

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text
            parsed = json.loads(strip_json_markdown(raw))
            record.update(parsed)
            break
        except json.JSONDecodeError:
            if attempt == 2:
                record.update({"programme_name": "annotation parse error",
                               "functional_description": raw[:300],
                               "key_evidence": [],
                               "confidence": "LOW",
                               "suggested_validation": ""})
        except Exception as e:
            if attempt == 2:
                record.update({"programme_name": "api call failed",
                               "functional_description": str(e),
                               "key_evidence": [],
                               "confidence": "LOW",
                               "suggested_validation": ""})
            time.sleep(5)

    results_list.append(record)
    time.sleep(SLEEP_API)

    if (i + 1) % 10 == 0 or (i + 1) == len(todo_fids):
        with open(ann_path, "w") as f:
            json.dump(results_list, f, indent=2)
        prog = record.get("programme_name", "")
        print(f"  [{i+1}/{len(todo_fids)}] F#{fid} ({prompt_type}): {prog}", flush=True)

print(f"\nDone. {len(results_list)} features annotated.", flush=True)
print(f"Saved: {ann_path}", flush=True)

# Quick summary
unique_names = len({r.get("programme_name","") for r in results_list})
print(f"Unique programme names: {unique_names} / {len(results_list)}", flush=True)
