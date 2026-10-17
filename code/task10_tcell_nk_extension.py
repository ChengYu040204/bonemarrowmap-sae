#!/usr/bin/env python3
"""
Task 10: T Cell and NK Cell SAE Feature Discovery and Annotation
================================================================

This is Dr. Ng Action Item 2: extend the SAE analysis pipeline to T cell
and NK cell lineages, following the same architecture used for B cells.

Key difference from task9:
  Tasks 8-9 worked with 209 features pre-selected by B-cell ANOVA.
  T/NK cells are barely represented in that feature set (<5% composition
  in top-200 cells for most features).

  Task 10 therefore adds a dedicated ANOVA step:
    Step 1: Run ANOVA across all 2048 SAE features with T/NK cells as
            the foreground group and all other cells as background.
    Step 2: Select features with F-stat >= MIN_F_STAT (data-driven cutoff,
            consistent with B-cell analysis which annotated all active
            features rather than a fixed top-N).
    Steps 3-7: t-test gene signatures + MyGene.info RAG + LLM annotation.

T/NK cell types (11 subtypes from BoneMarrowMap):
  T cells: CD4 Naive, CD4 Central Memory, CD4 Effector Memory,
           CD4 Regulatory, CD8 Naive, CD8 Central Memory,
           CD8 Effector Memory 1, CD8 Effector Memory 2, T Proliferating
  NK cells: NK, NK CD56high

System prompt design:
  Two prompts selected based on feature's cell-type composition:
    T_DOMINANT (>50% T cells in top-200): emphasise CD4 vs CD8, naive vs
      memory, regulatory vs effector, exhaustion, cytokine programmes.
    NK_DOMINANT (>50% NK cells in top-200): emphasise CD56bright vs
      CD56dim, KIR expression, cytotoxicity vs cytokine production.
    MIXED: general lymphocyte programme prompt.

Usage:
  cd "Papers & Datas/SAE training/sae_code/"
  export ANTHROPIC_API_KEY="sk-ant-..."
  /opt/miniconda3/envs/scanpy-env/bin/python task10_tcell_nk_extension.py

Outputs:
  tcell_nk_anova_results.csv        F-stats for all 2048 features
  gene_signatures_tcell_nk.json     per-feature t-test gene signatures
  gene_function_cache.json          MyGene.info cache (extended in-place)
  sae_annotations_tcell_nk.json     RAG-LLM annotations for top T/NK features

Author: SAE x Haematopoiesis project (FP-Decode), UCL Gurdon Studentship 2026
"""

import os
import json
import pickle
import time
import warnings
import csv as csv_module

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

ADATA_PATH      = "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad"
SAE_PKL         = "sae_results_all.pkl"

# Output files (this task)
ANOVA_CSV       = "tcell_nk_anova_results.csv"
SIG_FILE        = "gene_signatures_tcell_nk.json"
GENE_CACHE      = "gene_function_cache.json"
OUTPUT          = "sae_annotations_tcell_nk.json"

API_KEY         = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL           = "claude-haiku-4-5-20251001"

# T/NK lineage definition (from BoneMarrowMap cell types)
T_CELL_TYPES = [
    "CD4 Naive",
    "CD4 Central Memory",
    "CD4 Effector Memory",
    "CD4 Regulatory",
    "CD8 Naive",
    "CD8 Central Memory",
    "CD8 Effector Memory 1",
    "CD8 Effector Memory 2",
    "T Proliferating",
]
NK_CELL_TYPES = [
    "NK",
    "NK CD56high",
]
T_NK_TYPES = T_CELL_TYPES + NK_CELL_TYPES

# Feature selection: F-stat threshold (data-driven, consistent with B-cell
# analysis that used all active features rather than fixed top-N).
# MIN_F_STAT = 50 is a conservative starting point.
# After the first run, inspect tcell_nk_anova_results.csv to verify
# the F-stat distribution and adjust if needed.
MIN_F_STAT      = 50       # minimum ANOVA F-stat to include a feature
TOP_CELLS       = 200      # foreground cells per feature in t-test
TOP_GENES       = 15       # genes sent to LLM per feature
N_GLOBAL_BG     = 2000     # random global background size

# NK dominant threshold: if NK fraction in top-200 cells exceeds this,
# use NK-specific prompt; else use T cell prompt; below T_MIN use mixed.
NK_DOMINANT_THRESH = 0.40
T_DOMINANT_THRESH  = 0.40

SLEEP_API       = 1.2
SLEEP_MYGENE    = 1.0


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

N_CELLS_TOTAL, N_SAE_FEATURES = X_sae.shape
print(f"      X_sae shape: {X_sae.shape}", flush=True)


# =============================================================================
# STEP 2: Load AnnData
#
# We always need AnnData here (unlike task9 where it was optional).
# We need:
#   cell_types  -> to define T/NK vs other masks for ANOVA
#   gene_names  -> for t-test gene signatures
#   X_genes     -> expression matrix for t-test
# =============================================================================
print("[2/7] Loading AnnData ...", flush=True)

adata      = sc.read_h5ad(ADATA_PATH)
gene_names = np.array(adata.var_names)
cell_types = adata.obs["CellType"].values.astype(str)
X_raw      = adata.X
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

# Build T/NK and other masks
tcnk_mask  = np.array([ct in T_NK_TYPES  for ct in cell_types])
t_mask     = np.array([ct in T_CELL_TYPES for ct in cell_types])
nk_mask    = np.array([ct in NK_CELL_TYPES for ct in cell_types])
other_mask = ~tcnk_mask

tcnk_indices  = np.where(tcnk_mask)[0]
other_indices = np.where(other_mask)[0]

n_tcnk  = tcnk_mask.sum()
n_other = other_mask.sum()

print(f"      T/NK cells:  {n_tcnk:>7} ({n_tcnk/N_CELLS_TOTAL*100:.1f}%)", flush=True)
print(f"      Other cells: {n_other:>7} ({n_other/N_CELLS_TOTAL*100:.1f}%)", flush=True)

# Per-subtype counts
for ct in sorted(T_NK_TYPES):
    n = (cell_types == ct).sum()
    if n > 0:
        print(f"        {ct:<35} {n:>6}", flush=True)


# =============================================================================
# STEP 3: ANOVA — find T/NK-specific SAE features
#
# One-way ANOVA: group1 = T/NK activations, group2 = non-T/NK activations
#
# Vectorised across all N_SAE_FEATURES simultaneously:
#   F = (SS_between / df_between) / (SS_within / df_within)
#   df_between = 1  (2 groups - 1)
#   df_within  = N_CELLS_TOTAL - 2
#
# Why F-stat rather than t-test here:
#   ANOVA on activation values identifies features where T/NK cells have
#   systematically higher or lower mean activation than other cells.
#   This is the same approach used for B-cell feature selection (task5).
# =============================================================================
print("[3/7] Running T/NK ANOVA across all 2048 SAE features ...", flush=True)

if os.path.exists(ANOVA_CSV):
    print(f"      Found existing {ANOVA_CSV}, loading ...", flush=True)
    anova_df = pd.read_csv(ANOVA_CSV)
    f_stats_all = anova_df["f_stat"].values
    print(f"      Loaded {len(anova_df)} F-stats", flush=True)
else:
    x1 = X_sae[tcnk_mask]    # (N_TCNK, N_SAE_FEATURES)
    x2 = X_sae[other_mask]   # (N_other, N_SAE_FEATURES)

    n1, n2 = x1.shape[0], x2.shape[0]
    N_total = n1 + n2

    m1 = x1.mean(axis=0)
    m2 = x2.mean(axis=0)
    m_overall = (n1 * m1 + n2 * m2) / N_total

    ss_between = n1 * (m1 - m_overall) ** 2 + n2 * (m2 - m_overall) ** 2
    ss_within  = (((x1 - m1) ** 2).sum(axis=0) +
                  ((x2 - m2) ** 2).sum(axis=0))

    df_between = 1
    df_within  = N_total - 2

    f_stats_all = (ss_between / df_between) / (ss_within / df_within + 1e-10)
    f_stats_all = np.where(np.isnan(f_stats_all) | np.isinf(f_stats_all),
                           0.0, f_stats_all)

    anova_df = pd.DataFrame({
        "feature_id": np.arange(N_SAE_FEATURES),
        "f_stat":     f_stats_all,
        "mean_tcnk":  m1,
        "mean_other": m2,
    })
    anova_df = anova_df.sort_values("f_stat", ascending=False).reset_index(drop=True)
    anova_df["rank"] = anova_df.index + 1
    anova_df.to_csv(ANOVA_CSV, index=False)
    print(f"      ANOVA done. Saved to {ANOVA_CSV}", flush=True)
    print(f"      Top-5 features by F-stat:", flush=True)
    for _, row in anova_df.head(5).iterrows():
        print(f"        F#{int(row['feature_id']):>4}  F={row['f_stat']:.1f}  "
              f"mean_tcnk={row['mean_tcnk']:.4f}  mean_other={row['mean_other']:.4f}",
              flush=True)

    del x1, x2

# Select features by F-stat threshold (data-driven, consistent with B-cell
# analysis which used all active features rather than fixed top-N).
selected_df  = anova_df[anova_df["f_stat"] >= MIN_F_STAT].copy()
n_select     = len(selected_df)
top_fids     = selected_df["feature_id"].astype(int).tolist()
fid_to_rank  = {int(row["feature_id"]): int(row["rank"])
                for _, row in selected_df.iterrows()}
fid_to_fstat = {int(row["feature_id"]): float(row["f_stat"])
                for _, row in selected_df.iterrows()}

# Report threshold result and T/NK enrichment direction
enriched = [(fid, fid_to_fstat[fid]) for fid in top_fids
            if anova_df.loc[anova_df["feature_id"] == fid, "mean_tcnk"].values[0] >
               anova_df.loc[anova_df["feature_id"] == fid, "mean_other"].values[0]]
print(f"      MIN_F_STAT threshold: {MIN_F_STAT}", flush=True)
print(f"      Features passing threshold: {n_select} / {N_SAE_FEATURES}", flush=True)
print(f"        T/NK-higher: {len(enriched)}   "
      f"T/NK-lower: {n_select - len(enriched)}", flush=True)
if n_select > 0:
    print(f"        F-stat range: {selected_df['f_stat'].min():.1f} to "
          f"{selected_df['f_stat'].max():.1f}", flush=True)


# =============================================================================
# STEP 4: Check signature cache
# =============================================================================
print("[4/7] Checking signature cache ...", flush=True)

gene_signatures = {}   # fid -> list of gene symbols
ct_compositions = {}   # fid -> dict {celltype: pct}

if os.path.exists(SIG_FILE):
    with open(SIG_FILE) as fh:
        sig_raw = json.load(fh)
    for fid in top_fids:
        key = str(fid)
        if key in sig_raw:
            gene_signatures[fid] = sig_raw[key]["genes"]
            ct_compositions[fid] = sig_raw[key].get("celltype_composition", {})
    print(f"      Loaded {len(gene_signatures)} cached signatures", flush=True)
else:
    sig_raw = {}

fids_to_compute = [f for f in top_fids if f not in gene_signatures]
print(f"      Features needing t-test: {len(fids_to_compute)}", flush=True)


# =============================================================================
# STEP 5: Global t-test gene signatures for top T/NK features
#
# Foreground: top TOP_CELLS cells from ALL 161K by X_sae[:, fid] activation.
#   (We use ALL cells, not just T/NK, so we find the cells where this feature
#   is most strongly activated regardless of lineage. The composition analysis
#   tells us if those cells are predominantly T/NK.)
#
# Background: N_GLOBAL_BG random cells from the remaining 161K.
#   Global background ensures that T/NK-specific genes (e.g. CD3D, GNLY, NKG7)
#   stand out relative to erythroid/myeloid cells.
# =============================================================================
if fids_to_compute:
    print("[5/7] Computing global t-test signatures ...", flush=True)

    all_cell_indices = np.arange(N_CELLS_TOTAL)

    for i, fid in enumerate(fids_to_compute):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"      t-test: {i+1}/{len(fids_to_compute)} (fid={fid}) ...",
                  flush=True)

        acts_f = X_sae[:, fid]

        fg_idx = np.argsort(acts_f)[::-1][:TOP_CELLS]
        fg_set = set(fg_idx.tolist())
        non_fg = np.array([idx for idx in all_cell_indices if idx not in fg_set])
        bg_idx = np.random.choice(non_fg, size=min(N_GLOBAL_BG, len(non_fg)),
                                   replace=False)

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

        top_idx             = np.argsort(t_stats)[::-1][:TOP_GENES]
        gene_signatures[fid] = gene_names[top_idx].tolist()

        cts_fg = cell_types[fg_idx]
        ct_counts = pd.Series(cts_fg).value_counts(normalize=True)
        ct_compositions[fid] = {
            k: round(float(v) * 100, 1)
            for k, v in ct_counts.items()
            if float(v) >= 0.01
        }

    # Save signatures
    for fid in fids_to_compute:
        sig_raw[str(fid)] = {
            "genes":               gene_signatures[fid],
            "celltype_composition": ct_compositions[fid],
        }
    with open(SIG_FILE, "w") as fh:
        json.dump(sig_raw, fh, indent=2)
    print(f"      Saved {len(sig_raw)} signatures to {SIG_FILE}", flush=True)

else:
    print("[5/7] All signatures cached. Skipping.", flush=True)

# Free memory
del X_genes
del X_sae


# =============================================================================
# STEP 6: MyGene.info lookup
# =============================================================================
print("[6/7] Querying MyGene.info for gene functions ...", flush=True)

if os.path.exists(GENE_CACHE):
    with open(GENE_CACHE) as fh:
        gene_cache = json.load(fh)
    print(f"      Loaded {len(gene_cache)} cached gene entries", flush=True)
else:
    gene_cache = {}

all_genes_needed = set()
for fid in top_fids:
    if fid in gene_signatures:
        all_genes_needed.update(gene_signatures[fid])

uncached = sorted(all_genes_needed - set(gene_cache.keys()))
print(f"      Unique genes needed: {len(all_genes_needed)} | New lookups: {len(uncached)}",
      flush=True)

if uncached:
    mg         = mygene.MyGeneInfo()
    chunk_size = 500
    n_batches  = (len(uncached) - 1) // chunk_size + 1

    for batch_i in range(n_batches):
        chunk = uncached[batch_i * chunk_size : (batch_i + 1) * chunk_size]
        print(f"      Batch {batch_i+1}/{n_batches} ({len(chunk)} genes) ...", flush=True)
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
    print(f"      Cache updated: {len(gene_cache)} total", flush=True)


# =============================================================================
# STEP 7: RAG-enhanced LLM annotation
#
# Three system prompts chosen by cell-type composition of each feature:
#   T_DOMINANT  (>40% T cells in top-200): CD4/CD8 differentiation focus
#   NK_DOMINANT (>40% NK cells in top-200): cytotoxicity / CD56 focus
#   MIXED: general T/NK lymphocyte programme
# =============================================================================
print("[7/7] Running RAG-enhanced LLM annotation ...", flush=True)

if not API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not set.\n"
        "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
        "  /opt/miniconda3/envs/scanpy-env/bin/python task10_tcell_nk_extension.py"
    )

client = anthropic.Anthropic(api_key=API_KEY)

SYSTEM_PROMPT_TCELL = (
    "You are an expert immunologist and computational biologist specialising in "
    "T cell differentiation and function within the human haematopoietic system. "
    "You are annotating features from a Sparse Autoencoder (SAE) trained on scVI latent "
    "representations of the BoneMarrowMap human bone marrow atlas (~160K cells). "
    "Each feature represents a candidate Functional Programme (FP).\n\n"
    "CONTEXT: The gene signature below distinguishes cells highly activating this SAE "
    "feature from the global haematopoietic atlas (global t-test). "
    "The top-200 activating cells are predominantly T cells.\n\n"
    "Focus on the specific T cell programme: distinguish precisely between:\n"
    "  CD4 vs CD8 lineage identity (e.g. CD4 vs CD8A/B expression)\n"
    "  Naive vs central memory vs effector memory vs terminal effector\n"
    "  Regulatory T cell (Treg) programme (FOXP3, IL2RA, CTLA4)\n"
    "  T cell exhaustion (PDCD1, LAG3, HAVCR2, TIGIT)\n"
    "  Cytotoxic programme (GZMB, GZMK, PRF1, GNLY)\n"
    "  T cell proliferation (MKI67, STMN1, TOP2A)\n"
    "  Metabolic state (mitochondrial, glycolytic, mTOR)\n\n"
    "Do NOT use generic labels like 'T cell activation programme'. "
    "Name the specific subtype and mechanism "
    "(e.g. 'CD8 effector memory cytotoxic programme', "
    "'CD4 Treg FOXP3-mediated suppression', "
    "'CD8 naive T cell quiescence programme').\n\n"
    "Respond ONLY as valid JSON:\n"
    "{\n"
    '  "programme_name": "concise mechanistic name maximum 7 words",\n'
    '  "functional_description": "2-3 sentences on the specific T cell programme",\n'
    '  "key_evidence": ["GENE1: why diagnostic for this subtype", "GENE2: ...", "GENE3: ..."],\n'
    '  "confidence": "HIGH | MEDIUM | LOW",\n'
    '  "suggested_validation": "one specific experiment or marker"\n'
    "}"
)

SYSTEM_PROMPT_NK = (
    "You are an expert immunologist and computational biologist specialising in "
    "NK cell biology and innate lymphocyte function in human bone marrow. "
    "You are annotating features from a Sparse Autoencoder (SAE) trained on scVI latent "
    "representations of the BoneMarrowMap human bone marrow atlas (~160K cells). "
    "Each feature represents a candidate Functional Programme (FP).\n\n"
    "CONTEXT: The gene signature below distinguishes cells highly activating this SAE "
    "feature from the global haematopoietic atlas (global t-test). "
    "The top-200 activating cells are predominantly NK cells.\n\n"
    "Focus on the specific NK cell programme: distinguish precisely between:\n"
    "  CD56bright NK cells: cytokine producers, tissue-resident, IFN-gamma secretion\n"
    "  CD56dim NK cells: cytotoxic, KIR-expressing, ADCC\n"
    "  NK cell cytotoxicity programme (GNLY, GZMB, PRF1, NKG7)\n"
    "  NK cell education / licensing (KIR2DL1/2/3, KIR3DL1, NKG2A/C)\n"
    "  NK cell maturation from CD56bright to CD56dim\n"
    "  Adaptive NK cells (FcR-gamma negative, NKG2C+)\n"
    "  NK cell metabolic programme\n\n"
    "Do NOT use generic labels like 'NK cell activation'. "
    "Name the specific subtype and mechanism "
    "(e.g. 'CD56dim NK cytotoxic effector programme', "
    "'CD56bright NK cytokine secretion programme', "
    "'NK cell KIR-mediated self-tolerance').\n\n"
    "Respond ONLY as valid JSON:\n"
    "{\n"
    '  "programme_name": "concise mechanistic name maximum 7 words",\n'
    '  "functional_description": "2-3 sentences on the specific NK programme",\n'
    '  "key_evidence": ["GENE1: why diagnostic for this NK subtype", "GENE2: ...", "GENE3: ..."],\n'
    '  "confidence": "HIGH | MEDIUM | LOW",\n'
    '  "suggested_validation": "one specific experiment or marker"\n'
    "}"
)

SYSTEM_PROMPT_MIXED = (
    "You are an expert immunologist and computational biologist specialising in "
    "lymphocyte biology within the human haematopoietic system. "
    "You are annotating features from a Sparse Autoencoder (SAE) trained on scVI latent "
    "representations of the BoneMarrowMap human bone marrow atlas (~160K cells). "
    "Each feature represents a candidate Functional Programme (FP).\n\n"
    "CONTEXT: The gene signature below distinguishes cells highly activating this SAE "
    "feature from the global haematopoietic atlas (global t-test). "
    "The top-200 activating cells are a mix of T cells and NK cells, or other lymphocytes.\n\n"
    "Identify the shared lymphocyte programme: this may be a cross-lineage programme "
    "shared between T and NK cells (e.g. cytotoxic machinery, proliferation, "
    "mitochondrial metabolism, or a progenitor state). "
    "Be specific about the functional mechanism and which cell types share it.\n\n"
    "Respond ONLY as valid JSON:\n"
    "{\n"
    '  "programme_name": "concise mechanistic name maximum 7 words",\n'
    '  "functional_description": "2-3 sentences on the shared lymphocyte programme",\n'
    '  "key_evidence": ["GENE1: why diagnostic", "GENE2: ...", "GENE3: ..."],\n'
    '  "confidence": "HIGH | MEDIUM | LOW",\n'
    '  "suggested_validation": "one specific experiment"\n'
    "}"
)


def classify_feature(ct_comp):
    """Return 'tcell', 'nk', or 'mixed' based on composition."""
    t_frac  = sum(ct_comp.get(ct, 0) for ct in T_CELL_TYPES)  / 100.0
    nk_frac = sum(ct_comp.get(ct, 0) for ct in NK_CELL_TYPES) / 100.0
    if t_frac >= T_DOMINANT_THRESH:
        return "tcell"
    if nk_frac >= NK_DOMINANT_THRESH:
        return "nk"
    return "mixed"


def build_prompt(fid, rank, f_stat, ct_comp, genes_with_context):
    celltype_lines = "\n".join(
        f"  {pct:5.1f}%  {ct}"
        for ct, pct in sorted(ct_comp.items(), key=lambda x: -x[1])
    ) or "  (no cell-type breakdown available)"
    gene_lines = "\n".join(
        f"  {j+1:2d}. {entry}" for j, entry in enumerate(genes_with_context)
    )
    ftype = classify_feature(ct_comp)
    return (
        f"Feature ID: {fid}\n"
        f"T/NK ANOVA rank: {rank}  F-stat: {f_stat:.1f}\n"
        f"Feature type (by composition): {ftype}\n\n"
        f"Cell-type composition of top {TOP_CELLS} activating cells:\n"
        f"{celltype_lines}\n\n"
        f"Top {len(genes_with_context)} genes (global t-test ranked, with NCBI Gene context):\n"
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


# Resume logic
if os.path.exists(OUTPUT):
    with open(OUTPUT) as fh:
        results_list = json.load(fh)
    done_fids = {
        r["feature_id"] for r in results_list
        if r.get("confidence", "LOW") != "LOW" or
           r.get("programme_name", "") not in ("api call failed", "annotation parse error", "")
    }
    results_list = [r for r in results_list if r["feature_id"] in done_fids]
    print(f"      Resuming: {len(done_fids)} features already annotated", flush=True)
else:
    results_list = []
    done_fids    = set()

n_total   = len(top_fids)
n_skipped = 0
n_errors  = 0

for i, fid in enumerate(top_fids):
    if fid in done_fids:
        n_skipped += 1
        continue

    if fid not in gene_signatures:
        print(f"      WARNING: no signature for fid={fid}, skipping", flush=True)
        n_errors += 1
        continue

    genes      = gene_signatures[fid]
    ct_comp    = ct_compositions.get(fid, {})
    rank       = fid_to_rank.get(fid, 0)
    f_stat     = fid_to_fstat.get(fid, 0.0)
    ftype      = classify_feature(ct_comp)

    system_p = (
        SYSTEM_PROMPT_TCELL if ftype == "tcell" else
        SYSTEM_PROMPT_NK    if ftype == "nk"    else
        SYSTEM_PROMPT_MIXED
    )

    genes_with_context = []
    for g in genes:
        entry   = gene_cache.get(g, {})
        summary = entry.get("summary", "").strip()
        if len(summary) > 120:
            summary = summary[:117] + "..."
        genes_with_context.append(
            f"{g}: {summary}" if summary else f"{g}: (no functional summary available)"
        )

    user_msg   = build_prompt(fid, rank, f_stat, ct_comp, genes_with_context)
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
        except json.JSONDecodeError:
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
        "anova_rank":             rank,
        "anova_f_stat":           f_stat,
        "feature_type":           ftype,
        "top_genes":              genes,
        "celltype_composition":   ct_comp,
        "programme_name":         annotation.get("programme_name", ""),
        "functional_description": annotation.get("functional_description", ""),
        "key_evidence":           annotation.get("key_evidence", []),
        "confidence":             annotation.get("confidence", "LOW"),
        "suggested_validation":   annotation.get("suggested_validation", ""),
        "background":             f"Global random {N_GLOBAL_BG} from {N_CELLS_TOTAL}",
    }
    results_list.append(record)
    done_fids.add(fid)

    features_done = i + 1 - n_skipped
    if features_done % 10 == 0 or features_done == 1:
        print(
            f"      [{features_done:>3}/{n_total - n_skipped}] "
            f"fid={fid:>4} rank={rank:>3} [{ftype:>6}] | "
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
    f"  MIN_F_STAT threshold:         {MIN_F_STAT}\n"
    f"  Total T/NK features selected: {n_total}\n"
    f"  Skipped (cached):             {n_skipped}\n"
    f"  Newly annotated:              {n_total - n_skipped - n_errors}\n"
    f"  Errors:                       {n_errors}\n"
    f"  ANOVA results:                {ANOVA_CSV}\n"
    f"  Gene signatures:              {SIG_FILE}\n"
    f"  Output annotations:           {OUTPUT}\n"
    f"\n  To adjust threshold: edit MIN_F_STAT in the config section.\n"
    f"  Delete {SIG_FILE} and {OUTPUT} before re-running with a new threshold\n"
    f"  (ANOVA CSV is reusable across threshold changes).",
    flush=True,
)
