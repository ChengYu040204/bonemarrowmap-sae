"""
task12_dc_annotation.py
========================
DC-specific SAE feature annotation pipeline.

Steps:
  1. Load DC SAE feature matrix from sae_dc_results_all.pkl
  2. Load AnnData, extract DC cell indices and gene expression
  3. Multi-class ANOVA: cDC1 vs cDC2 vs pDC (F-test across 3 DC subtypes)
     Features with F >= 50 selected for annotation
  4. DC-internal Welch t-test: foreground = top-200 activating DC cells
     background = remaining ~11K DC cells
     NaN handled: replaced with -inf before argsort
  5. MyGene.info RAG: batch query NCBI Gene summaries for top-15 genes
     Reuses gene_function_cache.json if available (skip already-fetched genes)
  6. Claude haiku API annotation (dc-tailored prompts):
     cDC1-dominant: cross-presentation, BATF3/IRF8, CLEC9A, XCR1
     cDC2-dominant: MHC-II loading, CLEC12A, CD1c, monocyte-instructed DCs
     pDC-dominant:  IFN-I production, TLR7/9, SPIB, IRF7, SIGLEC-H
     Progenitor:    DC commitment, FLT3L-responsive, ZBTB46, CX3CR1

Run in terminal (NOT in notebook kernel -- avoids AnnData memory conflict):
  cd "Papers & Datas/SAE training/sae_code"
  export ANTHROPIC_API_KEY="sk-ant-..."
  /opt/miniconda3/envs/scanpy-env/bin/python task12_dc_annotation.py

Outputs:
  dc_anova_routeB.csv
  gene_signatures_dc_routeB.json
  sae_annotations_dc_routeB.json

Resume logic: if sae_annotations_dc_routeB.json exists, skips features that
already have confidence != "LOW" and programme_name is not a failure string.
"""

import csv
import json
import os
import pickle
import sys
import time

import anthropic
import anndata as ad
import numpy as np
import requests
from scipy.stats import f_oneway

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------
TOP_CELLS   = 200     # foreground cells per feature (highest activating)
TOP_GENES   = 15      # genes per signature for RAG
ANOVA_F_THR = 50.0    # F-statistic threshold for feature selection

DC_MATURE_TYPES = ["cDC1", "cDC2", "pDC"]
DC_PROGENITOR_TYPES = ["Pre-cDC", "ASDC", "Pre-pDC", "Pre-pDC Cycling"]
DC_TYPES = DC_MATURE_TYPES + DC_PROGENITOR_TYPES

ADATA_PATH = (
    "/Users/elvainyu/Desktop/申请与学习/暑研和实习/暑研/‼️2026暑研资料"
    "/Computational Biology/Papers & Datas/Papers' Dataset"
    "/BoneMarrowMap_Unsorted_scVI.h5ad"
)

# -------------------------------------------------------------------------------
# System prompts per dominant DC subtype
# -------------------------------------------------------------------------------
SYSTEM_PROMPT_CDC1 = """You are a dendritic cell biology expert specialising in human bone marrow haematopoiesis.
You will be shown a gene signature from a Sparse Autoencoder (SAE) trained on ~11,400 DC cells from the BoneMarrowMap atlas.
The top-200 activating cells are predominantly cDC1 (classical DC type 1).

cDC1 context: CLEC9A+, XCR1+, CADM1+, BATF3+, IRF8+ cells specialised in cross-presentation of dead-cell antigens to CD8 T cells.
Develop from pre-cDC progenitors under FLT3L + IRF8 transcriptional control.
Key programmes: antigen uptake, MHC-I cross-loading, IRF8/BATF3 identity, CLEC9A-mediated dead-cell sensing.

Task: identify the specific functional programme this SAE feature represents.
Be precise: distinguish IRF8-driven identity vs. cross-presentation machinery vs. cytokine production.
Do NOT generalise to "DC activation" when the gene list is informative.

Respond ONLY as valid JSON (no markdown) with exactly these keys:
{
  "programme_name": "<specific name, max 6 words>",
  "functional_description": "<2-3 sentences, mechanistic>",
  "key_evidence": ["<GENE: specific function in DC biology>", ...],
  "confidence": "HIGH|MEDIUM|LOW",
  "suggested_validation": "<one concrete experiment>"
}"""

SYSTEM_PROMPT_CDC2 = """You are a dendritic cell biology expert specialising in human bone marrow haematopoiesis.
You will be shown a gene signature from a Sparse Autoencoder (SAE) trained on ~11,400 DC cells from the BoneMarrowMap atlas.
The top-200 activating cells are predominantly cDC2 (classical DC type 2).

cDC2 context: CLEC12A+, CD1c+, CX3CR1+, KLF4+, NOTCH2-instructed DCs specialised in MHC-II antigen presentation to CD4 T cells.
Two distinct subsets: cDC2A (T-bet-, Foxp3+ Treg-inducing) and cDC2B (T-bet+, inflammatory).
Monocyte-instructed DCs (MIDCs) often co-cluster with cDC2.
Key programmes: MHC-II loading, endosomal antigen processing, Notch/KLF4 identity, CD4 T cell priming.

Task: identify the specific functional programme this SAE feature represents.
Distinguish cDC2A vs cDC2B identity if evidence allows. Flag MIDC contamination if monocyte markers present.

Respond ONLY as valid JSON (no markdown) with exactly these keys:
{
  "programme_name": "<specific name, max 6 words>",
  "functional_description": "<2-3 sentences, mechanistic>",
  "key_evidence": ["<GENE: specific function in DC biology>", ...],
  "confidence": "HIGH|MEDIUM|LOW",
  "suggested_validation": "<one concrete experiment>"
}"""

SYSTEM_PROMPT_PDC = """You are a dendritic cell biology expert specialising in human bone marrow haematopoiesis.
You will be shown a gene signature from a Sparse Autoencoder (SAE) trained on ~11,400 DC cells from the BoneMarrowMap atlas.
The top-200 activating cells are predominantly pDC (plasmacytoid dendritic cells).

pDC context: SIGLEC-H+, BST2+, LILRA4+, GZMB low, TLR7/9+, SPIB+, IRF7+ cells specialised in
massive Type I IFN (IFN-alpha/beta) production upon viral sensing. Unique lymphoid-biased ontogeny
from IL-7R+ pre-pDC progenitors expressing both lymphoid and DC genes.
Key programmes: TLR7/9 endosomal sensing, IRF7 activation, IFN-I secretion, pDC identity (SPIB/E2-2),
pre-pDC commitment, IFN-stimulated gene (ISG) expression.

Task: identify the specific functional programme this SAE feature represents.
Distinguish IFN production vs. identity maintenance vs. TLR signalling vs. pre-pDC commitment.

Respond ONLY as valid JSON (no markdown) with exactly these keys:
{
  "programme_name": "<specific name, max 6 words>",
  "functional_description": "<2-3 sentences, mechanistic>",
  "key_evidence": ["<GENE: specific function in DC biology>", ...],
  "confidence": "HIGH|MEDIUM|LOW",
  "suggested_validation": "<one concrete experiment>"
}"""

SYSTEM_PROMPT_PROGENITOR = """You are a dendritic cell biology expert specialising in human bone marrow haematopoiesis.
You will be shown a gene signature from a Sparse Autoencoder (SAE) trained on ~11,400 DC cells from the BoneMarrowMap atlas.
The top-200 activating cells are predominantly DC progenitors (Pre-cDC, ASDC, or Pre-pDC).

DC progenitor context:
  Pre-cDC: committed cDC1/cDC2 progenitor, FLT3+, ZBTB46+, lacks lineage identity genes
  ASDC (AXL+SIGLEC-6+ DC): rare bipotent progenitor giving rise to both cDC2 and pDC
  Pre-pDC: pDC-committed progenitor retaining some lymphoid markers (IL7R low)
  Pre-pDC Cycling: proliferating pre-pDC, cell cycle markers prominent

Key programmes: FLT3L signalling, DC lineage commitment, MHC-I/II upregulation,
AXL-mediated efferocytosis in ASDC, IL7R expression in pre-pDC.

Task: identify the specific functional programme this SAE feature represents.
Flag cell cycle artefacts if cycling markers (STMN1/PCNA/TYMS) dominate.

Respond ONLY as valid JSON (no markdown) with exactly these keys:
{
  "programme_name": "<specific name, max 6 words>",
  "functional_description": "<2-3 sentences, mechanistic>",
  "key_evidence": ["<GENE: specific function in DC biology>", ...],
  "confidence": "HIGH|MEDIUM|LOW",
  "suggested_validation": "<one concrete experiment>"
}"""

SYSTEM_PROMPT_MIXED = """You are a dendritic cell biology expert specialising in human bone marrow haematopoiesis.
You will be shown a gene signature from a Sparse Autoencoder (SAE) trained on ~11,400 DC cells from the BoneMarrowMap atlas.
The top-200 activating cells are a mixture of DC subtypes.

Context: Mixed-subtype activation suggests a shared DC programme operating across subtypes.
Candidate shared programmes: antigen presentation machinery (MHC-I/II), co-stimulatory signalling
(CD80/CD86/CD40), DC survival (FLT3L/BCL2), metabolic state (OXPHOS vs glycolysis),
cytokine-responsive ISGs, or general DC maturation signals.

Task: identify the shared or cross-subtype functional programme this SAE feature represents.
Do not assign to a specific DC subtype if the activation is truly mixed.

Respond ONLY as valid JSON (no markdown) with exactly these keys:
{
  "programme_name": "<specific name, max 6 words>",
  "functional_description": "<2-3 sentences, mechanistic>",
  "key_evidence": ["<GENE: specific function in DC biology>", ...],
  "confidence": "HIGH|MEDIUM|LOW",
  "suggested_validation": "<one concrete experiment>"
}"""

USER_TEMPLATE = """Feature ID: {feature_id}
ANOVA F-statistic (cDC1 vs cDC2 vs pDC): {f_stat:.1f}
ANOVA rank (among all active DC features): {anova_rank}

Top {n_genes} genes by DC-internal Welch t-statistic (foreground = top-200 activating DC cells):
{gene_list}

Cell-type composition of top-200 activating DC cells:
{celltype_composition}

Gene functional context (NCBI Gene summaries):
{gene_functions}

Identify the specific DC functional programme this SAE feature represents."""

# -------------------------------------------------------------------------------
# Step 1: Load DC SAE feature matrix
# -------------------------------------------------------------------------------
print("[1/6] Loading DC SAE feature matrix...")
with open(os.path.join(SCRIPT_DIR, "sae_dc_results_all.pkl"), "rb") as f:
    data = pickle.load(f)

if hasattr(data[1], "detach"):
    X_sae = data[1].detach().cpu().numpy()
else:
    X_sae = np.array(data[1])

print(f"      X_sae shape: {X_sae.shape}")
n_cells_dc, n_features = X_sae.shape
active_features = (X_sae.max(axis=0) > 1e-6).sum()
print(f"      Active features: {active_features}/{n_features}")

# -------------------------------------------------------------------------------
# Step 2: Load AnnData, extract DC cells, get gene expression
# -------------------------------------------------------------------------------
print("[2/6] Loading AnnData and extracting DC cells...")
adata = ad.read_h5ad(ADATA_PATH)
print(f"      adata shape: {adata.shape}")

dc_mask = adata.obs["CellType"].isin(DC_TYPES)
dc_cell_types = adata.obs["CellType"][dc_mask].values

print(f"      DC cells in adata: {dc_mask.sum()}")
for ct in DC_TYPES:
    n = (dc_cell_types == ct).sum()
    print(f"        {ct:<25} {n:>6}")

# Load global indices from extraction script
with open(os.path.join(SCRIPT_DIR, "dc_global_indices.pkl"), "rb") as f:
    dc_global_indices = pickle.load(f)
print(f"      dc_global_indices: {len(dc_global_indices)} entries")

# Extract gene expression matrix for DC cells
X_genes = adata.X
gene_names = list(adata.var_names)

if X_sae.shape[0] != len(dc_global_indices):
    print(f"WARNING: X_sae rows ({X_sae.shape[0]}) != dc_global_indices ({len(dc_global_indices)})")
    sys.exit(1)

# -------------------------------------------------------------------------------
# Step 3: Multi-class ANOVA (cDC1 vs cDC2 vs pDC)
# Only 3 mature subtypes used for ANOVA; progenitors included in t-test background
# -------------------------------------------------------------------------------
print("[3/6] Computing ANOVA (cDC1 vs cDC2 vs pDC) across active features...")

active_fids = [i for i in range(n_features) if X_sae[:, i].max() > 1e-6]
print(f"      Active features to test: {len(active_fids)}")

ctype_local = np.array([dc_cell_types[i] for i in range(len(dc_global_indices))])
cdc1_mask = (ctype_local == "cDC1")
cdc2_mask = (ctype_local == "cDC2")
pdc_mask  = (ctype_local == "pDC")

print(f"      cDC1: {cdc1_mask.sum()}, cDC2: {cdc2_mask.sum()}, pDC: {pdc_mask.sum()}")

if cdc1_mask.sum() < 10 or cdc2_mask.sum() < 10 or pdc_mask.sum() < 10:
    print("WARNING: Insufficient cells for 3-class ANOVA. Falling back to binary (cDC vs pDC).")
    group_a = X_sae[cdc1_mask | cdc2_mask]
    group_b = X_sae[pdc_mask]
    anova_results = []
    for fid in active_fids:
        f, _ = f_oneway(group_a[:, fid], group_b[:, fid])
        if np.isnan(f) or np.isinf(f):
            f = 0.0
        anova_results.append((fid, f))
else:
    anova_results = []
    for fid in active_fids:
        f, _ = f_oneway(
            X_sae[cdc1_mask, fid],
            X_sae[cdc2_mask, fid],
            X_sae[pdc_mask,  fid],
        )
        if np.isnan(f) or np.isinf(f):
            f = 0.0
        anova_results.append((fid, f))

anova_results.sort(key=lambda x: x[1], reverse=True)
selected = [(fid, f) for fid, f in anova_results if f >= ANOVA_F_THR]
print(f"      Features with F >= {ANOVA_F_THR}: {len(selected)}")
if selected:
    print(f"      Top F-stat: {selected[0][1]:.1f} (feature #{selected[0][0]})")
    print(f"      Rank-10 F:  {selected[min(9,len(selected)-1)][1]:.1f}")

anova_csv_path = os.path.join(SCRIPT_DIR, "dc_anova_routeB.csv")
with open(anova_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["anova_rank", "feature_id", "f_stat"])
    for rank, (fid, f) in enumerate(anova_results, start=1):
        writer.writerow([rank, fid, f"{f:.2f}"])
print(f"      Saved {anova_csv_path}")

# -------------------------------------------------------------------------------
# Step 4: DC-internal Welch t-test gene signatures
# -------------------------------------------------------------------------------
print("[4/6] Computing DC-internal t-test gene signatures...")

sig_cache_path = os.path.join(SCRIPT_DIR, "gene_signatures_dc_routeB.json")
if os.path.exists(sig_cache_path):
    with open(sig_cache_path) as f:
        gene_signatures = json.load(f)
    print(f"      Loaded cached signatures for {len(gene_signatures)} features")
else:
    gene_signatures = {}

# Extract DC gene expression once (avoid repeated indexing into full adata.X)
import scipy.sparse as sp
if sp.issparse(X_genes):
    X_dc = X_genes[dc_global_indices].toarray().astype(np.float32)
else:
    X_dc = np.array(X_genes[dc_global_indices], dtype=np.float32)
print(f"      X_dc (gene matrix) shape: {X_dc.shape}")

n_dc_local = X_dc.shape[0]
all_local_idx = np.arange(n_dc_local)

new_sigs = 0
for rank, (fid, f_stat) in enumerate(selected, start=1):
    fid_str = str(fid)
    if fid_str in gene_signatures:
        continue

    acts = X_sae[:, fid]
    fg_local = np.argsort(acts)[::-1][:TOP_CELLS]
    fg_mask = np.zeros(n_dc_local, dtype=bool)
    fg_mask[fg_local] = True
    bg_local = all_local_idx[~fg_mask]

    fg_expr = X_dc[fg_local]
    bg_expr = X_dc[bg_local]
    n_fg, n_bg = len(fg_local), len(bg_local)

    m1 = fg_expr.mean(axis=0)
    m2 = bg_expr.mean(axis=0)
    v1 = fg_expr.var(axis=0) / n_fg
    v2 = bg_expr.var(axis=0) / n_bg
    t_stats = (m1 - m2) / np.sqrt(v1 + v2 + 1e-10)
    t_stats = np.where(np.isnan(t_stats) | np.isinf(t_stats), -np.inf, t_stats)

    top_idx = np.argsort(t_stats)[::-1][:TOP_GENES]
    top_genes = [gene_names[i] for i in top_idx]

    fg_types = ctype_local[fg_local]
    type_counts = {}
    for ct in DC_TYPES:
        c = (fg_types == ct).sum()
        if c > 0:
            type_counts[ct] = round(c / len(fg_local) * 100, 1)

    gene_signatures[fid_str] = {
        "anova_rank":          rank,
        "f_stat":              round(f_stat, 2),
        "genes":               top_genes,
        "celltype_composition": type_counts,
    }
    new_sigs += 1

    if new_sigs % 20 == 0:
        with open(sig_cache_path, "w") as f_out:
            json.dump(gene_signatures, f_out, indent=2)
        print(f"      Checkpoint: {new_sigs} new signatures computed", flush=True)

with open(sig_cache_path, "w") as f_out:
    json.dump(gene_signatures, f_out, indent=2)
print(f"      Saved gene_signatures_dc_routeB.json ({len(gene_signatures)} features)")

del X_dc

# -------------------------------------------------------------------------------
# Step 5: MyGene.info RAG (batch gene queries)
# -------------------------------------------------------------------------------
print("[5/6] Fetching gene function summaries from MyGene.info...")

cache_path = os.path.join(SCRIPT_DIR, "gene_function_cache.json")
if os.path.exists(cache_path):
    with open(cache_path) as f:
        gene_cache = json.load(f)
    print(f"      Loaded {len(gene_cache)} cached gene summaries")
else:
    gene_cache = {}

all_genes_needed = set()
for sig in gene_signatures.values():
    all_genes_needed.update(sig["genes"])
new_genes = [g for g in all_genes_needed if g not in gene_cache]
print(f"      New genes to query: {len(new_genes)}")

BATCH_SZ = 100
for i in range(0, len(new_genes), BATCH_SZ):
    batch = new_genes[i : i + BATCH_SZ]
    try:
        resp = requests.post(
            "https://mygene.info/v3/gene",
            json={"ids": batch, "fields": "symbol,summary", "species": "human"},
            timeout=30,
        )
        if resp.status_code == 200:
            for item in resp.json():
                sym = item.get("symbol", item.get("query", ""))
                summary = item.get("summary", "")
                if sym:
                    gene_cache[sym] = summary[:300] if summary else ""
        time.sleep(1)
    except Exception as e:
        print(f"      MyGene.info query error (batch {i//BATCH_SZ}): {e}")

with open(cache_path, "w") as f:
    json.dump(gene_cache, f, indent=2)
print(f"      gene_function_cache.json updated ({len(gene_cache)} total entries)")

# -------------------------------------------------------------------------------
# Step 6: Claude haiku API annotation
# -------------------------------------------------------------------------------
print("[6/6] Running Claude API annotation...")

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not set. Export it before running this script.")
    sys.exit(1)

client = anthropic.Anthropic(api_key=api_key)

ann_path = os.path.join(SCRIPT_DIR, "sae_annotations_dc_routeB.json")
if os.path.exists(ann_path):
    with open(ann_path) as f:
        results_list = json.load(f)
    done_fids = {
        r["feature_id"] for r in results_list
        if r.get("confidence", "LOW") != "LOW"
        and r.get("programme_name", "") not in (
            "", "api call failed", "annotation parse error"
        )
    }
    print(f"      Resuming: {len(done_fids)} already annotated successfully")
else:
    results_list = []
    done_fids = set()

def pick_prompt(composition: dict) -> str:
    cdc1_pct = composition.get("cDC1", 0)
    cdc2_pct = composition.get("cDC2", 0)
    pdc_pct  = composition.get("pDC", 0)
    prog_pct = sum(composition.get(ct, 0) for ct in DC_PROGENITOR_TYPES)
    dominant = max(cdc1_pct, cdc2_pct, pdc_pct, prog_pct)
    if dominant < 40:
        return SYSTEM_PROMPT_MIXED
    if cdc1_pct == dominant:
        return SYSTEM_PROMPT_CDC1
    if cdc2_pct == dominant:
        return SYSTEM_PROMPT_CDC2
    if pdc_pct  == dominant:
        return SYSTEM_PROMPT_PDC
    return SYSTEM_PROMPT_PROGENITOR

def strip_json_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()

total = len(selected)
for idx, (fid, f_stat) in enumerate(selected):
    if fid in done_fids:
        continue

    fid_str = str(fid)
    sig = gene_signatures.get(fid_str)
    if sig is None:
        continue

    rank      = sig["anova_rank"]
    genes     = sig["genes"]
    comp      = sig["celltype_composition"]
    gene_list = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(genes))

    comp_str  = "\n".join(
        f"  {ct}: {pct}%" for ct, pct in sorted(comp.items(), key=lambda x: -x[1])
    )
    func_lines = []
    for g in genes:
        summary = gene_cache.get(g, "")
        short   = summary[:150] if summary else "No summary available"
        func_lines.append(f"  {g}: {short}")
    gene_functions = "\n".join(func_lines)

    user_msg = USER_TEMPLATE.format(
        feature_id=fid,
        f_stat=f_stat,
        anova_rank=rank,
        n_genes=len(genes),
        gene_list=gene_list,
        celltype_composition=comp_str,
        gene_functions=gene_functions,
    )

    system_prompt = pick_prompt(comp)

    record = {
        "feature_id":          fid,
        "anova_rank":          rank,
        "f_stat":              f_stat,
        "top_genes":           genes,
        "celltype_composition": comp,
        "programme_name":      "api call failed",
        "functional_description": "",
        "key_evidence":        [],
        "confidence":          "LOW",
        "suggested_validation": "",
    }

    for attempt in range(3):
        try:
            time.sleep(1)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=900,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = strip_json_markdown(resp.content[0].text)
            parsed = json.loads(raw)
            record.update(parsed)
            record["feature_id"] = fid
            record["anova_rank"] = rank
            record["f_stat"]     = f_stat
            record["top_genes"]  = genes
            record["celltype_composition"] = comp
            break
        except json.JSONDecodeError:
            record["programme_name"] = "annotation parse error"
        except Exception as e:
            print(f"      API error (fid={fid}, attempt={attempt+1}): {e}", flush=True)
            time.sleep(5)

    results_list.append(record)
    done_fids.add(fid)

    prog_str = record.get("programme_name", "")[:55]
    print(
        f"  [{idx+1:>3}/{total}] F#{fid:<5} rank={rank:<4} "
        f"conf={record.get('confidence','?'):<7} {prog_str}",
        flush=True,
    )

    if len(results_list) % 10 == 0:
        with open(ann_path, "w") as f:
            json.dump(results_list, f, indent=2)
        print(f"      Checkpoint saved ({len(results_list)} annotations)", flush=True)

with open(ann_path, "w") as f:
    json.dump(results_list, f, indent=2)

# -------------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------------
total_ann   = len(results_list)
conf_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
names       = set()
for r in results_list:
    conf = r.get("confidence", "LOW")
    conf_counts[conf] = conf_counts.get(conf, 0) + 1
    names.add(r.get("programme_name", ""))

print("\n=== task12 Complete ===")
print(f"Total annotated:   {total_ann}")
print(f"Unique names:      {len(names)}/{total_ann}")
print(f"HIGH confidence:   {conf_counts['HIGH']}")
print(f"MEDIUM confidence: {conf_counts['MEDIUM']}")
print(f"LOW confidence:    {conf_counts['LOW']}")
print(f"\nOutput: sae_annotations_dc_routeB.json")
