"""
dc_full_pipeline.py
===================
DC Route B: Activation Extraction -> SAE Training -> Annotation
Run Cell by Cell in VSCode Python Interactive (Shift+Enter per Cell).

Stage 1 (Cell 0)        : Extract scVI encoder activations for ~11,433 DC cells
Stage 2 (Cells 1 to 4)  : Normalise, ablate, train, extract DC-specific SAE
Stage 3 (Cells 5 to 6)  : ANOVA + DC-internal t-test + RAG-LLM annotation

WORKFLOW
  Cell 0    Extract DC activations, save 4 pkl files
  Cell 0.5  Clear memory (scVI model + AnnData released before SAE training)
  Cell 1    Load pkl files, inspect distribution
  Cell 1.5  Normalise activations, save dc_activation_norm_params.pt
  Cell 2    Ablation sweep over l1_penalty candidates
  Cell 3    Train final SAE with best l1_penalty (seed=42)  <-- update BEST_L1 after Cell 2
  Cell 4    Extract full DC feature matrix (11K cells x 2048 features)
  Cell 5    ANOVA (cDC1 vs cDC2 vs pDC) + DC-internal Welch t-test gene signatures
  Cell 6    RAG via MyGene.info + Claude API annotation

REQUIREMENTS
  conda activate scanpy-env
  pip install anthropic --break-system-packages   (if not already installed)
  export ANTHROPIC_API_KEY="sk-ant-..."            (required for Cell 6)
"""

# %%
# Cell 0: DC Activation Extraction / 提取 DC 细胞 scVI 激活值
import os
import pickle
import random

import scanpy as sc
import scvi
import torch
from tqdm import tqdm
from scvi.dataloaders import AnnDataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
print("scvi-tools version:", scvi.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

DC_MATURE_TYPES = ["cDC1", "cDC2", "pDC"]
DC_PROGENITOR_TYPES = ["Pre-cDC", "ASDC", "Pre-pDC", "Pre-pDC Cycling"]
DC_TYPES = DC_MATURE_TYPES + DC_PROGENITOR_TYPES

EVAL_FRACTION = 0.1

DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "../../Papers' Dataset/BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad",
)
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../../scVI_model_unsorted")

print("[1/6] Loading full AnnData (backed='r')...")
adata = sc.read_h5ad(DATA_PATH, backed="r")
print(f"      Full dataset shape: {adata.shape}")

print("[2/6] Subsetting to Unsorted + Healthy_Reference...")
sorting_col = "Sorting" if "Sorting" in adata.obs.columns else "SortingStatus"
unsorted_mask = adata.obs[sorting_col].isin(["Unsorted", "Healthy_Reference"])
adata_unsorted = adata[unsorted_mask].to_memory()
print(f"      Unsorted subset shape: {adata_unsorted.shape}")

print("[3/6] Selecting HVGs (n_top_genes=4000, same as scVI training)...")
sc.pp.highly_variable_genes(
    adata_unsorted,
    n_top_genes=4000,
    subset=True,
    flavor="seurat_v3",
    layer="counts",
)
print(f"      HVG subset shape: {adata_unsorted.shape}")

print("[4/6] Setting up AnnData for scVI and loading model...")
scvi.model.SCVI.setup_anndata(
    adata_unsorted,
    layer="counts",
    batch_key="Study",
)
model = scvi.model.SCVI.load(MODEL_PATH, adata=adata_unsorted)
vae = model.module
vae.eval()
print("      Model loaded successfully.")

vae_encode = vae.z_encoder
activations = {}

def getActivation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def extract_activations(indices, shuffle, desc):
    hook = vae_encode.encoder.register_forward_hook(getActivation("encoder"))
    loader = iter(AnnDataLoader(
        model.adata_manager,
        indices=indices,
        shuffle=shuffle,
        batch_size=32,
    ))
    matrix = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            vae_encode(batch["X"].to(device))
            matrix.append(activations["encoder"].to("cpu"))
    hook.remove()
    return torch.cat(matrix, dim=0)

print("[5/6] Filtering DC cells from adata_unsorted...")
dc_mask = adata_unsorted.obs["CellType"].isin(DC_TYPES)
n_dc = dc_mask.sum()
print(f"      DC cell count: {n_dc}")

print("      Per-type breakdown:")
for ct in DC_TYPES:
    n = (adata_unsorted.obs["CellType"] == ct).sum()
    marker = " (mature)" if ct in DC_MATURE_TYPES else " (progenitor)"
    print(f"        {ct:<25}{marker:<14} {n:>6}")

if n_dc == 0:
    raise ValueError(
        "No DC cells found. Check CellType labels match DC_TYPES list. "
        "Run adata_unsorted.obs['CellType'].unique() to inspect."
    )

dc_int_idx = [
    adata_unsorted.obs_names.get_loc(bc)
    for bc in adata_unsorted.obs_names[dc_mask]
]
print(f"      Integer index range: {min(dc_int_idx)} to {max(dc_int_idx)}")

random.seed(42)
dc_shuffled = dc_int_idx.copy()
random.shuffle(dc_shuffled)
n_eval  = max(1, int(len(dc_shuffled) * EVAL_FRACTION))
n_train = len(dc_shuffled) - n_eval
dc_train_idx = sorted(dc_shuffled[:n_train])
dc_eval_idx  = sorted(dc_shuffled[n_train:])
print(f"      Train: {len(dc_train_idx)}, Eval: {len(dc_eval_idx)}, Total: {n_dc}")

print("[6/6] Extracting activations for DC cells...")
dc_train_matrix = extract_activations(dc_train_idx, shuffle=True,  desc="  DC train")
print(f"      dc_train shape: {dc_train_matrix.shape}")
with open(os.path.join(SCRIPT_DIR, "dc_train_activation_matrix.pkl"), "wb") as f:
    pickle.dump(dc_train_matrix, f)

dc_eval_matrix = extract_activations(dc_eval_idx, shuffle=True, desc="  DC eval ")
print(f"      dc_eval  shape: {dc_eval_matrix.shape}")
with open(os.path.join(SCRIPT_DIR, "dc_eval_activation_matrix.pkl"), "wb") as f:
    pickle.dump(dc_eval_matrix, f)

dc_all_matrix = extract_activations(dc_int_idx, shuffle=False, desc="  DC all  ")
print(f"      dc_all   shape: {dc_all_matrix.shape}")
with open(os.path.join(SCRIPT_DIR, "dc_all_activation_matrix.pkl"), "wb") as f:
    pickle.dump(dc_all_matrix, f)

with open(os.path.join(SCRIPT_DIR, "dc_global_indices.pkl"), "wb") as f:
    pickle.dump(dc_int_idx, f)
print(f"      dc_global_indices.pkl saved ({len(dc_int_idx)} indices)")

print("\nCell 0 complete. Output files:")
for fname in [
    "dc_train_activation_matrix.pkl",
    "dc_eval_activation_matrix.pkl",
    "dc_all_activation_matrix.pkl",
    "dc_global_indices.pkl",
]:
    fpath = os.path.join(SCRIPT_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {fname:<40}  {size_mb:.1f} MB")

# %%
# Cell 0.5: Memory Cleanup / 释放 scVI 模型和 AnnData 内存
import gc

del adata
del adata_unsorted
del model
del vae
del vae_encode
del activations
del dc_train_matrix
del dc_eval_matrix
del dc_all_matrix
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Memory released. Kernel is now ready for SAE training.")

# %%
# Cell 1: Load DC Activation pkl Files / 加载 DC 激活值 pkl 文件
import os
import pickle
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading DC activation pkl files...")
with open(os.path.join(SCRIPT_DIR, "dc_train_activation_matrix.pkl"), "rb") as f:
    train_raw = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, "dc_eval_activation_matrix.pkl"), "rb") as f:
    eval_raw = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, "dc_all_activation_matrix.pkl"), "rb") as f:
    all_raw = pickle.load(f)

def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)

train_np = to_numpy(train_raw)
eval_np  = to_numpy(eval_raw)
all_np   = to_numpy(all_raw)

print(f"dc_train shape: {train_np.shape}")
print(f"dc_eval  shape: {eval_np.shape}")
print(f"dc_all   shape: {all_np.shape}")

print("\nActivation distribution (train set):")
print(f"  mean = {train_np.mean():.4f}")
print(f"  std  = {train_np.std():.4f}")
print(f"  min  = {train_np.min():.4f}")
print(f"  max  = {train_np.max():.4f}")
print(f"  % > 0 = {(train_np > 0).mean() * 100:.1f}%")

# %%
# Cell 1.5: Normalise Activations / 激活值归一化（中心化 + L2 缩放）
print("Normalising DC activations...")

mean_act = train_np.mean(axis=0, keepdims=True)

train_centered = train_np - mean_act
eval_centered  = eval_np  - mean_act
all_centered   = all_np   - mean_act

train_norms = np.linalg.norm(train_centered, axis=1)
scale = np.percentile(train_norms, 95)
print(f"  95th percentile L2 norm (scale): {scale:.4f}")

train_normed = train_centered / scale
eval_normed  = eval_centered  / scale
all_normed   = all_centered   / scale

print("Post-normalisation stats (train):")
print(f"  mean = {train_normed.mean():.6f}  (target ~0)")
print(f"  std  = {train_normed.std():.4f}")
print(f"  max  = {train_normed.max():.4f}  (target < 5)")

norm_params = {
    "mean_act": torch.tensor(mean_act, dtype=torch.float32),
    "scale":    torch.tensor(scale,    dtype=torch.float32),
}
torch.save(norm_params, os.path.join(SCRIPT_DIR, "dc_activation_norm_params.pt"))
print("Saved dc_activation_norm_params.pt")

# %%
# Cell 2: Ablation Sweep over l1_penalty / 消融实验：选最优 l1_penalty
import sys
sys.path.insert(0, SCRIPT_DIR)
from dictionary_learning import AutoEncoder, trainSAE
from dictionary_learning.trainers.standard import StandardTrainer

INPUT_DIM  = 128
HIDDEN_DIM = 2048
BATCH_SIZE = 512
LR         = 1e-3

train_tensor = torch.tensor(train_normed, dtype=torch.float32)
eval_tensor  = torch.tensor(eval_normed,  dtype=torch.float32)

CANDIDATES = [1e-3, 5e-3, 1e-2, 2e-2]

print(f"{'l1_penalty':<14} {'Dead':>8} {'Active':>8} {'L0':>8} {'frac_var':>10}")
print("-" * 55)

for l1 in CANDIDATES:
    torch.manual_seed(0)
    ae = AutoEncoder(INPUT_DIM, HIDDEN_DIM)
    total_steps = len(train_tensor) // BATCH_SIZE * 8

    trainer_cfg = {
        "trainer":              StandardTrainer,
        "dict_size":            HIDDEN_DIM,
        "activation_dim":       INPUT_DIM,
        "lr":                   LR,
        "l1_penalty":           l1,
        "warmup_steps":         max(100, total_steps // 20),
        "sparsity_warmup_steps": max(200, total_steps // 10),
        "steps":                total_steps,
        "resample_steps":       max(1000, total_steps // 2),
        "device":               "cpu",
    }

    trainSAE(
        data=train_tensor,
        trainer_configs=[trainer_cfg],
        save_dir=os.path.join(SCRIPT_DIR, f"sae_dc_ablation_l1_{l1}"),
        steps=total_steps,
        log_steps=max(100, total_steps // 20),
        normalize_activations=False,
    )

    ae_path = os.path.join(SCRIPT_DIR, f"sae_dc_ablation_l1_{l1}", "trainer_0", "ae.pt")
    ae_loaded = torch.load(ae_path, map_location="cpu")
    ae_loaded.eval()

    with torch.no_grad():
        recons, features = ae_loaded(eval_tensor)

    features_np = features.numpy()
    dead         = (features_np.max(axis=0) < 1e-6).sum()
    active       = HIDDEN_DIM - dead
    l0           = (features_np > 0).sum(axis=1).mean()
    ss_res       = ((eval_tensor.numpy() - recons.numpy()) ** 2).sum()
    ss_tot       = ((eval_tensor.numpy() - eval_tensor.numpy().mean(axis=0)) ** 2).sum()
    frac_var     = 1 - ss_res / (ss_tot + 1e-10)

    print(f"{l1:<14.0e} {dead:>8}/{HIDDEN_DIM} {active:>8} {l0:>8.1f} {frac_var:>10.4f}")

print("\nTarget: Dead~0, L0=20-80, frac_var>0.85")
print("Update BEST_L1 in Cell 3 based on the table above.")

# %%
# Cell 3: Train Final DC SAE with Best l1_penalty / 训练最终 DC SAE
# Update BEST_L1 based on Cell 2 ablation table before running this cell.

BEST_L1 = 2e-2   # <--- UPDATE THIS based on Cell 2 output

torch.manual_seed(42)
print(f"Training final DC SAE (l1={BEST_L1}, seed=42)...")

train_tensor = torch.tensor(train_normed, dtype=torch.float32)
eval_tensor  = torch.tensor(eval_normed,  dtype=torch.float32)

total_steps = len(train_tensor) // BATCH_SIZE * 8
warmup      = max(100, total_steps // 20)
sp_warmup   = max(200, total_steps // 10)
resample    = max(1000, total_steps // 2)

print(f"  total_steps={total_steps}, warmup={warmup}, resample={resample}")

trainer_cfg = {
    "trainer":               StandardTrainer,
    "dict_size":             HIDDEN_DIM,
    "activation_dim":        INPUT_DIM,
    "lr":                    LR,
    "l1_penalty":            BEST_L1,
    "warmup_steps":          warmup,
    "sparsity_warmup_steps": sp_warmup,
    "steps":                 total_steps,
    "resample_steps":        resample,
    "device":                "cpu",
}

SAVE_DIR = os.path.join(SCRIPT_DIR, "sae_dc_final")
trainSAE(
    data=train_tensor,
    trainer_configs=[trainer_cfg],
    save_dir=SAVE_DIR,
    steps=total_steps,
    log_steps=max(100, total_steps // 20),
    normalize_activations=False,
)

ae_path  = os.path.join(SAVE_DIR, "trainer_0", "ae.pt")
ae_final = torch.load(ae_path, map_location="cpu")
ae_final.eval()

with torch.no_grad():
    recons_eval, features_eval = ae_final(eval_tensor)

features_eval_np  = features_eval.numpy()
dead_final        = (features_eval_np.max(axis=0) < 1e-6).sum()
active_final      = HIDDEN_DIM - dead_final
l0_final          = (features_eval_np > 0).sum(axis=1).mean()
ss_res            = ((eval_tensor.numpy() - recons_eval.numpy()) ** 2).sum()
ss_tot            = ((eval_tensor.numpy() - eval_tensor.numpy().mean(axis=0)) ** 2).sum()
frac_var_final    = 1 - ss_res / (ss_tot + 1e-10)

print("\nFinal model quality:")
print(f"  Dead features:             {dead_final}/{HIDDEN_DIM}")
print(f"  Active features:           {active_final}")
print(f"  Mean L0:                   {l0_final:.1f}")
print(f"  frac_variance_explained:   {frac_var_final:.4f}")
print(f"  Saved to:                  {ae_path}")

# %%
# Cell 4: Extract Full DC Feature Matrix / 提取全量 DC 特征矩阵
print("Extracting DC feature matrix from full DC set...")

all_tensor = torch.tensor(all_normed, dtype=torch.float32)
ae_loaded  = torch.load(
    os.path.join(SCRIPT_DIR, "sae_dc_final", "trainer_0", "ae.pt"),
    map_location="cpu",
)
ae_loaded.eval()

BATCH        = 4096
recons_list  = []
features_list = []

with torch.no_grad():
    for start in range(0, len(all_tensor), BATCH):
        batch = all_tensor[start : start + BATCH]
        r, f  = ae_loaded(batch)
        recons_list.append(r)
        features_list.append(f)
        if (start // BATCH) % 5 == 0:
            print(f"  Processed {start}/{len(all_tensor)}", flush=True)

recons_all   = torch.cat(recons_list,   dim=0)
features_all = torch.cat(features_list, dim=0)

print(f"\nfeatures_all shape: {features_all.shape}")
features_np  = features_all.numpy()
active_cells = (features_np.max(axis=0) > 1e-6).sum()
l0_all       = (features_np > 0).sum(axis=1).mean()
print(f"Active features (per-cell max > 1e-6): {active_cells}/{HIDDEN_DIM}")
print(f"Mean L0 (non-zero features per cell):  {l0_all:.1f}")

result_path = os.path.join(SCRIPT_DIR, "sae_dc_results_all.pkl")
with open(result_path, "wb") as f:
    pickle.dump((recons_all, features_all), f)
print(f"\nSaved sae_dc_results_all.pkl ({os.path.getsize(result_path)/1e6:.1f} MB)")
print("Ready for Cell 5: ANOVA + t-test gene signature extraction.")

# %%
# Cell 5: ANOVA + DC-Internal t-test Gene Signatures / ANOVA 筛选 + DC 内部 t-test 基因签名
import json
import csv
import scipy.stats as ss
import scanpy as sc
import numpy as np
import pickle

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ANOVA_F_THR = 50.0
TOP_CELLS   = 200
TOP_GENES   = 15

DC_MATURE_TYPES     = ["cDC1", "cDC2", "pDC"]
DC_PROGENITOR_TYPES = ["Pre-cDC", "ASDC", "Pre-pDC", "Pre-pDC Cycling"]
DC_TYPES            = DC_MATURE_TYPES + DC_PROGENITOR_TYPES

# --- Load SAE feature matrix ---
print("Loading sae_dc_results_all.pkl...")
with open(os.path.join(SCRIPT_DIR, "sae_dc_results_all.pkl"), "rb") as fh:
    _d = pickle.load(fh)
raw_feats = _d[1]
if hasattr(raw_feats, "detach"):
    X_sae = raw_feats.detach().cpu().numpy()
else:
    X_sae = np.array(raw_feats)
print(f"  X_sae shape: {X_sae.shape}")  # (N_dc, 2048)

# --- Load DC global indices ---
with open(os.path.join(SCRIPT_DIR, "dc_global_indices.pkl"), "rb") as fh:
    dc_global_indices = pickle.load(fh)
dc_global_indices = list(dc_global_indices)
print(f"  DC global indices: {len(dc_global_indices)}")

# --- Load AnnData for gene expression and CellType labels ---
DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad",
)
print("Loading AnnData (HVG subset)...")
adata_unsorted = sc.read_h5ad(DATA_PATH)
print(f"  adata shape: {adata_unsorted.shape}")

gene_names = list(adata_unsorted.var_names)
cell_types  = adata_unsorted.obs["CellType"].values

# DC cell types and local (within DC subset) to global index mapping
dc_celltypes = [cell_types[gi] for gi in dc_global_indices]

# Group local DC indices by mature subtype for ANOVA
cdc1_local = [i for i, ct in enumerate(dc_celltypes) if ct == "cDC1"]
cdc2_local = [i for i, ct in enumerate(dc_celltypes) if ct == "cDC2"]
pdc_local  = [i for i, ct in enumerate(dc_celltypes) if ct == "pDC"]

print(f"  cDC1: {len(cdc1_local)}, cDC2: {len(cdc2_local)}, pDC: {len(pdc_local)}")

n_dc     = X_sae.shape[0]
n_feats  = X_sae.shape[1]
all_local = np.arange(n_dc)

# Active feature IDs
active_fids = [
    fid for fid in range(n_feats)
    if X_sae[:, fid].max() > 1e-6
]
print(f"  Active features: {len(active_fids)}/{n_feats}")

# --- Multi-class ANOVA: cDC1 vs cDC2 vs pDC ---
print(f"\nRunning ANOVA (cDC1 vs cDC2 vs pDC), threshold F >= {ANOVA_F_THR}...")
anova_results = []
for fid in active_fids:
    acts = X_sae[:, fid]
    g1   = acts[cdc1_local]
    g2   = acts[cdc2_local]
    g3   = acts[pdc_local]
    if len(g1) < 2 or len(g2) < 2 or len(g3) < 2:
        continue
    f_stat, _ = ss.f_oneway(g1, g2, g3)
    if np.isnan(f_stat) or np.isinf(f_stat):
        continue
    anova_results.append({"feature_id": fid, "f_stat": f_stat})

anova_results.sort(key=lambda x: x["f_stat"], reverse=True)
selected = [r for r in anova_results if r["f_stat"] >= ANOVA_F_THR]
print(f"  Features passing F >= {ANOVA_F_THR}: {len(selected)}")
if len(selected) < 5:
    # Fall back to top 50 if threshold is too strict for DC (small dataset)
    selected = anova_results[:min(50, len(anova_results))]
    print(f"  Threshold too strict; using top {len(selected)} features instead.")

# Save ANOVA results
anova_path = os.path.join(SCRIPT_DIR, "dc_anova_routeB.csv")
with open(anova_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["feature_id", "f_stat", "anova_rank"])
    writer.writeheader()
    for rank, r in enumerate(selected, 1):
        writer.writerow({"feature_id": r["feature_id"], "f_stat": round(r["f_stat"], 2), "anova_rank": rank})
print(f"  Saved {anova_path}")

# --- DC-internal Welch t-test gene signatures ---
sig_cache_path = os.path.join(SCRIPT_DIR, "gene_signatures_dc_routeB.json")
if os.path.exists(sig_cache_path):
    with open(sig_cache_path) as fh:
        gene_sigs = json.load(fh)
    existing_fids = set(int(k) for k in gene_sigs.keys())
    print(f"  Loaded {len(existing_fids)} cached gene signatures.")
else:
    gene_sigs     = {}
    existing_fids = set()

# Precompute DC gene expression matrix once
print("Precomputing DC gene expression matrix...")
import scipy.sparse as sp_sparse

X_raw = adata_unsorted.X
# Extract rows for DC cells
dc_rows = []
for gi in dc_global_indices:
    row = X_raw[gi]
    if sp_sparse.issparse(row):
        row = row.toarray().ravel()
    dc_rows.append(row)
X_dc = np.array(dc_rows, dtype=np.float32)  # (N_dc, N_genes)
print(f"  X_dc shape: {X_dc.shape}")

# t-test for each selected feature
new_sigs = 0
for r in selected:
    fid = r["feature_id"]
    if fid in existing_fids:
        continue
    acts_dc  = X_sae[:, fid]
    sorted_l = np.argsort(acts_dc)[::-1]
    fg_local = sorted_l[:TOP_CELLS]
    bg_mask  = np.ones(n_dc, dtype=bool)
    bg_mask[fg_local] = False
    bg_local = all_local[bg_mask]

    fg_expr  = X_dc[fg_local]   # (TOP_CELLS, N_genes)
    bg_expr  = X_dc[bg_local]   # (~N_dc-200, N_genes)
    n_fg     = len(fg_local)
    n_bg     = len(bg_local)

    m1 = fg_expr.mean(axis=0)
    m2 = bg_expr.mean(axis=0)
    v1 = fg_expr.var(axis=0)  / n_fg
    v2 = bg_expr.var(axis=0)  / n_bg
    t_stats = (m1 - m2) / np.sqrt(v1 + v2 + 1e-10)
    t_stats = np.where(np.isnan(t_stats) | np.isinf(t_stats), -np.inf, t_stats)
    top_idx  = np.argsort(t_stats)[::-1][:TOP_GENES]
    top_genes = [gene_names[i] for i in top_idx]

    fg_celltypes    = [dc_celltypes[i] for i in fg_local]
    ct_counts       = {}
    for ct in fg_celltypes:
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    composition     = {ct: round(n / TOP_CELLS * 100, 1) for ct, n in ct_counts.items() if n / TOP_CELLS >= 0.01}

    gene_sigs[str(fid)] = {
        "genes":              top_genes,
        "celltype_composition": composition,
        "anova_rank":         next(i+1 for i, s in enumerate(selected) if s["feature_id"] == fid),
        "f_stat":             round(r["f_stat"], 2),
    }
    new_sigs += 1

if new_sigs > 0:
    with open(sig_cache_path, "w") as fh:
        json.dump(gene_sigs, fh, indent=2)
    print(f"  Saved {new_sigs} new gene signatures -> {sig_cache_path}")
else:
    print("  All gene signatures already cached.")

del X_dc
import gc; gc.collect()
print("\nCell 5 complete. Ready for Cell 6: RAG + LLM annotation.")

# %%
# Cell 6: RAG + Claude API Annotation / RAG 增强注释 (需要 ANTHROPIC_API_KEY)
import os
import json
import time
import pickle
import requests
import anthropic

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DC_MATURE_TYPES     = ["cDC1", "cDC2", "pDC"]
DC_PROGENITOR_TYPES = ["Pre-cDC", "ASDC", "Pre-pDC", "Pre-pDC Cycling"]
DC_TYPES            = DC_MATURE_TYPES + DC_PROGENITOR_TYPES

# -------------------------------------------------------------------
# DC-specific system prompts (five prompts, one per dominant subtype)
# -------------------------------------------------------------------
SYSTEM_PROMPT_CDC1 = """
You are a computational biology expert specialising in dendritic cell biology and haematopoiesis.
You will be shown a gene signature extracted by a Sparse Autoencoder trained on scVI latent
representations of the BoneMarrowMap atlas. The activating cells are predominantly cDC1
(CLEC9A+, XCR1+, BATF3+, IRF8-high). Focus your annotation on:
  - cDC1-specific transcription factors: BATF3, IRF8, ID2
  - Cross-presentation machinery: TAP1, TAP2, TAPBP, NLRC5
  - cDC1 surface markers: CLEC9A, XCR1, CADM1, CLNK
  - Antigen sensing: TLR3, STING pathway genes
Distinguish cDC1 from cDC2 and pDC. Output strict JSON only.
""".strip()

SYSTEM_PROMPT_CDC2 = """
You are a computational biology expert specialising in dendritic cell biology and haematopoiesis.
You will be shown a gene signature extracted by a Sparse Autoencoder trained on scVI latent
representations of the BoneMarrowMap atlas. The activating cells are predominantly cDC2
(CLEC12A+, CD1c+, CD14-low, KLF4+, NOTCH2-dependent). Focus your annotation on:
  - cDC2-specific transcription factors: KLF4, NOTCH2, ZEB2, IRF4
  - MHC-II and CD1 antigen presentation: HLA-DRA, HLA-DRB1, CD1C, CD1D
  - Toll-like receptor signalling: TLR4, TLR2, TLR5, MyD88 pathway
  - cDC2 surface markers: CLEC12A, ITGAM, CX3CR1, FCGR2A
  - Inflammatory cytokine production: IL12B, IL23A, TNF
Distinguish cDC2 from cDC1 (no XCR1) and pDC (no SIGLEC-H). Output strict JSON only.
""".strip()

SYSTEM_PROMPT_PDC = """
You are a computational biology expert specialising in plasmacytoid dendritic cell (pDC) biology
and haematopoiesis. You will be shown a gene signature extracted by a Sparse Autoencoder trained
on scVI latent representations of the BoneMarrowMap atlas. The activating cells are predominantly
pDC (SIGLEC-H+, BST2+, CLEC4C+, LILRA4+). Focus your annotation on:
  - Master pDC transcription factors: SPIB, IRF7, TCF4 (E2-2), RUNX2
  - Type I IFN production machinery: IRF7, MX1, OAS1, OAS2, OAS3, IFIT1, IFIT2
  - pDC-specific surface markers: SIGLEC-H, BST2, CLEC4C, LILRA4, IL3RA (CD123)
  - Nucleic acid sensing: TLR7, TLR9, UNC93B1
  - pDC maturation and migration: CCR9, CXCR4, PTPRS
Distinguish pDC from cDC1 and cDC2: pDC do NOT cross-present (no TAP1/XCR1) and produce
far more Type I IFN per cell than any other DC. Output strict JSON only.
""".strip()

SYSTEM_PROMPT_PROGENITOR = """
You are a computational biology expert specialising in dendritic cell progenitor biology
and haematopoiesis. You will be shown a gene signature extracted by a Sparse Autoencoder
trained on scVI latent representations of the BoneMarrowMap atlas. The activating cells are
predominantly DC progenitors (Pre-cDC, ASDC, Pre-pDC, or Pre-pDC Cycling). Focus your annotation on:
  - ASDC (Axl+Siglec-6+ DC): AXL, SIGLEC6, CX3CR1, CCR2 (bipotent pDC/cDC2 progenitor)
  - Pre-cDC: CLEC12A, PTPRS, LY86 (committed cDC progenitor, CCR7-low)
  - Pre-pDC: SIGLEC-H-low, SPIB-low, intermediate between ASDC and pDC
  - Cycling progenitors: PCNA, MKI67, STMN1, TYMS, TOP2A (S/G2-M cell cycle genes)
  - Lineage commitment signals: IRF8 (cDC1 bias), IRF4 (cDC2 bias), SPIB (pDC bias)
  - FLT3L pathway: FLT3, CSF1R
Distinguish mature DC from progenitors: progenitors lack full surface marker expression
and may co-express multiple lineage markers. Output strict JSON only.
""".strip()

SYSTEM_PROMPT_MIXED = """
You are a computational biology expert specialising in dendritic cell biology and haematopoiesis.
You will be shown a gene signature extracted by a Sparse Autoencoder trained on scVI latent
representations of the BoneMarrowMap atlas. The activating cells contain a mixture of DC subtypes
(cDC1, cDC2, pDC, and/or progenitors). Focus your annotation on:
  - Pan-DC programmes shared across subtypes: MHC-II presentation, FLT3 signalling,
    antigen uptake (endocytosis), CCR7-driven migration, DC maturation (CD83, CCR7, LAMP3)
  - Stress response: heat shock proteins, UPR genes in secretory DCs
  - Metabolic programmes: OXPHOS, glycolysis, fatty acid oxidation
  - Proliferative programmes: cell cycle genes in cycling DC progenitors
When a programme spans multiple subtypes, prioritise the shared molecular mechanism
over individual lineage identity. Output strict JSON only.
""".strip()

USER_TEMPLATE = """\
Feature ID: {fid}
ANOVA rank: {rank} (F-stat = {f_stat:.1f}, three-group: cDC1 vs cDC2 vs pDC)

Top {n_genes} genes by DC-internal Welch t-statistic:
{gene_block}

Cell-type composition of top-200 activating DC cells:
{comp_block}

Gene functional summaries (from NCBI Gene / MyGene.info):
{func_block}

Respond ONLY as valid JSON (no markdown fences) with this exact schema:
{{
  "programme_name": "string (6 words max)",
  "functional_description": "string (2-3 sentences, technically precise)",
  "key_evidence": ["GENE: rationale", "GENE: rationale", ...],
  "confidence": "HIGH or MEDIUM or LOW",
  "suggested_validation": "string"
}}
"""

def pick_prompt(composition: dict) -> str:
    cdc1_pct  = composition.get("cDC1",  0)
    cdc2_pct  = composition.get("cDC2",  0)
    pdc_pct   = composition.get("pDC",   0)
    prog_pct  = sum(composition.get(ct, 0) for ct in DC_PROGENITOR_TYPES)
    dominant  = max(cdc1_pct, cdc2_pct, pdc_pct, prog_pct)
    if dominant < 40:
        return SYSTEM_PROMPT_MIXED
    if cdc1_pct  == dominant:
        return SYSTEM_PROMPT_CDC1
    if cdc2_pct  == dominant:
        return SYSTEM_PROMPT_CDC2
    if pdc_pct   == dominant:
        return SYSTEM_PROMPT_PDC
    return SYSTEM_PROMPT_PROGENITOR

def query_mygene(genes: list, cache: dict) -> dict:
    needed = [g for g in genes if g not in cache]
    if needed:
        try:
            resp = requests.post(
                "https://mygene.info/v3/gene",
                json={"ids": needed, "fields": "symbol,summary", "species": "human"},
                timeout=20,
            )
            if resp.status_code == 200:
                for rec in resp.json():
                    sym = rec.get("symbol", "")
                    cache[sym] = rec.get("summary", "")
        except Exception:
            pass
        time.sleep(1)
    return {g: cache.get(g, "") for g in genes}

def strip_json_md(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()

# --- Load inputs ---
sig_path = os.path.join(SCRIPT_DIR, "gene_signatures_dc_routeB.json")
with open(sig_path) as fh:
    gene_sigs = json.load(fh)

func_cache_path = os.path.join(SCRIPT_DIR, "gene_function_cache.json")
if os.path.exists(func_cache_path):
    with open(func_cache_path) as fh:
        gene_func_cache = json.load(fh)
else:
    gene_func_cache = {}

ann_path = os.path.join(SCRIPT_DIR, "sae_annotations_dc_routeB.json")
if os.path.exists(ann_path):
    with open(ann_path) as fh:
        results_list = json.load(fh)
else:
    results_list = []

failure_names = {"api call failed", "annotation parse error", ""}
done_fids = {
    r["feature_id"] for r in results_list
    if r.get("confidence", "LOW") != "LOW"
    and r.get("programme_name", "") not in failure_names
}
print(f"Resuming: {len(done_fids)} features already annotated.")

# Sort by ANOVA rank
selected_items = sorted(gene_sigs.items(), key=lambda kv: kv[1].get("anova_rank", 9999))

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. "
        "Run: export ANTHROPIC_API_KEY='sk-ant-...' before this cell."
    )
client = anthropic.Anthropic(api_key=api_key)

total = len(selected_items)
done  = 0
for fid_str, sig in selected_items:
    fid = int(fid_str)
    if fid in done_fids:
        done += 1
        continue

    genes       = sig.get("genes", [])[:15]
    composition = sig.get("celltype_composition", {})
    rank        = sig.get("anova_rank", "?")
    f_stat      = sig.get("f_stat", 0.0)

    # RAG: fetch gene functions
    func_info   = query_mygene(genes, gene_func_cache)

    gene_block  = "\n".join(f"  {g}" for g in genes)
    comp_block  = "\n".join(
        f"  {ct}: {pct}%"
        for ct, pct in sorted(composition.items(), key=lambda x: -x[1])
    ) or "  (no cell type composition available)"
    func_block  = "\n".join(
        f"  {g}: {func_info.get(g, 'No summary available.')[:200]}"
        for g in genes
    )

    user_msg = USER_TEMPLATE.format(
        fid        = fid,
        rank       = rank,
        f_stat     = f_stat,
        n_genes    = len(genes),
        gene_block = gene_block,
        comp_block = comp_block,
        func_block = func_block,
    )

    system_prompt = pick_prompt(composition)

    record = {
        "feature_id":   fid,
        "anova_rank":   rank,
        "f_stat":       f_stat,
        "top_genes":    genes,
        "celltype_composition": composition,
        "prior_programme_name": None,
    }

    for attempt in range(3):
        try:
            time.sleep(2)
            response = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 900,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text
            parsed   = json.loads(strip_json_md(raw_text))
            record.update({
                "programme_name":        parsed.get("programme_name",        "annotation parse error"),
                "functional_description":parsed.get("functional_description",""),
                "key_evidence":          parsed.get("key_evidence",          []),
                "confidence":            parsed.get("confidence",            "LOW"),
                "suggested_validation":  parsed.get("suggested_validation",  ""),
            })
            break
        except Exception as exc:
            if attempt == 2:
                record.update({
                    "programme_name":        "api call failed",
                    "functional_description":"",
                    "key_evidence":          [],
                    "confidence":            "LOW",
                    "suggested_validation":  "",
                })
            else:
                time.sleep(5)

    results_list.append(record)
    done += 1

    if done % 10 == 0:
        with open(ann_path, "w") as fh:
            json.dump(results_list, fh, indent=2)
        with open(func_cache_path, "w") as fh:
            json.dump(gene_func_cache, fh, indent=2)
        conf = record.get("confidence", "?")
        name = record.get("programme_name", "?")[:60]
        print(f"  [{done}/{total}] F#{fid} ({conf}) {name}", flush=True)

with open(ann_path, "w") as fh:
    json.dump(results_list, fh, indent=2)
with open(func_cache_path, "w") as fh:
    json.dump(gene_func_cache, fh, indent=2)

# --- Summary ---
successful = [r for r in results_list if r.get("programme_name", "") not in failure_names]
unique_names = len({r["programme_name"] for r in successful})
conf_counts  = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
for r in successful:
    conf_counts[r.get("confidence", "LOW")] = conf_counts.get(r.get("confidence", "LOW"), 0) + 1

print(f"\nAnnotation complete.")
print(f"  Total annotated:    {len(results_list)}")
print(f"  Successful:         {len(successful)}")
print(f"  Unique names:       {unique_names}")
print(f"  HIGH confidence:    {conf_counts['HIGH']}")
print(f"  MEDIUM confidence:  {conf_counts['MEDIUM']}")
print(f"  LOW / failed:       {conf_counts['LOW']}")
print(f"\nOutputs saved:")
print(f"  {ann_path}")
print(f"  {func_cache_path}")
