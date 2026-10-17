"""
extract_scvi_activations_dc.py
================================
Extract encoder activations (128-dim) from trained scVI model,
filtered to Dendritic Cell populations only.

Outputs (saved as .pkl in same directory as this script):
  - dc_train_activation_matrix.pkl   shape: [N_dc_train, 128]
  - dc_eval_activation_matrix.pkl    shape: [N_dc_eval,  128]
  - dc_all_activation_matrix.pkl     shape: [N_dc_all,   128]
  - dc_global_indices.pkl            list: global integer indices into adata_unsorted

Purpose: Route B DC-specific SAE retraining.
  The full 161K-cell SAE heavily favours erythroid/monocyte variance.
  A dedicated SAE on ~11,433 DC cells ensures all 2048 features are
  allocated to DC-internal variation (cDC1 vs cDC2 vs pDC vs progenitors).

DC cell types used (7 subtypes, ~11,433 cells):
  Mature DCs:    cDC1, cDC2, pDC
  DC progenitors: Pre-cDC, ASDC, Pre-pDC, Pre-pDC Cycling

Adapted from extract_scvi_activations_tcnk.py (same model, same HVG setup).
"""

import os
import pickle
import random

import scanpy as sc
import scvi
import torch
from tqdm import tqdm
from scvi.dataloaders import AnnDataLoader

# Reproducibility
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
print("scvi-tools version:", scvi.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# DC cell type labels (must match adata.obs["CellType"] exactly)
DC_MATURE_TYPES = [
    "cDC1",
    "cDC2",
    "pDC",
]
DC_PROGENITOR_TYPES = [
    "Pre-cDC",
    "ASDC",
    "Pre-pDC",
    "Pre-pDC Cycling",
]
DC_TYPES = DC_MATURE_TYPES + DC_PROGENITOR_TYPES

# Train/eval split ratio
EVAL_FRACTION = 0.1

# -------------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------------
DATA_PATH = (
    "/Users/elvainyu/Desktop/申请与学习/暑研和实习/暑研/‼️2026暑研资料"
    "/Computational Biology/Papers & Datas/Papers' Dataset"
    "/BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad"
)
MODEL_PATH = (
    "/Users/elvainyu/Desktop/申请与学习/暑研和实习/暑研/‼️2026暑研资料"
    "/Computational Biology/Papers & Datas/scVI_model_unsorted"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------------------
# STEP 1: Load full annotated AnnData (backed mode, memory-efficient)
# -------------------------------------------------------------------------------
print("[1/6] Loading full AnnData (backed='r')...")
adata = sc.read_h5ad(DATA_PATH, backed="r")
print(f"      Full dataset shape: {adata.shape}")

# -------------------------------------------------------------------------------
# STEP 2: Subset to Unsorted + Healthy_Reference (same cells as scVI training)
# -------------------------------------------------------------------------------
print("[2/6] Subsetting to Unsorted + Healthy_Reference...")
sorting_col = "Sorting" if "Sorting" in adata.obs.columns else "SortingStatus"
unsorted_mask = adata.obs[sorting_col].isin(["Unsorted", "Healthy_Reference"])
adata_unsorted = adata[unsorted_mask].to_memory()
print(f"      Unsorted subset shape: {adata_unsorted.shape}")

# -------------------------------------------------------------------------------
# STEP 3: Select HVGs (must exactly match scVI training: n_top_genes=4000)
# -------------------------------------------------------------------------------
print("[3/6] Selecting HVGs (n_top_genes=4000, same as scVI training)...")
sc.pp.highly_variable_genes(
    adata_unsorted,
    n_top_genes=4000,
    subset=True,
    flavor="seurat_v3",
    layer="counts",
)
print(f"      HVG subset shape: {adata_unsorted.shape}")

# -------------------------------------------------------------------------------
# STEP 4: Setup AnnData for scVI and load trained model
# -------------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------
# Helper: extract activations for a given integer index list
# -------------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------
# STEP 5: Identify DC indices in adata_unsorted
# -------------------------------------------------------------------------------
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

# Convert boolean mask to integer positions within adata_unsorted
dc_int_idx = [
    adata_unsorted.obs_names.get_loc(bc)
    for bc in adata_unsorted.obs_names[dc_mask]
]
print(f"      Integer index range: {min(dc_int_idx)} to {max(dc_int_idx)}")

# Deterministic train/eval split
random.seed(42)
dc_shuffled = dc_int_idx.copy()
random.shuffle(dc_shuffled)
n_eval = max(1, int(len(dc_shuffled) * EVAL_FRACTION))
n_train = len(dc_shuffled) - n_eval
dc_train_idx = sorted(dc_shuffled[:n_train])
dc_eval_idx = sorted(dc_shuffled[n_train:])
print(f"      Train: {len(dc_train_idx)}, Eval: {len(dc_eval_idx)}, Total: {n_dc}")

# -------------------------------------------------------------------------------
# STEP 6: Extract and save
# -------------------------------------------------------------------------------
print("[6/6] Extracting activations for DC cells...")

dc_train_matrix = extract_activations(dc_train_idx, shuffle=True, desc="  DC train")
print(f"      dc_train shape: {dc_train_matrix.shape}")
with open(os.path.join(OUT_DIR, "dc_train_activation_matrix.pkl"), "wb") as f:
    pickle.dump(dc_train_matrix, f)

dc_eval_matrix = extract_activations(dc_eval_idx, shuffle=True, desc="  DC eval ")
print(f"      dc_eval  shape: {dc_eval_matrix.shape}")
with open(os.path.join(OUT_DIR, "dc_eval_activation_matrix.pkl"), "wb") as f:
    pickle.dump(dc_eval_matrix, f)

dc_all_matrix = extract_activations(dc_int_idx, shuffle=False, desc="  DC all  ")
print(f"      dc_all   shape: {dc_all_matrix.shape}")
with open(os.path.join(OUT_DIR, "dc_all_activation_matrix.pkl"), "wb") as f:
    pickle.dump(dc_all_matrix, f)

# Save global indices for downstream scripts (local DC index -> adata_unsorted index)
with open(os.path.join(OUT_DIR, "dc_global_indices.pkl"), "wb") as f:
    pickle.dump(dc_int_idx, f)
print(f"      dc_global_indices.pkl saved ({len(dc_int_idx)} indices)")

# -------------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------------
print("\nDone. Output files:")
for fname in [
    "dc_train_activation_matrix.pkl",
    "dc_eval_activation_matrix.pkl",
    "dc_all_activation_matrix.pkl",
    "dc_global_indices.pkl",
]:
    fpath = os.path.join(OUT_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {fname:<40}  {size_mb:.1f} MB")

print("\nNext step: open train_sae_dc.ipynb and run Cell 1 to Cell 4.")
