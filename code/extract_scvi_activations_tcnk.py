"""
extract_scvi_activations_tcnk.py
=================================
Extract encoder activations (128-dim) from trained scVI model,
filtered to T-cell and NK-cell populations only.

Outputs (saved as .pkl in same directory as this script):
  - tcnk_train_activation_matrix.pkl   shape: [N_tcnk_train, 128]
  - tcnk_eval_activation_matrix.pkl    shape: [N_tcnk_eval,  128]
  - tcnk_all_activation_matrix.pkl     shape: [N_tcnk_all,   128]

Purpose: Route B T/NK-specific SAE retraining.
  The full 161K-cell SAE allocates features proportionally to
  variance explained, heavily favouring erythroid trajectories.
  Training a dedicated SAE on 70K T/NK cells ensures all 2048
  features are allocated to T/NK-specific variation.

T/NK cell types used (11 subtypes, ~70,714 cells):
  T cells: CD4 Naive, CD4 Central Memory, CD4 Effector Memory,
           CD4 Regulatory, CD8 Naive, CD8 Central Memory,
           CD8 Effector Memory 1, CD8 Effector Memory 2,
           T Proliferating
  NK cells: NK, NK CD56high

Adapted from extract_scvi_activations_new.py (same model, same HVG setup).
"""

import os
import pickle

import scanpy as sc
import scvi
import torch
from tqdm import tqdm
from scvi.dataloaders import AnnDataLoader

# Reproducibility
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
print("scvi-tools version:", scvi.__version__)

# Device setup: GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# T/NK cell type labels (must match adata.obs["CellType"] exactly)
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

# Train/eval split ratio for T/NK subset (must match original scVI split ~90/10)
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
# STEP 1: Load AnnData and rebuild metadata
# -------------------------------------------------------------------------------
print("\n[1/6] Loading AnnData...")
adata = sc.read_h5ad(DATA_PATH)
adata.layers["counts"] = adata.X.copy()

def assign_sorting(study_name):
    if study_name in ["HCA_Hay2018", "HCA_BM", "Oetjen2018"]:
        return "Unsorted"
    elif study_name in ["Setty2019", "Mende2022", "Ainciburu2022", "Ainciburu2023"]:
        return "CD34_Sorted"
    elif study_name == "Granja2019":
        return "Mixed_Sorting"
    else:
        return "Unknown"

adata.obs["Sorting_Strategy"] = adata.obs["Study"].apply(assign_sorting)
adata.obs["Disease_Status"]   = "Healthy_Reference"


# -------------------------------------------------------------------------------
# STEP 2: Subset to Unsorted + Healthy_Reference (same as original training)
# -------------------------------------------------------------------------------
print("[2/6] Subsetting to Unsorted + Healthy_Reference...")
adata_unsorted = adata[
    (adata.obs["Sorting_Strategy"] == "Unsorted") &
    (adata.obs["Disease_Status"]   == "Healthy_Reference")
].copy()
print(f"      {adata_unsorted.n_obs} cells total (should be 161,604)")


# -------------------------------------------------------------------------------
# STEP 3: HVG selection (must match training exactly: n=4000)
# -------------------------------------------------------------------------------
print("[3/6] Selecting HVGs (n=4000, must match original training)...")
sc.pp.highly_variable_genes(
    adata_unsorted,
    n_top_genes=4000,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="Study",
)
print(f"      {adata_unsorted.n_vars} genes after HVG subset (should be 4000)")


# -------------------------------------------------------------------------------
# STEP 4: Register AnnData + load scVI model
# -------------------------------------------------------------------------------
print("[4/6] Setting up scVI and loading model...")
scvi.model.SCVI.setup_anndata(
    adata_unsorted,
    layer="counts",
    batch_key="Study",
)
model = scvi.model.SCVI.load(MODEL_PATH, adata=adata_unsorted)

vae = model.module
vae.eval()
print("Model loaded. Checking z_encoder structure...")

vae_encode  = vae.z_encoder
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
# STEP 5: Identify T/NK indices in adata_unsorted
# -------------------------------------------------------------------------------
print("[5/6] Filtering T/NK cells from adata_unsorted...")

tcnk_mask = adata_unsorted.obs["CellType"].isin(T_NK_TYPES)
n_tcnk    = tcnk_mask.sum()
print(f"      T/NK cell count: {n_tcnk}")

# Print per-type breakdown for verification
print("      Per-type breakdown:")
for ct in T_NK_TYPES:
    n = (adata_unsorted.obs["CellType"] == ct).sum()
    print(f"        {ct:<35} {n:>6}")

# Raise early if cell types are missing (typo guard)
if n_tcnk == 0:
    raise ValueError(
        "No T/NK cells found. Check that CellType labels in adata_unsorted "
        "match T_NK_TYPES list. Inspect adata_unsorted.obs['CellType'].unique()."
    )

# Convert boolean mask to integer positions within adata_unsorted
# (AnnDataLoader requires positional indices into adata_manager, not cell barcodes)
tcnk_int_idx = [
    adata_unsorted.obs_names.get_loc(bc)
    for bc in adata_unsorted.obs_names[tcnk_mask]
]
print(f"      Integer index range: {min(tcnk_int_idx)} to {max(tcnk_int_idx)}")

# Deterministic train/eval split of T/NK indices
import random
random.seed(42)
tcnk_shuffled = tcnk_int_idx.copy()
random.shuffle(tcnk_shuffled)
n_eval  = max(1, int(len(tcnk_shuffled) * EVAL_FRACTION))
n_train = len(tcnk_shuffled) - n_eval
tcnk_train_idx = sorted(tcnk_shuffled[:n_train])
tcnk_eval_idx  = sorted(tcnk_shuffled[n_train:])
print(f"      Train: {len(tcnk_train_idx)}, Eval: {len(tcnk_eval_idx)}, Total: {n_tcnk}")


# -------------------------------------------------------------------------------
# STEP 6: Extract and save
# -------------------------------------------------------------------------------
print("[6/6] Extracting activations for T/NK cells...")

tcnk_train_matrix = extract_activations(tcnk_train_idx, shuffle=True, desc="  T/NK train")
print(f"      tcnk_train shape: {tcnk_train_matrix.shape}")
with open(os.path.join(OUT_DIR, "tcnk_train_activation_matrix.pkl"), "wb") as f:
    pickle.dump(tcnk_train_matrix, f)

tcnk_eval_matrix = extract_activations(tcnk_eval_idx, shuffle=True, desc="  T/NK eval ")
print(f"      tcnk_eval  shape: {tcnk_eval_matrix.shape}")
with open(os.path.join(OUT_DIR, "tcnk_eval_activation_matrix.pkl"), "wb") as f:
    pickle.dump(tcnk_eval_matrix, f)

tcnk_all_matrix = extract_activations(tcnk_int_idx, shuffle=False, desc="  T/NK all  ")
print(f"      tcnk_all   shape: {tcnk_all_matrix.shape}")
with open(os.path.join(OUT_DIR, "tcnk_all_activation_matrix.pkl"), "wb") as f:
    pickle.dump(tcnk_all_matrix, f)

# Also save the global integer indices so downstream scripts can map back to adata_unsorted
with open(os.path.join(OUT_DIR, "tcnk_global_indices.pkl"), "wb") as f:
    pickle.dump(tcnk_int_idx, f)
print(f"      tcnk_global_indices.pkl saved ({len(tcnk_int_idx)} indices)")

# -------------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------------
print("\nDone. Output files:")
for fname in [
    "tcnk_train_activation_matrix.pkl",
    "tcnk_eval_activation_matrix.pkl",
    "tcnk_all_activation_matrix.pkl",
    "tcnk_global_indices.pkl",
]:
    fpath = os.path.join(OUT_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {fname:<42}  {size_mb:.1f} MB")

print(f"\nNext step: open train_sae_tcnk.ipynb and load tcnk_train/eval pkl files.")
