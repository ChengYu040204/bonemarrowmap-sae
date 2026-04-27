"""
extract_scvi_activations.py
===========================
Extract encoder activations (128-dim) from trained scVI model.

Outputs (saved as .pkl in same directory as this script):
  - train_activation_matrix.pkl   shape: [N_train, 128]
  - eval_activation_matrix.pkl    shape: [N_eval,  128]
  - all_activation_matrix.pkl     shape: [N_all,   128]  -- order matches adata_unsorted
  - bcell_activation_matrix.pkl   shape: [N_bcell, 128]  -- B-cell lineage subset

Adapted from Dr Ng's template to match BoneMarrowMap training setup:
  n_top_genes=4000, layer='counts', batch_key='Study', n_latent=30
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

# -------------------------------------------------------------------------------
# PATHS — update if you move files
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
OUT_DIR = os.path.dirname(os.path.abspath(__file__))   # same folder as this script

# -------------------------------------------------------------------------------
# STEP 1: Load AnnData + rebuild metadata
# -------------------------------------------------------------------------------
print("\n[1/6] Loading AnnData...")
adata = sc.read_h5ad(DATA_PATH)

# Backup raw counts — required by scVI
adata.layers["counts"] = adata.X.copy()

# Sorting strategy — same logic used during training
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
# STEP 2: Subset to Unsorted + Healthy_Reference
# -------------------------------------------------------------------------------
print("[2/6] Subsetting to Unsorted + Healthy_Reference...")
adata_unsorted = adata[
    (adata.obs["Sorting_Strategy"] == "Unsorted") &
    (adata.obs["Disease_Status"]   == "Healthy_Reference")
].copy()
print(f"      {adata_unsorted.n_obs} cells retained")

# -------------------------------------------------------------------------------
# STEP 3: HVG selection — must match training exactly
# -------------------------------------------------------------------------------
print("[3/6] Selecting HVGs (n=4000)...")
sc.pp.highly_variable_genes(
    adata_unsorted,
    n_top_genes=4000,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="Study",
)
print(f"      {adata_unsorted.n_vars} genes after HVG subset")

# -------------------------------------------------------------------------------
# STEP 4: Register AnnData + load model
# -------------------------------------------------------------------------------
print("[4/6] Setting up scVI and loading model...")
scvi.model.SCVI.setup_anndata(
    adata_unsorted,
    layer="counts",
    batch_key="Study",
)
model = scvi.model.SCVI.load(MODEL_PATH, adata=adata_unsorted)

# Sanity check: print model structure
vae = model.module
vae.eval()
print("\n-- Model structure (confirm 4000->128->128->30) --")
print(vae)
print("--------------------------------------------------\n")

# Grab the encoder — hooks go here
vae_encode  = vae.z_encoder
activations = {}

def getActivation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# -------------------------------------------------------------------------------
# Helper: extract activations for a given index list
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
            vae_encode(batch["X"].to(device))          # triggers hook
            matrix.append(activations["encoder"].to("cpu"))   # move to CPU immediately
    hook.remove()
    result = torch.cat(matrix, dim=0)
    return result   # shape: [N, 128]

# -------------------------------------------------------------------------------
# STEP 5: Extract train / eval / all
# -------------------------------------------------------------------------------
print("[5/6] Extracting activations...")

train_matrix = extract_activations(model.train_indices,      shuffle=True,  desc="  train")
print(f"      train shape: {train_matrix.shape}")
with open(os.path.join(OUT_DIR, "train_activation_matrix.pkl"), "wb") as f:
    pickle.dump(train_matrix, f)

eval_matrix = extract_activations(model.validation_indices,  shuffle=True,  desc="  eval ")
print(f"      eval  shape: {eval_matrix.shape}")
with open(os.path.join(OUT_DIR, "eval_activation_matrix.pkl"), "wb") as f:
    pickle.dump(eval_matrix, f)

all_matrix = extract_activations(list(range(len(adata_unsorted))), shuffle=False, desc="  all  ")
print(f"      all   shape: {all_matrix.shape}")
with open(os.path.join(OUT_DIR, "all_activation_matrix.pkl"), "wb") as f:
    pickle.dump(all_matrix, f)

# -------------------------------------------------------------------------------
# STEP 6: Extract B-cell lineage subset (bonus — makes downstream easier)
# -------------------------------------------------------------------------------
print("[6/6] Extracting B-cell lineage subset...")
B_CELL_TYPES = [
    "HSC", "LMPP", "MLP", "MLP-II", "CLP",
    "Pro-B Cycling", "Pro-B VDJ", "Large Pre-B",
    "Small Pre-B", "Immature B", "Mature B", "Plasma Cell",
]
bcell_mask    = adata_unsorted.obs["CellType"].isin(B_CELL_TYPES)
bcell_int_idx = [adata_unsorted.obs_names.get_loc(i)
                 for i in adata_unsorted.obs_names[bcell_mask]]

bcell_matrix = extract_activations(bcell_int_idx, shuffle=False, desc="  B-cell")
print(f"      B-cell shape: {bcell_matrix.shape}")
with open(os.path.join(OUT_DIR, "bcell_activation_matrix.pkl"), "wb") as f:
    pickle.dump(bcell_matrix, f)

# -------------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------------
print("\n✅ Done. Output files:")
for fname in [
    "train_activation_matrix.pkl",
    "eval_activation_matrix.pkl",
    "all_activation_matrix.pkl",
    "bcell_activation_matrix.pkl",
]:
    fpath = os.path.join(OUT_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {fname:<35}  {size_mb:.1f} MB")
