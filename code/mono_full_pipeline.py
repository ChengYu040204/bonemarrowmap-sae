"""
mono_full_pipeline.py
=====================
Monocyte Route B: Activation Extraction -> SAE Training -> Annotation
Run Cell by Cell in VSCode Python Interactive (Shift+Enter per Cell).

Stage 1 (Cell 0)        : Extract scVI encoder activations for ~35,643 Mono cells
Stage 2 (Cells 1 to 4)  : Normalise, ablate, train, extract Mono-specific SAE
Stage 3 (Cells 5 to 6)  : 3-way ANOVA + Mono-internal t-test + RAG-LLM annotation

ANOVA DESIGN (3 groups):
  Group 1: CD14 Mono       (Classical,     ~21854 cells)
  Group 2: CD16 Mono       (Non-classical,  ~4834 cells)
  Group 3: Progenitor pool (~8955 cells: Early GMP + GMP-Cycle + GMP-Mono
                                       + Early ProMono + Late ProMono)
  Excluded: GMP-Neut (613 cells, neutrophil-biased, will be analysed
                      separately in future granulocyte pipeline)

WORKFLOW
  Cell 0    Extract Mono activations, save 4 pkl files
  Cell 0.5  Clear memory (scVI model + AnnData released before SAE training)
  Cell 1    Load pkl files, inspect distribution
  Cell 1.5  Normalise activations, save mono_activation_norm_params.pt
  Cell 2    Ablation sweep over l1_penalty candidates
  Cell 3    Train final SAE with best l1_penalty (seed=42, resample=5000)
  Cell 4    Extract full Mono feature matrix
  Cell 5    3-way ANOVA + Mono-internal Welch t-test gene signatures
  Cell 6    RAG via MyGene.info + Claude API annotation (5 prompts)

REQUIREMENTS
  conda activate scanpy-env
  pip install anthropic --break-system-packages
  Set ANTHROPIC_API_KEY in Cell 6 inline (Jupyter kernel does NOT inherit
  terminal env vars; see Session 26 lesson).

LESSONS APPLIED FROM PRIOR PIPELINES
  - Session 24: seed=42 (avoid Initialization Lottery), resample_steps=5000
                (avoid Resample Cascade)
  - Session 26: ae.encode/ae.decode (avoid forward() single-tensor trap),
                proper random batch shuffling, coerce_summary helper for
                shared gene_function_cache.json (mixed dict/string values)
"""

# %%
# Cell 0: Mono Activation Extraction / 提取 Mono 细胞 scVI 激活值
import os
import pickle
import random

import scanpy as sc
import scvi
import torch
from tqdm import tqdm
from scvi.dataloaders import AnnDataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath("mono_full_pipeline.py"))

scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
print("scvi-tools version:", scvi.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Cell-type definitions / 细胞类型定义 ===
MONO_MATURE_TYPES   = ["CD14 Mono", "CD16 Mono"]
MONO_PROGENITOR_GMP = ["Early GMP", "GMP-Cycle", "GMP-Mono"]
MONO_PROGENITOR_PRO = ["Early ProMono", "Late ProMono"]
MONO_PROGENITOR_ALL = MONO_PROGENITOR_GMP + MONO_PROGENITOR_PRO
MONO_TYPES          = MONO_MATURE_TYPES + MONO_PROGENITOR_ALL
# Note: GMP-Neut (613 cells) excluded; belongs to neutrophil lineage.

EVAL_FRACTION = 0.1

# === Use pre-built 161,604 x 4000 subset (already HVG'd + has 'counts' layer) ===
# Bonus: skips Sorting filter (Annotated h5ad's 'Sorting' values no longer match
# ["Unsorted", "Healthy_Reference"]) and HVG step.
DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad",
)
MODEL_PATH = os.path.join(SCRIPT_DIR, "../../scVI_model_unsorted")  # 2x ../ not 3x

print("[1/4] Loading pre-built Unsorted_scVI subset (no Sorting filter needed)...")
adata_unsorted = sc.read_h5ad(DATA_PATH)
print(f"      Shape: {adata_unsorted.shape}  (expected 161604 x 4000)")
assert "counts" in adata_unsorted.layers, "Missing 'counts' layer in h5ad."

# Removed: backed-mode load, Sorting filter, layers["counts"] copy, HVG selection.
# All of those are pre-applied in BoneMarrowMap_Unsorted_scVI.h5ad.

print("[2/4] Setting up AnnData for scVI and loading model...")
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

print("[3/4] Filtering Mono cells from adata_unsorted...")
mono_mask = adata_unsorted.obs["CellType"].isin(MONO_TYPES)
n_mono = mono_mask.sum()
print(f"      Mono cell count: {n_mono}")

print("      Per-type breakdown:")
for ct in MONO_TYPES:
    n = (adata_unsorted.obs["CellType"] == ct).sum()
    if ct in MONO_MATURE_TYPES:        role = "(mature)"
    elif ct in MONO_PROGENITOR_GMP:    role = "(GMP)"
    else:                              role = "(ProMono)"
    print(f"        {ct:<25}{role:<12} {n:>6}")

if n_mono == 0:
    raise ValueError(
        "No Mono cells found. Check CellType labels match MONO_TYPES list. "
        "Run adata_unsorted.obs['CellType'].unique() to inspect."
    )

mono_int_idx = [
    adata_unsorted.obs_names.get_loc(bc)
    for bc in adata_unsorted.obs_names[mono_mask]
]
print(f"      Integer index range: {min(mono_int_idx)} to {max(mono_int_idx)}")

random.seed(42)
mono_shuffled = mono_int_idx.copy()
random.shuffle(mono_shuffled)
n_eval  = max(1, int(len(mono_shuffled) * EVAL_FRACTION))
n_train = len(mono_shuffled) - n_eval
mono_train_idx = sorted(mono_shuffled[:n_train])
mono_eval_idx  = sorted(mono_shuffled[n_train:])
print(f"      Train: {len(mono_train_idx)}, Eval: {len(mono_eval_idx)}, Total: {n_mono}")

print("[4/4] Extracting activations for Mono cells...")
mono_train_matrix = extract_activations(mono_train_idx, shuffle=True, desc="  Mono train")
print(f"      mono_train shape: {mono_train_matrix.shape}")
with open(os.path.join(SCRIPT_DIR, "mono_train_activation_matrix.pkl"), "wb") as f:
    pickle.dump(mono_train_matrix, f)

mono_eval_matrix = extract_activations(mono_eval_idx, shuffle=True, desc="  Mono eval ")
print(f"      mono_eval  shape: {mono_eval_matrix.shape}")
with open(os.path.join(SCRIPT_DIR, "mono_eval_activation_matrix.pkl"), "wb") as f:
    pickle.dump(mono_eval_matrix, f)

mono_all_matrix = extract_activations(mono_int_idx, shuffle=False, desc="  Mono all  ")
print(f"      mono_all   shape: {mono_all_matrix.shape}")
with open(os.path.join(SCRIPT_DIR, "mono_all_activation_matrix.pkl"), "wb") as f:
    pickle.dump(mono_all_matrix, f)

with open(os.path.join(SCRIPT_DIR, "mono_global_indices.pkl"), "wb") as f:
    pickle.dump(mono_int_idx, f)
print(f"      mono_global_indices.pkl saved ({len(mono_int_idx)} indices)")

print("\nCell 0 complete. Output files:")
for fname in [
    "mono_train_activation_matrix.pkl",
    "mono_eval_activation_matrix.pkl",
    "mono_all_activation_matrix.pkl",
    "mono_global_indices.pkl",
]:
    fpath = os.path.join(SCRIPT_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {fname:<42}  {size_mb:.1f} MB")

# %%
# Cell 0.5: Memory Cleanup / 释放 scVI 模型和 AnnData 内存
import gc

del adata
del adata_unsorted
del model
del vae
del vae_encode
del activations
del mono_train_matrix
del mono_eval_matrix
del mono_all_matrix
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Memory released. Kernel is now ready for SAE training.")

# %%
# Cell 1: Load Mono Activation pkl Files / 加载 Mono 激活值 pkl 文件
import os
import pickle
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath("mono_full_pipeline.py"))

print("Loading Mono activation pkl files...")
with open(os.path.join(SCRIPT_DIR, "mono_train_activation_matrix.pkl"), "rb") as f:
    train_raw_t = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, "mono_eval_activation_matrix.pkl"), "rb") as f:
    eval_raw_t = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, "mono_all_activation_matrix.pkl"), "rb") as f:
    all_raw_t = pickle.load(f)

def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)

train_np = to_numpy(train_raw_t)
eval_np  = to_numpy(eval_raw_t)
all_np   = to_numpy(all_raw_t)

print(f"mono_train shape: {train_np.shape}")
print(f"mono_eval  shape: {eval_np.shape}")
print(f"mono_all   shape: {all_np.shape}")

print("\nActivation distribution (train set):")
print(f"  mean = {train_np.mean():.4f}")
print(f"  std  = {train_np.std():.4f}")
print(f"  min  = {train_np.min():.4f}")
print(f"  max  = {train_np.max():.4f}")
print(f"  % > 0 = {(train_np > 0).mean() * 100:.1f}%")

# %%
# Cell 1.5: Normalise Activations / 激活值归一化（中心化 + L2 缩放）
print("Normalising Mono activations...")

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
torch.save(norm_params, os.path.join(SCRIPT_DIR, "mono_activation_norm_params.pt"))
print("Saved mono_activation_norm_params.pt")

# %%
# Cell 2: Ablation Sweep over l1_penalty / 消融实验
import gc
from dictionary_learning.trainers.standard import StandardTrainer

INPUT_DIM  = 128
HIDDEN_DIM = 2048
LR         = 1e-3
STEPS      = 10000
BATCH_SIZE = 512
CANDIDATES = [1e-3, 5e-3, 1e-2, 2e-2]

train_norm = torch.tensor(train_normed, dtype=torch.float32)
eval_norm  = torch.tensor(eval_normed,  dtype=torch.float32)

# === Robust extractor (handles AE forward signature differences) ===
def extract_recons_features(ae, x):
    """
    Three AE forward conventions:
      1) ae.encode() + ae.decode() exist (Marks-lab dictionary_learning default)
      2) ae(x, output_features=True) returns (recons, features)
      3) ae(x) directly returns (recons, features) tuple
    """
    if hasattr(ae, "encode") and hasattr(ae, "decode"):
        f = ae.encode(x)
        r = ae.decode(f)
        return r, f
    try:
        out = ae(x, output_features=True)
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1]
    except TypeError:
        pass
    out = ae(x)
    if isinstance(out, tuple) and len(out) >= 2:
        return out[0], out[1]
    raise RuntimeError(
        f"Cannot extract features. ae returned {type(out).__name__}."
    )

print(f"{'l1_penalty':>12} | {'Dead':>6} | {'L0':>6} | {'frac_var':>9}")
print("-" * 44)

for l1 in CANDIDATES:
    torch.manual_seed(0)
    trainer = StandardTrainer(
        steps=STEPS,
        activation_dim=INPUT_DIM,
        dict_size=HIDDEN_DIM,
        layer=0,
        lm_name="scVI",
        lr=LR,
        l1_penalty=l1,
        warmup_steps=500,
        sparsity_warmup_steps=1000,
        resample_steps=3000,
        seed=0,
    )
    ae = trainer.ae

    # Training loop with shuffling
    n_train = len(train_norm)
    perm    = torch.randperm(n_train)
    cursor  = 0
    for step in range(STEPS):
        if cursor + BATCH_SIZE > n_train:
            perm   = torch.randperm(n_train)
            cursor = 0
        batch   = train_norm[perm[cursor : cursor + BATCH_SIZE]]
        cursor += BATCH_SIZE
        trainer.update(step, batch)

    # Evaluation
    ae.eval()
    with torch.no_grad():
        recons, features = extract_recons_features(ae, eval_norm)
    features_np = features.cpu().numpy()
    recons_np   = recons.cpu().numpy()

    dead     = int((features_np.max(axis=0) < 1e-6).sum())
    l0       = float((features_np > 1e-6).sum(axis=1).mean())
    ss_res   = float(((eval_normed - recons_np) ** 2).sum())
    ss_tot   = float(((eval_normed - eval_normed.mean(axis=0)) ** 2).sum())
    frac_var = float(1 - ss_res / (ss_tot + 1e-10))

    print(f"{l1:>12.0e} | {dead:>6} | {l0:>6.1f} | {frac_var:>9.4f}")

    del trainer, ae, features, recons, features_np, recons_np
    gc.collect()

print("\nCell 2 complete. Pick BEST_L1: Dead~0, L0=20-80, frac_var>0.5")

# %%
# Cell 3: 训练最终 Mono SAE / Train Final Mono SAE
# UPDATE BEST_L1 based on Cell 2 ablation table before running

BEST_L1 = 5e-3   # <<< UPDATE based on Cell 2 output

RESAMPLE = 5000   # Session 24 lesson: 3000 caused Resample Cascade

torch.manual_seed(42)
print(f"\nTraining final Mono SAE")
print(f"  l1_penalty:      {BEST_L1}")
print(f"  steps:           {STEPS}")
print(f"  resample_steps:  {RESAMPLE}  (raised from 3000 per Session 24)")
print(f"  torch seed:      42")

trainer = StandardTrainer(
    steps=STEPS,
    activation_dim=INPUT_DIM,
    dict_size=HIDDEN_DIM,
    layer=0,
    lm_name="scVI",
    lr=LR,
    l1_penalty=BEST_L1,
    warmup_steps=500,
    sparsity_warmup_steps=1000,
    resample_steps=RESAMPLE,
    seed=42,
)
ae = trainer.ae

n_train = len(train_norm)
perm    = torch.randperm(n_train)
cursor  = 0
print("\nTraining...")
for step in range(STEPS):
    if cursor + BATCH_SIZE > n_train:
        perm   = torch.randperm(n_train)
        cursor = 0
    batch   = train_norm[perm[cursor : cursor + BATCH_SIZE]]
    cursor += BATCH_SIZE
    trainer.update(step, batch)
    if (step + 1) % 1000 == 0:
        print(f"  step {step+1:>5}/{STEPS}", flush=True)

SAVE_DIR = os.path.join(SCRIPT_DIR, "sae_mono_final", "trainer_0")
os.makedirs(SAVE_DIR, exist_ok=True)
AE_PATH = os.path.join(SAVE_DIR, "ae.pt")
torch.save(ae, AE_PATH)
print(f"\nSaved trained AE to: {AE_PATH}")

ae.eval()
with torch.no_grad():
    recons_eval, features_eval = extract_recons_features(ae, eval_norm)
features_eval_np = features_eval.cpu().numpy()
recons_eval_np   = recons_eval.cpu().numpy()

dead_final     = int((features_eval_np.max(axis=0) < 1e-6).sum())
active_final   = HIDDEN_DIM - dead_final
l0_final       = float((features_eval_np > 1e-6).sum(axis=1).mean())
ss_res         = float(((eval_normed - recons_eval_np) ** 2).sum())
ss_tot         = float(((eval_normed - eval_normed.mean(axis=0)) ** 2).sum())
frac_var_final = float(1 - ss_res / (ss_tot + 1e-10))

print("\n========= Final Mono SAE quality =========")
print(f"  l1_penalty:                 {BEST_L1}")
print(f"  Dead features:              {dead_final}/{HIDDEN_DIM}")
print(f"  Active features:            {active_final}")
print(f"  Mean L0:                    {l0_final:.1f}")
print(f"  frac_variance_explained:    {frac_var_final:.4f}")
print(f"  Saved to:                   {AE_PATH}")
print("==========================================")
print(repr({"dead": dead_final, "active": active_final, "L0": l0_final, "frac_var": frac_var_final}))

del trainer, features_eval, recons_eval, features_eval_np, recons_eval_np
gc.collect()

# %%
# Cell 4: 提取全量 Mono 特征矩阵 / Extract Full Mono Feature Matrix
print("Extracting Mono feature matrix from full set...")

all_tensor = torch.tensor(all_normed, dtype=torch.float32)
ae_loaded  = torch.load(
    os.path.join(SCRIPT_DIR, "sae_mono_final", "trainer_0", "ae.pt"),
    map_location="cpu",
    weights_only=False,
)
ae_loaded.eval()

BATCH        = 4096
recons_list  = []
features_list = []

with torch.no_grad():
    for start in range(0, len(all_tensor), BATCH):
        batch = all_tensor[start : start + BATCH]
        r, f  = extract_recons_features(ae_loaded, batch)
        recons_list.append(r)
        features_list.append(f)
        done_n = min(start + BATCH, len(all_tensor))
        print(f"  processed {done_n}/{len(all_tensor)}", flush=True)

recons_all   = torch.cat(recons_list,   dim=0)
features_all = torch.cat(features_list, dim=0)

features_np  = features_all.cpu().numpy()
active_cells = int((features_np.max(axis=0) > 1e-6).sum())
l0_all       = float((features_np > 1e-6).sum(axis=1).mean())
print(f"\nfeatures_all shape: {tuple(features_all.shape)}")
print(f"Active features (col max > 1e-6): {active_cells}/{HIDDEN_DIM}")
print(f"Mean L0 (per-cell non-zero):       {l0_all:.1f}")

result_path = os.path.join(SCRIPT_DIR, "sae_mono_results_all.pkl")
with open(result_path, "wb") as f:
    pickle.dump((recons_all, features_all), f)
print(f"\nSaved sae_mono_results_all.pkl ({os.path.getsize(result_path)/1e6:.1f} MB)")
print("Ready for Cell 5: 3-way ANOVA + Mono-internal t-test.")

del recons_list, features_list, recons_all, all_tensor
gc.collect()

# %%
# Cell 5: 3-way ANOVA + Mono 内部 t-test / 3-way ANOVA + Mono-internal Welch t-test
import json
import csv
import scipy.stats as ss
import scipy.sparse as sp_sparse
import scanpy as sc

SCRIPT_DIR  = os.path.dirname(os.path.abspath("mono_full_pipeline.py"))
ANOVA_F_THR = 50.0
TOP_CELLS   = 200
TOP_GENES   = 15

MONO_MATURE_TYPES   = ["CD14 Mono", "CD16 Mono"]
MONO_PROGENITOR_GMP = ["Early GMP", "GMP-Cycle", "GMP-Mono"]
MONO_PROGENITOR_PRO = ["Early ProMono", "Late ProMono"]
MONO_PROGENITOR_ALL = MONO_PROGENITOR_GMP + MONO_PROGENITOR_PRO
MONO_TYPES          = MONO_MATURE_TYPES + MONO_PROGENITOR_ALL

# === Load SAE feature matrix ===
print("Loading sae_mono_results_all.pkl...")
with open(os.path.join(SCRIPT_DIR, "sae_mono_results_all.pkl"), "rb") as fh:
    _d = pickle.load(fh)
raw_feats = _d[1]
X_sae = raw_feats.detach().cpu().numpy() if hasattr(raw_feats, "detach") else np.array(raw_feats)
print(f"  X_sae shape: {X_sae.shape}")

# === Load Mono global indices ===
with open(os.path.join(SCRIPT_DIR, "mono_global_indices.pkl"), "rb") as fh:
    mono_global_indices = pickle.load(fh)
mono_global_indices = list(mono_global_indices)
n_mono = len(mono_global_indices)
print(f"  Mono global indices: {n_mono}")

assert X_sae.shape[0] == n_mono, (
    f"Shape mismatch: X_sae has {X_sae.shape[0]} rows but Mono indices = {n_mono}."
)

# === Load AnnData ===
DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad",
)
print("Loading AnnData...")
adata_unsorted = sc.read_h5ad(DATA_PATH)
print(f"  adata shape: {adata_unsorted.shape}")

gene_names = list(adata_unsorted.var_names)
cell_types = adata_unsorted.obs["CellType"].values

mono_celltypes = [cell_types[gi] for gi in mono_global_indices]

print("\n  Per-type cell counts in Mono subset:")
total_known = 0
for ct in MONO_TYPES:
    n = mono_celltypes.count(ct)
    total_known += n
    marker = "" if n > 0 else "   <-- ZERO (label mismatch?)"
    if ct in MONO_MATURE_TYPES:        role = "(mature)"
    elif ct in MONO_PROGENITOR_GMP:    role = "(GMP)"
    else:                              role = "(ProMono)"
    print(f"    {ct:<25} {role:<12} {n:>6}{marker}")
print(f"    SUM matching MONO_TYPES: {total_known} / {n_mono}")

non_mono = [ct for ct in mono_celltypes if ct not in MONO_TYPES]
if len(non_mono) > 0:
    unique_non = list(set(non_mono))[:10]
    print(f"\n  WARNING: {len(non_mono)} cells have non-Mono labels.")
    print(f"    Sample unmatched labels: {unique_non}")

# === Build 3 ANOVA groups: CD14 vs CD16 vs Progenitor (pooled) ===
cd14_local = [i for i, ct in enumerate(mono_celltypes) if ct == "CD14 Mono"]
cd16_local = [i for i, ct in enumerate(mono_celltypes) if ct == "CD16 Mono"]
prog_local = [i for i, ct in enumerate(mono_celltypes) if ct in MONO_PROGENITOR_ALL]
print(f"\n  ANOVA groups: CD14={len(cd14_local)}, CD16={len(cd16_local)}, Progenitor(pool)={len(prog_local)}")

if min(len(cd14_local), len(cd16_local), len(prog_local)) < 2:
    raise ValueError("At least one ANOVA group has < 2 cells; cannot run ANOVA.")

n_feats = X_sae.shape[1]
all_local = np.arange(n_mono)

active_fids = [fid for fid in range(n_feats) if X_sae[:, fid].max() > 1e-6]
print(f"\n  Active features: {len(active_fids)}/{n_feats}")

# === 3-way ANOVA ===
print(f"\nRunning 3-way ANOVA (CD14 vs CD16 vs Progenitor-pool), F >= {ANOVA_F_THR}...")
anova_results = []
for fid in active_fids:
    acts = X_sae[:, fid]
    g1   = acts[cd14_local]
    g2   = acts[cd16_local]
    g3   = acts[prog_local]
    f_stat, _ = ss.f_oneway(g1, g2, g3)
    if np.isnan(f_stat) or np.isinf(f_stat):
        continue
    anova_results.append({"feature_id": fid, "f_stat": float(f_stat)})

anova_results.sort(key=lambda x: x["f_stat"], reverse=True)
selected = [r for r in anova_results if r["f_stat"] >= ANOVA_F_THR]
print(f"  Features passing F >= {ANOVA_F_THR}: {len(selected)}")
if len(selected) < 5:
    selected = anova_results[:min(50, len(anova_results))]
    print(f"  Threshold too strict; falling back to top {len(selected)} features.")

print("\n  F-stat distribution:")
for r in selected[:5]:
    print(f"    F#{r['feature_id']:<5}  F={r['f_stat']:.1f}")
rank50 = min(49, len(selected) - 1)
print(f"    rank-50 F = {selected[rank50]['f_stat']:.1f}")
print(f"    dynamic range top1/rank50 = {selected[0]['f_stat']/selected[rank50]['f_stat']:.1f}x")

# Save ANOVA CSV
anova_path = os.path.join(SCRIPT_DIR, "mono_anova_routeB.csv")
with open(anova_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["feature_id", "f_stat", "anova_rank"])
    writer.writeheader()
    for rank, r in enumerate(selected, 1):
        writer.writerow({
            "feature_id": r["feature_id"],
            "f_stat":     round(r["f_stat"], 2),
            "anova_rank": rank,
        })
print(f"\n  Saved {anova_path}")

# === Bulk-extract Mono gene expression matrix ===
print("\nExtracting Mono gene expression matrix (bulk index)...")
X_raw = adata_unsorted.X
X_mono = X_raw[mono_global_indices]
if sp_sparse.issparse(X_mono):
    X_mono = X_mono.toarray()
X_mono = X_mono.astype(np.float32)
print(f"  X_mono shape: {X_mono.shape}")

# === Mono-internal Welch t-test gene signatures ===
sig_cache_path = os.path.join(SCRIPT_DIR, "gene_signatures_mono_routeB.json")
if os.path.exists(sig_cache_path):
    with open(sig_cache_path) as fh:
        gene_sigs = json.load(fh)
    existing_fids = set(int(k) for k in gene_sigs.keys())
    print(f"  Loaded {len(existing_fids)} cached signatures (resume mode).")
else:
    gene_sigs     = {}
    existing_fids = set()

new_sigs = 0
for rank, r in enumerate(selected, 1):
    fid = r["feature_id"]
    if fid in existing_fids:
        continue
    acts_mono = X_sae[:, fid]
    sorted_l  = np.argsort(acts_mono)[::-1]
    fg_local  = sorted_l[:TOP_CELLS]
    bg_mask   = np.ones(n_mono, dtype=bool)
    bg_mask[fg_local] = False
    bg_local  = all_local[bg_mask]

    fg_expr = X_mono[fg_local]
    bg_expr = X_mono[bg_local]
    n_fg    = len(fg_local)
    n_bg    = len(bg_local)

    m1 = fg_expr.mean(axis=0)
    m2 = bg_expr.mean(axis=0)
    v1 = fg_expr.var(axis=0) / n_fg
    v2 = bg_expr.var(axis=0) / n_bg
    t_stats = (m1 - m2) / np.sqrt(v1 + v2 + 1e-10)
    t_stats = np.where(np.isnan(t_stats) | np.isinf(t_stats), -np.inf, t_stats)
    top_idx   = np.argsort(t_stats)[::-1][:TOP_GENES]
    top_genes = [gene_names[i] for i in top_idx]

    fg_cts = [mono_celltypes[i] for i in fg_local]
    ct_counts = {}
    for ct in fg_cts:
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    composition = {
        ct: round(n / TOP_CELLS * 100, 1)
        for ct, n in ct_counts.items()
        if n / TOP_CELLS >= 0.01
    }

    gene_sigs[str(fid)] = {
        "genes":                top_genes,
        "celltype_composition": composition,
        "anova_rank":           rank,
        "f_stat":               round(r["f_stat"], 2),
    }
    new_sigs += 1

if new_sigs > 0:
    with open(sig_cache_path, "w") as fh:
        json.dump(gene_sigs, fh, indent=2)
    print(f"\n  Saved {new_sigs} new gene signatures -> {sig_cache_path}")
else:
    print("\n  All gene signatures already cached.")

del X_mono, adata_unsorted
gc.collect()
print("\nCell 5 complete. Ready for Cell 6: RAG + LLM annotation.")

# %%
# Cell 6: RAG + Claude API 注释 / RAG-Enhanced LLM Annotation Pipeline
import json, time, gc
import requests
import anthropic

# === API key inline (Jupyter kernel does NOT inherit terminal env var) ===
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-你的key'   # <<< PASTE YOUR KEY

SCRIPT_DIR = os.path.dirname(os.path.abspath("mono_full_pipeline.py"))

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key or not api_key.startswith("sk-ant"):
    raise EnvironmentError("ANTHROPIC_API_KEY not set or malformed.")
client = anthropic.Anthropic(api_key=api_key)
print(f"API client initialised. Model: claude-haiku-4-5-20251001")

MONO_MATURE_TYPES   = ["CD14 Mono", "CD16 Mono"]
MONO_PROGENITOR_GMP = ["Early GMP", "GMP-Cycle", "GMP-Mono"]
MONO_PROGENITOR_PRO = ["Early ProMono", "Late ProMono"]
MONO_PROGENITOR_ALL = MONO_PROGENITOR_GMP + MONO_PROGENITOR_PRO
MONO_TYPES          = MONO_MATURE_TYPES + MONO_PROGENITOR_ALL

# === Five Mono-specific system prompts / 五套 Mono 专用 prompt ===
SYSTEM_PROMPT_CD14 = """
You are a computational biology expert specialising in monocyte biology and haematopoiesis.
You will be shown a gene signature extracted by a Sparse Autoencoder trained on scVI latent
representations of the BoneMarrowMap atlas. The activating cells are predominantly CD14 Mono
(classical, inflammatory monocytes). Focus your annotation on:
  - Master TFs: SPI1 (PU.1), CEBPB, IRF5, KLF4
  - Bacterial PAMP sensing: TLR2, TLR4, TLR5, CD14, LBP, MyD88
  - Inflammasome and IL-1 family: NLRP3, CASP1, IL1B, IL18, GSDMD
  - Pro-inflammatory cytokines: TNF, IL6, IL12B, CCL2, CXCL8 (IL8)
  - Surface markers: CD14 (high), FCGR1A (CD64), CCR2, CD163
  - Effector function: phagocytosis (FCGR family), ROS via NCF1/NCF2/NCF4, NADPH oxidase
  - Recruitment to inflammation: CCR2-CCL2 axis (bone marrow egress)
Distinguish CD14 Mono from CD16 Mono (low CCR2, high CX3CR1) and progenitors.
Output strict JSON only.
""".strip()

SYSTEM_PROMPT_CD16 = """
You are a computational biology expert specialising in non-classical monocyte biology
and haematopoiesis. You will be shown a gene signature extracted by a Sparse Autoencoder
trained on scVI latent representations of the BoneMarrowMap atlas. The activating cells are
predominantly CD16 Mono (non-classical, patrolling monocytes). Focus your annotation on:
  - Master TFs: NR4A1 (Nur77), NR4A2, NR4A3 (orphan nuclear receptors driving patrolling)
  - Patrolling and vascular surveillance: CX3CR1 (high), LFA-1 (ITGAL/ITGB2), crawling
  - Antiviral: IFI27, IFITM3, RSAD2 (viperin), ISG15, type I IFN-induced genes
  - Surface markers: FCGR3A (CD16), CX3CR1 high, CD14 low or absent, HLA class I
  - Antibody-dependent cellular cytotoxicity (ADCC): FCGR3A, NCAM1 surface
  - Endothelial interaction: ITGAX, ITGAM, JAM family
Distinguish CD16 Mono from CD14 Mono (high CD14, CCR2, inflammatory cytokines) and
intermediate Mono. Output strict JSON only.
""".strip()

SYSTEM_PROMPT_GMP = """
You are a computational biology expert specialising in early myeloid progenitor biology
and haematopoiesis. You will be shown a gene signature extracted by a Sparse Autoencoder
trained on scVI latent representations of the BoneMarrowMap atlas. The activating cells are
predominantly GMP-stage progenitors (Early GMP, GMP-Cycle, GMP-Mono).
GMP = Granulocyte-Monocyte Progenitor: bipotent precursor giving rise to both monocytes
and neutrophils. Focus your annotation on:
  - Master TFs: SPI1 (PU.1), CEBPA, GATA2 (early), MYB, IRF8 (mono-bias) vs CEBPE (neut-bias)
  - HSC/progenitor markers: CD34, KIT (declining in late GMP)
  - Cell cycle (GMP-Cycle): CDK1, CDK2, MCM2-7, PCNA, TOP2A, MKI67
  - Early myeloid lineage: MPO, ELANE (priming, both mono and neut)
  - Lineage commitment signals: IRF8 (mono bias), CEBPE (neut bias), KLF4 onset
  - FLT3 signalling: FLT3 high, CSF1R rising
Distinguish GMP from ProMono (more committed, MPO downregulated, CSF1R/CD64 upregulated)
and from mature mono (full CD14/CD16 expression). Output strict JSON only.
""".strip()

SYSTEM_PROMPT_PROMONO = """
You are a computational biology expert specialising in committed monocyte progenitor biology
and haematopoiesis. You will be shown a gene signature extracted by a Sparse Autoencoder
trained on scVI latent representations of the BoneMarrowMap atlas. The activating cells are
predominantly ProMono-stage progenitors (Early ProMono or Late ProMono).
ProMono = committed monocyte progenitor downstream of GMP, upstream of CD14 Mono.
Focus your annotation on:
  - Mono commitment TFs: KLF4 (high), IRF8 (cooperates with PU.1 to drive mono fate)
  - Surface markers transitioning: CSF1R (M-CSF receptor) up, MS4A3 high (transient ProMono marker)
  - Loss of neutrophil markers: MPO down, ELANE down (vs GMP-Neut)
  - Gain of mono markers: LYZ, CTSL, CTSS, FCGR1A (CD64) up
  - Late ProMono vs Early ProMono: CD14 onset (Late), surface marker maturation
  - Pre-egress programmes: CCR2 expression, S1PR1 modulation
Distinguish ProMono from CD14 Mono (full surface marker expression, exit cell cycle)
and from GMP (still bipotent, MPO/ELANE high). Output strict JSON only.
""".strip()

SYSTEM_PROMPT_MIXED = """
You are a computational biology expert specialising in monocyte biology and haematopoiesis.
You will be shown a gene signature extracted by a Sparse Autoencoder trained on scVI latent
representations of the BoneMarrowMap atlas. The activating cells span multiple Mono subtypes
or developmental stages. Focus your annotation on:
  - Pan-Mono shared programmes: MHC-II antigen presentation (HLA-DRA, HLA-DRB1, CIITA),
    Fc receptor family (FCGR2A, FCGR2B), CD68 (LAMP4), ITGAM (CD11b)
  - M-CSF signalling pathway: CSF1R (CD115), CSF2RA, FLT3
  - Antigen presentation and processing: HLA class II, CD74 (invariant chain), CTSS, LGMN
  - Common metabolic programmes: OXPHOS, glycolysis, fatty acid oxidation
  - Stress and proteostasis: heat shock proteins, UPR genes
  - Cell cycle for cycling progenitor subsets: CDK family, MCM, PCNA
  - DC/Mono shared TFs: IRF8 (also drives cDC1), KLF4 (also drives cDC2)
When a programme spans multiple subtypes, prioritise the shared molecular mechanism
over individual lineage identity. Output strict JSON only.
""".strip()

USER_TEMPLATE = """\
Feature ID: {fid}
ANOVA rank: {rank} (F-stat = {f_stat:.1f}, three-group: CD14 Mono vs CD16 Mono vs Progenitor-pool)

Top {n_genes} genes by Mono-internal Welch t-statistic:
{gene_block}

Cell-type composition of top-200 activating Mono cells:
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
    cd14_pct = composition.get("CD14 Mono", 0)
    cd16_pct = composition.get("CD16 Mono", 0)
    gmp_pct  = sum(composition.get(ct, 0) for ct in MONO_PROGENITOR_GMP)
    pro_pct  = sum(composition.get(ct, 0) for ct in MONO_PROGENITOR_PRO)
    dominant = max(cd14_pct, cd16_pct, gmp_pct, pro_pct)
    if dominant < 40:
        return SYSTEM_PROMPT_MIXED
    if cd14_pct == dominant: return SYSTEM_PROMPT_CD14
    if cd16_pct == dominant: return SYSTEM_PROMPT_CD16
    if gmp_pct  == dominant: return SYSTEM_PROMPT_GMP
    return SYSTEM_PROMPT_PROMONO

# === Defensive helper: coerce cache value to safe string ===
def coerce_summary(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return (val.get("summary") or val.get("description")
                or val.get("text") or "")
    return str(val)

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
        except Exception as e:
            print(f"    MyGene query failed: {type(e).__name__}; using empty summaries.")
        time.sleep(1)
    return {g: coerce_summary(cache.get(g, "")) for g in genes}

def strip_json_md(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()

# === Load inputs ===
sig_path = os.path.join(SCRIPT_DIR, "gene_signatures_mono_routeB.json")
with open(sig_path) as fh:
    gene_sigs = json.load(fh)
print(f"Loaded {len(gene_sigs)} gene signatures from {os.path.basename(sig_path)}")

func_cache_path = os.path.join(SCRIPT_DIR, "gene_function_cache.json")
if os.path.exists(func_cache_path):
    with open(func_cache_path) as fh:
        gene_func_cache = json.load(fh)
    print(f"Loaded {len(gene_func_cache)} cached gene summaries (shared cache).")
else:
    gene_func_cache = {}

# Normalise cache to all-strings (defensive, even if DC pipeline already migrated)
_fixed = 0
for k, v in list(gene_func_cache.items()):
    if not isinstance(v, str):
        gene_func_cache[k] = coerce_summary(v)
        _fixed += 1
if _fixed > 0:
    print(f"  Normalised {_fixed} non-string cache entries.")
    with open(func_cache_path, "w") as fh:
        json.dump(gene_func_cache, fh, indent=2)

ann_path = os.path.join(SCRIPT_DIR, "sae_annotations_mono_routeB.json")
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
print(f"Resume: {len(done_fids)} already annotated; {len(gene_sigs) - len(done_fids)} remaining.")

selected_items = sorted(gene_sigs.items(), key=lambda kv: kv[1].get("anova_rank", 9999))

total = len(selected_items)
done  = 0
print(f"\nAnnotating {total - len(done_fids)} features...\n")

for fid_str, sig in selected_items:
    fid = int(fid_str)
    if fid in done_fids:
        done += 1
        continue

    genes       = sig.get("genes", [])[:15]
    composition = sig.get("celltype_composition", {})
    rank        = sig.get("anova_rank", "?")
    f_stat      = sig.get("f_stat", 0.0)

    func_info = query_mygene(genes, gene_func_cache)

    gene_block = "\n".join(f"  {g}" for g in genes)
    comp_block = "\n".join(
        f"  {ct}: {pct}%"
        for ct, pct in sorted(composition.items(), key=lambda x: -x[1])
    ) or "  (no cell type composition available)"
    func_block = "\n".join(
        f"  {g}: {(coerce_summary(func_info.get(g, '')) or 'No summary available.')[:200]}"
        for g in genes
    )

    user_msg      = USER_TEMPLATE.format(
        fid=fid, rank=rank, f_stat=f_stat,
        n_genes=len(genes), gene_block=gene_block,
        comp_block=comp_block, func_block=func_block,
    )
    system_prompt = pick_prompt(composition)

    record = {
        "feature_id":           fid,
        "anova_rank":           rank,
        "f_stat":               f_stat,
        "top_genes":            genes,
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
                "programme_name":         parsed.get("programme_name",         "annotation parse error"),
                "functional_description": parsed.get("functional_description", ""),
                "key_evidence":           parsed.get("key_evidence",           []),
                "confidence":             parsed.get("confidence",             "LOW"),
                "suggested_validation":   parsed.get("suggested_validation",   ""),
            })
            break
        except Exception as exc:
            if attempt == 2:
                record.update({
                    "programme_name":         "api call failed",
                    "functional_description": "",
                    "key_evidence":           [],
                    "confidence":             "LOW",
                    "suggested_validation":   f"Error: {type(exc).__name__}",
                })
            else:
                time.sleep(5)

    results_list.append(record)
    done += 1

    if done % 5 == 0:
        with open(ann_path, "w") as fh:
            json.dump(results_list, fh, indent=2)
        with open(func_cache_path, "w") as fh:
            json.dump(gene_func_cache, fh, indent=2)
        conf = record.get("confidence", "?")
        name = record.get("programme_name", "?")[:55]
        print(f"  [{done:>3}/{total}] F#{fid:<5} ({conf:<6}) {name}", flush=True)

with open(ann_path, "w") as fh:
    json.dump(results_list, fh, indent=2)
with open(func_cache_path, "w") as fh:
    json.dump(gene_func_cache, fh, indent=2)

# === Summary ===
successful   = [r for r in results_list if r.get("programme_name", "") not in failure_names]
unique_names = len({r["programme_name"] for r in successful})
conf_counts  = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
for r in successful:
    c = r.get("confidence", "LOW")
    conf_counts[c] = conf_counts.get(c, 0) + 1

subtype_counts = {"CD14-dom": 0, "CD16-dom": 0, "GMP-dom": 0, "ProMono-dom": 0, "mixed": 0}
for r in successful:
    comp = r.get("celltype_composition", {})
    cd14, cd16 = comp.get("CD14 Mono", 0), comp.get("CD16 Mono", 0)
    gmp = sum(comp.get(ct, 0) for ct in MONO_PROGENITOR_GMP)
    pro = sum(comp.get(ct, 0) for ct in MONO_PROGENITOR_PRO)
    dom = max(cd14, cd16, gmp, pro)
    if dom < 40:           subtype_counts["mixed"]       += 1
    elif cd14 == dom:      subtype_counts["CD14-dom"]    += 1
    elif cd16 == dom:      subtype_counts["CD16-dom"]    += 1
    elif gmp  == dom:      subtype_counts["GMP-dom"]     += 1
    else:                  subtype_counts["ProMono-dom"] += 1

print(f"\n{'='*50}")
print(f"Mono Annotation Complete")
print(f"{'='*50}")
print(f"  Total annotated:        {len(results_list)}")
print(f"  Successful:             {len(successful)}")
print(f"  Unique programme names: {unique_names}")
print(f"\nConfidence breakdown:")
print(f"  HIGH:    {conf_counts['HIGH']:>3}  ({100*conf_counts['HIGH']/max(1,len(successful)):.1f}%)")
print(f"  MEDIUM:  {conf_counts['MEDIUM']:>3}  ({100*conf_counts['MEDIUM']/max(1,len(successful)):.1f}%)")
print(f"  LOW:     {conf_counts['LOW']:>3}  ({100*conf_counts['LOW']/max(1,len(successful)):.1f}%)")
print(f"\nDominant subtype distribution:")
for st, n in subtype_counts.items():
    print(f"  {st:<14} {n:>3}")
print(f"\nOutputs:")
print(f"  {ann_path}")
print(f"  {func_cache_path}")
print(f"\nReady for cross-lineage joint analysis (Dr. Ng Action Item 3).")

gc.collect()
