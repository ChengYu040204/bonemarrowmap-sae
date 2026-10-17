"""
train_sae_tcnk.py
=================
Route B: Train a dedicated Sparse Autoencoder on T/NK cell scVI activations.

Background (why Route B):
  The full-dataset SAE (161K cells, 16x expansion 128->2048) allocates features
  proportionally to variance explained. Erythroid differentiation dominates the
  bone marrow, so only 3/2048 SAE features are T/NK-enriched despite T/NK being
  43.8% of cells. Training exclusively on 70,714 T/NK activations dedicates all
  2048 features to T/NK-specific variation (CD4 vs CD8 vs NK sub-programmes).

Prerequisites:
  - tcnk_train_activation_matrix.pkl  (from extract_scvi_activations_tcnk.py)
  - tcnk_eval_activation_matrix.pkl
  - tcnk_all_activation_matrix.pkl

Pipeline mirrors train_sae.ipynb (B-cell SAE):
  Cell 1   : Load pkl files, inspect shapes and activation statistics
  Cell 1.5 : Normalize activations (center + scale by 95th percentile L2 norm)
  Cell 2   : Ablation sweep over l1_penalty {1e-3, 5e-4, 1e-4}
  Cell 3   : Train final SAE with best l1_penalty
  Cell 4   : Extract T/NK SAE feature matrix (70,714 x 2048), save pkl

Run this file section by section in VSCode Python Interactive (Jupyter-style),
or convert to .ipynb with jupytext.

Author: SAE x Haematopoiesis project (FP-Decode), UCL Gurdon Studentship 2026
"""

# %%
# ==============================================================================
# Cell 1: Load pkl files + inspect activation statistics
# ==============================================================================
import os
import pickle
import numpy as np
import torch

# Adjust if running from a different working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
PKL_DIR    = SCRIPT_DIR

def load_pkl(fname):
    path = os.path.join(PKL_DIR, fname)
    with open(path, "rb") as f:
        data = pickle.load(f)
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.array(data)

train_acts = load_pkl("tcnk_train_activation_matrix.pkl")
eval_acts  = load_pkl("tcnk_eval_activation_matrix.pkl")
all_acts   = load_pkl("tcnk_all_activation_matrix.pkl")

print(f"tcnk_train shape: {train_acts.shape}")
print(f"tcnk_eval  shape: {eval_acts.shape}")
print(f"tcnk_all   shape: {all_acts.shape}")

# Inspect raw activation distribution (critical before normalization)
flat = all_acts.flatten()
print(f"\nRaw activation stats:")
print(f"  mean  = {flat.mean():.4f}")
print(f"  std   = {flat.std():.4f}")
print(f"  min   = {flat.min():.4f}")
print(f"  max   = {flat.max():.4f}")
print(f"  > 0   = {(flat > 0).mean()*100:.1f}%")
# Expected: mean ~1-2, std ~5-10, min=0 (ReLU), some large outliers
# If these look very different from B-cell values, check the extraction script.


# %%
# ==============================================================================
# Cell 1.5: Normalize activations
# Mirrors Session 5 fix: scVI ReLU output is non-zero-centered with outliers.
# dictionary_learning library assumes approximately zero-centered inputs.
# Two steps: (1) subtract per-feature mean, (2) divide by 95th-percentile L2 norm.
# ==============================================================================
import torch as th

# Compute normalization parameters from TRAINING set only
# (avoid data leakage from eval/all sets)
train_t = th.tensor(train_acts, dtype=th.float32)
mean_act = train_t.mean(dim=0, keepdim=True)     # shape (1, 128): per-feature mean

# Center
train_centered = train_t - mean_act
eval_t  = th.tensor(eval_acts,  dtype=th.float32) - mean_act
all_t   = th.tensor(all_acts,   dtype=th.float32) - mean_act

# Scale by 95th percentile of row L2 norms (robust to outliers)
row_norms = train_centered.norm(dim=1)
scale = th.quantile(row_norms, 0.95).item()
print(f"Normalization scale (95th pctile L2 norm of training rows): {scale:.4f}")

train_norm = train_centered / scale
eval_norm  = eval_t  / scale
all_norm   = all_t   / scale

print(f"\nPost-normalization stats (train):")
flat_norm = train_norm.numpy().flatten()
print(f"  mean  = {flat_norm.mean():.4f}  (target: ~0)")
print(f"  std   = {flat_norm.std():.4f}   (expect ~0.03-0.06 based on B-cell: 0.041)")
print(f"  max   = {flat_norm.max():.4f}   (target: < 5)")

# Save normalization parameters for inference/reproducibility
norm_params = {"mean_act": mean_act, "scale": scale}
th.save(norm_params, os.path.join(SCRIPT_DIR, "tcnk_activation_norm_params.pt"))
print("Saved: tcnk_activation_norm_params.pt")

# Convert back to numpy for dictionary_learning
train_np = train_norm.numpy().astype(np.float32)
eval_np  = eval_norm.numpy().astype(np.float32)
all_np   = all_norm.numpy().astype(np.float32)


# %%
# ==============================================================================
# Cell 2: Ablation sweep over l1_penalty
# Run all three values and record: dead features, L0, frac_variance_explained.
# Target: dead~0, L0=20-80, frac_var>0.5
#
# IMPORTANT: run each sub-block separately and read the printed metrics before
# proceeding. The right l1_penalty for T/NK may differ from B-cell (5e-4)
# because T/NK activation scale may differ.
# ==============================================================================
import sys
sys.path.insert(0, SCRIPT_DIR)

from dictionary_learning import AutoEncoder, StandardTrainer, trainSAE

INPUT_DIM      = train_np.shape[1]   # 128
EXPANSION      = 16
DICT_SIZE      = INPUT_DIM * EXPANSION   # 2048

print(f"SAE architecture: {INPUT_DIM} -> {DICT_SIZE} (expansion ratio {EXPANSION}x)")
print(f"Training samples: {len(train_np)}")
print(f"Eval samples:     {len(eval_np)}")

# Sub-function: train one SAE and return quality metrics
def run_ablation(l1_penalty, label):
    import torch as th
    from torch.utils.data import TensorDataset, DataLoader

    train_t_local = th.tensor(train_np, dtype=th.float32)
    eval_t_local  = th.tensor(eval_np,  dtype=th.float32)

    train_ds = TensorDataset(train_t_local)
    eval_ds  = TensorDataset(eval_t_local)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,  drop_last=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=2048, shuffle=False, drop_last=False)

    total_steps = len(train_loader)

    ae = AutoEncoder(INPUT_DIM, DICT_SIZE)
    trainer_cfg = {
        "lr":                   1e-3,
        "l1_penalty":           l1_penalty,
        "steps":                total_steps,
        "warmup_steps":         max(1, total_steps // 45),
        "sparsity_warmup_steps": max(1, total_steps // 22),
        "resample_steps":       max(1, total_steps // 15),
    }
    trainer = StandardTrainer(ae, **trainer_cfg)
    save_dir = os.path.join(SCRIPT_DIR, f"sae_tcnk_l1pen{l1_penalty}")
    os.makedirs(save_dir, exist_ok=True)

    trainSAE(
        data=train_loader,
        trainer=trainer,
        save_dir=save_dir,
        log_steps=max(1, total_steps // 10),
    )

    # Evaluate on eval set
    ae.eval()
    all_t_local = th.tensor(all_np, dtype=th.float32)
    with th.no_grad():
        recons, features = ae(all_t_local)

    feats_np = features.detach().numpy()
    dead  = (feats_np.std(axis=0) < 1e-6).sum()
    l0    = (feats_np > 0).sum(axis=1).mean()

    # Fraction of variance explained
    recons_np = recons.detach().numpy()
    all_flat  = all_np
    ss_res = ((all_flat - recons_np) ** 2).mean()
    ss_tot = all_flat.var()
    frac_var = 1.0 - ss_res / (ss_tot + 1e-10)

    print(f"\n[{label}] l1_penalty={l1_penalty}")
    print(f"  Dead features: {dead}/{DICT_SIZE}")
    print(f"  Mean L0:       {l0:.1f}  (target: 20-80)")
    print(f"  Frac var expl: {frac_var:.4f}  (target: >0.5)")

    return {
        "l1_penalty": l1_penalty,
        "dead": int(dead),
        "l0": float(l0),
        "frac_var": float(frac_var),
        "save_dir": save_dir,
    }

# ---- Run ablation ----
# Run one at a time; read metrics before running next.
# Comment out the ones you have already run.

print("\n=== Ablation 1/3 ===")
result_1e3 = run_ablation(1e-3, "Ablation 1/3")

print("\n=== Ablation 2/3 ===")
result_5e4 = run_ablation(5e-4, "Ablation 2/3")

print("\n=== Ablation 3/3 ===")
result_1e4 = run_ablation(1e-4, "Ablation 3/3")

print("\n=== Ablation Summary ===")
for r in [result_1e3, result_5e4, result_1e4]:
    print(f"  l1={r['l1_penalty']:.0e}  dead={r['dead']:4d}  L0={r['l0']:5.1f}  frac_var={r['frac_var']:.4f}")

print("\nChoose the l1_penalty where: dead~0, L0=20-80, frac_var>0.5")
print("Set BEST_L1 in Cell 3 to the chosen value.")


# %%
# ==============================================================================
# Cell 3: Train final SAE with chosen l1_penalty
# Set BEST_L1 to the value that gave dead~0, L0=20-80, frac_var>0.5.
# Based on B-cell experience, expect 5e-4 to be a good starting point.
# ==============================================================================
import torch as th
from torch.utils.data import TensorDataset, DataLoader

BEST_L1 = 5e-4   # Update after reading ablation results

train_t = th.tensor(train_np, dtype=th.float32)
eval_t  = th.tensor(eval_np,  dtype=th.float32)

train_ds = TensorDataset(train_t)
eval_ds  = TensorDataset(eval_t)
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,  drop_last=True)
eval_loader  = DataLoader(eval_ds,  batch_size=2048, shuffle=False, drop_last=False)

total_steps = len(train_loader)
print(f"Total training steps: {total_steps}")

ae_final = AutoEncoder(INPUT_DIM, DICT_SIZE)
trainer_cfg_final = {
    "lr":                    1e-3,
    "l1_penalty":            BEST_L1,
    "steps":                 total_steps,
    "warmup_steps":          max(1, total_steps // 45),
    "sparsity_warmup_steps": max(1, total_steps // 22),
    "resample_steps":        max(1, total_steps // 15),
}
trainer_final = StandardTrainer(ae_final, **trainer_cfg_final)

FINAL_SAVE_DIR = os.path.join(SCRIPT_DIR, "sae_tcnk_final")
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)

print(f"Training final T/NK SAE (l1_penalty={BEST_L1})...")
trainSAE(
    data=train_loader,
    trainer=trainer_final,
    save_dir=FINAL_SAVE_DIR,
    log_steps=max(1, total_steps // 10),
)
print(f"Saved to: {FINAL_SAVE_DIR}/ae.pt")


# %%
# ==============================================================================
# Cell 4: Extract T/NK SAE features for all 70,714 T/NK cells
# ==============================================================================
import torch as th
import pickle, os, numpy as np
from dictionary_learning import AutoEncoder

# Load final model
FINAL_SAVE_DIR = os.path.join(SCRIPT_DIR, "sae_tcnk_final")
ae_loaded = AutoEncoder(INPUT_DIM, DICT_SIZE)
ae_loaded.load_state_dict(th.load(os.path.join(FINAL_SAVE_DIR, "ae.pt")))
ae_loaded.eval()

all_t = th.tensor(all_np, dtype=th.float32)

with th.no_grad():
    recons_all, features_all = ae_loaded(all_t)

feats_np  = features_all.detach().numpy()
recons_np = recons_all.detach().numpy()

print(f"Feature matrix shape: {feats_np.shape}")  # expect (70714, 2048)

# Quality check
dead    = (feats_np.std(axis=0) < 1e-6).sum()
l0_mean = (feats_np > 0).sum(axis=1).mean()
ss_res  = ((all_np - recons_np) ** 2).mean()
ss_tot  = all_np.var()
frac_var = 1.0 - ss_res / (ss_tot + 1e-10)

print(f"Dead features:    {dead}/{DICT_SIZE}")
print(f"Mean L0:          {l0_mean:.1f}")
print(f"Frac var explained: {frac_var:.4f}")

# Save result tuple (same format as original sae_results_all.pkl)
result_tuple = (recons_all, features_all)
out_path = os.path.join(SCRIPT_DIR, "sae_tcnk_results_all.pkl")
with open(out_path, "wb") as f:
    pickle.dump(result_tuple, f)
print(f"Saved: sae_tcnk_results_all.pkl  ({os.path.getsize(out_path)/1e6:.1f} MB)")

print("\nNext step: run task11_tcnk_annotation.py")
print("  Input:  sae_tcnk_results_all.pkl + tcnk_global_indices.pkl")
print("  Output: tcnk_anova_results_routeB.csv + sae_annotations_tcnk_routeB.json")
