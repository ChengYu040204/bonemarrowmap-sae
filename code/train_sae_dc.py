"""
train_sae_dc.py
================
Interactive Python script for DC-specific SAE training.
Run Cell by Cell in VSCode Python Interactive mode (Shift+Enter per Cell).

Cell structure:
  Cell 1:   Load DC pkl files, inspect activation distribution
  Cell 1.5: Normalize activations (center + scale), save norm params
  Cell 2:   Ablation sweep: l1_penalty in {1e-3, 5e-3, 1e-2, 2e-2}
  Cell 3:   Train final SAE with best l1_penalty (seed=42)
  Cell 4:   Extract full DC feature matrix, save sae_dc_results_all.pkl

Target metrics (from B-cell and T/NK experience):
  Dead features: ~0 to 10% of 2048
  L0 norm:       20 to 80 (mean active features per cell)
  frac_variance: > 0.85
"""

# %%
# Cell 1: Load DC activation pkl files
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
# Cell 1.5: Normalize activations (identical method to B-cell and T/NK SAE)
# Step 1: Center by subtracting per-feature mean (computed on train set)
# Step 2: Scale by 95th percentile of row L2 norms (robust to outliers)
# Save normalization params for reproducibility

print("Normalizing DC activations...")

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

print("Post-normalization stats (train):")
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
# Cell 2: Ablation sweep over l1_penalty values
# Adjust the candidate list if initial results are unexpected.
# Based on T/NK experience (best=2e-2), start search in similar range.
# B-cell experience: best was 5e-4 (different cell type, always ablate independently).

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
        "sparsity_warmup_steps":max(200, total_steps // 10),
        "steps":                total_steps,
        "resample_steps":       max(1000, total_steps // 3),
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

    features_np  = features.numpy()
    dead         = (features_np.max(axis=0) < 1e-6).sum()
    active        = HIDDEN_DIM - dead
    l0           = (features_np > 0).sum(axis=1).mean()
    ss_res       = ((eval_tensor.numpy() - recons.numpy()) ** 2).sum()
    ss_tot       = ((eval_tensor.numpy() - eval_tensor.numpy().mean(axis=0)) ** 2).sum()
    frac_var     = 1 - ss_res / (ss_tot + 1e-10)

    print(f"{l1:<14.0e} {dead:>8}/{HIDDEN_DIM} {active:>8} {l0:>8.1f} {frac_var:>10.4f}")

print("\nTarget: Dead~0, L0=20-80, frac_var>0.85")
print("Set BEST_L1 below and run Cell 3.")

# %%
# Cell 3: Train final DC SAE with best l1_penalty (seed=42 for reproducibility)
# Update BEST_L1 based on Cell 2 ablation results.

BEST_L1 = 2e-2  # UPDATE THIS based on Cell 2 output

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

ae_path = os.path.join(SAVE_DIR, "trainer_0", "ae.pt")
ae_final = torch.load(ae_path, map_location="cpu")
ae_final.eval()

with torch.no_grad():
    recons_eval, features_eval = ae_final(eval_tensor)

features_eval_np = features_eval.numpy()
dead_final   = (features_eval_np.max(axis=0) < 1e-6).sum()
active_final  = HIDDEN_DIM - dead_final
l0_final     = (features_eval_np > 0).sum(axis=1).mean()
ss_res       = ((eval_tensor.numpy() - recons_eval.numpy()) ** 2).sum()
ss_tot       = ((eval_tensor.numpy() - eval_tensor.numpy().mean(axis=0)) ** 2).sum()
frac_var_final = 1 - ss_res / (ss_tot + 1e-10)

print("\nFinal model quality:")
print(f"  Dead features: {dead_final}/{HIDDEN_DIM}")
print(f"  Active features: {active_final}")
print(f"  Mean L0: {l0_final:.1f}")
print(f"  frac_variance_explained: {frac_var_final:.4f}")
print(f"  Saved to: {ae_path}")

# %%
# Cell 4: Extract full DC feature matrix (70K cells, 2048 features)
# Saves sae_dc_results_all.pkl as tuple (recons_all, features_all)

print("Extracting DC feature matrix from full DC set...")

all_tensor = torch.tensor(all_normed, dtype=torch.float32)
ae_loaded = torch.load(
    os.path.join(SCRIPT_DIR, "sae_dc_final", "trainer_0", "ae.pt"),
    map_location="cpu",
)
ae_loaded.eval()

BATCH = 4096
recons_list  = []
features_list = []

with torch.no_grad():
    for start in range(0, len(all_tensor), BATCH):
        batch = all_tensor[start : start + BATCH]
        r, f  = ae_loaded(batch)
        recons_list.append(r)
        features_list.append(f)
    if start % (BATCH * 5) == 0:
        print(f"  Processed {start}/{len(all_tensor)}", flush=True)

recons_all   = torch.cat(recons_list,   dim=0)
features_all = torch.cat(features_list, dim=0)

print(f"\nfeatures_all shape: {features_all.shape}")
features_np = features_all.numpy()
active_cells = (features_np.max(axis=0) > 1e-6).sum()
l0_all = (features_np > 0).sum(axis=1).mean()
print(f"Active features (per-cell max > 1e-6): {active_cells}/{HIDDEN_DIM}")
print(f"Mean L0 (non-zero features per cell): {l0_all:.1f}")

result_path = os.path.join(SCRIPT_DIR, "sae_dc_results_all.pkl")
with open(result_path, "wb") as f:
    pickle.dump((recons_all, features_all), f)
print(f"\nSaved sae_dc_results_all.pkl ({os.path.getsize(result_path)/1e6:.1f} MB)")
print("Next step: run task12_dc_annotation.py in terminal.")
