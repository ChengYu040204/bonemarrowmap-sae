import os
import tempfile

import scanpy as sc
import scvi
import seaborn as sns
import torch

from tqdm import tqdm

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)
torch.set_float32_matmul_precision("high")

# Read adata
adata_path = "cui_immune_dictionary.h5ad"
adata = sc.read(
    adata_path
)
sc.pp.highly_variable_genes(
    adata,
    # layer="counts",
    flavor="seurat_v3",
    n_top_genes=7500,
    span = 0.5,
    # batch_key="donorID_unified", # batch = library
    subset=True,
)

scvi.model.SCVI.setup_anndata(adata)

# load trained model .pt file
model = scvi.model.SCVI.load("ImmuneDictionary_scvi", adata=adata)

# model.module looks like a vanilla pytorch model
vae = model.module
vae.eval()
vae_encode = vae.z_encoder

activations = {}

def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activations[name] = output.detach()
  return hook

# Choose the layer you want to extract features from and Register the hook
hook = vae_encode.encoder.register_forward_hook(getActivation("encoder"))

# some exploration: this for instance gives you the latent space representation of one batch:
from scvi.dataloaders import AnnDataLoader
adl = iter(AnnDataLoader(model.adata_manager, indices = model.train_indices, shuffle=True, batch_size=32))
activation_matrix = []
# Forward pass through the model
with torch.no_grad():
    for batch in tqdm(adl):
        _, encoded = vae_encode(batch['X'].to('cuda')) # This triggers the forward hook
        activation_matrix.append(activations["encoder"])
    hook.remove()
    activation_matrix = torch.cat(activation_matrix, axis = 0)

activation_matrix.to('cpu')

# The `activation_matrix` can now be used as input to train the sparse autoencoder
import pickle
with open('train_activation_matrix.pkl', 'wb') as outstream:
   pickle.dump(activation_matrix, outstream)

# do the same for eval set
hook = vae_encode.encoder.register_forward_hook(getActivation("encoder"))
adl = iter(AnnDataLoader(model.adata_manager, indices = model.validation_indices, shuffle=True, batch_size=32))
activation_matrix = []
# Forward pass through the model
with torch.no_grad():
    for batch in tqdm(adl):
        _, encoded = vae_encode(batch['X'].to('cuda')) # This triggers the forward hook
        activation_matrix.append(activations["encoder"])
    hook.remove()
    activation_matrix = torch.cat(activation_matrix, axis = 0)

activation_matrix.to('cpu')
with open('eval_activation_matrix.pkl', 'wb') as outstream:
   pickle.dump(activation_matrix, outstream)

# do the same for the whole dataset
hook = vae_encode.encoder.register_forward_hook(getActivation("encoder"))
adl = iter(AnnDataLoader(model.adata_manager, indices = range(len(adata)), shuffle=False, batch_size=32))
activation_matrix = []
# Forward pass through the model
with torch.no_grad():
    for batch in tqdm(adl):
        _, encoded = vae_encode(batch['X'].to('cuda')) # This triggers the forward hook
        activation_matrix.append(activations["encoder"])
    hook.remove()
    activation_matrix = torch.cat(activation_matrix, axis = 0)

activation_matrix.to('cpu')
with open('all_activation_matrix.pkl', 'wb') as outstream:
   pickle.dump(activation_matrix, outstream)
