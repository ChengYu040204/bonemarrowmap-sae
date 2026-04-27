import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE

# custom class based on 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Device set dynamically for CPU/GPU compatibility
torch.manual_seed(42)

with open('train_activation_matrix.pkl', 'rb') as instream:
    train_data = pickle.load(instream)

with open('eval_activation_matrix.pkl', 'rb') as instream:
    eval_data = pickle.load(instream)

with open('all_activation_matrix.pkl', 'rb') as instream:
    all_data = pickle.load(instream)

# eval_size = int(eval_split * len(dataset))
# train_data, eval_data = dataset[:-eval_size], dataset[-eval_size:]

# force training by epoch: repeat the same data <n_epoch> times
n_epoch = 20
train_data_orig = train_data.clone()
train_data = train_data.repeat((n_epoch, 1))
train_data.to(device)
eval_data.to(device)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False)

activation_dim = 128 # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data must be an iterator that outputs strings
trainer_cfg = {
    "trainer": StandardTrainer,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,
    "dict_size": dictionary_size,
    "steps": 10000, 
    "warmup_steps": 100,
    "sparsity_warmup_steps": 200,
    "layer": 1, 
    "lm_name": "test", 
    "lr": 1e-3,
    "device": device,
}

# train the sparse autoencoder (SAE)
# final weights and config saved to save_dir
trainSAE(
    data=train_loader,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
    steps = len(train_loader),
    save_dir="sae_dictionary-learning/",
    verbose = True,
    log_steps = 500,
)

torch.cuda.empty_cache() 
# get features and reconstructed activations
from dictionary_learning import AutoEncoder
ae = AutoEncoder.from_pretrained("sae_dictionary-learning/trainer_0/ae.pt")
ae.to(device)
ae.eval()
train_data_orig.to(device)
all_data.to(device)
recons_act_train, features_train = ae(train_data_orig, output_features=True)
recons_act_eval, features_eval = ae(eval_data, output_features=True)
recons_act_all, features_all = ae(all_data, output_features=True)

recons_act_train.to('cpu')
features_train.to('cpu')
recons_act_eval.to('cpu')
features_eval.to('cpu')
recons_act_all.to('cpu')
features_all.to('cpu')

with open("sae_dictionary-learning_results.pkl", "wb") as outstream:
    pickle.dump((recons_act_train, features_train, recons_act_eval, features_eval), outstream)

with open("sae_dictionary-learning_results_all.pkl", "wb") as outstream:
    pickle.dump((recons_act_all, features_all), outstream)

