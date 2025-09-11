import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from pathlib import Path

from src.whitening.factorvae_train import MLPEncoder
from src.whitening.pca_ica import PCAICAWhiteningModel

# ----------------------------
# Models info
# ----------------------------
models_info = {
    "BERT-base": {"repo": "bert-base-uncased", "subfolder": None},
    "WhitenedCSE": {"repo": "SupstarZh/whitenedcse-bert-large", "subfolder": "unsup-whitenedcse-bert-large"},
    "UNSEE-Barlow": {"repo": "asparius/UNSEE-Barlow", "subfolder": None},
}

# ----------------------------
# Load models and tokenizers
# ----------------------------
print("\n=== Loading Models and Tokenizers ===")
models_hf, tokenizers = {}, {}
for name, info in models_info.items():
    print(f"-> Loading {name} from {info['repo']}" + (f"/{info['subfolder']}" if info["subfolder"] else ""))
    models_hf[name] = (
        AutoModel.from_pretrained(info["repo"], subfolder=info["subfolder"])
        if info["subfolder"]
        else AutoModel.from_pretrained(info["repo"])
    )
    tokenizers[name] = (
        AutoTokenizer.from_pretrained(info["repo"], subfolder=info["subfolder"])
        if info["subfolder"]
        else AutoTokenizer.from_pretrained(info["repo"])
    )
print("✓ All models loaded.\n")

# ----------------------------
# Load QQP sample
# ----------------------------
print("=== Loading QQP Dataset Sample ===")
dataset = load_dataset("glue", "qqp", split="train[:200]")
sentences = list(dataset["question1"]) + list(dataset["question2"])
print(f"Loaded {len(sentences)} sentences from QQP.\n")

# ----------------------------
# Embedding function
# ----------------------------
def embed_sentences(tokenizer, model, sentences, batch_size=64, device='cpu'):
    print(f"   Encoding {len(sentences)} sentences...")
    model.eval()
    model.to(device)
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        all_embeddings.append(batch_embeddings.cpu().numpy())
        print(f"     Encoded batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}")
    return np.vstack(all_embeddings)

# ----------------------------
# Generate embeddings
# ----------------------------
print("=== Generating Embeddings ===")
embeddings = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for name in models_hf:
    print(f"-> {name}")
    embeddings[name] = embed_sentences(tokenizers[name], models_hf[name], sentences, device=device)
    print(f"   Shape: {embeddings[name].shape}\n")

# ----------------------------
# PCA+ICA whitening
# ----------------------------
print("=== Fitting PCA+ICA Whitening Models ===")
whitened_embeddings = {}
for base_name in ["WhitenedCSE", "UNSEE-Barlow"]:
    print(f"-> Training PCA+ICA whitening on {base_name} embeddings...")
    whitening_model = PCAICAWhiteningModel.fit(embeddings[base_name], pca_dim=0.96)
    whitened_embeddings[f"{base_name}+PCAICA"] = whitening_model.transform(embeddings[base_name])
    print(f"   Done. Whitened shape: {whitened_embeddings[f'{base_name}+PCAICA'].shape}\n")

embeddings.update(whitened_embeddings)

# ----------------------------
# Load pretrained FactorVAE encoder
# ----------------------------
factorvae_ckpt_path = Path("factorvae_qqp_ckpts/factorvae_epoch20.pt")  # adjust path
z_dim = 32  # match the trained FactorVAE

print(f"=== Loading pretrained FactorVAE from {factorvae_ckpt_path} ===")
# assume embeddings["WhitenedCSE"] as input to FactorVAE
sample_input_dim = embeddings["WhitenedCSE"].shape[1]
factorvae_encoder = MLPEncoder(input_dim=sample_input_dim, z_dim=z_dim).to(device)
ckpt = torch.load(factorvae_ckpt_path, map_location=device)
factorvae_encoder.load_state_dict(ckpt["encoder"])
factorvae_encoder.eval()

# Encode embeddings to FactorVAE latent space
factorvae_embeddings = {}
with torch.no_grad():
    x_tensor = torch.tensor(embeddings["WhitenedCSE"], dtype=torch.float32).to(device)
    mu, logvar = factorvae_encoder(x_tensor)
    z_latent = mu.cpu().numpy()  # use mean as representation
    factorvae_embeddings["WhitenedCSE+FactorVAE"] = z_latent
    print(f"-> FactorVAE latent shape: {z_latent.shape}\n")

embeddings.update(factorvae_embeddings)

# ----------------------------
# Correlation statistics
# ----------------------------
def compute_corr_stats(embed):
    corr = np.corrcoef(embed, rowvar=False)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return {
        "mean_abs_corr": np.mean(np.abs(off_diag)),
        "median_abs_corr": np.median(np.abs(off_diag)),
        "max_abs_corr": np.max(np.abs(off_diag)),
        "std_abs_corr": np.std(np.abs(off_diag)),
    }

print("=== Computing Correlation Statistics ===")
for name, embed in embeddings.items():
    print(f"\n>>> {name}")
    stats = compute_corr_stats(embed)
    print(f"   Embedding shape: {embed.shape}")
    for k, v in stats.items():
        print(f"   {k:>15}: {v:.4f}")
