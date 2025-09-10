#!/usr/bin/env python3
"""
FactorVAE for two text tasks: QQP and MS-MARCO.
- Trains a FactorVAE on sentence / passage embeddings from a SentenceTransformer (RoBERTa variants).
- Includes latent correlation diagnostics per epoch.

Usage examples:
    # QQP
    python factorvae_text_tasks.py --task qqp --epochs 20 --batch-size 256

    # MS-MARCO (default text fields "query,passage"; change with --msmarco-text-fields)
    python factorvae_text_tasks.py --task msmarco --epochs 10 --batch-size 512 --msmarco-max-samples 50000
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# Optional imports for text tasks
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def permute_dims_across_batch(z: torch.Tensor) -> torch.Tensor:
    """
    Permute each latent dimension independently across the batch.
    z: (B, D)
    returns z_perm: (B, D)
    """
    B, D = z.size()
    z_perm = []
    for j in range(D):
        idx = torch.randperm(B, device=z.device)
        z_perm.append(z[idx, j])
    z_perm = torch.stack(z_perm, dim=1)
    return z_perm

# ---------------------------
# MLP Encoder/Decoder (for embeddings)
# ---------------------------
class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, z_dim: int = 32, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden // 2, z_dim)
        self.logvar = nn.Linear(hidden // 2, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class MLPDecoder(nn.Module):
    def __init__(self, output_dim: int, z_dim: int = 32, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------
# Discriminator for TC estimation
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, z_dim: int, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z):
        return self.net(z).squeeze(1)  # logits


# ---------------------------
# VAE utilities
# ---------------------------
def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    # per-sample KL (sum over latent dims)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


# ---------------------------
# Embedding builders
# ---------------------------
def build_embeddings_from_pairs(
    ds_split,
    text_fields: List[str],
    embed_model: str,
    device: str,
    max_samples: int = None,
    embed_batch_size: int = 256
) -> TensorDataset:
    """
    Given a HuggingFace dataset split and list of text_fields (e.g. ['question1','question2'] or ['query','passage']),
    compute sentence-transformer embeddings and produce a TensorDataset of concatenated embeddings.
    - text_fields: list of 1 or 2 fields. If 2 fields, embeddings are concatenated [emb1 || emb2].
    """
    print(f"[embed builder] Using model: {embed_model} on device {device}")
    sbert = SentenceTransformer(embed_model, device=device)

    # prepare text pairs
    texts_a = ds_split[text_fields[0]]
    texts_b = None
    if len(text_fields) > 1:
        texts_b = ds_split[text_fields[1]]

    N = len(texts_a)
    if max_samples:
        N = min(max_samples, N)

    embeddings_out = []
    for i in tqdm(range(0, N, embed_batch_size), desc="Embedding batches"):
        end = min(i + embed_batch_size, N)
        batch_a = [t if t is not None else "" for t in texts_a[i:end]]
        emb_a = sbert.encode(batch_a, convert_to_tensor=True, show_progress_bar=False, device=device)

        if texts_b is not None:
            batch_b = [t if t is not None else "" for t in texts_b[i:end]]
            emb_b = sbert.encode(batch_b, convert_to_tensor=True, show_progress_bar=False, device=device)
            emb_cat = torch.cat([emb_a, emb_b], dim=1)
        else:
            emb_cat = emb_a

        embeddings_out.append(emb_cat.cpu())

    embeddings = torch.cat(embeddings_out, dim=0)
    print(f"[embed builder] Built embeddings shape: {embeddings.shape}")
    return TensorDataset(embeddings)


# ---------------------------
# Latent correlation diagnostics
# ---------------------------
def latent_correlation_diagnostics(z_tensor: torch.Tensor, top_k: int = 10) -> dict:
    """
    z_tensor: (N, D) numpy or torch tensor (on CPU)
    Returns: dict with covariance, correlation, mean_abs_offdiag, top_k pairs (abs corr)
    """
    if isinstance(z_tensor, torch.Tensor):
        z = z_tensor.detach().cpu().numpy()
    else:
        z = z_tensor
    import numpy as np

    # zero-mean
    zc = z - z.mean(axis=0, keepdims=True)
    cov = np.cov(zc, rowvar=False)
    diag = np.sqrt(np.diag(cov) + 1e-12)
    corr = cov / (diag[:, None] * diag[None, :])
    # numerical cleanup
    corr = np.clip(corr, -1.0, 1.0)
    D = corr.shape[0]
    abs_corr = np.abs(corr)
    # remove diagonal
    abs_corr_no_diag = abs_corr.copy()
    np.fill_diagonal(abs_corr_no_diag, 0.0)
    mean_abs_offdiag = abs_corr_no_diag.sum() / (D * (D - 1))
    # find top_k pairs
    flat_idx = abs_corr_no_diag.flatten().argsort()[::-1]
    top_pairs = []
    seen = set()
    for ind in flat_idx:
        if len(top_pairs) >= top_k:
            break
        i = ind // D
        j = ind % D
        if i == j:
            continue
        if (j, i) in seen:
            continue
        seen.add((i, j))
        top_pairs.append((int(i), int(j), float(corr[i, j])))
    return {
        "cov": cov,
        "corr": corr,
        "mean_abs_offdiag": float(mean_abs_offdiag),
        "top_pairs": top_pairs
    }


# ---------------------------
# Training loops
# ---------------------------
def train_factorvae_on_embeddings(
    dataset: TensorDataset,
    device: str,
    epochs: int = 20,
    batch_size: int = 256,
    z_dim: int = 32,
    gamma: float = 10.0,
    lr: float = 1e-4,
    save_dir: str = "factorvae_ckpts",
    seed: int = 42,
    eval_sample_for_diag: int = 4096
):
    """
    General FactorVAE training on a TensorDataset of embeddings (each item is one embedding vector).
    """

    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    input_dim = dataset.tensors[0].shape[1]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    encoder = MLPEncoder(input_dim=input_dim, z_dim=z_dim).to(device)
    decoder = MLPDecoder(output_dim=input_dim, z_dim=z_dim).to(device)
    discriminator = Discriminator(z_dim=z_dim).to(device)

    opt_vae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)

    global_step = 0
    for epoch in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for (emb,) in pbar:
            emb = emb.to(device)

            # ---- VAE forward ----
            mu, logvar = encoder(emb)
            z = reparameterize(mu, logvar)
            recon = decoder(z)
            recon_loss = F.mse_loss(recon, emb, reduction="mean")
            kl = kl_divergence(mu, logvar).mean()

            # ---- Discriminator training ----
            with torch.no_grad():
                z_detach = z.detach()
            z_perm = permute_dims_across_batch(z_detach)
            disc_real_logits = discriminator(z_detach)
            disc_perm_logits = discriminator(z_perm)

            disc_loss = 0.5 * (F.binary_cross_entropy_with_logits(disc_real_logits, torch.ones_like(disc_real_logits)) +
                               F.binary_cross_entropy_with_logits(disc_perm_logits, torch.zeros_like(disc_perm_logits)))

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            # ---- TC estimate for generator ----
            real_logits_for_tc = discriminator(z)
            perm_logits_for_tc = discriminator(permute_dims_across_batch(z))
            tc_est = (real_logits_for_tc - perm_logits_for_tc).mean()

            vae_loss = recon_loss + kl + gamma * tc_est

            opt_vae.zero_grad()
            vae_loss.backward()
            opt_vae.step()

            global_step += 1
            pbar.set_postfix({
                "recon_mse": f"{recon_loss.item():.6f}",
                "kl": f"{kl.item():.3f}",
                "tc": f"{tc_est.item():.3f}",
                "disc": f"{disc_loss.item():.3f}"
            })

        # -- End epoch: diagnostics & checkpoint --
        # sample a subset of embeddings, run encoder to get z for diagnostics
        sample_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        zs_for_diag = []
        taken = 0
        for (emb_sample,) in sample_loader:
            emb_sample = emb_sample.to(device)
            mu_s, logvar_s = encoder(emb_sample)
            z_s = reparameterize(mu_s, logvar_s).detach().cpu()
            zs_for_diag.append(z_s)
            taken += z_s.size(0)
            if taken >= eval_sample_for_diag:
                break
        zs_for_diag = torch.cat(zs_for_diag, dim=0)[:eval_sample_for_diag]

        diag = latent_correlation_diagnostics(zs_for_diag, top_k=10)
        print(f"[epoch {epoch}] mean_abs_offdiag_corr = {diag['mean_abs_offdiag']:.6f}")
        print(f"top correlated pairs (i, j, corr): {diag['top_pairs']}")

        # checkpoint
        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "disc": discriminator.state_dict(),
            "opt_vae": opt_vae.state_dict(),
            "opt_disc": opt_disc.state_dict(),
            "epoch": epoch
        }
        torch.save(ckpt, os.path.join(save_dir, f"factorvae_epoch{epoch}.pt"))

    print("Training finished. Checkpoints saved to:", save_dir)


# ---------------------------
# Task-specific dataset loaders
# ---------------------------
def prepare_qqp_embeddings(device: str, embed_model: str, max_samples: int = None, embed_batch_size: int = 512) -> TensorDataset:
    print("[QQP] loading dataset...")
    ds = load_dataset("glue", "qqp", split="train")
    # QQP fields: 'question1', 'question2'
    text_fields = ['question1', 'question2']
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return build_embeddings_from_pairs(ds, text_fields, embed_model, device, max_samples, embed_batch_size)

def prepare_msmarco_embeddings(device: str, embed_model: str, max_samples: int = None, text_fields: List[str] = None, embed_batch_size: int = 512) -> TensorDataset:
    """
    Loads MS-MARCO and builds embeddings.
    By default text_fields = ['query', 'passage'] (concatenate). You can pass a single field as well.
    Note: MS-MARCO dataset configs vary; we attempt 'ms_marco' with default split 'train'.
    """
    print("[MS-MARCO] loading dataset...")
    # Attempt to load a commonly used ms_marco config; user can override text_fields
    ds = load_dataset("ms_marco", "v1.1", split="train") if "v1.1" in load_dataset.__doc__ else load_dataset("ms_marco", split="train")
    if text_fields is None:
        text_fields = ["query", "passage"]  # default; may need adjusting depending on dataset config
    # if ms marco has multiple columns, ensure they exist; fallback to first text column available
    available_cols = list(ds.column_names)
    chosen_fields = []
    for f in text_fields:
        if f in available_cols:
            chosen_fields.append(f)
    if len(chosen_fields) == 0:
        # fallback: pick first string column
        for c in available_cols:
            if ds.features[c].dtype == 'string' or str(ds.features[c]).lower().startswith("value"):
                chosen_fields = [c]
                break
    print("[MS-MARCO] using text fields:", chosen_fields)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return build_embeddings_from_pairs(ds, chosen_fields, embed_model, device, max_samples, embed_batch_size)


# ---------------------------
# CLI and main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["qqp", "msmarco"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--embed-model", type=str, default="sentence-transformers/paraphrase-roberta-base-v1")
    p.add_argument("--max-embed-samples", type=int, default=None)
    p.add_argument("--msmarco-text-fields", type=str, default="query,passage",
                   help="Comma-separated fields to use from MS-MARCO (default 'query,passage'). You can pass a single field.")
    p.add_argument("--eval-samples-diag", type=int, default=4096,
                   help="Number of z samples to collect for latent diagnostics each epoch (may be expensive).")
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = args.save_dir or f"factorvae_{args.task}_ckpts"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(args.seed)

    if args.task == "qqp":
        dataset = prepare_qqp_embeddings(device=device, embed_model=args.embed_model, max_samples=args.max_embed_samples)
    elif args.task == "msmarco":
        fields = [f.strip() for f in args.msmarco_text_fields.split(",") if f.strip()]
        dataset = prepare_msmarco_embeddings(device=device, embed_model=args.embed_model, max_samples=args.max_embed_samples, text_fields=fields)
    else:
        raise ValueError("Unknown task")

    print("Dataset embeddings ready. Starting training...")
    train_factorvae_on_embeddings(
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        gamma=args.gamma,
        lr=args.lr,
        save_dir=save_dir,
        seed=args.seed,
        eval_sample_for_diag=args.eval_samples_diag
    )

if __name__ == "__main__":
    main()
