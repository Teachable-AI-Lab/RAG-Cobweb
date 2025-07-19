import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from your_model_file import FullEncoder  # Make sure to import your saved model

def whiteness_report(embeddings, title=""):
    with torch.no_grad():
        emb = embeddings.cpu().numpy()

        mean = np.mean(emb, axis=0)
        cov = np.cov(emb, rowvar=False)
        std = np.std(emb, axis=0)

        # Normalize cov matrix for comparison with identity
        identity = np.eye(cov.shape[0])
        diff_from_identity = np.linalg.norm(cov - identity)
        condition_number = np.linalg.cond(cov)

        print(f"\n== Whitening Diagnostics: {title} ==")
        print(f"Mean norm:           {np.linalg.norm(mean):.4f}")
        print(f"Covariance deviation: {diff_from_identity:.4f}")
        print(f"Covariance cond #:    {condition_number:.2f}")
        print(f"Std (should ≈ 1):     min {std.min():.2f}, max {std.max():.2f}")

        # Optional: plot correlation matrix
        corr = np.corrcoef(emb.T)
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Embedding Correlation Matrix")
        plt.show()


def embed_texts(texts, model, tokenizer, device):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            emb = model(**encoded)
            all_embeddings.append(emb.squeeze(0).cpu())

    return torch.stack(all_embeddings)