from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_qqp_texts(limit=32000):
    dataset = load_dataset("glue", "qqp")["train"]
    dataset = dataset.filter(lambda x: x["label"] == 1)

    print("Loaded", len(dataset), "instances through QQP dataset!")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    texts1 = list(dataset["question1"])

    return texts1

def whiteness_report(embeddings, title=""):
    emb = embeddings.cpu().numpy()
    mean = np.mean(emb, axis=0)
    cov = np.cov(emb, rowvar=False)
    std = np.std(emb, axis=0)

    diff_from_identity = np.linalg.norm(cov - np.eye(cov.shape[0]))
    cond_num = np.linalg.cond(cov)

    print(f"\n== Whitening Diagnostics: {title} ==")
    print(f"Mean norm:            {np.linalg.norm(mean):.4f}")
    print(f"Covariance deviation: {diff_from_identity:.4f}")
    print(f"Covariance cond #:     {cond_num:.2e}")
    print(f"Std (should ≈ 1):      min {std.min():.2f}, max {std.max():.2f}")

    corr = np.corrcoef(emb.T)
    plt.figure(figsize=(6,6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"{title} – Embedding Correlation")
    plt.show()

def embed_texts(texts, model, tokenizer, device, batch_size=64):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = model(**enc)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhitenedRoberta("whitening_roberta").to(device)
    tokenizer = AutoTokenizer.from_pretrained("whitening_roberta")

    # Load QQP pairs + flatten to a list of texts
    all_texts = load_qqp_texts(limit=3000)
    print(f"Analyzing {len(all_texts)} total texts...")

    embeddings = embed_texts(all_texts, model, tokenizer, device=device)
    whiteness_report(embeddings, title="QQP Embeddings (n=3000)")
