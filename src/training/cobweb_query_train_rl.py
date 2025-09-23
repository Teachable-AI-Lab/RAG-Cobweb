"""
Reinforcement Learning Fine-tuning of a Query Encoder with Cobweb Ranking
=========================================================================

This script uses REINFORCE to train a query encoder so that queries
produce embeddings that maximize retrieval reward against a Cobweb tree
of document embeddings.

Reward = reciprocal rank (1 / (rank+1)) of the ground-truth passage.
"""

import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from datasets import load_dataset

from src.cobweb.CobwebWrapper import CobwebWrapper
from src.whitening.pca_ica import PCAICAWhiteningModel

from tqdm import tqdm
from random import sample as randsample

# --------------------------
# Run setup
# --------------------------
UNIQUE_ID = uuid.uuid4().hex
SAVE_DIR = f"./models/cobweb_query_runs_rl/{UNIQUE_ID}"
os.makedirs(SAVE_DIR, exist_ok=True)

CORPUS_SIZE = 20000
TARGET_SIZE = 2000
DATASET_NAME = "msmarco"
SPLIT = "validation"
MODEL_NAME = "all-roberta-large-v1"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Run ID: {UNIQUE_ID}")
print(f"[INFO] Saving outputs to {SAVE_DIR}")

# --------------------------
# Dataset prep
# --------------------------
ds = load_dataset("ms_marco", "v2.1", split=SPLIT)
ds.shuffle()

all_passages, positive_pairs, corpus = [], [], []

for ex in ds:
    query = ex["query"]
    passage_texts = ex["passages"]["passage_text"]
    is_selected_flags = ex["passages"]["is_selected"]

    if any(is_selected_flags) and len(positive_pairs) < TARGET_SIZE:
        positive_pairs.append((query, passage_texts[is_selected_flags.index(1)]))
        corpus.extend(passage_texts)
    elif len(corpus) < CORPUS_SIZE:
        all_passages.extend(passage_texts)
    else:
        break

if len(corpus) < CORPUS_SIZE:
    corpus.extend(randsample(all_passages, CORPUS_SIZE - len(corpus)))

queries = [pair[0] for pair in positive_pairs]
targets = [pair[1] for pair in positive_pairs]

print(f"[INFO] Loaded {len(corpus)} passages, {len(queries)} queries, {len(targets)} targets.")

# --------------------------
# Document encoder + whitening
# --------------------------
document_encoder = SentenceTransformer(MODEL_NAME)
corpus_embs = document_encoder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
pca_ica_model = PCAICAWhiteningModel.fit(corpus_embs, pca_dim=0.96)

pca_ica_corp_embs = pca_ica_model.transform(corpus_embs)
pca_dim_size = pca_ica_corp_embs.shape[1]

def encode_func(text):
    emb = document_encoder.encode([text])
    return pca_ica_model.transform(emb)[0]

document_cobweb = CobwebWrapper(corpus, pca_ica_corp_embs, encode_func=encode_func)
sentence_to_leaf = {sent: leaf_idx for leaf_idx, sent in enumerate(document_cobweb.sentences)}

print("[INFO] Finished Cobweb building!")

# --------------------------
# Query encoder (policy)
# --------------------------
class StochasticQueryEncoder(nn.Module):
    def __init__(self, base_model: str, out_dim: int):
        super().__init__()
        self.base = SentenceTransformer(base_model)
        in_dim = self.base.get_sentence_embedding_dimension()
        self.projection = nn.Linear(in_dim, out_dim)

        # exploration control
        self.log_std = nn.Parameter(torch.zeros(out_dim) + 0.5)

    def forward(self, features):
        base_emb = self.base(features)["sentence_embedding"]
        projected = self.projection(base_emb)
        std = torch.exp(self.log_std)
        noise = torch.randn_like(projected) * std
        action = projected + noise
        return action, projected, std


# --------------------------
# Reward function
# --------------------------
def compute_reward(query_emb, cobweb, true_leaf, device):
    scores = cobweb.cobweb_rank_scores(query_emb, is_embedding=True)
    sorted_indices = torch.argsort(scores, descending=True)

    if true_leaf in sorted_indices:
        rank_pos = sorted_indices.tolist().index(true_leaf)
    else:
        rank_pos = len(sorted_indices)

    reward = 1.0 / (rank_pos + 1)   # reciprocal rank
    reward = torch.tensor(reward * 10.0, device=device)  # scaled
    return reward


# --------------------------
# Evaluation
# --------------------------
def evaluate(query_encoder, cobweb, queries, targets, sentence_to_leaf, device="cpu", top_k=10):
    query_encoder.eval()
    results = []
    recall_hits = 0
    reciprocal_ranks = []
    true_scores = []
    rank_positions = []

    with torch.no_grad():
        for i, (q, t) in enumerate(zip(queries, targets)):
            if t not in sentence_to_leaf:
                continue

            features = query_encoder.base.tokenize([q])
            features = {k: v.to(device) for k, v in features.items()}
            q_emb, _, _ = query_encoder(features)
            q_emb = q_emb[0]

            scores = cobweb.cobweb_rank_scores(q_emb, is_embedding=True)

            true_leaf = sentence_to_leaf[t]
            true_score = scores[true_leaf].item()
            true_scores.append(true_score)

            sorted_indices = torch.argsort(scores, descending=True).tolist()
            if true_leaf in sorted_indices:
                rank_pos = sorted_indices.index(true_leaf) + 1
                rank_positions.append(rank_pos)
            else:
                rank_pos = None

            top_indices = sorted_indices[:top_k]
            if true_leaf in top_indices:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / rank_pos)
            else:
                reciprocal_ranks.append(0.0)

            results.append({
                "query": q,
                "target": t,
                "true_score": true_score,
                "rank_pos": rank_pos,
                "top_docs": [cobweb.sentences[idx] for idx in top_indices],
                "top_scores": [scores[idx].item() for idx in top_indices],
            })

    total = len(results)
    recall_at_k = recall_hits / total if total > 0 else 0
    mrr_at_k = sum(reciprocal_ranks) / total if total > 0 else 0
    avg_true_score = sum(true_scores) / total if total > 0 else 0
    avg_rank_pos = sum(rank_positions) / len(rank_positions) if rank_positions else float("inf")

    print("\n" + "#"*100)
    print(f"[EVAL] Evaluated {total} queries")
    print(f"[EVAL] Recall@{top_k}: {recall_at_k:.4f}")
    print(f"[EVAL] MRR@{top_k}: {mrr_at_k:.4f}")
    print(f"[EVAL] Avg True Similarity Score: {avg_true_score:.4f}")
    print(f"[EVAL] Avg Rank Position: {avg_rank_pos:.2f}")
    print("#"*100 + "\n")

    return {
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr_at_k,
        "avg_true_score": avg_true_score,
        "avg_rank_pos": avg_rank_pos,
        "rank_positions": rank_positions,
    }


# --------------------------
# Training loop (REINFORCE + eval)
# --------------------------
def collate_inputexample(batch):
    queries = [ex.texts[0] for ex in batch]
    labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
    return queries, labels

def train_rl_query_encoder(
    encoder, cobweb, dataloader, optimizer, device,
    queries, targets, sentence_to_leaf,
    num_epochs=5, save_dir="./models/cobweb_query_runs_rl"
):
    os.makedirs(save_dir, exist_ok=True)
    running_baseline = 0.0
    beta = 0.9

    for epoch in range(num_epochs):
        total_loss = 0.0
        count = 0

        for step, (batch_queries, batch_labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            batch_loss = 0.0

            for q, true_leaf in zip(batch_queries, batch_labels):
                features = encoder.base.tokenize([q])
                features = {k: v.to(device) for k, v in features.items()}
                action, mean_emb, std = encoder(features)

                reward = compute_reward(action[0], cobweb, true_leaf.item(), device)

                running_baseline = beta * running_baseline + (1 - beta) * reward.item()
                adv = reward - running_baseline

                var = std.pow(2)
                log_prob = -((action - mean_emb) ** 2 / (2 * var)).sum() - torch.log(std).sum()
                loss = -log_prob * adv

                batch_loss += loss

            batch_loss = batch_loss / len(batch_queries)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            count += 1

            if step % 10 == 0:
                print(f"[Batch {step}] Loss = {batch_loss.item():.4f}")

        avg_loss = total_loss / count
        print(f"[INFO] Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(save_dir, f"rl_query_encoder_epoch{epoch+1}.pt")
        torch.save(encoder.state_dict(), ckpt_path)
        print(f"[INFO] Saved checkpoint -> {ckpt_path}")

        # Evaluate after each epoch
        evaluate(encoder, cobweb, queries, targets, sentence_to_leaf, device=device, top_k=10)


# --------------------------
# Build dataloader
# --------------------------
train_examples = []
for i, (q, t) in enumerate(zip(queries, targets)):
    if t in sentence_to_leaf:
        train_examples.append(InputExample(texts=[q], label=sentence_to_leaf[t]))

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=12,
    collate_fn=collate_inputexample
)

# --------------------------
# Run training
# --------------------------
query_encoder = StochasticQueryEncoder(MODEL_NAME, out_dim=pca_dim_size).to(device)
optimizer = optim.AdamW(query_encoder.parameters(), lr=2e-5)

train_rl_query_encoder(
    query_encoder, document_cobweb, train_dataloader, optimizer,
    device=device, queries=queries, targets=targets,
    sentence_to_leaf=sentence_to_leaf, num_epochs=10, save_dir=SAVE_DIR
)

print(f"[INFO] Training finished. Models + metrics saved in {SAVE_DIR}")
