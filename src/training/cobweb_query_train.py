"""
Finetune a query encoder via Contrastive Loss where the similarity metric is the
matchup between simulated and ground-truth rankings!

Dataset: MS-Marco
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from datasets import load_dataset

from src.cobweb.CobwebWrapper import CobwebWrapper
from src.whitening.pca_ica import PCAICAWhiteningModel

from tqdm import tqdm
from random import sample as randsample
import uuid

############ VARIABLES ############
UNIQUE_ID = uuid.uuid4().hex
SAVE_DIR = f"./models/cobweb_query_runs/{UNIQUE_ID}"
os.makedirs(SAVE_DIR, exist_ok=True)

CORPUS_SIZE = 20000
TARGET_SIZE = 1000
DATASET_NAME = "msmarco"
SPLIT = "validation"
MODEL_NAME = "all-roberta-large-v1"
###################################

print(f"[INFO] Run ID: {UNIQUE_ID}")
print(f"[INFO] Saving all outputs to {SAVE_DIR}")

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
corpus_embs = document_encoder.encode(corpus)
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
# Query encoder
# --------------------------
class QueryEncoderWithProjection(nn.Module):
    def __init__(self, base_model: SentenceTransformer, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.base = base_model
        in_dim = base_model.get_sentence_embedding_dimension()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, features):
        base_emb = self.base(features)["sentence_embedding"]
        return self.projection(base_emb)

class FixedDocsRankingLoss(nn.Module):
    def __init__(self, query_model, cobweb, temperature=1.0):
        super().__init__()
        self.query_model = query_model
        self.cobweb = cobweb
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, sentence_features, labels):
        query_emb = self.query_model(sentence_features)
        batch_loss = []

        for b in range(query_emb.size(0)):
            q = query_emb[b]
            leaf_scores = self.cobweb.cobweb_rank_scores(q, is_embedding=True)
            logits = leaf_scores.unsqueeze(0) / self.temperature
            logits = logits - logits.max() # normalization

            target = labels[b].unsqueeze(0)
            batch_loss.append(self.ce(logits, target))

        if not batch_loss:
            return torch.tensor(0.0, requires_grad=True, device=query_emb.device)

        return torch.stack(batch_loss).mean()

def collate_inputexample(batch):
    return batch

base_qencoder = SentenceTransformer(MODEL_NAME)
query_encoder = QueryEncoderWithProjection(base_qencoder, out_dim=pca_dim_size)

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

loss = FixedDocsRankingLoss(query_model=query_encoder, cobweb=document_cobweb)

# --------------------------
# Training
# --------------------------
def train(query_encoder, loss_fn, train_dataloader, num_epochs=3, lr=2e-5, device="cpu"):
    optimizer = torch.optim.AdamW(query_encoder.parameters(), lr=lr)
    query_encoder.to(device)

    for epoch in range(num_epochs):
        query_encoder.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            queries = [ex.texts[0] for ex in batch]
            labels = torch.tensor([ex.label for ex in batch], dtype=torch.long, device=device)

            features = query_encoder.base.tokenize(queries)
            features = {k: v.to(device) for k, v in features.items()}

            optimizer.zero_grad()
            loss = loss_fn(features, labels)
            loss.backward()

            # added gradient clipping
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"  [Batch {step}] Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"[INFO] Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(SAVE_DIR, f"cobweb_query_encoder_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": query_encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss
        }, ckpt_path)
        print(f"[INFO] Saved checkpoint -> {ckpt_path}")

    # Save final model
    final_path = os.path.join(SAVE_DIR, "cobweb_query_encoder_final.pt")
    torch.save(query_encoder.state_dict(), final_path)
    print(f"[INFO] Final model saved -> {final_path}")

train(
    query_encoder, loss, train_dataloader, 
    num_epochs=10,
    lr=2e-5,
    device="cpu"
)

"""
How to load:
---------------
model_path = FILENAME HERE
state_dict = torch.load(model_path, map_location=device)
query_encoder.load_state_dict(state_dict)
query_encoder.to(device)
query_encoder.eval()
"""

# --------------------------
# Evaluation (metrics + per-query reporting)
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

            # encode query
            features = query_encoder.base.tokenize([q])
            features = {k: v.to(device) for k, v in features.items()}
            q_emb = query_encoder(features)[0]  # (D,)

            # compute Cobweb scores
            scores = cobweb.cobweb_rank_scores(q_emb, is_embedding=True)

            # get ground-truth leaf index + score
            true_leaf = sentence_to_leaf[t]
            true_score = scores[true_leaf].item()
            true_scores.append(true_score)

            # rank documents
            sorted_indices = torch.argsort(scores, descending=True).tolist()
            if true_leaf in sorted_indices:
                rank_pos = sorted_indices.index(true_leaf) + 1  # 1-based rank
                rank_positions.append(rank_pos)
            else:
                rank_pos = None

            top_indices = sorted_indices[:top_k]
            top_scores = [scores[idx].item() for idx in top_indices]
            top_docs = [cobweb.sentences[idx] for idx in top_indices]

            # recall@k
            if true_leaf in top_indices:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / (rank_pos if rank_pos is not None else 1e9))
            else:
                reciprocal_ranks.append(0.0)

            # store results
            results.append({
                "query": q,
                "target": t,
                "true_score": true_score,
                "rank_pos": rank_pos,
                "top_docs": top_docs,
                "top_scores": top_scores
            })

            # print sample
            print("="*100)
            print(f"[Query {i}] {q}")
            print(f"  Ground Truth: {t}")
            print(f"  True Similarity Score: {true_score:.4f} | Rank: {rank_pos}")
            print("  Top Predictions:")
            for rank, (doc, score) in enumerate(zip(top_docs, top_scores), 1):
                marker = "<-- GT" if doc == t else ""
                print(f"    {rank}. {score:.4f} | {doc[:80]} {marker}")

    # --------------------------
    # Aggregate statistics
    # --------------------------
    total = len(results)
    recall_at_k = recall_hits / total if total > 0 else 0
    mrr_at_k = sum(reciprocal_ranks) / total if total > 0 else 0
    avg_true_score = sum(true_scores) / total if total > 0 else 0
    avg_rank_pos = sum(rank_positions) / len(rank_positions) if rank_positions else float("inf")

    print("\n" + "#"*100)
    print(f"[STATS] Evaluated {total} queries")
    print(f"[STATS] Recall@{top_k}: {recall_at_k:.4f}")
    print(f"[STATS] MRR@{top_k}: {mrr_at_k:.4f}")
    print(f"[STATS] Avg True Similarity Score: {avg_true_score:.4f}")
    print(f"[STATS] Avg Rank Position of True Doc: {avg_rank_pos:.2f}")
    print(f"[STATS] Rank Distribution (sample): {rank_positions[:50]}")
    print("#"*100 + "\n")

    return {
        "results": results,
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr_at_k,
        "avg_true_score": avg_true_score,
        "avg_rank_pos": avg_rank_pos,
        "rank_positions": rank_positions
    }

# Run evaluation after training
metrics = evaluate(
    query_encoder, document_cobweb,
    queries, targets, sentence_to_leaf,
    device="cpu", top_k=10
)

print(metrics)

# TODO need to do whitening analysis on the embeddings space