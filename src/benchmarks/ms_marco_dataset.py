from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import numpy as np
import faiss
import time
from random import sample as randsample
# from your_module import CobwebAsADatabase, encode_and_whiten_pcaica # Assuming these are in your module
from functools import partial




# Assuming msmarco_passage_train_embs and msmarco_query_train_embs are loaded
# def train_msmarco_pcaica_whitening_model(msmarco_passage_embs, msmarco_query_embs, emb_dim=256):
#     """
#     Trains the PCAICAWhiteningModel on MS MARCO embeddings.

#     Args:
#         msmarco_passage_embs (np.ndarray): Embeddings from MS MARCO passages.
#         msmarco_query_embs (np.ndarray): Embeddings from MS MARCO queries.
#         emb_dim (int): The desired dimensionality after PCA.

#     Returns:
#         PCAICAWhiteningModel: The trained whitening model.
#     """
#     print(f"Training PCAICA Whitening Model on MS MARCO with EMB_DIM={emb_dim}...")
#     combined_embeddings = np.concatenate((msmarco_passage_embs, msmarco_query_embs), axis=0)
#     whitening_transform_model = PCAICAWhiteningModel.fit(combined_embeddings, emb_dim)
#     print("PCAICA Whitening Model training complete.")
#     return whitening_transform_model

# # To run this independently:
# # from your_module import PCAICAWhiteningModel, load_embeddings # Assuming your modules
# # # Assuming msmarco_passage_embeddings.npy and msmarco_query_embeddings.npy exist
# # msmarco_passage_train_embs = load_embeddings("msmarco_passage_embeddings.npy")
# # msmarco_query_train_embs = load_embeddings("msmarco_query_embeddings.npy")
# # if msmarco_passage_train_embs is not None and msmarco_query_train_embs is not None:
# #     EMB_DIM = 256 # Or the dimension you used for training
# #     whitening_model = train_msmarco_pcaica_whitening_model(msmarco_passage_train_embs, msmarco_query_train_embs, EMB_DIM)

# def run_msmarco_benchmark(st_model, whitening_model=None, sample_size=300, distractor_size=1200, top_k=3):
#     """
#     Runs the MS MARCO benchmark comparing different retrieval methods.

#     Args:
#         st_model (SentenceTransformer): The SentenceTransformer model to use.
#         whitening_model (PCAICAWhiteningModel, optional): The whitening model to use for whiteCAAD.
#         sample_size (int): The number of positive query-passage pairs to sample.
#         distractor_size (int): The number of distractor passages to include in the corpus.
#         top_k (int): The number of results to retrieve.
#     """
#     print(f"\n--- Running MS MARCO Benchmark (TOP_K={top_k}) ---")

#     # Load MS MARCO passage-ranking validation split
#     ds = load_dataset("ms_marco", "v2.1", split="validation")

#     # Build (query, passage, is_selected) triples where passages are <= 60 words
#     all_passages = []
#     for ex in ds:
#         query = ex["query"]
#         passage_texts = ex["passages"]["passage_text"]
#         is_selected_flags = ex["passages"]["is_selected"]

#         for txt, is_sel in zip(passage_texts, is_selected_flags):
#             all_passages.append((query, txt, is_sel))

#     # Remove duplicates (optional, but safe)
#     all_passages = list({(q, t, sel) for q, t, sel in all_passages})

#     # Collect positive pairs and distractor passages
#     positive_pairs = [(q, t) for q, t, sel in all_passages if sel == 1]
#     print(f"Total positive pairs: {len(positive_pairs)}")

#     # Sample positive pairs for benchmarking
#     sampled_pairs = randsample(positive_pairs, sample_size)
#     queries, targets = zip(*sampled_pairs)

#     # Build corpus: unique positive targets + distractors (non-relevant passages)
#     unique_targets = set(targets)
#     distractors = [t for _, t, sel in all_passages if sel == 0 and t not in unique_targets]
#     distractors_sample = randsample(distractors, min(distractor_size, len(distractors)))

#     corpus = list(unique_targets) + distractors_sample
#     print(f"Corpus size: {len(corpus)}")

#     # Encode corpus
#     print("Encoding corpus embeddings...")
#     start = time.time()
#     corpus_embeddings = st_model.encode(corpus, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
#     corpus_time = time.time() - start
#     print(f"Corpus Embedding Time: {corpus_time}")

#     dim = corpus_embeddings.shape[1]

#     # === FAISS Setup ===
#     faiss_index = faiss.IndexFlatIP(dim)
#     start = time.time()
#     faiss_index.add(corpus_embeddings)
#     faiss_build = time.time() - start

#     def retrieve_faiss(q, k):
#         emb = st_model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
#         _, ids = faiss_index.search(np.expand_dims(emb, 0), k)
#         return [corpus[i] for i in ids[0]]

#     # === CAAD setup ===
#     encode_func = partial(st_model.encode, convert_to_numpy=True)
#     start = time.time()
#     regCAAD = CobwebAsADatabase(corpus, corpus_embeddings=corpus_embeddings, similarity_type="manhattan", encode_func=encode_func, verbose=False)
#     caad_build = time.time() - start

#     # === whiteCAAD setup ===
#     if whitening_model:
#         white_encode_func = partial(encode_and_whiten_pcaica, st_model=st_model, whitening_model=whitening_model)
#         start = time.time()
#         whiteCAAD = CobwebAsADatabase(
#             corpus,
#             corpus_embeddings=whitening_model.transform(corpus_embeddings),
#             similarity_type="manhattan",
#             encode_func=white_encode_func,
#             verbose=False
#         )
#         whitecaad_build = time.time() - start

#         def retrieve_whitecaad(q, k):
#             return whiteCAAD.cobweb_predict(q, k=k, verbose=False)
#     else:
#         whitecaad_build = None
#         retrieve_whitecaad = None


#     def evaluate(fn, name, build_time):
#         if fn is None:
#             return {"method": name, f"recall@{top_k}": "N/A", f"mrr@{top_k}": "N/A", f"ndcg@{top_k}": "N/A", "avg_latency_ms": "N/A", "build_time_s": "N/A"}

#         hits = mrr = ndcg = 0
#         latencies = []
#         for q, t in tqdm(zip(queries, targets), total=len(queries), desc=name):
#             st = time.time()
#             res = fn(q, top_k)
#             latencies.append(time.time() - st)
#             if t in res:
#                 hits += 1
#                 rank = res.index(t) + 1
#                 mrr += 1 / rank
#             rel = [1 if doc == t else 0 for doc in res]
#             # Ensure relevance has at least one relevant item for NDCG calculation if possible
#             if sum(rel) > 0:
#                  # Use ideal ranking for NDCG, higher scores first
#                 ideal_relevance = sorted(rel, reverse=True)
#                 # Provide a simple ranking based on position for the actual relevance
#                 actual_ranking = list(range(len(rel), 0, -1))
#                 ndcg += ndcg_score([ideal_relevance], [actual_ranking])

#         n = len(queries)
#         return {
#             "method": name,
#             f"recall@{top_k}": round(hits/n, 4),
#             f"mrr@{top_k}": round(mrr/n, 4),
#             f"ndcg@{top_k}": round(ndcg/n, 4),
#             "avg_latency_ms": round(1000 * np.mean(latencies), 2) if latencies else "N/A",
#             "build_time_s": round(build_time, 2) if build_time is not None else "N/A"
#         }

#     results = [
#         evaluate(retrieve_faiss,   "FAISS",    faiss_build),
#         evaluate(retrieve_caad,    "CAAD",     caad_build),
#         evaluate(retrieve_whitecaad,"whiteCAAD",whitecaad_build),
#     ]

#     print(f"\n--- Retrieval Benchmark (MS MARCO, {len(queries)} queries, K={top_k}) ---")
#     for r in results:
#         print(r)

# # To run this independently:
# # from sentence_transformers import SentenceTransformer
# # from your_module import PCAICAWhiteningModel, load_embeddings # Assuming your modules
# # st_model = SentenceTransformer('all-roberta-large-v1', trust_remote_code=True)
# # # Load data and train whitening model (assuming these functions exist and work independently)
# # # Assuming msmarco_passage_embeddings.npy and msmarco_query_embeddings.npy exist
# # msmarco_passage_train_embs = load_embeddings("msmarco_passage_embeddings.npy")
# # msmarco_query_train_embs = load_embeddings("msmarco_query_embeddings.npy")
# # if msmarco_passage_train_embs is not None and msmarco_query_train_embs is not None:
# #     EMB_DIM = 256 # Or the dimension you used for training
# #     whitening_model = PCAICAWhiteningModel.fit(np.concatenate((msmarco_passage_train_embs, msmarco_query_train_embs), axis=0), EMB_DIM)
# #     run_msmarco_benchmark(st_model, whitening_model=whitening_model, sample_size=300, distractor_size=1200, top_k=3)
# # else:
# #     run_msmarco_benchmark(st_model, sample_size=300, distractor_size=1200, top_k=3) # Run without whitening