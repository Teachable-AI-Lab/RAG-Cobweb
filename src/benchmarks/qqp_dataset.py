from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import numpy as np
import faiss
import time
from random import sample as randsample
from annoy import AnnoyIndex
import hnswlib
# from your_module import CobwebAsADatabase, encode_and_whiten_pcaica # Assuming these are in your module
from functools import partial

# Load QQP dataset and extract duplicates
# dataset = load_dataset("quora", split="train", trust_remote_code=True)
# duplicates = [ex for ex in dataset if ex["is_duplicate"] == 1]

# def run_qqp_benchmark(st_model, whitening_model=None, subset_size=800, target_size=200, top_k=3):
#     """
#     Runs the QQP benchmark comparing different retrieval methods.

#     Args:
#         st_model (SentenceTransformer): The SentenceTransformer model to use.
#         whitening_model (PCAICAWhiteningModel, optional): The whitening model to use for whiteCAAD.
#         subset_size (int): The size of the sampled dataset.
#         target_size (int): The number of queries to use.
#         top_k (int): The number of results to retrieve.
#     """
#     print(f"\n--- Running QQP Benchmark (TOP_K={top_k}) ---")

#     dataset = load_dataset("quora", split="train", trust_remote_code=True)
#     duplicates = [ex for ex in dataset if ex["is_duplicate"] == 1]
#     shuffle(duplicates)

#     sampled = randsample(duplicates, subset_size)
#     queries = [ex["questions"]["text"][0] for ex in sampled[:target_size]]
#     targets = [ex["questions"]["text"][1] for ex in sampled[:target_size]]
#     corpus = [ex["questions"]["text"][1] for ex in sampled]

#     print("Length of Corpus:", len(corpus))

#     # Encode corpus
#     print("Encoding corpus embeddings...")
#     start_time = time.time()
#     corpus_embeddings = st_model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
#     encoding_time = time.time() - start_time
#     print(f"Corpus embedding time: {round(encoding_time, 2)} seconds")

#     # === FAISS Setup ===
#     dim = corpus_embeddings.shape[1]
#     faiss_index = faiss.IndexFlatIP(dim)
#     start_time = time.time()
#     faiss_index.add(corpus_embeddings)
#     faiss_build_time = time.time() - start_time
#     print(f"FAISS index build time: {round(faiss_build_time, 2)} seconds")

#     def retrieve_faiss(query, k):
#         query_emb = st_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
#         _, ids = faiss_index.search(np.expand_dims(query_emb, axis=0), k)
#         return [corpus[i] for i in ids[0]]

#     # === Annoy Setup ===
#     annoy_index = AnnoyIndex(dim, 'angular')
#     for i, emb in enumerate(corpus_embeddings):
#         annoy_index.add_item(i, emb)
#     start_time = time.time()
#     annoy_index.build(10)
#     annoy_build_time = time.time() - start_time
#     print(f"Annoy index build time: {round(annoy_build_time, 2)} seconds")

#     def retrieve_annoy(query, k):
#         query_emb = st_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
#         ids = annoy_index.get_nns_by_vector(query_emb, k)
#         return [corpus[i] for i in ids]

#     # === HNSWLIB Setup ===
#     hnsw_index = hnswlib.Index(space='cosine', dim=dim)
#     hnsw_index.init_index(max_elements=len(corpus), ef_construction=100, M=16)
#     start_time = time.time()
#     hnsw_index.add_items(corpus_embeddings, np.arange(len(corpus)))
#     hnsw_index.set_ef(50)
#     hnsw_build_time = time.time() - start_time
#     print(f"HNSWLIB index build time: {round(hnsw_build_time, 2)} seconds")

#     def retrieve_hnsw(query, k):
#         query_emb = st_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
#         ids, _ = hnsw_index.knn_query(query_emb, k=k)
#         return [corpus[i] for i in ids[0]]

#     # === CAAD Setup (Assumes these classes exist in your env) ===
#     encode_func = partial(st_model.encode, convert_to_numpy=True)
#     start_time = time.time()
#     regCAAD = CobwebAsADatabase(corpus, corpus_embeddings=corpus_embeddings, similarity_type="manhattan", encode_func=encode_func, verbose=False)
#     caad_build_time = time.time() - start_time
#     print(f"CAAD build time: {round(caad_build_time, 2)} seconds")

#     def retrieve_caad(query, k):
#         return regCAAD.cobweb_predict(query, k=k, verbose=False)

#     # === whiteCAAD Setup ===
#     if whitening_model:
#         white_encode_func = partial(encode_and_whiten_pcaica, st_model=st_model, whitening_model=whitening_model)
#         start_time = time.time()
#         whiteCAAD = CobwebAsADatabase(corpus, corpus_embeddings=whitening_model.transform(corpus_embeddings), similarity_type="manhattan", encode_func=white_encode_func, verbose=False)
#         whitecaad_build_time = time.time() - start_time
#         print(f"whiteCAAD build time: {round(whitecaad_build_time, 2)} seconds")

#         def retrieve_whitecaad(query, k):
#             return whiteCAAD.cobweb_predict(query, k=k, verbose=False)
#     else:
#         whitecaad_build_time = None
#         retrieve_whitecaad = None


#     # === Evaluation Function ===
#     def evaluate(retrieve_fn, name):
#         if retrieve_fn is None:
#             return {"method": name, f"recall@{top_k}": "N/A", f"mrr@{top_k}": "N/A", f"ndcg@{top_k}": "N/A", "avg_latency_ms": "N/A"}

#         hits, mrr_total, ndcg_total = 0, 0, 0
#         latencies = []

#         for query, target in tqdm(zip(queries, targets), total=len(queries), desc=name):
#             start = time.time()
#             retrieved = retrieve_fn(query, top_k)
#             latencies.append(time.time() - start)

#             if target in retrieved:
#                 hits += 1
#                 rank = retrieved.index(target) + 1
#                 mrr_total += 1 / rank

#             relevance = [1 if doc == target else 0 for doc in retrieved]
#             # Ensure relevance has at least one relevant item for NDCG calculation if possible
#             if sum(relevance) > 0:
#                  # Use ideal ranking for NDCG, higher scores first
#                 ideal_relevance = sorted(relevance, reverse=True)
#                 # Provide a simple ranking based on position for the actual relevance
#                 actual_ranking = list(range(len(relevance), 0, -1))
#                 ndcg_total += ndcg_score([ideal_relevance], [actual_ranking])


#         n = len(queries)
#         return {
#             "method": name,
#             f"recall@{top_k}": round(hits / n, 4),
#             f"mrr@{top_k}": round(mrr_total / n, 4),
#             f"ndcg@{top_k}": round(ndcg_total / n, 4),
#             "avg_latency_ms": round(1000 * np.mean(latencies), 2) if latencies else "N/A"
#         }

#     # === Run All Evaluations ===
#     faiss_results = evaluate(retrieve_faiss, "FAISS")
#     annoy_results = evaluate(retrieve_annoy, "Annoy")
#     hnsw_results = evaluate(retrieve_hnsw, "HNSWLIB")
#     caad_results = evaluate(retrieve_caad, "CAAD")
#     whitecaad_results = evaluate(retrieve_whitecaad, "whiteCAAD")

#     # === Print Summary ===
#     print(f"\n--- QQP Benchmark Results (TOP_K={top_k}) ---")
#     for res in [faiss_results, annoy_results, hnsw_results, caad_results, whitecaad_results]:
#         print(res)

# # To run this independently:
# # from sentence_transformers import SentenceTransformer
# # from your_module import PCAICAWhiteningModel, load_embeddings, load_sts_embeddings # Assuming your modules
# # st_model = SentenceTransformer('all-roberta-large-v1', trust_remote_code=True)
# # # Load data and train whitening model (assuming these functions exist and work independently)
# # convo_embs = load_embeddings("convo_emb.npy")
# # sts_embeddings, _ = load_sts_embeddings(st_model, split='train', score_threshold=0.0)
# # if convo_embs is not None and sts_embeddings.size > 0:
# #     EMB_DIM = 512 # Or the dimension you used for training
# #     whitening_model = PCAICAWhiteningModel.fit(np.concatenate((sts_embeddings, convo_embs), axis=0), EMB_DIM)
# #     run_qqp_benchmark(st_model, whitening_model=whitening_model, subset_size=800, target_size=200, top_k=3)
# # else:
# #      run_qqp_benchmark(st_model, subset_size=800, target_size=200, top_k=3) # Run without whitening if data not available