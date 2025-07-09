from src.cobweb.CobwebWrapper import CobwebWrapper
import json
import os
from time import time
import numpy as np

def get_embedding_path(model_name: str, dataset: str, split: str):
    os.makedirs("data/embeddings", exist_ok=True)
    model_name = model_name.replace('/', '-')
    return f"data/embeddings/{model_name}_{dataset}_{split}.npy"

def load_cobweb_model(model_name, corpus, corpus_embs, split, mode):
    cobweb_path = f"models/cobweb_wrappers/{model_name}_{split}_{mode}.json"
    if os.path.exists(cobweb_path):
        print(f"Loading Cobweb model from {cobweb_path}")
        with open(cobweb_path, 'r') as f:
            cobweb_json = json.load(f)
        return CobwebWrapper.load_json(cobweb_json, encode_func=lambda x: x)
    else:
        print(f"Computing Cobweb model and saving to {cobweb_path}")
        cobweb = CobwebWrapper(corpus=corpus, corpus_embeddings=corpus_embs, encode_func=lambda x: x)
        cobweb.dump_json(cobweb_path)
        return cobweb
def load_or_compute_embeddings(texts, model_name, dataset, split, compute = False):
    path = get_embedding_path(model_name, dataset, split)
    if os.path.exists(path) and not compute:
        print(f"Loading embeddings from {path}")
        return np.load(path)
    else:
        print(f"Computing embeddings and saving to {path}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        np.save(path, embeddings)
        return embeddings
if __name__ == "__main__":
    model_name = "all-roberta-large-v1"
    split = "train"
    mode = "base"

    corpus = load_or_compute_embeddings(None, model_name=model_name, dataset="qqp_corpus", split=split, compute=False)
    corpus_embs = load_or_compute_embeddings(corpus, model_name=model_name, dataset="qqp_corpus", split=split, compute=False)
    cobweb = load_cobweb_model(model_name, corpus = None, corpus_embs = corpus_embeddings, split = split, mode = mode)
    queries_embs = load_or_compute_embeddings(None, model_name = model_name, dataset = "qqp_queries", split=split, compute=False)

    print(queries_embs[:5])

    s = time()
    old_implementation = [cobweb.cobweb_predict(query, k=5) for query in queries_embs[:10]]
    print(f"Old implementation took {time() - s:.2f} seconds")
    print("Old Implementation Results:")
    print(old_implementation)
    s = time()
    new_implementation = [cobweb.cobweb_predict(query, use_fast=True, retrieve_k=5) for query in queries_embs[0:10]]
    print(f"New implementation took {time() - s:.2f} seconds")
    print("New Implementation Results:")
    print(new_implementation)
