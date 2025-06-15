from sentence_transformers import SentenceTransformer
from cobweb.CobwebTorchTree import CobwebTorchTree
import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import heapq
from queue import PriorityQueue
import itertools
import functools
from tqdm import tqdm

class CobwebAsADatabase:
    def __init__(self, st_model, corpus=None, corpus_embeddings=None, similarity_type="manhattan", encode_func=None, verbose=False):
        """
        Initializes the CobwebDatabase with an optional corpus.

        Args:
            corpus (list of str): The list of initial sentences.
            similarity_type (str): The distance metric to use ('cosine', 'euclidean', etc.).
            verbose (bool): If True, prints detailed logs.
        """
        self.encode_func = encode_func
        if not encode_func:
            self.encode_func = functools.partial(st_model.encode, convert_to_numpy=True)
        self.similarity_type = similarity_type
        self.verbose = verbose

        self.sentences = []
        self.embeddings = []
        self.sentence_to_node = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_init_search = 50 # CONSTANT!!!

        if not encode_func:
            self.tree = CobwebTorchTree(shape=(1024,), device=self.device) # CHANGE SHAPE DEPENDING ON FINAL VALUES!
        else:
            self.tree = CobwebTorchTree(shape=(EMB_DIM,), device=self.device)

        if corpus:
            self.add_sentences(corpus, corpus_embeddings)

    def add_sentences(self, new_sentences, new_vectors=None):
        """
        Adds new sentences to the database and updates the Cobweb tree.
        """
        if new_vectors is None:
            new_embeddings = self.encode_func(new_sentences)
        else:
            if self.tree.shape[0] != new_vectors.shape[1]:
                new_embeddings = self.encode_func(new_vectors)
            else:
                new_embeddings = new_vectors
        start_index = len(self.sentences)

        for i, (sentence, emb) in tqdm(enumerate(zip(new_sentences, new_embeddings)), total=len(new_sentences), desc="Training CobwebAsADatabase"):
            self.sentences.append(sentence)
            self.embeddings.append(emb)
            leaf = self.tree.ifit(torch.tensor(emb, device=self.device))
            leaf.sentence_id = start_index + i
            self.sentence_to_node[start_index + i] = leaf

        print()

        if self.verbose:
            print(f"Added {len(new_sentences)} sentences.")

    def predict(self, input_sentence, k, sort_by_date=False, verbose=True):
        """
        Predict similar sentences using optimized DFS from Cobweb tree.
        Prioritizes children closest to input embedding.
        Sorts results by node timestamp.
        """
        emb = self.encode_func([input_sentence])
        input_vec = emb[0].reshape(1, -1)
        tensor = torch.tensor(emb[0], device=self.device)
        leaf = self.tree.categorize(tensor, use_best=True, max_nodes=self.max_init_search)

        candidates = []
        visited = set()

        def dfs(node):
            if len(candidates) >= k or node in visited:
                return
            visited.add(node)

            if hasattr(node, "sentence_id"):
                idx = node.sentence_id
                if idx is not None:
                    dist_to_input = pairwise_distances(
                        input_vec,
                        self.embeddings[idx].reshape(1, -1),
                        metric=self.similarity_type
                    )[0][0]
                    candidates.append((dist_to_input, idx, node.timestamp))
                return

            children = getattr(node, "children", [])
            if not children:
                return

            child_vecs = [child.mean.cpu().numpy().reshape(1, -1) for child in children]
            dists = pairwise_distances(input_vec, np.vstack(child_vecs), metric=self.similarity_type)[0]
            sorted_children = [child for _, child in sorted(zip(dists, children), key=lambda x: x[0])]

            for child in sorted_children:
                dfs(child)

        current = leaf
        while current is not None and len(candidates) < k:
            dfs(current)
            current = getattr(current, "parent", None)

        if verbose:
            print("Sentences ranked by order found:")
            [print("- " + s) for s in [self.sentences[idx] for _, idx, _ in sorted(heapq.nsmallest(k, candidates), key=lambda x: x[0])]]

        sort_idx = 2 if sort_by_date else 0

        top_k = sorted(heapq.nsmallest(k, candidates), key=lambda x: x[sort_idx])
        return [self.sentences[idx] for _, idx, _ in top_k]

    def predict_multiple(self, input_sentences, k, sort_by_date=False, verbose=True):
        """
        Predicts k similar documents for each input sentence.
        Avoids re-visiting already explored leaves and uses optimized DFS.
        Sorts the final output by node timestamp.
        """
        final_candidates = []
        visited_leaves = set()

        for input_sentence in input_sentences:
            emb = self.encode_func([input_sentence])
            input_vec = emb[0].reshape(1, -1)
            tensor = torch.tensor(emb[0], device=self.device)
            leaf = self.tree.categorize(tensor, use_best=True, max_nodes=self.max_init_search)

            candidates = []
            visited_nodes = set()

            def dfs_from_node(node, edge_sum):
                if node in visited_nodes or len(candidates) >= k:
                    return
                visited_nodes.add(node)

                if hasattr(node, "sentence_id") and node.sentence_id is not None and node.sentence_id not in visited_leaves:
                    idx = node.sentence_id
                    dist_to_input = pairwise_distances(
                        input_vec,
                        self.embeddings[idx].reshape(1, -1),
                        metric=self.similarity_type
                    )[0][0]
                    candidates.append((dist_to_input, node.timestamp, idx))
                    visited_leaves.add(idx)
                    return

                children = getattr(node, "children", [])
                if not children:
                    return

                parent_vec = node.mean.cpu().numpy().reshape(1, -1)
                child_vecs = [child.mean.cpu().numpy().reshape(1, -1) for child in children]
                edge_dists = pairwise_distances(input_vec, np.vstack(child_vecs), metric=self.similarity_type)[0]

                sorted_children = [child for _, child in sorted(zip(edge_dists, children), key=lambda x: x[0])]

                for child in sorted_children:
                    dfs_from_node(child, edge_sum)

            current = leaf
            while current is not None and len(candidates) < k:
                dfs_from_node(current, 0.0)
                current = getattr(current, "parent", None)

            final_candidates.extend(candidates)

        key1 = 1 if sort_by_date else 0
        key2 = 0 if sort_by_date else 1

        final_sorted = sorted(final_candidates, key=lambda x: (x[key1], x[key2]))
        return [self.sentences[idx] for _, _, idx in final_sorted]

    def print_tree(self):
        """
        Prints the structure of the Cobweb tree.
        """
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"Sentence ID: {getattr(node, 'sentence_id', 'N/A')}"
            print(f"{indent}- Node ID {node.id} {label}")
            if hasattr(node, "sentence_id"):
                idx = node.sentence_id
                print(f"{indent}    \"{self.sentences[idx]}\"")
            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nSentence clustering tree:")
        _print_node(self.tree.root)
