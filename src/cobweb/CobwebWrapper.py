import torch
import json
import math
from src.cobweb.CobwebTorchTree import CobwebTorchTree
from tqdm import tqdm

class CobwebWrapper:
    def __init__(self, corpus=None, corpus_embeddings=None, encode_func=lambda x: x):
        """
        Initializes the CobwebWrapper with optional sentences and/or embeddings.
        """

        self.encode_func = encode_func

        self.sentences = []
        self.sentence_to_node = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_init_search = 100000

        # Determine embedding shape
        if corpus_embeddings is not None:
            corpus_embeddings = torch.tensor(corpus_embeddings) if isinstance(corpus_embeddings, list) else corpus_embeddings
            embedding_shape = corpus_embeddings.shape[1:]
        elif corpus and len(corpus) > 0:
            sample_emb = self.encode_func([corpus[0]])
            embedding_shape = sample_emb.shape[1:]


        self.tree = CobwebTorchTree(shape=embedding_shape, device=self.device)

        if corpus_embeddings is not None:
            if corpus is None:
                corpus = [None] * len(corpus_embeddings)
            self.add_sentences(corpus, corpus_embeddings)
        elif corpus is not None and len(corpus) > 0:
            self.add_sentences(corpus)

    def add_sentences(self, new_sentences, new_vectors=None):
        """
        Adds new sentences and/or embeddings to the Cobweb tree.
        If a sentence is None, it is treated as an embedding-only entry.
        """
        if new_vectors is None:
            new_embeddings = self.encode_func(new_sentences)
        else:
            new_embeddings = new_vectors
            if isinstance(new_embeddings, list):
                new_embeddings = torch.tensor(new_embeddings)
            if new_embeddings.shape[1] != self.tree.shape[0]:
                print(f"[Warning] Provided vector dim {new_embeddings.shape[1]} != tree dim {self.tree.shape[0]}, re-encoding...")
                new_embeddings = self.encode_func(new_sentences)

        start_index = len(self.sentences)

        for i, (sent, emb) in tqdm(enumerate(zip(new_sentences, new_embeddings)),
                                   total=len(new_sentences),
                                   desc="Training CobwebTree"):
            self.sentences.append(sent)
            leaf = self.tree.ifit(torch.tensor(emb, device=self.device))
            leaf.sentence_id = start_index + i
            self.sentence_to_node[start_index + i] = leaf


    def cobweb_predict_fast(self, input, k=5, return_ids=False, is_embedding=False):
        """
        Ultra-fast prediction using sentence_to_node mapping directly.
        Bypasses tree traversal entirely by using the leaf node mapping.
        """
        if is_embedding:
            emb = input
        else:
            emb = self.encode_func([input])[0]

        x = torch.tensor(emb, device=self.device).unsqueeze(0)  # (1, D)
        
        # Get all leaf nodes directly from sentence_to_node
        leaf_nodes = list(self.sentence_to_node.values())
        
        if not leaf_nodes:
            return []
        
        # Pre-allocate tensors for vectorized computation
        num_leaves = len(leaf_nodes)
        leaf_means = torch.zeros(num_leaves, self.tree.shape[0], device=self.device)
        leaf_vars = torch.zeros(num_leaves, self.tree.shape[0], device=self.device)
        
        # Fill tensors efficiently
        for i, node in enumerate(leaf_nodes):
            leaf_means[i] = node.mean
            leaf_vars[i] = torch.clamp(node.meanSq / node.count, min=1e-8)
        
        # Vectorized log probability computation
        log_2pi = math.log(2 * math.pi)
        diff_sq = (x - leaf_means) ** 2  # (N, D)
        
        log_probs = -0.5 * (
            log_2pi * self.tree.shape[0] +
            torch.log(leaf_vars).sum(dim=1) +
            (diff_sq / leaf_vars).sum(dim=1)
        )
        
        # Get top-k results
        if k >= num_leaves:
            sorted_indices = torch.argsort(log_probs, descending=True)
            selected_nodes = [leaf_nodes[idx] for idx in sorted_indices]
        else:
            _, topk_indices = torch.topk(log_probs, k, largest=True)
            selected_nodes = [leaf_nodes[idx] for idx in topk_indices]
        
        # Convert to sentence IDs or sentences
        results = []
        for node in selected_nodes:
            sid = getattr(node, 'sentence_id', None)
            if sid is None or sid >= len(self.sentences):
                continue
            results.append(sid if return_ids else self.sentences[sid])
        
        return results

    def cobweb_predict(self, input, k=5, return_ids=False, is_embedding=False):
        """
        Predict top-k similar entries from the tree.
        
        Args:
            input (str or tensor): Sentence or embedding vector
            k (int): Number of similar items to return
            return_ids (bool): If True, return sentence IDs. Else, return sentences.
            is_embedding (bool): Set True if `input` is already an embedding vector.
        """
        if is_embedding:
            emb = input
        else:
            emb = self.encode_func([input])[0]

        tensor = torch.tensor(emb, device=self.device)
        leaves = self.tree.categorize(tensor, use_best=True, max_nodes=self.max_init_search, retrieve_k=k)

        results = []
        for leaf in leaves:
            sid = getattr(leaf, 'sentence_id', None)
            if sid is None or sid >= len(self.sentences):
                continue
            results.append(sid if return_ids else self.sentences[sid])
        return results

    def print_tree(self):
        """
        Recursively prints the tree structure.
        """
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"Sentence ID: {getattr(node, 'sentence_id', 'N/A')}"
            print(f"{indent}- Node ID {node.id} {label}")
            sid = getattr(node, "sentence_id", None)
            if sid is not None and sid < len(self.sentences):
                sentence = self.sentences[sid]
                if sentence is not None:
                    print(f"{indent}    \"{sentence}\"")
                else:
                    print(f"{indent}    [Embedding only]")
            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nCobweb Sentence Clustering Tree:")
        _print_node(self.tree.root)

    def dump_json(self, save_path=None):
        """
        Serializes the CobwebWrapper into a JSON string.
        Only serializes essential data: tree, sentences, and sentence-to-node mapping.
        """
        wrapper_state = {
            "tree": json.loads(self.tree.dump_json()),
            "sentences": self.sentences,
            "embedding_dim": self.tree.shape[0] if hasattr(self.tree, 'shape') else None
        }
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(wrapper_state, f, indent=2)
        return json.dumps(wrapper_state, indent=2)


    @staticmethod
    def load_json(json_data, encode_func=lambda x: x):
        """
        Loads a CobwebWrapper from a JSON string or dict.

        Args:
            json_data (str or dict): The saved wrapper state.
            encode_func (callable): The encoding function to be used.

        Returns:
            CobwebWrapper instance.
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        # Load tree
        tree = CobwebTorchTree(1024)
        print("Loading tree from JSON...")
        tree.load_json(json.dumps(data["tree"]))

        # Initialize wrapper with minimal setup
        wrapper = CobwebWrapper.__new__(CobwebWrapper)
        wrapper.tree = tree
        wrapper.encode_func = encode_func

        # Restore attributes
        wrapper.sentences = data.get("sentences", [])
        wrapper.device = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        wrapper.max_init_search = data.get("max_init_search", 100000)

        # Reconstruct sentence_to_node mapping
        sentence_to_node = {}
        def _index_nodes(node, st):
            if hasattr(node, 'sentence_id') and node.sentence_id is not None:
                sentence_to_node[node.sentence_id] = node
            for child in getattr(node, "children", []):
                _index_nodes(child, st)

        _index_nodes(wrapper.tree.root, sentence_to_node)
        wrapper.sentence_to_node = sentence_to_node
        wrapper.print_tree()

        return wrapper

    def __len__(self):
        """
        Returns the number of sentences in the Cobweb tree.
        """
        return len(self.sentences)