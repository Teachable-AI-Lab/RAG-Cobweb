import torch
from src.cobweb.CobwebTorchTree import CobwebTorchTree
from tqdm import tqdm

class CobwebWrapper:
    def __init__(self, corpus=None, corpus_embeddings=None, encode_func=None):
        """
        Initializes the CobwebDatabase with an optional corpus.

        Args:
            corpus (list of str): The list of initial sentences.
            similarity_type (str): The distance metric to use ('cosine', 'euclidean', etc.).
            encode_func (callable): A function that takes a list of sentences and returns their embeddings.
        """
        if encode_func is None:
             raise ValueError("encode_func must be provided during initialization.")
        self.encode_func = encode_func

        self.sentences = []
        self.embeddings = []
        self.sentence_to_node = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_init_search = 100000 # Can set this to a really high number because now the CobwebTorchTree has a retrieve_k argument!

        # Determine shape from first embedding or assume a default if corpus is empty
        if corpus_embeddings is not None and len(corpus_embeddings) > 0:
            embedding_shape = corpus_embeddings.shape[1:]
        elif corpus is not None and len(corpus) > 0:
            # Encode a sample to determine shape if corpus is provided but not embeddings
            sample_embedding = self.encode_func([corpus[0]])
            embedding_shape = sample_embedding.shape[1:]
        else:
            # Default shape if no initial data is provided - might need adjustment
            embedding_shape = (768,) # A common embedding dimension, adjust if needed


        self.tree = CobwebTorchTree(shape=embedding_shape, prior_var=1e2, device=self.device)

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
                # If provided vectors have different shape, encode sentences instead
                print(f"Warning: Provided vector shape {new_vectors.shape[1]} does not match tree shape {self.tree.shape[0]}. Encoding sentences instead.")
                new_embeddings = self.encode_func(new_sentences)
            else:
                new_embeddings = new_vectors

        start_index = len(self.sentences)

        for i, (sentence, emb) in tqdm(enumerate(zip(new_sentences, new_embeddings)), total=len(new_sentences), desc="Training CobwebAsADatabase"):
            self.sentences.append(sentence)
            self.embeddings.append(emb)
            leaf = self.tree.ifit(torch.tensor(emb, device=self.device))
            leaf.sentence_id = start_index + i
            self.sentence_to_node[start_index + i] = leaf


    def cobweb_predict(self, input_sentence, k, sort_by_date=False):
        """
        Predict similar sentences using Cobweb categorization hierarchy rather
        than BFS after finding an initial node to leapfrog from.
        """
        emb = self.encode_func([input_sentence])
        input_vec = emb[0].reshape(1, -1)
        tensor = torch.tensor(emb[0], device=self.device)
        leaves = self.tree.categorize(tensor, use_best=True, max_nodes=self.max_init_search, retrieve_k=k)

        if sort_by_date:
            # Assuming leaves is a list of nodes
            return [self.sentences[leaf.sentence_id] for leaf in sorted(leaves, key=lambda x: x.timestamp)]

        return [self.sentences[leaf.sentence_id] for leaf in leaves]

    def print_tree(self):
        """
        Prints the structure of the Cobweb tree.
        """
        def _print_node(node, depth=0):
            indent = "  " * depth
            label = f"Sentence ID: {getattr(node, 'sentence_id', 'N/A')}"
            print(f"{indent}- Node ID {node.id} {label}")
            if hasattr(node, "sentence_id") and node.sentence_id is not None and node.sentence_id < len(self.sentences):
                idx = node.sentence_id
                print(f"{indent}    \"{self.sentences[idx]}\"")
            for child in getattr(node, "children", []):
                _print_node(child, depth + 1)

        print("\nSentence clustering tree:")
        _print_node(self.tree.root)