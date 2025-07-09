import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# sentences = test_corpus # Assuming test_corpus is defined elsewhere

def run_faiss_example(sentences, test_query, top_k=10):
    """
    Runs a FAISS example for comparison.

    Args:
        sentences (list of str): The corpus to index.
        test_query (str): The query sentence.
        top_k (int): The number of results to retrieve.
    """
    print("\nRunning FAISS Example...")
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True)

    # 3. Embed the sentences
    embeddings = st_model.encode(sentences, convert_to_numpy=True)

    # 4. Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 5. Function to query similar sentences
    def search(query, k):
        query_embedding = st_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k)
        print(f"\nQuery: {query}\nTop {k} similar sentences (FAISS):")
        for i, idx in enumerate(indices[0]):
            print(f"{i+1}: {sentences[idx]} (distance: {distances[0][i]:.4f})")

    # Example usage
    search(test_query, top_k)

if __name__ == "__main__":
    from src.utils.datasets import load_sample_corpuses
    sample_corpuses = load_sample_corpuses()
    test_corpus = sample_corpuses["user_corpus2"]
    test_query = "What should the user have for dinner?"
    run_faiss_example(test_corpus, test_query, top_k=4)