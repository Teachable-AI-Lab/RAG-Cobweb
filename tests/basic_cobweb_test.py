from src.cobweb.CobwebWrapper import CobwebWrapper


test_query = "What should the user have for dinner?"

def run_cobweb_database_example(test_corpus, encode_func):
    """
    Runs a minimal working example of the CobwebWrapper.

    Args:
        test_corpus (list of str): The corpus to use for the database.
        encode_func (callable): The function to use for encoding sentences.
    """
    print("Running CobwebWrapper Example...")
    new_db = CobwebWrapper(test_corpus, encode_func=encode_func)
    new_db.print_tree()
    return new_db


def run_cobweb_prediction_example(cobweb_db, test_query, k=4):
    """
    Runs a prediction example using the CobwebWrapper.

    Args:
        cobweb_db (CobwebWrapper): An initialized CobwebWrapper instance.
        test_query (str): The query sentence.
        k (int): The number of results to retrieve.
    """
    print(f"\nRunning Cobweb prediction example for query: '{test_query}'")
    results = cobweb_db.cobweb_predict(test_query, k=k, verbose=True)
    print("\nRetrieved sentences:")
    for res in results:
        print(f"- {res}")

# To run this example independently:
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from src.utils.datasets import load_sample_corpuses
    from functools import partial
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True)
    encode_func = partial(st_model.encode, convert_to_numpy=True)
    # Assuming load_sample_corpuses is defined and accessible
    sample_corpuses = load_sample_corpuses()
    test_corpus = sample_corpuses["user_corpus2"]
    new_db = run_cobweb_database_example(test_corpus, encode_func)
    run_cobweb_prediction_example(new_db, test_query, k=4)