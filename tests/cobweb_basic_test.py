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


def run_cobweb_prediction_example(cobweb_db, test_query, k=4, use_fast=True):
    """
    Runs a prediction example using the CobwebWrapper.

    Args:
        cobweb_db (CobwebWrapper): An initialized CobwebWrapper instance.
        test_query (str): The query sentence.
        k (int): The number of results to retrieve.
    """
    print(f"\nRunning Cobweb prediction example for query: '{test_query}'")
    if use_fast:
        results = cobweb_db.cobweb_predict_fast(test_query, k=k)
    else:
        results = cobweb_db.cobweb_predict(test_query, k=k)
    print("\nRetrieved sentences:")
    for res in results:
        print(f"- {res}")

def test_save_load(cobweb_db, save_path, query=None):
    """
    Tests saving the CobwebWrapper to a JSON file.

    Args:
        cobweb_db (CobwebWrapper): An initialized CobwebWrapper instance.
        save_path (str): The path to save the JSON file.
    """
    print(f"\nSaving CobwebWrapper to {save_path}")
    json_string = cobweb_db.dump_json(save_path=save_path)
    print("CobwebWrapper saved successfully.")
    cobweb_db_loaded = CobwebWrapper.load_json(json_string, encode_func=cobweb_db.encode_func)
    print("CobwebWrapper loaded successfully from JSON.")
    if query:
        print(f"\nRunning prediction on loaded CobwebWrapper for query: '{query}'")
        results = cobweb_db_loaded.cobweb_predict(query, k=4)
        print("\nRetrieved sentences from loaded CobwebWrapper:")

        for res in results:
            print(f"- {res}")
    return json_string


# To run this example independently:
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from src.utils.datasets import load_sample_corpuses
    from functools import partial
    st_model = SentenceTransformer('all-roberta-large-v1', trust_remote_code=True)
    encode_func = partial(st_model.encode, convert_to_numpy=True)
    # Assuming load_sample_corpuses is defined and accessible
    sample_corpuses = load_sample_corpuses()
    test_corpus = sample_corpuses["user_corpus2"]
    new_db = run_cobweb_database_example(test_corpus, encode_func)
    run_cobweb_prediction_example(new_db, test_query, k=4, use_fast=False)
    run_cobweb_prediction_example(new_db, test_query, k=4, use_fast=True)

    print("\nTesting save and load functionality...")
    save_path = "outputs/tests/cobweb_test.json"
    json_string = test_save_load(new_db, save_path, query=test_query)
    print("\nTesting completed successfully.")