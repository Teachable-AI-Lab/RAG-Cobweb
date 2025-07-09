import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import load_dataset

nltk.download('punkt_tab')  # Download sentence tokenizer model

def load_sample_corpuses():
    """Loads and returns the sample corpuses."""
    small_corpus1 = sent_tokenize("""The James Webb Space Telescope, launched in December 2021, has significantly advanced our understanding of the early universe. Its infrared capabilities allow it to peer through cosmic dust, revealing galaxies that formed over 13 billion years ago. Unlike the Hubble Telescope, which operates primarily in visible and ultraviolet light, Webb specializes in the infrared spectrum, providing complementary data that expands our astronomical knowledge.
                                  Intermittent fasting has gained popularity in recent years as a dietary intervention with potential metabolic and cognitive benefits. Studies have suggested that time-restricted eating can improve insulin sensitivity, reduce inflammation, and potentially enhance neuroplasticity. However, the long-term effects of such eating patterns remain under active investigation, especially in diverse populations with different baseline health conditions.
                                  The Treaty of Versailles, signed in 1919, formally ended World War I but sowed the seeds for further global conflict. By imposing harsh reparations on Germany and redrawing borders across Europe and the Middle East, the treaty inadvertently contributed to economic instability and nationalist resentment. Historians often debate whether these conditions directly facilitated the rise of authoritarian regimes in the 1930s.
                                  Cryptocurrencies like Bitcoin and Ethereum operate on decentralized blockchain technology, enabling peer-to-peer transactions without intermediaries. While proponents argue that cryptocurrencies provide financial freedom and privacy, critics cite volatility, energy consumption, and regulatory concerns. The rise of central bank digital currencies (CBDCs) reflects a shift in how governments are responding to these innovations.
                                  Coral bleaching is a phenomenon caused by oceanic temperature rise, often linked to climate change. When water is too warm, corals expel the algae (zooxanthellae) living in their tissues, causing them to turn completely white. While bleaching doesn't immediately kill coral, it leaves them vulnerable to disease and mortality. Global conservation efforts aim to reduce emissions and implement marine protected areas to preserve biodiversity.
                                  James Joyce’s Ulysses is a modernist novel that chronicles a single day—June 16, 1904—in the life of Leopold Bloom. The narrative is celebrated for its stream-of-consciousness style, linguistic experimentation, and intertextual references. Although controversial upon publication, Ulysses is now regarded as a cornerstone of 20th-century literature and a seminal work in the development of the modern novel.""")

    small_corpus2 = sent_tokenize("Photosynthesis is the process by which green plants convert sunlight into chemical energy. Photosynthesis primarily takes place in the chloroplasts of plant cells, using chlorophyll to absorb light. Photosynthesis involves the transformation of carbon dioxide and water into glucose and oxygen. Photosynthesis is essential for life on Earth because it provides oxygen and is the foundation of most food chains. The water cycle describes how water moves through the Earth's atmosphere, surface, and underground. The water cycle consists of processes such as evaporation, condensation, precipitation, and collection. Solar energy drives the water cycle by heating water in oceans and lakes, causing it to evaporate into the air. The water cycle plays a crucial role in regulating climate and supporting all forms of life on Earth. World War II was a global conflict that lasted from 1939 to 1945. World War II involved most of the world’s nations, forming two major opposing military alliances: the Allies and the Axis. Key events of World War II include the invasion of Poland, the Battle of Stalingrad, D-Day, and the dropping of atomic bombs on Hiroshima and Nagasaki. World War II drastically reshaped global politics and led to the formation of the United Nations. The human digestive system is responsible for breaking down food into nutrients the body can use. The human digestive system includes organs such as the mouth, esophagus, stomach, intestines, liver, and pancreas. Enzymes and digestive juices in the digestive system aid in the breakdown of carbohydrates, proteins, and fats. Nutrients are absorbed in the small intestine, while waste is excreted through the large intestine. Renewable energy comes from sources that are naturally replenished, like solar, wind, and hydro power. Renewable energy offers a sustainable alternative to fossil fuels, helping to reduce greenhouse gas emissions. Advances in technology have made renewable energy more efficient and affordable. Transitioning to renewable energy is vital for combating climate change and promoting energy security.")

    small_add_corpus2 = [
        "Photosynthesis is the process by which green plants use sunlight to produce food from carbon dioxide and water.",
        "This process occurs mainly in the chloroplasts of plant cells and releases oxygen as a byproduct.",
        "World War II began in 1939 and involved most of the world’s nations, including the major powers divided into the Allies and the Axis.",
        "The war ended in 1945 after the defeat of Nazi Germany and the surrender of Japan following the atomic bombings of Hiroshima and Nagasaki."
    ]

    user_corpus1 = sent_tokenize(
        "User prefers vegetarian recipes. "
        "User enjoys hiking on weekends. "
        "User works as a freelance graphic designer. "
        "User asks for Indian or Italian cuisine suggestions. "
        "User listens to jazz and lo-fi music. "
        "User frequently reads science fiction novels. "
        "User avoids gluten in meals. "
        "User owns a golden retriever. "
        "User likes visiting art museums. "
        "User practices yoga every morning. "
        "User uses a MacBook for creative work. "
        "User follows a low-carb diet. "
        "User is interested in learning Japanese. "
        "User commutes by bicycle. "
        "User enjoys indie films and documentaries. "
        "User plays the acoustic guitar. "
        "User volunteers at the local animal shelter. "
        "User prefers using eco-friendly products. "
        "User tracks daily habits in a bullet journal. "
        "User often asks for personal finance tips."
    )

    user_corpus2 = [
        "User’s name is Alex Johnson.",
        "User is 29 years old.",
        "User is male.",
        "User lives in Seattle, Washington.",
        "User works as a software engineer.",
        "User is employed at TechNova Inc.",
        "User enjoys hiking, photography, and coding.",
        "User’s favorite programming language is Python.",
        "User holds a B.S. degree in Computer Science.",
        "User graduated in 2018.",
        "User is single.",
        "User speaks English and Spanish.",
        "User has one dog named Max.",
        "User has visited 12 countries.",
        "User uses Python, JavaScript, React, and Docker.",
        "User’s GitHub username is alexj.",
        "User is passionate about technology and innovation.",
        "User often contributes to open-source projects.",
        "User values continuous learning and self-improvement.",
        "User's LOVES to eat pasta.",
        "User is gluten-free.",
        "User’s favorite food is sushi.",
    ]

    return {
        "small_corpus1": small_corpus1,
        "small_corpus2": small_corpus2,
        "small_add_corpus2": small_add_corpus2,
        "user_corpus1": user_corpus1,
        "user_corpus2": user_corpus2,
    }

def generate_and_save_convo_embeddings(corpus = 'user_corpus1', model = 'all-roberta-large-v1', output_path=None):
    st_model = SentenceTransformer(model, trust_remote_code=True)
    corpus = load_sample_corpuses()[corpus] if isinstance(corpus, str) else corpus
    if corpus:
        print("Encoding conversation corpus...")
        convo_embs = st_model.encode(corpus, convert_to_numpy=True, batch_size=100, show_progress_bar=True)
        if output_path:
            np.save(output_path, convo_embs)
            print(f"Saved embeddings to {output_path}")
        return convo_embs
    else:
        print("No corpus loaded for encoding.")

def load_embeddings(filepath):
    """Loads embeddings from a .npy file."""
    try:
        embeddings = np.load(filepath)
        print(f"Loaded embeddings from {filepath} with shape {embeddings.shape}")
        return embeddings
    except FileNotFoundError:
        print(f"Error: Embedding file not found at {filepath}")
        return None
    


def load_sts_embeddings(model, split='train', score_threshold=None, device=None):
    """
    Loads STS dataset, encodes sentences, and returns embeddings and labels.

    Args:
        model: An initialized SentenceTransformer model.
        split (str): Dataset split to load ('train', 'test', 'validation').
        score_threshold (float, optional): Minimum similarity score to include pairs.
        device (str, optional): Device to use for encoding.

    Returns:
        tuple: (embeddings, labels) as numpy arrays.
    """
    # Load STS dataset
    dataset = load_dataset("stsb_multi_mt", name="en", split=split, trust_remote_code=True)
    if isinstance(model, str):
        model = SentenceTransformer(model, trust_remote_code=True)

    embeddings = []
    labels = []

    print(f"Processing STS {split} split...")
    for item in tqdm(dataset):
        s1 = item['sentence1']
        s2 = item['sentence2']
        score = item['similarity_score'] / 5.0  # Normalize to [0, 1]

        # Optional: Only use highly similar pairs
        if score_threshold is not None and score < score_threshold:
            continue

        # Get embeddings
        # Ensure the model is on the correct device if specified
        model.to(device if device else model.device)
        emb1 = model.encode(s1, convert_to_numpy=True, device=device)
        emb2 = model.encode(s2, convert_to_numpy=True, device=device)

        # Use both individually
        embeddings.append(emb1)
        embeddings.append(emb2)

        labels.append(score)
        labels.append(score)  # Both emb1 and emb2 share the score

    if embeddings:
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        return embeddings, labels
    else:
        print(f"No data loaded for STS {split} split.")
        return np.array([]), np.array([])


def combine_and_save_sts_embeddings(output_path="sts_embeddings.npy"):
    st_model = SentenceTransformer('all-roberta-large-v1', trust_remote_code=True)
    sts_train_emb, _ = load_sts_embeddings(st_model, split='train', score_threshold=0.0)
    sts_test_emb, _ = load_sts_embeddings(st_model, split='test', score_threshold=0.0)
    sts_dev_emb, _ = load_sts_embeddings(st_model, split='validation', score_threshold=0.0) # Corrected split name

    if sts_train_emb.size > 0 or sts_test_emb.size > 0 or sts_dev_emb.size > 0:
        all_sts_embeddings = np.concatenate((sts_train_emb, sts_test_emb, sts_dev_emb), axis=0)
        np.save(output_path, all_sts_embeddings)
        print(f"Saved combined STS embeddings to {output_path} with shape {all_sts_embeddings.shape}")
    else:
        print("No STS embeddings loaded to combine.")