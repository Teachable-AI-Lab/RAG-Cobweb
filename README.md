# Language Cobweb
This project implements and evaluates a Cobweb-based approach for language tasks, specifically focusing on its potential as an incremental, concept-based database for semantic search and retrieval.
The core idea is to use the Cobweb concept formation algorithm, traditionally applied to categorical or continuous data, to build a hierarchical knowledge base of sentence embeddings. This structure allows for novel methods of categorization and retrieval compared to traditional vector stores.

## Whitening Branch - for attempting to process embeddings to work with Cobweb
Initial Plan:
- Create embeddings that function with cobweb through a whitening process (VICReg + Categorical Utility based loss)
- Update Cobweb's prediction to become a matrix-based function for GPU acceleration

## Components
- **Cobweb Implementation (src/cobweb)**: Contains a PyTorch-based implementation of the Cobweb algorithm (CobwebTorchNode.py, CobwebTorchTree.py) adapted for handling continuous vector embeddings, along with a CobwebWrapper.py class that wraps the tree for database-like operations (adding data, querying).
- **Whitening and VAE Models (src/whitening)**: Includes implementations for whitening sentence embeddings using PCA and ICA (pca_ica.py) and a Beta Variational Autoencoder (beta_vae.py) to explore dimensionality reduction and latent space representations for improved clustering.
- **Benchmarking (src/benchmarks)**: Contains scripts and logic (eval.py) to evaluate the performance of the Cobweb-based database against standard vector search methods (FAISS, Annoy, HNSWLIB) on various datasets like Quora Question Pairs and MS-MARCO.
Utilities (src/utils): Houses helper functions (utils.py) for tasks such as encoding and transforming embeddings.
- **Scripts (scripts)**: Placeholder for various scripts to run training, evaluation, and PCA/ICA fitting processes.


## Installation and Usage

Details on how to install the required dependencies and use the package components will be added in future updates.
