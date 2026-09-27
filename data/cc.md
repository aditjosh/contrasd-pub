# Datasets
The datasets curated in this project are described below. 

## Blockchain Articles
Using web crawl, text was extracted from articles publicly hosted at these sites: 
- aave.com (AAVE protocol blog)
- blog.ethereum.org (Ethereum blogs), blog.ethswarm.org (Ethereum Swarm), eips.ethereum.org (Ethereum Improvement Proposals)
- ethereum-magicians.org (Ethereum magicians blog), ethresear.ch (Ethereum research)
- filecoin.io (Filecoin)
- hedera.com (Hedera blockchain)
- hyperledger-fabric.readthedocs.io (Hyperledger Fabric), lfdecentralizedtrust.org/blog (Hyperledger)
- vitalik.eth.limo (Vitalik Buterin's blog)

The extracted text files can be found in the [blockchain subdirectory](blockchain/blogs_raw/)

## Dictionary of Human Values and Non-Values
Natural language relies heavily on context to determine word meaning, particularly regarding whether a term conveys a specific human value within a given sentence. To formalize these distinctions, terms are paired with standardized *sense definitions* that delineate their human value and non-value senses.

A dictionary is constructed by mapping these senses using a multi-step pipeline: First, a vocabulary $$V$$ of human values-centric terms was shortlisted from the top 50 search engine results across two query platforms (totaling 100 source pages). Second, two benchmarking corpora were compiled of clean English sentences - a general-domain corpus ($$T_{CC}$$) sampled from the Common Crawl's C4 dataset, and a domain-specific corpus ($$T_{BC}$$) extracted from a curated collection of blockchain literature.

For each term in the vocabulary $$V$$, a set of sentences was manually selected from the corpora $$T_{BC}$$ and $$T_{CC}$$. Then, three different LLM agents – Qwen3, Llama 3.2, and Gemma3 – were prompted to identify whether each term conveys human values in those sentences, and associate the terms with the relevant *sense definitions*. Following a RAG approach, the LLM agents are provided with a Python utility script to query sense definitions from WordNet. If no definition is found in WordNet, the LLM is asked to generate an appropriate sense definition.

The embeddings for the human values and non-values senses of the vocabulary $$V$$ are in the parquet files under [experiments](data/experiments).


## Directory Map
```
/
├── data/                                                              # Shared project data
|   ├── blockchain
│       ├── blogs_raw                                                  # Downloaded blockchain articles
│       └── sense_definitions                                          # Sense definitions assigned using LLM
|   ├── experiments    
|       ├── wordnet_human_values_sense_embeddings_qwen38b.parquet      # Qwen embeddings for dataset human values
|       ├── wordnet_non_values_sense_embeddings_qwen38b.parquet        # Qwen embeddings for dataset non values
|       ├── sbert_wordnet_human_values_allminilm.parquet               # SBERT (all-MiniLM-L6-v2) embeddings for human values 
|       ├── sbert_wordnet_non_values_allminilm.parquet                 # SBERT (all-MiniLM-L6-v2) embeddings for non values
|       ├── ethics_virtues_train_dataset_embeddings_allminilm.parquet  # SBERT (all-MiniLM-L6-v2) embeddings for ETHICS train data  
|       ├── ethics_virtues_test_dataset_embeddings_allminilm.parquet   # SBERT (all-MiniLM-L6-v2) embeddings for ETHICS test data
|       ├── ethics_virtues_train_dataset_embeddings_qwen38b.parquet    # Qwen embeddings for ETHICS train data
|       ├── ethics_virtues_test_dataset_embeddings_qwen38b.parquet     # Qwen embeddings for ETHICS test data

```

