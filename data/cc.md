# Project Data 

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

## Overview
This location contains the data files associated with the ContraSD project.