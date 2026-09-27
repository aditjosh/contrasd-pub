from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import pandas as pd
import dask.dataframe as dd
import dask.bag as db
import pyarrow as pa
import pyarrow.parquet as pq
import json
import torch
import torch.nn.functional as F
from torch import Tensor
import time, json
import sys, os
from pathlib import Path
from .pretrained_model import PretrainedEmbeddingsGenerator

class ValueSense():
    def __init__(self, model_name, vec_db_name, vec_db_host='milvus-standalone', vec_db_port='19530'):
        self.db_name = vec_db_name
        vec_db_uri = "http://" + vec_db_host + ':' + vec_db_port
        self.mvs_client = MilvusClient(uri=vec_db_uri, token="root:Milvus", db_name=self.db_name)
        self.encoder = PretrainedEmbeddingsGenerator(model_name)

    def search(self, collection_name, query: str, top_k: int = 5):
        """
        Performs a vector search on the Milvus collection.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        #print(f"Searching {collection_name} for query: '{query}'...")
        # Generate embedding for the query
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0].tolist()

        # Perform the search
        res = self.mvs_client.search(
            collection_name=collection_name,
            data=[query_embedding],
            anns_field="sense_embedding",
            output_fields=["id","word","sense","is_human_value"],
            limit=top_k,
            search_params={"metric_type": "COSINE"}
        )
        #print("Search completed.")
        return res

def read_lines_from_file(file_path, lines_to_read=20, start_line_number=1):
    read_lines = []
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, 1):  # start enumeration from 1
                if i >= start_line_number and i < start_line_number + lines_to_read:
                    read_lines.append(line)
                elif i >= start_line_number + lines_to_read:
                    break  # Stop after reading the desired number of lines
        return read_lines
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_similar_senses(in_file_path, topk=4):
    search_sense = ValueSense(model_name='Qwen/Qwen3-Embedding-8B', vec_db_name='word_sense_embeddings', vec_db_host='milvus-standalone')
    vecdb_collection_names = [
        'wordnet_human_values_sense_embeddings_qwen38b', 'wordnet_non_values_sense_embeddings_qwen38b'
    ]
    similar_senses = []
    for line in read_lines_from_file(in_file_path, lines_to_read = 2, start_line_number=1):
        json_ws = json.loads(json.loads(line.rstrip(',\n')))
        sentence = json_ws['sentence']
        for entry in json_ws['terms']:
            term = entry['word'].lower()
            pos = entry['part_of_speech']
            search_query = entry['sense_definition']
        
            all_hits = []
            for collection_name in vecdb_collection_names:
                results = search_sense.search(collection_name, search_query, top_k=topk)
                all_hits.extend(results[0])  # The search returns a list of results per query vector
        
            top_hits = sorted(all_hits, key=lambda x: x['distance'], reverse=True)
            
            for hit in top_hits: 
                #print(f"  - Score: {hit['distance']:.4f}, Word: {hit['entity']['word']}, Text: {hit['entity']['sense']}, Human value?: {hit['entity']['is_human_value']}")
                input_dict = {'term': term, 'sense': search_query, 'similar_term': hit['entity']['word'], 'similar_sense': hit['entity']['sense'], 'similar_term_is_human_value':  hit['entity']['is_human_value'], 'cosine_distance_milvus': f"{hit['distance']:.4f}", 'sentence': sentence, 'part_of_speech': pos }
                similar_senses.append(input_dict)

    return similar_senses

def save_results_to_parquet(json_results, parquet_file_path, senses_file_path=None):
    col_names = ['term', 'sense', 'similar_term', 'similar_sense', 'similar_term_is_human_value', 'cosine_distance_milvus', 'sentence', 'part_of_speech']
    pd_df = pd.DataFrame(json_results)
    new_ddf = dd.from_pandas(pd_df, npartitions=1)

    if senses_file_path:
        senses_list = list(set(new_ddf['sense'].compute().tolist()))
        try:
            with open(senses_file_path, 'a') as file:
                for sense in senses_list:
                    file.write(sense + '\n')
        except FileNotFoundError:
            print(f"Error: The file '{senses_file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Load the existing Parquet file (create a dummy one if it doesn't exist for testing)
    try:
        existing_ddf = dd.read_parquet(parquet_file_path)
    except FileNotFoundError:
        print("Existing Parquet file not found. Creating a new one.")
        existing_ddf = dd.from_pandas(pd.DataFrame(columns=col_names), npartitions=1)

    # Concatenate the existing Dask DataFrame with the new Dask DataFrame
    updated_ddf = dd.concat([existing_ddf, new_ddf], ignore_index=True)

    custom_schema = {
        "term": pa.string(),  
        "sense": pa.string(),   
        'similar_term': pa.string(), 
        'similar_sense': pa.string(), 
        'similar_term_is_human_value': pa.bool_(),
        'cosine_distance_milvus': pa.float32(), 
        'sentence': pa.string(), 
        'part_of_speech': pa.string()
    }

    # Write the combined data to Parquet
    # Convert to Arrow Table for writing
    updated_ddf.to_parquet(parquet_file_path, compression='snappy', schema=custom_schema)

    print(f"Data appended and saved to {parquet_file_path}")

def read_parquet_with_criteria(filepath, filter_condition):
    # Define the filter condition: e.g., 'column_name' equals 'specific_value'
    #my_filter = [('column_name', '==', 'specific_value')]

    # Read the parquet file(s) with the filter applied
    filtered_df = dd.read_parquet(
        filepath,
        filters=filter_condition,
        engine='pyarrow' # 'pyarrow' is often recommended
    )

    # The result is a lazy Dask DataFrame. To get the actual data, call .compute()
    # Note: Computing will load the data into memory, so be mindful of size
    result = filtered_df.compute()
    return result