import pandas as pd
from pymilvus import MilvusClient, DataType
from pymilvus import connections, utility
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, db
from pymilvus.bulk_writer import get_import_progress
from pymilvus.bulk_writer import bulk_import
import time, os

HOSTNAME = "localhost"
MILVUS_PORT = "19530"
MILVUS_URI = "http://" +  HOSTNAME + ":" + MILVUS_PORT

def create_database(milvus_client, db_name):
    existing_databases = milvus_client.list_databases()
    if db_name in existing_databases:
        print(f"Database '{db_name}' already exists.")
        milvus_client.using_database(db_name)
    else:
        # Create the new database
        try:
            milvus_client.create_database(db_name=db_name)
            print(f"Database '{db_name}' created successfully.")
        except Exception as e:
            print(f"Could not create database, it might already exist or an error occurred: {e}")
        # Use the newly created database
        milvus_client.using_database(db_name)

def import_collection(db_host, db_port, db_name, collection_name, fields, column_order, input_parquet_file_path):
    df = pd.read_parquet(input_parquet_file_path)
    print(df.columns)
    df = df[column_order]
    print(f"Loaded {len(df)} rows from {input_parquet_file_path}")

    # Define Milvus Collection Schema (Must match data)
    if not connections.has_connection(alias='conn'):
        print("Did not find existing connection.")
        connections.connect(alias='conn', db_name=db_name, host=db_host, port=db_port)
    db.using_database(db_name, using='conn')

    if utility.has_collection(collection_name, using='conn'):
        print("Found existing collection.")
        collection = Collection(collection_name, using='conn')
    else:
        schema = CollectionSchema(fields, description="Parquet import example")
        collection = Collection(name=collection_name, schema=schema, using='conn')

    # Prepare Data for Bulk Insert
    files = [os.path.abspath(input_parquet_file_path)]

    collections_list = utility.list_collections(using='conn')
    print(f"Found collections: {collections_list}")

    # Import Data
    insert_result = collection.insert(data=df)
    collection.flush() 

    print(f'Insert state: {insert_result}')
    print("Import initiated successfully.")


db_name = "test_db"
PARQUET_FILE_LOC = "../../data/experiments"

# Create database
milvus_client = MilvusClient(uri=MILVUS_URI)
create_database(milvus_client, db_name)

# Import files 
DIMENSION = 384
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="label", dtype=DataType.INT64),
    FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION), # Adjust dim
]
collection_name = "ethics_virtues_test_dataset_embeddings_allminilm"
file_name = collection_name + ".parquet"
file_path = PARQUET_FILE_LOC + "/" + file_name
col_order = ["id","label","input","embedding"]
import_collection(HOSTNAME, MILVUS_PORT, db_name, collection_name, fields, col_order, file_path)

DIMENSION = 384
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="label", dtype=DataType.INT64),
    FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION), # Adjust dim
]
collection_name = "ethics_virtues_train_dataset_embeddings_allminilm"
file_name = collection_name + ".parquet"
file_path = PARQUET_FILE_LOC + "/" + file_name
col_order = ["id","label","input","embedding"]
import_collection(HOSTNAME, MILVUS_PORT, db_name, collection_name, fields, col_order,file_path)

DIMENSION = 4096
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="label", dtype=DataType.INT64),
    FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION), # Adjust dim
]
collection_name = "ethics_virtues_train_dataset_embeddings_qwen38b"
file_name = collection_name + ".parquet"
file_path = PARQUET_FILE_LOC + "/" + file_name
col_order = ["id","label","input","embedding"]
import_collection(HOSTNAME, MILVUS_PORT, db_name, collection_name, fields, col_order,file_path)

DIMENSION = 4096
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="label", dtype=DataType.INT64),
    FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION), # Adjust dim
]
collection_name = "ethics_virtues_test_dataset_embeddings_qwen38b"
file_name = collection_name + ".parquet"
file_path = PARQUET_FILE_LOC + "/" + file_name
col_order = ["id","label","input","embedding"]
import_collection(HOSTNAME, MILVUS_PORT, db_name, collection_name, fields, col_order,file_path)

DIMENSION = 384
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sense", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sense_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION), # Adjust dim
    FieldSchema(name="is_human_value", dtype=DataType.BOOL)
]
collection_name = "sbert_wordnet_human_values_allminilm"
file_name = collection_name + ".parquet"
file_path = PARQUET_FILE_LOC + "/" + file_name
col_order = ["id","word","sense","sense_embedding","is_human_value"]
import_collection(HOSTNAME, MILVUS_PORT, db_name, collection_name, fields, col_order,file_path)

DIMENSION = 384
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sense", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sense_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION), # Adjust dim
    FieldSchema(name="is_human_value", dtype=DataType.BOOL)
]
collection_name = "sbert_wordnet_non_values_allminilm"
file_name = collection_name + ".parquet"
file_path = PARQUET_FILE_LOC + "/" + file_name
col_order = ["id","word","sense","sense_embedding","is_human_value"]
import_collection(HOSTNAME, MILVUS_PORT, db_name, collection_name, fields, col_order,file_path)

# Human values
DIMENSION = 4096
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sense", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sense_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION), # Adjust dim
    FieldSchema(name="is_human_value", dtype=DataType.BOOL)
]
collection_name = "wordnet_human_values_sense_embeddings_qwen38b"
file_name = collection_name + ".parquet"
file_path = PARQUET_FILE_LOC + "/" + file_name
col_order = ["id","word","sense","sense_embedding","is_human_value"]
import_collection(HOSTNAME, MILVUS_PORT, db_name, collection_name, fields, col_order,file_path)

DIMENSION = 4096
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sense", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="sense_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION), # Adjust dim
    FieldSchema(name="is_human_value", dtype=DataType.BOOL)
]
collection_name = "wordnet_non_values_sense_embeddings_qwen38b"
file_name = collection_name + ".parquet"
file_path = PARQUET_FILE_LOC + "/" + file_name
col_order = ["id","word","sense","sense_embedding","is_human_value"]
import_collection(HOSTNAME, MILVUS_PORT, db_name, collection_name, fields, col_order,file_path)






