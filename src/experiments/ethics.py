import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import random, math, time
import numpy as np

class EthicsDataset(Dataset):
    def __init__(self, db_name, batch_size = 1024, output_fields=["id", "embedding"], specific_ids=None): 
        super(EthicsDataset, self).__init__()
        self.vec_db_host ='milvus-standalone'
        self.vec_db_port ='19530'
        self.db_name = db_name
        vec_db_uri = "http://" + self.vec_db_host + ':' + self.vec_db_port
        self.mvs_client = MilvusClient(uri=vec_db_uri, token="root:Milvus", db_name=self.db_name)
        self.batch_size = batch_size
        self.output_fields = output_fields
        self.specific_ids = specific_ids

    def get_items(self, collection, num_items=1):
        res = self.mvs_client.query(
            collection_name = collection,
            filter = "id>0",
            output_fields = ["label", "input", "embedding"],
            limit=num_items
        )
        return res

def run_main():
    collection_name = "ethics_test_dataset_embeddings_qwen38b"
    ds = EthicsDataset(db_name='word_sense_embeddings',batch_size=64)
    res = ds.get_items(collection=collection_name, ids=[462649061268802165,462649061268802166,462649061268802167])





