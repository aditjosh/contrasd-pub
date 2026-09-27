import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import random, math, time
import numpy as np
from model.datasets import MilvusDataset
from sentence_transformers import InputExample

## Dataset for contrastive learning and calibration
class MilvusSbertContrastiveDataset(Dataset):
    def __init__(self, db_name, human_values_col_name='sbert_wordnet_human_values_sense_embeddings_allminilm',
                non_values_col_name='sbert_wordnet_non_values_sense_embeddings_allminilm',batch_size = 1024):
        self.vec_db_host ='milvus-standalone'
        self.vec_db_port ='19530'
        self.db_name = db_name
        vec_db_uri = "http://" + self.vec_db_host + ':' + self.vec_db_port
        self.mvs_client = MilvusClient(uri=vec_db_uri, token="root:Milvus", db_name=self.db_name)
        self.batch_size = batch_size
        
        # Load the collection
        self.collection_b_name = non_values_col_name
        self.collection_b = self.mvs_client.load_collection(collection_name=self.collection_b_name)
        self.collection_a_name = human_values_col_name
        self.collection_a = self.mvs_client.load_collection(collection_name=self.collection_a_name)
        
        self.full_dataset_a = MilvusDataset(db_name=db_name, collection_name=self.collection_a_name, 
                                output_fields=["id", "sense", "sense_embedding", "is_human_value", "word"], batch_size = self.batch_size)
        self.full_dataset_b = MilvusDataset(db_name=db_name, collection_name=self.collection_b_name, 
                                output_fields=["id", "sense", "sense_embedding", "is_human_value", "word"], batch_size = self.batch_size)
        self.collection_a_ids_val = []
        self.collection_b_ids_val = []
        self.collection_a_ids_calib = []
        self.collection_b_ids_calib = []
        self.collection_a_ids_train = []
        self.collection_b_ids_train = []

        # Separate human values and non values entities for contrastive training
        self.humval_entities = None
        self.nonval_entities = None

        self.train_entities = None
        self.validation_entities = None
        self.calib_entities = None
        self.total_entities = None
        self.pairs_similar_different = []
        self.pairs_labels = None

    def __len__(self):
        return (self.total_entities + self.batch_size - 1) // self.batch_size

    def set_schwartz_entities(self, val):
        self.schwartz_entities = val

    def set_total_entities(self):
        self.total_entities = self.get_total_entities()

    def get_total_entities(self):
        # Count the total number of entities in the collection
        res_a = self.mvs_client.query(
            collection_name=self.collection_a_name,
            filter="id > 0",  # Empty filter to count all entities
            output_fields=["count(*)"]
        )
        res_b = self.mvs_client.query(
            collection_name=self.collection_b_name,
            filter="id > 0",  # Empty filter to count all entities
            output_fields=["count(*)"]
        )
        count_a = res_a[0]["count(*)"] if res_a else 0
        count_b = res_b[0]["count(*)"] if res_b else 0
        return count_a + count_b

    ### -------------------------------- Spliting utils. Each set contains (entity_id, collection_name, "humval"|"nonval") 
    def set_humval_entities(self):
        combined_res = []
        schwartz_items = []
        # Count the total number of entities in the collection
        res = self.mvs_client.query(
            collection_name=self.collection_a_name,
            filter="is_human_value == true",  
            output_fields=["id","word"]
        )
        for entity in res:
            combined_res.append((entity["id"], self.collection_a_name, "humval"))

        res = self.mvs_client.query(
            collection_name=self.collection_b_name,
            filter="is_human_value == true",  
            output_fields=["id"]
        )
        for entity in res:
            combined_res.append((entity["id"], self.collection_b_name, "humval"))
        
        self.humval_entities = combined_res
        self.set_schwartz_entities(schwartz_items)

    def set_nonval_entities(self):        # Count the total number of entities in the collection
        combined_res = []
        res = self.mvs_client.query(
            collection_name=self.collection_a_name,
            filter="is_human_value == false",  # Empty filter to count all entities
            output_fields=["id"]
        )
        for entity in res:
            combined_res.append((entity["id"], self.collection_a_name, "nonval"))
        res = self.mvs_client.query(
            collection_name=self.collection_b_name,
            filter="is_human_value == false",  # Empty filter to count all entities
            output_fields=["id"]
        )
        for entity in res:
            combined_res.append((entity["id"], self.collection_b_name, "nonval"))
            
        self.nonval_entities = combined_res

    def split_train_val_calib_sets(self, train_part=0.7, val_part=0.2):
        all_entities_sets = self.nonval_entities + self.humval_entities
        # Shuffle the list randomly in-place
        random.shuffle(all_entities_sets)
        train_size = int(len(all_entities_sets) * train_part)
        val_size = int(len(all_entities_sets) * val_part)
        calib_size = len(all_entities_sets) - train_size - val_size # Ensure all data is used

        train_set = all_entities_sets[:train_size] 
        val_set = all_entities_sets[train_size : train_size + val_size] 
        calib_set = all_entities_sets[train_size + val_size:]

        # Store the ids for the two collections in order to use the dataset (Subset) later for loading
        for entity in val_set:
            id_c = entity[0]
            collection_name = entity[1]
            if collection_name == self.collection_a_name:
                self.collection_a_ids_val.append(id_c)
            elif collection_name == self.collection_b_name:
                self.collection_b_ids_val.append(id_c)
        for entity in calib_set:
            id_c = entity[0]
            collection_name = entity[1]
            if collection_name == self.collection_a_name:
                self.collection_a_ids_calib.append(id_c)
            elif collection_name == self.collection_b_name:
                self.collection_b_ids_calib.append(id_c)
        for entity in train_set:
            id_c = entity[0]
            collection_name = entity[1]
            if collection_name == self.collection_a_name:
                self.collection_a_ids_train.append(id_c)
            elif collection_name == self.collection_b_name:
                self.collection_b_ids_train.append(id_c)

        # Include Schwartz values in the validation collection
        for entity in self.schwartz_entities:
            item = entity[0]
            collection_name = entity[1]
            if collection_name == self.collection_a_name:
                self.collection_a_ids_val.append(entity[0]['id'])
                val_set.append(item)
        val_size = len(val_set)

        ## Reset the humval and nonval sets to contain only the train set items
        nonval_entities = []
        humval_entities = []
        for entity in train_set:
            if entity[2] == "humval":
                humval_entities.append(entity)
            elif entity[2] == "nonval":
                nonval_entities.append(entity)
        self.nonval_entities = nonval_entities
        self.humval_entities = humval_entities

        # Validation and calibration sets
        self.train_entities = train_set
        self.calib_entities = calib_set
        self.validation_entities = val_set

        assert(len(calib_set) == calib_size)
        assert(len(val_set) == val_size)

        all_entities_size = train_size + val_size + calib_size
        print(f"Total entities: {all_entities_size}")
        print(f"Number of train entities: {train_size}")
        print(f"Number of validation entities: {val_size}")
        print(f"Number of calibration entities: {calib_size}")
        print(f"{all_entities_size} = {train_size} + {val_size} + {calib_size} = {train_size + val_size + calib_size}")

    ## To generate calibration set without using Torch utilities (random_split)
    def generate_calibration_set(self, fraction=0.1):
        all_entities_sets = self.nonval_entities + self.humval_entities
        random.shuffle(all_entities_sets)
        # 10% 
        self.ids_calib_entities = random.sample(all_entities_sets, math.floor(fraction * len(all_entities_sets)))
        ## Repopulate the remaining entities
        nonval_entities = []
        humval_entities = []
        for entity in all_entities_sets:
            for calib_entity in self.ids_calib_entities:
                if entity[0] == calib_entity[0] and entity[1] == calib_entity[1]:
                    continue
                else:
                    if entity[2] == "humval":
                        humval_entities.append(entity)
                    elif entity[2] == "nonval":
                        nonval_entities.append(entity)
        self.nonval_entities = nonval_entities
        self.humval_entities = humval_entities
        
        print(f"Generated calibration set of size: {len(self.ids_calib_entities)}")

    def generate_training_pairs(self, train_part=0.8, val_part=0.1, num_pairs_per_type=2):
        self.split_train_val_calib_sets(train_part=train_part, val_part=val_part)

        # --- Generate Labeled Pairs from the Two Classes ---
        NUM_PAIRS_PER_TYPE = num_pairs_per_type # Total 10,000 pairs (5000 similar, 5000 dissimilar)
        indices_list = []
        labels_list = []
    
        # 1. Generate SIMILAR pairs (Label=True/1, samples from the same class)
        for _ in range(NUM_PAIRS_PER_TYPE):
            # Randomly choose a class to sample from (0 for negative class, 1 for positive class)
            class_idx = random.choice([0, 1])
    
            if class_idx == 0: # Sample from Class 2 (Negative class embeddings)
                idx1, idx2 = random.sample(self.nonval_entities, 2)
                indices_list.append([idx1, idx2])
            else: # Sample from Class 1 (Positive class embeddings)
                idx1, idx2 = random.sample(self.humval_entities, 2)
                indices_list.append([idx1, idx2])
    
            labels_list.append(1) # Similar pair
    
        # 2. Generate DISSIMILAR pairs (Label=False/0, samples from different classes)
        for _ in range(NUM_PAIRS_PER_TYPE):
            # Sample A from positive class and B from negative class
            idx_humval = random.sample(self.humval_entities, 1)
            idx_nonval = random.sample(self.nonval_entities, 1)
    
            indices_list.append([idx_humval[0], idx_nonval[0]])
            labels_list.append(0) # Dissimilar pair
    
        # Final tensors
        self.pairs_similar_different = indices_list
        self.pairs_labels = labels_list
        assert len(self.pairs_similar_different) == len(self.pairs_labels)

        NUM_TOTAL_PAIRS = len(indices_list)
        print(f"Generated {NUM_TOTAL_PAIRS} pairs ({len(self.pairs_labels)} similar) for training.")

    ### ------------------------------- 
    def get_training_pairs(self):
        senses_a = []
        senses_b = []

        for i in range(0, len(self.pairs_similar_different), self.batch_size):
        #for batch in batched(self.pairs_pos_neg_ids, self.batch_size):
            batch_pairs = self.pairs_similar_different[i:i + self.batch_size]
            for pairs in batch_pairs:
                id_a = pairs[0][0]
                collection_a = pairs[0][1]
                expr_value = f"id == {id_a}"
                #print(f"query {collection_a}")
                res = self.mvs_client.query(
                    collection_name=collection_a,
                    filter=expr_value,
                    output_fields=["id", "sense", "is_human_value"]
                )
                res = res[0]
                senses_a.append(res['sense'])
                id_b = pairs[1][0]
                collection_b = pairs[1][1]
                expr_value = f"id == {id_b}"
                #print(f"query {collection_b}")
                res = self.mvs_client.query(
                    collection_name=collection_b,
                    filter=expr_value,
                    output_fields=["id", "sense", "is_human_value"]
                )
                res = res[0]
                senses_b.append(res['sense'])

        labels = self.pairs_labels                
        return senses_a, senses_b, labels

    def get_training_embed_pairs(self):
        senses_a = []
        senses_b = []

        for i in range(0, len(self.pairs_similar_different), self.batch_size):
        #for batch in batched(self.pairs_pos_neg_ids, self.batch_size):
            batch_pairs = self.pairs_similar_different[i:i + self.batch_size]
            for pairs in batch_pairs:
                id_a = pairs[0][0]
                collection_a = pairs[0][1]
                expr_value = f"id == {id_a}"
                #print(f"query {collection_a}")
                res = self.mvs_client.query(
                    collection_name=collection_a,
                    filter=expr_value,
                    output_fields=["id", "sense_embedding", "is_human_value"]
                )
                res = res[0]
                senses_a.append(res['sense_embedding'])

                id_b = pairs[1][0]
                collection_b = pairs[1][1]
                expr_value = f"id == {id_b}"
                #print(f"query {collection_b}")
                res = self.mvs_client.query(
                    collection_name=collection_b,
                    filter=expr_value,
                    output_fields=["id", "sense_embedding", "is_human_value"]
                )
                res = res[0]
                senses_b.append(res['sense_embedding'])

        labels = self.pairs_labels                
        return senses_a, senses_b, labels


    def get_subsets(self, set_name="validation"):
        sub_set_a = None
        sub_set_b = None
        output_fields_names=["id", "word", "sense", "sense_embedding", "is_human_value"]

        if set_name == "validation":
            sub_set_a = MilvusDataset(db_name=self.db_name, collection_name=self.collection_a_name, specific_ids=self.collection_a_ids_val, output_fields=output_fields_names)
            sub_set_b = MilvusDataset(db_name=self.db_name, collection_name=self.collection_b_name, specific_ids=self.collection_b_ids_val, output_fields=output_fields_names)
        elif set_name == "calibration":
            sub_set_a = MilvusDataset(db_name=self.db_name, collection_name=self.collection_a_name, specific_ids=self.collection_a_ids_calib, output_fields=output_fields_names)
            sub_set_b = MilvusDataset(db_name=self.db_name, collection_name=self.collection_b_name, specific_ids=self.collection_b_ids_calib, output_fields=output_fields_names)
        elif set_name == "train":
            sub_set_a = MilvusDataset(db_name=self.db_name, collection_name=self.collection_a_name, specific_ids=self.collection_a_ids_train, output_fields=output_fields_names)
            sub_set_b = MilvusDataset(db_name=self.db_name, collection_name=self.collection_b_name, specific_ids=self.collection_b_ids_train, output_fields=output_fields_names)
        return sub_set_a, sub_set_b

    ## We don't use this as dataloader queries do not work with 2 different collections
    def __getitem__(self, idx):
        pass

## Dataset for senses
class CustomSensesDataset(Dataset):
    def __init__(self, senses1_list, senses2_list, labels):
        self.senses1 = senses1_list
        self.senses2 = senses2_list
        self.labels = labels

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve the sample and label for a given index
        s1 = self.senses1[idx]
        s2 = self.senses2[idx]
        label = self.labels[idx]
        
        # In a real NLP scenario, you would perform tokenization/vectorization here
        # For this example, we'll return the raw strings and label
        sample = {'text1': s1, 'text2': s2, 'label': label}
        return sample

def setup_contrastive_ablation_dataset(db_name='word_sense_embeddings', batch_size=64, 
                            generate_train_pairs = True, num_pairs_per_type=5, train_part=0.8, val_part=0.1,
                            human_values_col_name='sbert_wordnet_human_values_sense_embeddings_allminilm',
                            non_values_col_name='sbert_wordnet_non_values_sense_embeddings_allminilm'):
    dataset = MilvusSbertContrastiveDataset(
        db_name = db_name,
        batch_size = batch_size,
        human_values_col_name=human_values_col_name,
        non_values_col_name=non_values_col_name
    )
    dataset.set_humval_entities()
    dataset.set_nonval_entities()
    dataset.set_total_entities()

    if generate_train_pairs:
        dataset.generate_training_pairs(train_part=train_part, val_part=val_part, num_pairs_per_type=num_pairs_per_type)
    else:
        dataset.split_train_val_calib_sets(train_part=train_part, val_part=val_part)

    return dataset

def setup_ablation_study(num_pairs_per_type=50, batch_size=64, train_part=0.8, val_part=0.1,
                        human_values_col_name='sbert_wordnet_human_values_sense_embeddings_allminilm',
                        non_values_col_name='sbert_wordnet_non_values_sense_embeddings_allminilm'):
    # Generate training data
    if num_pairs_per_type:
        milvus_dataset = setup_contrastive_ablation_dataset(generate_train_pairs = True, num_pairs_per_type=num_pairs_per_type, 
                    train_part=train_part, val_part=val_part,
                    human_values_col_name=human_values_col_name,
                    non_values_col_name=non_values_col_name)  

        ## Embedded senses
        senses_a, senses_b, train_labels = milvus_dataset.get_training_embed_pairs()

        contrastive_train_dataset = CustomSensesDataset(senses_a, senses_b, train_labels)
        contrastive_train_dataloader = DataLoader(contrastive_train_dataset, batch_size=batch_size, shuffle=False)
        milvus_dataset.specific_ids = True
        return contrastive_train_dataloader,train_labels,milvus_dataset
    else:
        milvus_dataset = setup_contrastive_ablation_dataset(generate_train_pairs = False, num_pairs_per_type=num_pairs_per_type, 
                    train_part=train_part, val_part=val_part,
                    human_values_col_name=human_values_col_name,
                    non_values_col_name=non_values_col_name)  
        return milvus_dataset 

def fetch_subset(milvus_dataset, subset_type):
    if subset_type == "train":
        train_dataset_a, train_dataset_b = milvus_dataset.get_subsets(set_name="train") 
        train_dataset_a_dataloader = DataLoader(train_dataset_a, batch_size=None, shuffle=True)
        train_dataset_b_dataloader = DataLoader(train_dataset_b, batch_size=None, shuffle=True)
        return train_dataset_a_dataloader, train_dataset_b_dataloader
    elif subset_type == "validation":
        validation_dataset_a, validation_dataset_b = milvus_dataset.get_subsets(set_name="validation") 
        validation_dataset_a_dataloader = DataLoader(validation_dataset_a, batch_size=None, shuffle=True)   # batch_size is None for dataloader because the dataset queries are already batched 
        validation_dataset_b_dataloader = DataLoader(validation_dataset_b, batch_size=None, shuffle=True)
        return validation_dataset_a_dataloader,validation_dataset_b_dataloader
    elif subset_type == "calibration":
        calib_dataset_a, calib_dataset_b = milvus_dataset.get_subsets(set_name="calibration") 
        calib_dataset_a_dataloader = DataLoader(calib_dataset_a, batch_size=None, shuffle=True)
        calib_dataset_b_dataloader = DataLoader(calib_dataset_b, batch_size=None, shuffle=True)
        return calib_dataset_a_dataloader,calib_dataset_b_dataloader

def extract_from_ablation_dataloader(data_loader):
    labels = []
    words = []
    embeddings_tensor = torch.empty(0)
    dataiter = iter(data_loader)
    batch = next(dataiter)
    
    embeddings_batch = torch.tensor(batch['sense_embeddings'])
    if 'is_human_values' in batch:
        is_human_values = np.array(batch['is_human_values'], dtype=int)
    else:
        is_human_values = np.full(len(batch['ids']), fill_value = -1, dtype = int) 
    labels.extend(is_human_values)
    words.extend(list(zip(batch['ids'],batch['words'])))
    if embeddings_tensor.numel() == 0:
        embeddings_tensor = embeddings_batch
    else:
        embeddings_tensor = torch.vstack((embeddings_tensor, embeddings_batch))
        
    assert len(labels) == embeddings_tensor.shape[0]
    return words, embeddings_tensor, labels

def extract_from_ablation_dataloader_for_finetune(data_loader):
    labels = []
    words = []
    senses = []
    dataiter = iter(data_loader)
    batch = next(dataiter)
    
    if 'is_human_values' in batch:
        is_human_values = np.array(batch['is_human_values'], dtype=int)
    else:
        is_human_values = np.full(len(batch['ids']), fill_value = -1, dtype = int) 
    labels.extend(is_human_values)
    words.extend(list(zip(batch['ids'],batch['words'])))

    if 'senses' in batch:
        senses = np.array(batch['senses'], dtype=object)
    else:
        senses = np.full(len(batch['ids']), fill_value = "", dtype = object) 
        
    assert len(labels) == len(senses)
    return words, senses, labels