import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import random, math, time
import numpy as np

## Dataset for vector db
class MilvusDataset(Dataset):
    def __init__(self, db_name, collection_name, batch_size = 1024, output_fields=["id", "sense_embedding", "is_human_value"], 
        items2exclude=None, specific_ids=None):
        self.vec_db_host ='milvus-standalone'
        self.vec_db_port ='19530'
        self.db_name = db_name
        vec_db_uri = "http://" + self.vec_db_host + ':' + self.vec_db_port
        self.mvs_client = MilvusClient(uri=vec_db_uri, token="root:Milvus", db_name=self.db_name)
        self.batch_size = batch_size
        self.output_fields = output_fields
        self.items2exclude = items2exclude
        self.specific_ids = specific_ids
        
        # Load the collection
        self.collection_name = collection_name
        self.collection = self.mvs_client.load_collection(collection_name=collection_name)
        self.total_entities = self.get_total_entities()

    def get_total_entities(self):
        # Count the total number of entities in the collection
        res = self.mvs_client.query(
            collection_name=self.collection_name,
            filter="id > 0",  # Empty filter to count all entities
            output_fields=["count(*)"]
        )
        return res[0]["count(*)"] if res else 0

    def __len__(self):
        return (self.total_entities + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        #offset = idx * self.batch_size
        # Retrieve entities in batches
        if self.specific_ids is not None:
            filter_expression = f"id in {self.specific_ids}" 
            res = self.mvs_client.query(
                collection_name=self.collection_name,
                filter=filter_expression,
                output_fields=self.output_fields, # Replace with your actual field names
            )
        else:
            filter_expression = ""
            res = self.mvs_client.query(
                collection_name=self.collection_name,
                filter=filter_expression,
                output_fields=self.output_fields, # Replace with your actual field names
                limit=self.batch_size,
                offset=idx * self.batch_size
            )
        # Extract embeddings and metadata
        result = {}
        if self.items2exclude is not None:
            for field in self.output_fields:
                curr_field_vals = []
                for entity in res:
                    if not (entity["id"], entity["word"]) in self.items2exclude:
                        curr_field_vals.append(entity[field])
                    result[f"{field}s"] = curr_field_vals
        else:
            for field in self.output_fields:
                result[f"{field}s"] = [entity[field] for entity in res]
        return result

    def extract_fields(self, data_loader, fields):
        res = {}
        for field in fields:
            res[f"{field}s"] = []
        for i, batch in enumerate(data_loader):
            for field in fields:
                res[f"{field}s"].extend(batch[f"{field}s"])            
            #is_human_values = batch['is_human_values']
            #labels.extend(is_human_values)
        #converted_labels = [int(item) for item in labels]
        #return np.array(labels)
        return res



## Dataset for contrastive learning and calibration
class MilvusContrastiveDataset(Dataset):
    def __init__(self, db_name, batch_size = 1024):
        self.vec_db_host ='milvus-standalone'
        self.vec_db_port ='19530'
        self.db_name = db_name
        vec_db_uri = "http://" + self.vec_db_host + ':' + self.vec_db_port
        self.mvs_client = MilvusClient(uri=vec_db_uri, token="root:Milvus", db_name=self.db_name)
        self.batch_size = batch_size
        
        # Load the collection
        self.collection_a_name = 'wordnet_human_values_sense_embeddings_qwen38b'
        self.collection_a = self.mvs_client.load_collection(collection_name=self.collection_a_name)
        self.collection_b_name = 'wordnet_non_values_sense_embeddings_qwen38b'
        self.collection_b = self.mvs_client.load_collection(collection_name=self.collection_b_name)

        self.full_dataset_a = MilvusDataset(db_name=db_name, collection_name=self.collection_a_name, 
                                output_fields=["id", "sense_embedding", "is_human_value", "word"], batch_size = self.batch_size)
        self.full_dataset_b = MilvusDataset(db_name=db_name, collection_name=self.collection_b_name, 
                                output_fields=["id", "sense_embedding", "is_human_value", "word"], batch_size = self.batch_size)
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

        ## List of items to exclude from the training
        self.schwartz_vals = ['self-direction', 'stimulation','hedonism','achievement','power','security','conformity','tradition','benevolence','universalism']
        self.schwartz_entities = []


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
        schwartz_vals = ['self-direction','stimulation','hedonism','achievement','power','security','conformity','tradition','benevolence','universalism']
        for entity in res:
            if entity["word"] in schwartz_vals:
                schwartz_items.append((entity, self.collection_a_name, "humval"))
                # Don't include Schwartz values in the results list
                continue
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
        #POSITIVE_CLASS_SIZE = self.total_positive_entities  # The user's specified size for each class
        #NEGATIVE_CLASS_SIZE = self.total_negative_entities
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
                #indices_2_list.append(idx2)
            else: # Sample from Class 1 (Positive class embeddings)
                idx1, idx2 = random.sample(self.humval_entities, 2)
                indices_list.append([idx1, idx2])
                #indices_2_list.append(idx2)
    
            labels_list.append(True) # Similar pair
    
        # 2. Generate DISSIMILAR pairs (Label=False/0, samples from different classes)
        for _ in range(NUM_PAIRS_PER_TYPE):
            # Sample A from positive class and B from negative class
            idx_humval = random.sample(self.humval_entities, 1)
            idx_nonval = random.sample(self.nonval_entities, 1)
    
            indices_list.append([idx_humval[0], idx_nonval[0]])
            #indices_2_list.append(idx_neg)
            labels_list.append(False) # Dissimilar pair
    
        # Final tensors
        self.pairs_similar_different = indices_list
        #self.pairs_neg_indices = indices_2_list
        self.pairs_labels = torch.tensor(labels_list, dtype=torch.bool)
        assert len(self.pairs_similar_different) == len(self.pairs_labels)

        NUM_TOTAL_PAIRS = len(indices_list)
        print(f"Generated {NUM_TOTAL_PAIRS} pairs ({self.pairs_labels.sum().item()} similar) for training.")
        #print(self.pairs_pos_neg_ids)

    ### ------------------------------- 
    def get_training_pairs(self):
        #labels_list = []
        embeddings_a_tensor = torch.empty(0)
        embeddings_b_tensor = torch.empty(0)
        #labels_tensor = torch.empty(0)
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
                #batch_res_list = []
                #for entity in batch_results:
                #    batch_res_list.append(entity)
                if embeddings_a_tensor.numel() == 0:
                    embeddings_a_tensor = torch.tensor(res['sense_embedding'])
                    #labels_list.append(res['is_human_value'])
                else:
                    embeddings_a_tensor = torch.vstack((embeddings_a_tensor, torch.tensor(res['sense_embedding'])))
                    #labels_list.append(res['is_human_value'])

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
                if embeddings_b_tensor.numel() == 0:
                    embeddings_b_tensor = torch.tensor(res['sense_embedding'])
                    #labels_list.append(res['is_human_value'])
                else:
                    embeddings_b_tensor = torch.vstack((embeddings_b_tensor, torch.tensor(res['sense_embedding'])))
                    #labels_list.append(res['is_human_value'])
                    
                #labels_tensor = torch.tensor(labels_list, dtype=torch.bool)
        labels_tensor = self.pairs_labels
        assert labels_tensor.shape[0] == embeddings_a_tensor.shape[0]
                
        return embeddings_a_tensor, embeddings_b_tensor, labels_tensor


    def get_subsets(self, set_name="validation"):
        sub_set_a = None
        sub_set_b = None
        output_fields_names=["id", "word", "sense_embedding", "is_human_value"]

        if set_name == "validation":
            #sub_set_a = Subset(self.full_dataset_a, self.collection_a_ids_val)
            #sub_set_b = Subset(self.full_dataset_b, self.collection_b_ids_val)
            sub_set_a = MilvusDataset(db_name=self.db_name, collection_name=self.collection_a_name, specific_ids=self.collection_a_ids_val, output_fields=output_fields_names)
            sub_set_b = MilvusDataset(db_name=self.db_name, collection_name=self.collection_b_name, specific_ids=self.collection_b_ids_val, output_fields=output_fields_names)
        elif set_name == "calibration":
            #sub_set_a = Subset(self.full_dataset_a, self.collection_a_ids_calib)
            #sub_set_b = Subset(self.full_dataset_b, self.collection_b_ids_calib)
            sub_set_a = MilvusDataset(db_name=self.db_name, collection_name=self.collection_a_name, specific_ids=self.collection_a_ids_calib, output_fields=output_fields_names)
            sub_set_b = MilvusDataset(db_name=self.db_name, collection_name=self.collection_b_name, specific_ids=self.collection_b_ids_calib, output_fields=output_fields_names)
        elif set_name == "train":
            sub_set_a = MilvusDataset(db_name=self.db_name, collection_name=self.collection_a_name, specific_ids=self.collection_a_ids_train, output_fields=output_fields_names)
            sub_set_b = MilvusDataset(db_name=self.db_name, collection_name=self.collection_b_name, specific_ids=self.collection_b_ids_train, output_fields=output_fields_names)
        return sub_set_a, sub_set_b

    def get_entities(self, set_name="validation"):
        entities = []
        if set_name == "validation":
            entities = self.validation_entities
        elif set_name == "calibration":
            entities = self.calib_entities

        embeddings_tensor = torch.empty(0)
        labels_list = []
        words_list = []
        for entity in entities:
            id_c = entity[0]
            collection = entity[1]
            expr_value = f"id == {id_c}"
            res = self.mvs_client.query(
                collection_name=collection,
                filter=expr_value,
                output_fields=["id", "sense_embedding", "is_human_value", "word"]
            )
            res = res[0]
            if embeddings_tensor.numel() == 0:
                embeddings_tensor = torch.tensor(res['sense_embedding'])
            else:
                embeddings_tensor = torch.vstack((embeddings_tensor, torch.tensor(res['sense_embedding'])))
            labels_list.append(res['is_human_value'])
            words_list.append((res["id"], res['word']))
        labels_tensor = torch.tensor(labels_list, dtype=torch.bool)
        assert labels_tensor.shape[0] == embeddings_tensor.shape[0]
                
        return words_list, embeddings_tensor, labels_tensor

    ## We don't use this as dataloader queries do not work with 2 different collections
    def __getitem__(self, idx):
        pass

def setup_contrastive_dataset(db_name='word_sense_embeddings', batch_size=64, generate_train_pairs = True, num_pairs_per_type=5, train_part=0.8, val_part=0.1):
    COLLECTION_NAME = 'wordnet_human_values_sense_embeddings_qwen38b'
    DB_NAME = 'word_sense_embeddings'
    VECTOR_FIELD_NAME = 'sense_embedding' # Your vector field name
    LABEL_FIELD_NAME = 'is_human_value' # Your label field name, set to None if no labels
    BATCH_SIZE = 64
    SHUFFLE = True # Shuffle data every epoch
    NUM_WORKERS = 0 # Use multi-process data loading (optional, set to 0 for single process)

    dataset = MilvusContrastiveDataset(
        db_name = db_name,
        batch_size = batch_size
    )

    if generate_train_pairs:
        dataset.set_humval_entities()
        dataset.set_nonval_entities()
        dataset.set_total_entities()
        dataset.generate_training_pairs(train_part=train_part, val_part=val_part, num_pairs_per_type=num_pairs_per_type)

    return dataset

def setup_contrastive_training(num_pairs_per_type=50, batch_size=64, train_part=0.8, val_part=0.1):
    # Generate training data
    milvus_dataset = setup_contrastive_dataset(generate_train_pairs = True, num_pairs_per_type=num_pairs_per_type, train_part=train_part, val_part=val_part)   # Hard-coded: Train 80%, Validation 10%, Calibration 10%
    #words_validation, embeddings_validation, labels_validation = milvus_dataset.get_entities(set_name="validation")
    #words_calibration, embeddings_calibration, labels_calibration = milvus_dataset.get_entities(set_name="calibration") 
    embeddings_a, embeddings_b, train_labels = milvus_dataset.get_training_pairs()
    contrastive_train_dataset = TensorDataset(embeddings_a, embeddings_b, train_labels)
    contrastive_train_dataloader = DataLoader(contrastive_train_dataset, batch_size=batch_size, shuffle=False)
    milvus_dataset.specific_ids = True
    return contrastive_train_dataloader,train_labels,milvus_dataset

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

def extract_from_dataloader(data_loader):
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
        #inputs = inputs.to(device)
        #encoded = model.encoder(inputs)
    if embeddings_tensor.numel() == 0:
        embeddings_tensor = embeddings_batch
    else:
        embeddings_tensor = torch.vstack((embeddings_tensor, embeddings_batch))
        
    assert len(labels) == embeddings_tensor.shape[0]
    return words, embeddings_tensor, labels