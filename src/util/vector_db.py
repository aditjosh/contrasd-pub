from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import json

## Class to perform vector db opeartions
class VectorDBOps():
    def __init__(self, vec_db_name, vec_db_host='localhost', vec_db_port='19530'):
        self.db_name = vec_db_name
        vec_db_uri = "http://" + vec_db_host + ':' + vec_db_port
        self.mvs_client = MilvusClient(uri=vec_db_uri, token="root:Milvus", db_name=self.db_name)

    def prepare_word_sense_collection(self, collection_name, max_text_len=4096, embedding_dim=2560):
        """
        Creates a Milvus collection if it doesn't already exist.
        """
        # Check if the collection exists
        if self.mvs_client.has_collection(collection_name=collection_name):
            print(f"Collection '{collection_name}' already exists. Using it.")
            return

        # Define the schema for the new collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="sense", dtype=DataType.VARCHAR, max_length=max_text_len),
            FieldSchema(name="sense_embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="is_human_value", dtype=DataType.BOOL)
        ]
        schema = CollectionSchema(fields, description="Human-value senses collection")

        index_params = self.mvs_client.prepare_index_params()
        index_params.add_index(
            field_name="id",
            index_type="AUTOINDEX"
        )
        # Index for the vector field
        index_params.add_index(
            field_name="word",
            index_type="AUTOINDEX"
        )
        index_params.add_index(
            field_name="sense",
            index_type="AUTOINDEX"
        )
        index_params.add_index(
            field_name="is_human_value",
            index_type="AUTOINDEX"
        )
        # Index for embedding
        index_params.add_index(
            field_name="sense_embedding",
            index_type="AUTOINDEX", # Or specify a specific scalar index type
            metric_type="COSINE"
        )

        # Create the collection with the defined schema
        print(f"Creating new collection: '{collection_name}'...")
        self.mvs_client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,
            schema=schema,
            index_params=index_params
        )
        print(f"Collection '{collection_name}' created successfully.")

    def prepare_sentences_collection(self, collection_name, max_text_len=4096, embedding_dim=2560):
        """
        Creates a Milvus collection if it doesn't already exist.
        """
        # Check if the collection exists
        if self.mvs_client.has_collection(collection_name=collection_name):
            print(f"Collection '{collection_name}' already exists. Using it.")
            return

        # Define the schema for the new collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="sense", dtype=DataType.VARCHAR, max_length=max_text_len),
            FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=max_text_len),
            FieldSchema(name="part_of_speech", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="sense_embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        ]
        schema = CollectionSchema(fields, description="Sentences collection")

        index_params = self.mvs_client.prepare_index_params()
        index_params.add_index(
            field_name="id",
            index_type="AUTOINDEX"
        )
        # Index for the vector field
        index_params.add_index(
            field_name="word",
            index_type="AUTOINDEX"
        )
        index_params.add_index(
            field_name="sense",
            index_type="AUTOINDEX"
        )
        index_params.add_index(
            field_name="sentence",
            index_type="AUTOINDEX"
        )
        index_params.add_index(
            field_name="part_of_speech",
            index_type="AUTOINDEX"
        )
        # Index for embedding
        index_params.add_index(
            field_name="sense_embedding",
            index_type="AUTOINDEX", # Or specify a specific scalar index type
            metric_type="COSINE"
        )

        # Create the collection with the defined schema
        print(f"Creating new collection: '{collection_name}'...")
        self.mvs_client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,
            schema=schema,
            index_params=index_params
        )
        print(f"Collection '{collection_name}' created successfully.")

    def insert_word_senses(self, encoder, collection_name, json_list_values):
        """
        Generates embeddings for a list of sentences and inserts them into Milvus.

        Args:
            encoder: To generate embeddings
            collection_name: Where to save the embeddings
            word_senses (list[str]): A list of json words and sense glosses
        """
        if not json_list_values:
            print("No word senses to insert.")
            return

        #glosses = []
        senses = []
        is_human_values = []
        words = []

        ## For each word, concatenate all its glosses. This will increase the variety of words that occur in the word's gloss.
        combined_hv_gloss = ''
        combined_hnv_gloss = ''

        # Sense is formed by combining all glosses, one per value of is_hv
        for ws in json_list_values:
            is_hv = ws['is_human_value'] if 'is_human_value' in ws else 0
            if is_hv == 1:
                if combined_hv_gloss == '':
                    combined_hv_gloss = ws['word'] + ': '
                    combined_hv_gloss = combined_hv_gloss + ws['gloss']
                else:
                    combined_hv_gloss = combined_hv_gloss + '. ' + ws['gloss']
            elif is_hv == 0:
                if combined_hnv_gloss == '':
                    combined_hnv_gloss = ws['word'] + ': '
                    combined_hnv_gloss = combined_hnv_gloss + ws['gloss']
                else:
                    combined_hnv_gloss = combined_hnv_gloss + '. ' + ws['gloss']

        ## For each word, one entry per value of is_hv
        seen_hv = False
        seen_hnv = False
        for ws in json_list_values:
            curr_word = ws['word']
            is_hv = ws['is_human_value'] if 'is_human_value' in ws else 0
            if (not seen_hv) and is_hv == 1:
                # Create an entry
                words.append(curr_word)
                is_human_values.append(True)
                senses.append(combined_hv_gloss)
                seen_hv = True
            if (not seen_hnv) and is_hv == 0:
                words.append(curr_word)
                is_human_values.append(False)
                senses.append(combined_hnv_gloss)
                seen_hnv = True

        print(f"Generating embeddings for {len(senses)} sense glosses of {curr_word}...")
        embeddings = encoder.encode(senses)

        # Prepare the data for insertion
        data_to_insert = [
            {"word": word, "sense" : sense, "sense_embedding": embedding.tolist(), "is_human_value" : is_human_value}
            for word, sense, embedding, is_human_value in zip(words, senses, embeddings, is_human_values)
        ]

        print("Inserting data into Milvus...")
        self.mvs_client.insert(
            collection_name=collection_name,
            data=data_to_insert
        )
        print("Data inserted successfully.")

    def insert_sentences(self, encoder, collection_name, json_list_values):
        """
        Generates embeddings for a list of sentences and inserts them into Milvus.

        Args:
            encoder: To generate embeddings
            collection_name: Where to save the embeddings
            word_senses (list[str]): A list of json words and sense glosses
        """
        if not json_list_values:
            print("No word senses to insert.")
            return
        
        word = None        

        # Sense is formed by combining all glosses, one per value of is_hv
        for ws in json_list_values:
            senses = []
            combined_gloss = ''
            word = ws['word']
            if combined_gloss == '':
                combined_gloss = ws['word'] + ': '
                combined_gloss = combined_gloss + ws['gloss']
            else:
                combined_gloss = combined_gloss + '. ' + ws['gloss']

            senses.append(combined_gloss)
            print(f"Generating embeddings for {len(senses)} sense glosses of {word}...")
            embeddings = encoder.encode(senses)

            # Prepare the data for insertion
            sentence = ws['sentence']
            pos = ws['part_of_speech']
            data_to_insert = [
                {"word": word, "sentence" : sentence, "part_of_speech" : pos, "sense" : sense, "sense_embedding": embedding.tolist()}
                for sense, embedding in zip(senses, embeddings)
            ]

            print("Inserting data into Milvus...")
            self.mvs_client.insert(
                collection_name=collection_name,
                data=data_to_insert
            )
        print("Data inserted successfully.")


    def search(self, encoder, collection_name, query: str, top_k: int = 5):
        """
        Performs a vector search on the Milvus collection.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        print(f"Searching for query: '{query}'...")
        # Generate embedding for the query
        query_embedding = encoder.encode([query], normalize_embeddings=True)[0].tolist()

        # Perform the search
        res = self.mvs_client.search(
            collection_name=collection_name,
            data=[query_embedding],
            anns_field="sense_embedding",
            output_fields=["id","word","sense","is_human_value"],
            limit=top_k
        )
        print("Search completed.")
        return res

    def read_file_in_batches(self, file_path, lines_to_read=100, start_line_number=1):
        batch = []
        try:
            with open(file_path, 'r') as file:
                for i, line in enumerate(file, 1):  # start enumeration from 1
                    if i >= start_line_number and i < start_line_number + lines_to_read:
                        batch.append(line)
                    elif i >= start_line_number + lines_to_read:
                        break  # Stop after reading the desired number of lines
            return batch
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def read_file_in_batch_streams(self, file_path, lines_to_read=20):
        try:
            with open(file_path, 'r') as file:
                batch = []
                for i, line in enumerate(file):
                    batch.append(line)
                    if len(batch) == lines_to_read:
                        yield batch
                        batch = []
                if batch:
                    yield batch
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


## Vectorize sentences and save in vector database
def vectorize_and_save_sentences(in_file_path, collection_name, milvus_db, encoder, embedding_dim):
    has_sentences = False
    # Data to insert
    for batch_of_strings in milvus_db.read_file_in_batch_streams(in_file_path, lines_to_read = 1):  ## Leave lines_to_read=1
        json_batch = [json.loads(s) for s in batch_of_strings]
        # Each line in the file contained word and its senses
        values_to_insert = []
        for json_ws in json_batch:
            json_ws = json.loads(json_ws)
            
            if 'senses' in json_ws:
                word = json_ws['word']
                ## File with wordnet senses
                for sense in json_ws['senses']:
                    word_is_hv = 0
                    if 'human_value_senses' in json_ws:
                        for hv_sense in json_ws['human_value_senses']:
                            if sense == hv_sense:
                                word_is_hv = 1
                                break
                    # Gloss is sense definition.
                    values_to_insert.append({'word': word, 'gloss': sense, 'is_human_value' : word_is_hv})
            else:
                ## File for unk/misc phrase senses
                if 'human_value_senses' in json_ws:
                    word = json_ws['word']
                    for hv_sense in json_ws['human_value_senses']:
                        values_to_insert.append({'word': word, 'gloss': hv_sense, 'is_human_value' : 1})
                if 'non_value_senses' in json_ws:
                    word = json_ws['word']
                    for nonval_sense in json_ws['non_value_senses']:
                        values_to_insert.append({'word': word, 'gloss': nonval_sense, 'is_human_value' : 0})
                if 'terms' in json_ws:
                    has_sentences = True
                    sentence = json_ws['sentence']
                    for entry in json_ws['terms']:
                        word = entry['word'].lower()
                        pos = entry['part_of_speech']
                        gloss = entry['sense_definition']
                        values_to_insert.append({'word': word, 'gloss': gloss, 'part_of_speech' : pos, 'sentence': sentence})

        if has_sentences:
            milvus_db.insert_sentences(encoder, collection_name, values_to_insert)
        else:
            milvus_db.insert_word_senses(encoder, collection_name, values_to_insert)