from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F
from torch import Tensor

## Generate embedddings for input text using a black-box model
class PretrainedEmbeddingsGenerator():
    def __init__(self, model_name=None, custom_cache_directory='/home/jovyan/sense_encoder_cache'):
        # Download the model to the custom directory
        if torch.cuda.is_available():
            print('using cuda')
            self.model = AutoModel.from_pretrained(model_name, device_map="auto", cache_dir=custom_cache_directory)
            print('Model is on device ', self.model.device)
            #self.model.to(torch.device("cuda"))
            self.device = torch.device('cuda')
        else:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=custom_cache_directory)
            self.device = torch.device('cpu')
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_directory)

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def encode(self, texts, normalize_embeddings=True, max_length=4096):
        batch_texts = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length) # padding, truncation ensure same vector length
        batch_texts.to(self.model.device)
        outputs = self.model(**batch_texts)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_texts['attention_mask'])

        # normalize embeddings
        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        cpu_embeddings = embeddings.to('cpu')
        del embeddings
        return cpu_embeddings


