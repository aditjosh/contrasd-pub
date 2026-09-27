import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss import LatentPairwiseContrastiveLoss
from model.models import ContrastiveClusterLearner, EmbeddingsAutoencoderService
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.stats import scoreatpercentile
from scipy.optimize import linear_sum_assignment

## Autoencoder models
class AutoencoderE1(nn.Module):
    def __init__(self, feature_dim, latent_dim=128):
        super(AutoencoderE1, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim),  # Latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x

# For testing with synthetic data
class ContrastiveEmbeddingsTransformerE1(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ContrastiveEmbeddingsTransformerE1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, latent_dim)
        )

    def forward(self, x):
        # Mandatory L2-Normalization
        embedding = self.fc(x)
        return F.normalize(embedding, p=2, dim=1)

class AutoencoderE2(nn.Module):
    def __init__(self, feature_dim, latent_dim=16, embed_transformer=None):
        super(AutoencoderE2, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.encoder = embed_transformer if embed_transformer else ContrastiveEmbeddingsTransformerE1(feature_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 48),
            nn.ReLU(),
            nn.Linear(48, self.feature_dim),
            nn.ReLU()
        )
    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded


class EmbeddingsAutoencoderServiceExt(EmbeddingsAutoencoderService):
    def __init__(self,autoencoder=None, embed_transformer=None, device="cpu"):
        super().__init__(autoencoder, embed_transformer, device)

    # Generate embeddings for data loader input
    def ethics_batch_encode_transform(self, data_loader) -> (torch.Tensor, list, np.ndarray):
        self.ae_model.eval()
        embeddings_tensor = torch.empty(0)
        output_tensor = torch.empty(0)
        labels = []
        words = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                inputs = torch.tensor(batch['embeddings'])
                inputs = inputs.to(self.device)
                if 'labels' in batch:
                    is_human_values = np.array(batch['labels'], dtype=int)
                else:
                    is_human_values = np.full(len(batch['ids']), fill_value = -1, dtype = int) 
                labels.extend(is_human_values)
                words.extend(batch['inputs'])
                
                # Encode and Transform
                transformed_batch = self.ae_model.encoder(inputs)

                if output_tensor.numel() == 0:
                    output_tensor = transformed_batch
                else:
                    output_tensor = torch.vstack((output_tensor, transformed_batch))

                assert len(labels) == output_tensor.shape[0]

        return output_tensor, words, np.array(labels)


class ContrastiveClusterLearnerExt(ContrastiveClusterLearner):
    def __init__(self, embed_transformer):
        super().__init__(embed_transformer) 

    def transform_ethics_embeddings(self, data_loader):
        return self.model.ethics_batch_encode_transform(data_loader)