import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import LatentPairwiseContrastiveLoss
import torch.optim as optim
import numpy as np
import umap
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.stats import scoreatpercentile
from scipy.optimize import linear_sum_assignment

class ContrastiveEmbeddingsTransformer(nn.Module):
    """MLP that outputs L2-normalized embeddings suitable for cosine distance CP."""
    def __init__(self, feature_dim, latent_dim):
        super(ContrastiveEmbeddingsTransformer, self).__init__()
        self.fc = nn.Sequential(
            #nn.Linear(feature_dim, embedding_dim),
            #nn.ReLU(),
            #nn.Linear(32, embedding_dim)
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        # Mandatory L2-Normalization for cosine-based non-conformity
        embedding = self.fc(x)
        return F.normalize(embedding, p=2, dim=1)

class EmbeddingsAutoencoder(nn.Module):
    def __init__(self, feature_dim, latent_dim=128, embed_transformer=None):
        super(EmbeddingsAutoencoder, self).__init__()
        self.encoder = embed_transformer if embed_transformer else ContrastiveEmbeddingsTransformer(feature_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        # Mandatory L2-Normalization for cosine-based non-conformity
        #latent_norm = F.normalize(latent, p=2, dim=1)
        return x

## Wrapper for training with contrastive dataset
class EmbeddingsAutoencoderService(nn.Module):
    def __init__(self, autoencoder=None, embed_transformer=None, device="cpu"):
        super(EmbeddingsAutoencoderService, self).__init__()
        # Hyperparameters and model definition parameters
        self.device = device
        self.input_dim = 4096
        self.latent_dim = 128
        self.embed_transformer = embed_transformer if embed_transformer else ContrastiveEmbeddingsTransformer(self.input_dim, self.latent_dim)
        self.ae_model = autoencoder if autoencoder else EmbeddingsAutoencoder(self.input_dim, self.latent_dim, embed_transformer=self.embed_transformer)

    # Generate embeddings for data loader input
    def batch_encode_transform(self, data_loader) -> (torch.Tensor, list, np.ndarray):
        self.ae_model.eval()
        embeddings_tensor = torch.empty(0)
        output_tensor = torch.empty(0)
        labels = []
        words = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                inputs = torch.tensor(batch['sense_embeddings'])
                inputs = inputs.to(self.device)
                if 'is_human_values' in batch:
                    is_human_values = np.array(batch['is_human_values'], dtype=int)
                else:
                    is_human_values = np.full(len(batch['ids']), fill_value = -1, dtype = int) 
                labels.extend(is_human_values)
                words.extend(batch['words'])
                
                # Encode and Transform
                transformed_batch = self.ae_model.encoder(inputs)

                if output_tensor.numel() == 0:
                    output_tensor = transformed_batch
                else:
                    output_tensor = torch.vstack((output_tensor, transformed_batch))

                assert len(labels) == output_tensor.shape[0]

        return output_tensor, words, np.array(labels)

    # Reduce dimensions of the input embeddings
    def encode_transform(self, input_embeddings: torch.Tensor):
        self.ae_model.eval()
        embeddings = []
        with torch.no_grad():
            inputs = input_embeddings
            inputs = inputs.to(self.device)
            transformed = self.ae_model.encoder(inputs)
            embeddings.append(transformed)
        return torch.vstack(embeddings)
    
#### Clustering Model
class ContrastiveClusterLearner():
    """
    Wrapper class to make a PyTorch model compatible with the scikit-learn 
    API (including fit, predict, and predict_proba).
    """
    def __init__(self, embed_transformer, num_clusters=2, device="cpu"):
        super(ContrastiveClusterLearner, self).__init__()
        # Hyperparameters and model definition parameters
        self.model = embed_transformer 
        self.device = device
        self.num_clusters = num_clusters
        self.kmeans_model = None

    def cosine_distance_matrix(self, x, y):
        """
        Calculates the Cosine Distance matrix between two sets of vectors, x and y.
        X: (N, D) tensor of data points
        Y: (K, D) tensor of cluster centroids
        Returns: (N, K) tensor of distances
        """
        # 1. Normalize both tensors (crucial for accurate cosine similarity)
        X_norm = F.normalize(x, p=2, dim=1)
        Y_norm = F.normalize(y, p=2, dim=1)
        
        # 2. Calculate Cosine Similarity (equivalent to X_norm @ Y_norm.T)
        # The output shape is (N, K)
        #similarity = X_norm @ Y_norm.T
        similarity = torch.mm(X_norm, Y_norm.t())
        
        # 3. Convert similarity to distance
        distance = 1.0 - similarity
        return distance

    # Custom kMeans model that uses cosine distances.
    def cosine_kmeans_fit(self, embeddings, num_clusters=2, max_iter=100, initial_indices=[]):
        """
        Performs K-Means clustering using Cosine Distance.

        Args:
            embeddings (torch.Tensor): The data to cluster, shape (N, D).
            num_clusters (int): The number of clusters (K).
            max_iter (int): Maximum iterations for convergence.

        Returns:
            tuple: (cluster_ids, centroids)
        """
        N, D = embeddings.shape
        device = embeddings.device
        
        # Initialize Centroids: Randomly select K points from the data
        if len(initial_indices) == 0:
            initial_indices = torch.randperm(N)[:num_clusters]
        
        centroids = embeddings[initial_indices]
        
        for i in range(max_iter):
            # --- E-STEP: Assignment ---
            # Calculate Cosine Distance from every point to every centroid
            # shape: (N, K)
            distances = self.cosine_distance_matrix(embeddings, centroids)
            
            # Find the index (cluster ID) of the minimum distance for each point
            # shape: (N,)
            cluster_ids = torch.argmin(distances, dim=1)
            
            # --- M-STEP: Update Centroids ---
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(num_clusters, 1, dtype=torch.float32, device=device)
            
            for k in range(num_clusters):
                # Find all points assigned to cluster k
                points_in_cluster = embeddings[cluster_ids == k]
                
                if len(points_in_cluster) > 0:
                    # Calculate the new centroid as the mean of the points in the cluster
                    new_centroids[k] = points_in_cluster.mean(dim=0)
                    counts[k] = len(points_in_cluster)
                else:
                    # Handle empty cluster: re-initialize centroid (optional: just keep old one)
                    new_centroids[k] = centroids[k]

            # Check for convergence: If centroids haven't moved significantly
            centroid_movement = torch.norm(new_centroids - centroids, dim=1).mean().item()
            centroids = new_centroids
            
            if centroid_movement < 1e-5:
                print(f"K-Means converged after {i+1} iterations.")
                break
                
        # Final step: Normalize centroids (important for final interpretation/usage)
        final_centroids = F.normalize(centroids, p=2, dim=1)
        
        return cluster_ids, final_centroids

    def predict_clusters(self, data_embeddings, learned_centroids):
        """
        Predicts the cluster ID for new data based on the minimum cosine distance 
        to the learned centroids.

        Args:
            data_embeddings (torch.Tensor): New data points, shape (N_val, D).
            learned_centroids (torch.Tensor): Final centroids from clustering, shape (K, D).

        Returns:
            torch.Tensor: Cluster ID assigned to each validation point, shape (N_val,).
        """
        # 1. Calculate the distance matrix between validation data and centroids
        # Output shape: (N_val, K)
        distances = self.cosine_distance_matrix(data_embeddings, learned_centroids)
        
        # 2. Assign cluster ID based on the minimum distance
        # The argmin function returns the index (cluster ID) corresponding to the minimum distance.
        predicted_cluster_ids = torch.argmin(distances, dim=1)
        
        return predicted_cluster_ids

    def lean_manifold_tsne(self, embeddings, n_components=2, perplexity=30):
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        embeddings_tsne_2d = tsne.fit_transform(embeddings.cpu().numpy())
        return torch.tensor(embeddings_tsne_2d)

    def lean_manifold_umap(self, embeddings, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
        embeddings_umap_2d = umap_model.fit_transform(embeddings.cpu().numpy())
        return embeddings_umap_2d
        
    def transform_embeddings(self, data_loader):
        return self.model.batch_encode_transform(data_loader)

    ############################################################################

    def fit(self, inputs: np.ndarray):
        kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=42, n_init=10, batch_size=256) 
        clusters = kmeans.fit_predict(inputs)
        self.kmeans_model = kmeans
        return clusters

    def predict(self, inputs: np.ndarray):
        cluster_labels = self.kmeans_model.predict(inputs)
        return cluster_labels

    # Transformation returning distances to cluster centers
    def distance_to_centers_transform(self, inputs: np.ndarray):
        transformed_data = self.kmeans_model.transform(inputs)
        return transformed_data

    ############################################################################ CP Related

    def calculate_calibration_conformity_scores_loo(self, calib_embeddings: torch.tensor, calib_labels: torch.tensor):
        """
        Calculates the conformity score (alpha) for the calibration set using 
        Leave-One-Out (LOO) L2-Normalized Euclidean Distance.
        
        Alpha_i = min_j!=i, y_j=y_i { Euclidean Distance(emb_i, emb_j) }
        
        This prevents the distance being 0 (distance to self), which causes tiny q_hat.
        
        Returns: A dictionary mapping class IDs to a list of non-conformity scores.
        """
        unique_classes = torch.unique(calib_labels)
        calib_embeddings_tensor = calib_embeddings.to(self.device)
        
        # Pre-calculate distances between ALL pairs in the calibration set
        # The embeddings are already L2-normalized
        distances_matrix = torch.cdist(calib_embeddings_tensor, calib_embeddings_tensor, p=2)
    
        scores = {c.item(): [] for c in unique_classes}
    
        for i in range(len(calib_embeddings)):
            current_label = calib_labels[i]
            
            # Mask for points of the SAME class, excluding the current point (i != j)
            mask_tensor = (calib_labels == current_label)
            
            # Create a boolean tensor mask from the numpy mask
            #mask_tensor = same_class_mask.detach().clone()
            
            # Ensure we exclude the point itself (i != i)
            mask_tensor[i] = False 
            
            if mask_tensor.sum() > 0:
                # Look at the distances from point 'i' to all other points
                row_i = distances_matrix[i]
                
                # Find the minimum distance to any OTHER point of the SAME class
                min_distance = row_i[mask_tensor].min()
                scores[current_label.item()].append(min_distance.item())
    
        return scores

    def calculate_test_conformity_scores(self, calib_embeddings, calib_labels, test_embeddings, test_labels):
        """
        Calculates the conformity score (alpha) for the test set.
        
        Alpha_i(c) = min_j, y_j=c { Euclidean Distance(emb_i, emb_j) }
        
        Returns: A list of dictionaries: [{class: score}, ...]
        """
        unique_calib_classes = torch.unique(calib_labels)
        unique_test_classes = torch.unique(test_labels)
        calib_embeddings_tensor = calib_embeddings.to(self.device)
        
        all_test_scores = []
    
        for i in range(len(test_embeddings)):
            test_emb = (test_embeddings[i]).to(self.device).unsqueeze(0)
            
            # Calculate Euclidean distance between the test sample and ALL calibration samples
            # Shape: (1, num_calib_samples)
            distances = torch.cdist(test_emb, calib_embeddings_tensor, p=2).squeeze(0)
            
            test_scores_i = {}
    
            for c in unique_test_classes:
                # Mask for test points belonging to class 'c'
                mask = (calib_labels == c)
                #mask = torch.tensor(same_class_mask.detach().clone(), dtype=torch.bool).to(self.device)
                
                if mask.sum() > 0:
                    # Find the MINIMUM distance (most similar/closest point) in class 'c'
                    min_distance = distances[mask].min()
                    test_scores_i[c.item()] = min_distance.item()
                else:
                     # If no calibration points exist for class c, score is effectively infinite
                     test_scores_i[c.item()] = np.inf 
    
            all_test_scores.append(test_scores_i)
    
        return all_test_scores

    def critical_quantile_mc(self, calib_alphas, confidence_level=0.9, n_bootstrap_samples=500):
        """
        Calculates Quantile q_hat with Monte Carlo (Bootstrap) Sampling
        """
        q_hat = {}
        gamma = 1 - confidence_level # Significance level
        N_BOOTSTRAP = n_bootstrap_samples # Number of Monte Carlo samples for robustness
        quantile_percentile = confidence_level * 100
    
        for c, alphas in calib_alphas.items():
            if not alphas: continue
            
            n = len(alphas)
            # Target index for the (1-gamma) quantile
            idx = int(np.ceil((n + 1) * gamma)) - 1
            idx = max(0, min(idx, n - 1))
            
            # Monte Carlo (Bootstrap) Loop to stabilize q_hat
            # We resample the calibration scores to ensure the chosen threshold 
            # is not overly sensitive to specific outliers in the calibration set.
            mc_quantiles = []
            alphas_array = np.array(alphas)
            
            for _ in range(N_BOOTSTRAP):
                # Sample with replacement
                resampled = np.random.choice(alphas_array, size=n, replace=True)
                resampled.sort()
                critical_quantile = np.percentile(resampled, quantile_percentile, axis=0)
                #mc_quantiles.append(resampled[idx])
                mc_quantiles.append(critical_quantile)
                
            # Use the mean of the MC quantiles for a robust estimate
            q_hat[c] = np.mean(mc_quantiles)
            
            #print(f"Class {c}: n={n}, MC Robust q_hat = {q_hat[c]:.4f} (std: {np.std(mc_quantiles):.4f})")

        return q_hat

    def compute_prediction_set(self, test_scores, q_hat):
        """
        Computes the prediction set for a single test sample based on the quantile q_hat.
        
        The prediction set includes all classes 'c' where the test sample's non-conformity 
        score for that class is less than or equal to the quantile q_hat.
        """
        prediction_set = []
        
        # test_scores is a dictionary: {class: score_for_this_test_sample}
        # Example: {'0': 0.1, '1': 0.8}
        
        for c, alpha_score in test_scores.items():
            #print(alpha_score)
            #print(f"q_hat[c]: {q_hat[c]}")
            ## Scores are negatively-oriented (smaller scores -- distances -- are better) 
            if alpha_score <= q_hat[c]:
                prediction_set.append(c)
                
        return sorted(prediction_set)

    def compute_conditional_coverage(self, q_hat, all_test_scores, test_labels):
        prediction_sets = []
        coverage_count = 0
        total_sets = len(test_labels)
        
        # Initialize structures for Conditional Coverage tracking
        unique_classes = np.unique(test_labels)
        conditional_coverage_total = {c: 0 for c in unique_classes}
        conditional_coverage_correct = {c: 0 for c in unique_classes}

        for i in range(total_sets):
            current_true_label = test_labels[i]
            scores_i = all_test_scores[i] 
            
            # Find the set of possible labels (C(x))
            C_x = self.compute_prediction_set(scores_i, q_hat)
            prediction_sets.append(C_x)
            
            # Update conditional coverage tracking
            conditional_coverage_total[current_true_label.item()] += 1
            
            if current_true_label in C_x:
                coverage_count += 1 # Marginal Coverage
                conditional_coverage_correct[current_true_label.item()] += 1

        return prediction_sets, coverage_count, conditional_coverage_total, conditional_coverage_correct

    ##### Coverage Gap prediction at the cluster level
    def cluster_non_conformity_score(self, embedding, center):
        """
        Non-conformity score: 1 - Cosine Similarity (distance to cluster center).
        """
        embedding = embedding.reshape(1, -1)
        center = center.reshape(1, -1)
        
        non_similarity = self.cosine_distance_matrix(embedding, center)
        return non_similarity

    def predict_non_conforming(self, test_embeddings, learned_centroids, q_hat_dict):
        """
        Predict non-conformity status.
        True = Non-conforming (Outlier), False = Conforming.
        """    
        test_embeddings_tensor = test_embeddings.to(self.device)
        test_cluster_labels = self.predict_clusters(test_embeddings_tensor, learned_centroids)
        non_conforming_status = []

        for i, emb in enumerate(test_embeddings_tensor):
            k = test_cluster_labels[i]
            if k not in q_hat_dict:
                non_conforming_status.append(True)
                continue

            center_k = learned_centroids[k]
            alpha_test = self.cluster_non_conformity_score(emb, center_k)
            threshold_k = q_hat_dict[k]

            is_non_conforming = alpha_test > threshold_k
            non_conforming_status.append(is_non_conforming)

        return np.array(non_conforming_status)

    def calculate_coverage_gaps(self, val_embeddings, learned_centroids, q_hat_dict, confidence_level=0.9):
        """
        Calculates the class-conditional coverage gap.
        Coverage Gap = |(1 - alpha) - Empirical Coverage|
        """
        val_embeddings_tensor = val_embeddings.to(self.device)
        val_labels = self.predict_clusters(val_embeddings_tensor, learned_centroids)
        is_non_conforming = self.predict_non_conforming(val_embeddings_tensor, learned_centroids, q_hat_dict)
        
        # Conforming points are those where is_non_conforming is False
        is_conforming = ~is_non_conforming
        
        gaps = {}
        print("\n--- Coverage Gap Analysis ---")
        print(f"Target Coverage (1 - alpha): {confidence_level:.4f}")
        
        for k in range(learned_centroids.shape[0]):
            # Points belonging to this cluster
            mask = (val_labels == k)
            n_k = torch.sum(mask)
            
            if n_k == 0:
                continue
            
            # How many were correctly identified as conforming?
            empirical_coverage = np.sum(is_conforming[mask]) / n_k
            gap = (empirical_coverage - confidence_level) # Signed gap
            gaps[k] = {
                'empirical_coverage': empirical_coverage,
                'gap': gap,
                'absolute_gap': 100 * abs(gap)
            }
            
            print(f"Cluster {k}: Empirical={empirical_coverage:.4f}, Gap={gap:+.4f}")
            
        return gaps

    


##########################################################################################################



