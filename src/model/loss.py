import torch
import torch.nn as nn

class LatentPairwiseContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks using the cosine similarity metric
    on the latent (re-encoded) vectors.

    It enforces:
    1. Similar pairs (label=1) to have a similarity close to 1.
    2. Dissimilar pairs (label=0) to have a similarity below a margin.

    Note: This is an energy-based approach where y=1 means "pull" and y=0 means "push".
    """
    def __init__(self, margin=0.5):
        super(LatentPairwiseContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, latent_a, latent_b, labels):
        # Calculate the cosine similarity for all pairs in the batch
        sim = nn.functional.cosine_similarity(latent_a, latent_b, dim=1) # Shape: (B,)

        # Convert labels to float for multiplication (0 for False/Dissimilar, 1 for True/Similar)
        labels = labels.float()

        # Loss term for SIMILAR pairs (label = 1):
        # We want similarity (sim) to be close to 1. 
        # This term is active only when labels=1.
        similar_pairs_mask = (labels == 1)
        #loss_similar = labels * torch.pow((1 - sim), 2)
        loss_similar = (1.0 - sim[similar_pairs_mask])

        # Loss term for DISSIMILAR pairs (label = 0):
        # We want similarity (sim) to be far from 1.
        # This term is active only when labels=0.
        #loss_dissimilar = (1 - labels) * torch.pow(torch.relu(sim - self.margin), 2)
        dissimilar_pairs_mask = (labels == 0)
        loss_dissimilar = torch.clamp(sim[dissimilar_pairs_mask] - self.margin, min=0.0)

        # Total loss is the mean of both terms
        #loss = (loss_similar + loss_dissimilar).mean()
        loss = torch.cat((loss_similar, loss_dissimilar)).mean()
        return loss


class LatentPairwiseContrastiveSAELoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks using the cosine similarity metric
    on the latent (re-encoded) vectors.

    It enforces:
    1. Similar pairs (label=1) to have a similarity close to 1.
    2. Dissimilar pairs (label=0) to have a similarity below a margin.

    Note: This is an energy-based approach where y=1 means "pull" and y=0 means "push".
    """
    def __init__(self, margin=0.5, target_sparsity=0.05, sparsity_wight=0.5):
        super(LatentPairwiseContrastiveSAELoss, self).__init__()
        self.margin = margin
        self.rho = target_sparsity   # Rho: % of neurons active
        self.beta = sparsity_wight   # Beta: How much weight to sparsity vs reconstruction

    def forward(self, latent_a, latent_b, labels):
        contrastive_loss = self.contrastive_loss(latent_a, latent_b, labels)
        sparsity_loss_a = self.sparsity_loss(latent_a, self.rho, self.beta)
        sparsity_loss_b = self.sparsity_loss(latent_b, self.rho, self.beta)
        return contrastive_loss + sparsity_loss_a + sparsity_loss_b

    def contrastive_loss(self, latent_a, latent_b, labels):
        # Calculate the cosine similarity for all pairs in the batch
        sim = nn.functional.cosine_similarity(latent_a, latent_b, dim=1) # Shape: (B,)

        # Convert labels to float for multiplication (0 for False/Dissimilar, 1 for True/Similar)
        labels = labels.float()

        # Loss term for SIMILAR pairs (label = 1):
        # We want similarity (sim) to be close to 1. 
        # This term is active only when labels=1.
        similar_pairs_mask = (labels == 1)
        #loss_similar = labels * torch.pow((1 - sim), 2)
        loss_similar = (1.0 - sim[similar_pairs_mask])

        # Loss term for DISSIMILAR pairs (label = 0):
        # We want similarity (sim) to be far from 1.
        # This term is active only when labels=0.
        #loss_dissimilar = (1 - labels) * torch.pow(torch.relu(sim - self.margin), 2)
        dissimilar_pairs_mask = (labels == 0)
        loss_dissimilar = torch.clamp(sim[dissimilar_pairs_mask] - self.margin, min=0.0)

        # Total loss is the mean of both terms
        #loss = (loss_similar + loss_dissimilar).mean()
        loss = torch.cat((loss_similar, loss_dissimilar)).mean()
        return loss

    def kl_divergence(self, rho, rho_hat):
        """
        Kullback-Leibler divergence to measure deviation from target sparsity.
        rho: Target sparsity probability (e.g., 0.05)
        rho_hat: Average activation of the hidden layer
        """
        rho_hat = torch.clamp(rho_hat, 1e-10, 0.9999) # Avoid log(0) errors
        return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

    def sparsity_loss(self, latent, sparsity_target, sparsity_weight):
        # Sparsity Penalty
        # Calculate average activation of hidden neurons across the batch
        rho_hat = torch.mean(latent, dim=0) 
        sparsity_penalty = torch.sum(self.kl_divergence(sparsity_target, rho_hat))
        
        # Total Loss
        return sparsity_weight * sparsity_penalty

class SupervisedInfoNCELoss(nn.Module):
    """
    Supervised InfoNCE Loss (NT-Xent) adapted for metric learning.
    It takes an entire batch of embeddings and their labels, treating samples 
    from the same class as positives and all others as negatives.
    """
    def __init__(self, temperature=0.07):
        super(SupervisedInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features shape: (N, D) where N=BATCH_SIZE, D=EMBEDDING_DIM
        # labels shape: (N,)

        # 1. Calculate similarity matrix S (N, N)
        # Cosine Similarity is equivalent to Dot Product for L2-normalized features.
        similarity_matrix = torch.matmul(features, features.T)
        
        # 2. Identify positive pairs (where labels match)
        # mask shape: (N, N)
        labels = labels.unsqueeze(0)
        positive_mask = torch.eq(labels, labels.T).float()
        
        # 3. Create a mask to remove self-comparison (i != j)
        # The diagonal elements should not be considered positive or negative
        logits_mask = torch.scatter(
            torch.ones_like(positive_mask), 
            1, 
            torch.arange(features.shape[0]).view(-1, 1).to(features.device),
            0
        )
        positive_mask = positive_mask * logits_mask # Apply self-comparison mask to positive mask

        # 4. Scale logits by temperature
        logits = similarity_matrix / self.temperature
        
        # Apply the logits mask (set self-comparison to a large negative number for stability)
        logits = logits - torch.eye(features.shape[0], device=features.device) * 1e9

        # 5. Compute the Log-Sum-Exp term for the denominator (all other samples in batch)
        # Numerator: Log similarity of positive pairs. Denominator: Sum of exp(similarities).
        
        # Sum of similarities for all samples (the log-sum-exp denominator)
        log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
        
        # Select the log-probability of the positive samples (numerator components)
        # The sum over dim=1 handles multiple positive samples per anchor (supervised InfoNCE)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1)
        
        # Final loss is the negative mean log-probability
        loss = -mean_log_prob_pos
        return loss.mean()