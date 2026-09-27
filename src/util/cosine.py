import numpy as np

def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two embedding vectors.
    Similarity (S) = (e1 . e2) / (||e1|| * ||e2||)
    """
    # Ensure vectors are non-zero to avoid division by zero
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)

    if norm_e1 == 0 or norm_e2 == 0:
        # If one vector is zero, similarity is undefined, treat as 0
        return 0.0

    return np.dot(e1, e2) / (norm_e1 * norm_e2)

def contrastive_cosine_sim_loss(
    similarity: float, 
    label: int, 
    margin: float = 0.5
) -> float:
    """
    Calculates the Contrastive Loss based on Cosine Similarity.

    The loss function combines two terms:
    1. Positive Pair Loss (label = 0, similar): 
       Loss = 1 - Similarity. Drives similarity towards 1.
    2. Negative Pair Loss (label = 1, dissimilar): 
       Loss = max(0, Similarity - margin). Drives similarity below the margin.

    Args:
        embedding_a: First feature vector (N,).
        embedding_b: Second feature vector (N,).
        label: 0 if embeddings are from the same class (similar), 1 otherwise (dissimilar).
        margin: The threshold similarity for dissimilar pairs.

    Returns:
        The calculated scalar loss value.
    """
    # ----------------------------------------------------
    # 1. Positive Pair Loss (Y = 0)
    # Target: Similarity (S) -> 1
    # Loss: 1 - S (Loss is 0 when S=1)
    # ----------------------------------------------------
    if label == 0:
        positive_loss = 1.0 - similarity
        return positive_loss
    
    # ----------------------------------------------------
    # 2. Negative Pair Loss (Y = 1)
    # Target: Similarity (S) -> < margin
    # Loss: max(0, S - margin) (Loss is 0 when S <= margin)
    # ----------------------------------------------------
    elif label == 1:
        # The margin should be a value between -1 and 1.
        negative_loss = max(0.0, similarity - margin)
        return negative_loss
    
    else:
        raise ValueError("Label must be 0 (similar) or 1 (dissimilar).")

def contrastive_cosine_loss(
    embedding_a: np.ndarray, 
    embedding_b: np.ndarray, 
    label: int, 
    margin: float = 0.5
) -> float:
    """
    Calculates the Contrastive Loss based on Cosine Similarity.

    The loss function combines two terms:
    1. Positive Pair Loss (label = 0, similar): 
       Loss = 1 - Similarity. Drives similarity towards 1.
    2. Negative Pair Loss (label = 1, dissimilar): 
       Loss = max(0, Similarity - margin). Drives similarity below the margin.

    Args:
        embedding_a: First feature vector (N,).
        embedding_b: Second feature vector (N,).
        label: 0 if embeddings are from the same class (similar), 1 otherwise (dissimilar).
        margin: The threshold similarity for dissimilar pairs.

    Returns:
        The calculated scalar loss value.
    """
    
    # Calculate the Cosine Similarity (S)
    similarity = cosine_similarity(embedding_a, embedding_b)
    
    # ----------------------------------------------------
    # 1. Positive Pair Loss (Y = 0)
    # Target: Similarity (S) -> 1
    # Loss: 1 - S (Loss is 0 when S=1)
    # ----------------------------------------------------
    if label == 0:
        positive_loss = 1.0 - similarity
        return positive_loss
    
    # ----------------------------------------------------
    # 2. Negative Pair Loss (Y = 1)
    # Target: Similarity (S) -> < margin
    # Loss: max(0, S - margin) (Loss is 0 when S <= margin)
    # ----------------------------------------------------
    elif label == 1:
        # The margin should be a value between -1 and 1.
        negative_loss = max(0.0, similarity - margin)
        return negative_loss
    
    else:
        raise ValueError("Label must be 0 (similar) or 1 (dissimilar).")