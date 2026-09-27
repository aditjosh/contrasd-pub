import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from .models import EmbeddingsAutoencoderService, EmbeddingsAutoencoder, ContrastiveEmbeddingsTransformer, ContrastiveClusterLearner
from .datasets import setup_contrastive_dataset, MilvusDataset, fetch_subset, extract_from_dataloader
from experiments.synthetic_datasets import generate_synthetic_data, setup_synthetic_dataloaders
from .loss import LatentPairwiseContrastiveLoss

def train_autoencoder_prepared_data(embed_trasformer, milvus_dataset, input_dim=4096, latent_dim=128, 
                            device = "cpu", batch_size=32, epochs=10, learning_rate=0.001):
    DEVICE = device
    FEATURE_DIM = input_dim
    LATENT_DIM = latent_dim
    VECTOR_FIELD_NAME = 'sense_embedding' 
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LEARNING_RATE = learning_rate

    # Dataset
    train_dataset_a_dataloader, train_dataset_b_dataloader = fetch_subset(milvus_dataset, 'train')
    words_a_train, embeddings_a_train, labels_a_train = extract_from_dataloader(train_dataset_a_dataloader)
    words_b_train, embeddings_b_train, labels_b_train = extract_from_dataloader(train_dataset_b_dataloader)

    embeddings_train = torch.vstack((embeddings_a_train, embeddings_b_train))
    train_dataset = TensorDataset(embeddings_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ## Training function for basic applications such as dimensionality reduction
    #model = SenseAutoencoder(feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    model = EmbeddingsAutoencoder(embed_transformer=embed_trasformer, feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader):
            #inputs = torch.tensor(batch[VECTOR_FIELD_NAME])
            #inputs = inputs.to(DEVICE)
            inputs = batch[0]      # TensorDataset itself returns a tuple or a list of items for each sample
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model, train_dataloader

def train_contrastive_embeddings_transformer_prepared_data(contrastive_train_loader, input_dim=32, 
                        embedding_dim = 8, device = "cpu", epochs=10, learning_rate=0.001, margin=0.5):
    DEVICE = device
    FEATURE_DIM = input_dim
    EMBEDDING_DIM = embedding_dim
    BATCH_SIZE = 32
    EPOCHS = epochs
    TEMPERATURE = 0.07
    LEARNING_RATE = learning_rate
    MARGIN = margin

    model = ContrastiveEmbeddingsTransformer(FEATURE_DIM, EMBEDDING_DIM).to(DEVICE)
    
    criterion = LatentPairwiseContrastiveLoss(margin=MARGIN)
    #criterion = SupervisedInfoNCELoss(temperature=TEMPERATURE)
    #criterion = LatentPairwiseContrastiveSAELoss(margin=0.5,target_sparsity=0.5,sparsity_wight=0.05)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Contrastive Model Training (Optimizing for Cosine Distance) ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for data1, data2, label in contrastive_train_loader:
            data1, data2, label = data1.to(DEVICE), data2.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            emb1 = model(data1)
            emb2 = model(data2)
            loss = criterion(emb1, emb2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(contrastive_train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Contrastive Loss: {avg_loss:.4f}")

    print("--- Training Complete ---")
    return model


def majority_vote_relabeling(y_true, y_pred):
    """
    Remaps predicted cluster labels (y_pred) to match true category labels (y_true)
    based on the majority true class found within each cluster.
    
    This function is specifically designed for a binary (two-class) problem where 
    true labels are 0 and 1.
    """
    # Ensure inputs are NumPy arrays of integer type
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    # Get the unique cluster labels assigned by K-Means (should be 0 and 1)
    cluster_labels = np.unique(y_pred)
    
    # Initialize the mapping dictionary: {predicted_cluster_id: true_category_id}
    mapping = {}
    
    #for cluster_id in cluster_labels:
    # 1. Isolate the true labels for all points in the current cluster
    # As there are only 2 clusters, let's check the first cluster
    cluster_id = cluster_labels[0]
    true_labels_in_cluster = y_true[y_pred == cluster_id]
    
    if len(true_labels_in_cluster) == 0:
        # Handle empty clusters by mapping them arbitrarily (e.g., to 0)
        mapping[cluster_id] = 0
        #continue
        
    # 2. Find the frequency of true labels (0s and 1s)
    # Use bincount, which counts occurrences of non-negative integers
    counts = np.bincount(true_labels_in_cluster)
    
    # 3. Determine the majority true category
    # The index of the largest count is the majority label (0 or 1)
    majority_label = np.argmax(counts)
    
    # Store the mapping
    mapping[cluster_id] = majority_label

    # Assign the opposite to the other cluster
    mapping[cluster_labels[1]] = 1-majority_label
    
    
    # 4. Apply the mapping to the entire y_pred array
    y_relabeled = np.array([mapping[pred_label] for pred_label in y_pred])
    
    return y_relabeled, mapping

def majority_vote_relabeling_full(y_true, y_pred):
    """
    Remaps predicted cluster labels (y_pred) to match true category labels (y_true)
    based on the majority true class found within each cluster.
    
    This function is specifically designed for a binary (two-class) problem where 
    true labels are 0 and 1.
    """
    # Ensure inputs are NumPy arrays of integer type
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    # Get the unique cluster labels assigned by K-Means (should be 0 and 1)
    cluster_labels = np.unique(y_pred)
    
    # Initialize the mapping dictionary: {predicted_cluster_id: true_category_id}
    mapping = {}
    
    for cluster_id in cluster_labels:
        # 1. Isolate the true labels for all points in the current cluster
        true_labels_in_cluster = y_true[y_pred == cluster_id]
        
        if len(true_labels_in_cluster) == 0:
            # Handle empty clusters by mapping them arbitrarily (e.g., to 0)
            mapping[cluster_id] = 0
            continue
            
        # 2. Find the frequency of true labels (0s and 1s)
        # Use bincount, which counts occurrences of non-negative integers
        counts = np.bincount(true_labels_in_cluster)
        
        # 3. Determine the majority true category
        # The index of the largest count is the majority label (0 or 1)
        majority_label = np.argmax(counts)
        
        # Store the mapping
        mapping[cluster_id] = majority_label
    
    # 4. Apply the mapping to the entire y_pred array
    y_relabeled = np.array([mapping[pred_label] for pred_label in y_pred])
    
    return y_relabeled, mapping


def main():
    print(f"No implementation")

if __name__=="__main__":
    main()