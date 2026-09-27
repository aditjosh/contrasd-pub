import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model.models import EmbeddingsAutoencoderService, EmbeddingsAutoencoder, ContrastiveEmbeddingsTransformer, ContrastiveClusterLearner
from .eval_models import ContrastiveClusterLearnerExt, ContrastiveEmbeddingsTransformerE1, AutoencoderE1, AutoencoderE2, EmbeddingsAutoencoderServiceExt, ContrastiveClusterLearnerExt
from .synthetic_datasets import generate_synthetic_data, setup_synthetic_dataloaders
from model.loss import LatentPairwiseContrastiveLoss
from model.training import majority_vote_relabeling


def train_autoencoder_e1_synthetic_data(train_loader, input_dim=128, latent_dim=32, device = "cpu", epochs=10):
    DEVICE = device
    FEATURE_DIM = input_dim
    LATENT_DIM = latent_dim
    EPOCHS = epochs
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    model = AutoencoderE1(feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop 
    print(f"--- Starting Training on {DEVICE} ---")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels in train_loader:
            # Input data is the target for an autoencoder (X = Y)
            inputs = inputs.to(DEVICE) 
            
            # Forward pass
            outputs = model(inputs)
            
            # Loss calculation (output vs. original data)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    print("\nTraining Complete.")
    return model

def train_autoencoder_e2_synthetic_data(train_loader, embed_transformer, input_dim=128, latent_dim=32, device = "cpu", epochs=10):
    DEVICE = device
    FEATURE_DIM = input_dim
    LATENT_DIM = latent_dim
    EPOCHS = epochs
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    model = AutoencoderE2(embed_transformer=embed_transformer, feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop 
    print(f"--- Starting Training on {DEVICE} ---")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels in train_loader:
            # Input data is the target for an autoencoder (X = Y)
            inputs = inputs.to(DEVICE) 
            
            # Forward pass
            outputs = model(inputs)
            
            # Loss calculation (output vs. original data)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    print("\nTraining Complete.")
    return model

def train_contrastive_embeddings_transformer_with_synthetic_data(contrastive_train_loader, input_dim=32, latent_dim = 8, batch_size=32, device = "cpu", epochs=10):
    DEVICE = device
    FEATURE_DIM = input_dim
    LATENT_DIM = latent_dim
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    TEMPERATURE = 0.07
    LEARNING_RATE = 0.001
    MARGIN = 0.5

    model = ContrastiveEmbeddingsTransformerE1(input_dim=FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    
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


########################################################################## ContraSD

def extract_embeddings_synthetic(model, data_loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            #X_batch = F.normalize(X_batch, p=2, dim=1)
            emb = model.encoder(X_batch)
            embeddings.append(emb)
            labels.append(y_batch)
    return torch.vstack(embeddings), torch.concatenate(labels, axis=0)

# ContraSD
def conformal_prediction_contrastive_with_synthetic_data(ae_model, embed_transformer_service, train_loader, calib_loader, val_loader, device, alpha=0.1, print_stats=False):
    ########################### Step 1: Extract L2-Normalized Embeddings from the Autoencoder
    train_emb, train_labels_extracted = extract_embeddings_synthetic(ae_model, train_loader, device)
    calib_emb, calib_labels_extracted = extract_embeddings_synthetic(ae_model, calib_loader, device)
    val_emb, val_labels = extract_embeddings_synthetic(ae_model, val_loader, device)
    combined_emb = torch.vstack((train_emb, calib_emb, val_emb))
    combined_labels = torch.concatenate((train_labels_extracted, calib_labels_extracted, val_labels), axis=0)

    len_train_data = train_emb.shape[0]
    len_calib_data = calib_emb.shape[0]
    len_val_data = val_emb.shape[0]
    assert(len_train_data+len_calib_data+len_val_data == combined_emb.shape[0])

    ccl_model = ContrastiveClusterLearnerExt(embed_transformer=embed_transformer_service, device=device)

    list_q_hat, list_emp_marginal_coverage, list_avg_pred_set_size, list_set_sizes_per_class, list_cc_per_class = [], [], [], [], []
    N_ITER = 10
    for i in range(0, N_ITER):
        ############################# Step 2: Clustering model
        ## Find clusters
        K=2
        clusters_train, centroids_train = ccl_model.cosine_kmeans(train_emb, K)
        clusters_train = clusters_train.detach().cpu()
        # Label matching
        clusters_train, clusters_labels_mapping = majority_vote_relabeling(y_true=train_labels_extracted, y_pred=clusters_train)
        
        # Predict centers
        cluster_labels_valid = ccl_model.predict_clusters(val_emb, centroids_train)
        cluster_labels_valid = cluster_labels_valid.detach().cpu()
        clusters_valid_relabeled = np.vectorize(clusters_labels_mapping.get)(cluster_labels_valid)
    
        
        clusters_calib = ccl_model.predict_clusters(calib_emb, centroids_train)
        clusters_calib = clusters_calib.detach().cpu()
        clusters_calib_relabeled = []
        for cluster_id in clusters_calib:
            cluster_id = cluster_id.item()
            if cluster_id == clusters_labels_mapping[cluster_id]:
                # cluster ids map, do nothing
                continue
            else:
                #print('Relabling calib set')
                cluster_id = clusters_labels_mapping[cluster_id]
                clusters_calib_relabeled.append(cluster_id)
        clusters_calib_relabeled = np.array(clusters_calib_relabeled)
    
        if print_stats:
            print(f"Relabled clusters for valid set: {np.unique(clusters_valid_relabeled)}")
            print(f"Relabled clusters for calib set: {np.unique(clusters_calib_relabeled)}")
        
        # Predicted y_hat
        clusters = np.concat((clusters_train, clusters_valid_relabeled, clusters_calib_relabeled))
    
        ## Transform the emeddings
        train_calib_val_emb = torch.vstack((train_emb, calib_emb, val_emb))
    
        ## Predicted labels
        pred_calib_labels = torch.tensor(clusters_calib_relabeled)
        pred_val_labels = torch.tensor(clusters_valid_relabeled)

        if len(pred_calib_labels) < 1:
            print('Insufficient predicted labels, skipping this iteration')
            continue
        
        ############################### Step 3: Conformal Calibration
        # Compute LOO Conformity Scores (Alpha Values) for Calibration Set
        # This calculation ensures no 0 distance scores are included.
        calib_alphas = ccl_model.calculate_calibration_conformity_scores_loo(calib_emb, pred_calib_labels)
    
        # Calculate Quantile q_hat with Monte Carlo (Bootstrap) Sampling
        CONFIDENCE_LEVEL = 1 - alpha
        n_bootstrap_samples = 500
        q_hat = ccl_model.critical_quantile_mc(calib_alphas, confidence_level=CONFIDENCE_LEVEL, n_bootstrap_samples=n_bootstrap_samples)
        list_q_hat.append(q_hat)
        
        if print_stats:
            print(f"q_hat values: {q_hat}")
    
        # Compute Test Set Conformity Scores
        # Distance from test point to the nearest calibration point in each class.
        all_test_scores = ccl_model.calculate_test_conformity_scores(calib_emb, pred_calib_labels, val_emb)
        #print(all_test_scores)
        #print(test_labels)
    
        # 5. Determine Prediction Sets
        try:
            prediction_sets, coverage_count, conditional_coverage_total, conditional_coverage_correct = ccl_model.compute_conditional_coverage(q_hat, all_test_scores, pred_val_labels)
        except KeyError:
            print('Key error, continue to next iter')
            continue
            
        # --- 3.4. Evaluation ---
        total_sets = len(prediction_sets)
        coverage = coverage_count / total_sets
        avg_set_size = np.mean([len(s) for s in prediction_sets])
        set_sizes = [len(s) for s in prediction_sets]


        list_emp_marginal_coverage.append(coverage)
        list_avg_pred_set_size.append(avg_set_size)
        #list_set_sizes.append(set_sizes)
        
        if print_stats:
            print("\n--- Conformal Prediction Results ---")
            print(f"Target Confidence (1-alpha): {CONFIDENCE_LEVEL:.2f}")
            print(f"Empirical Marginal Coverage: {coverage:.4f}")
            print(f"Average Prediction Set Size: {avg_set_size:.3f}")
    
        counts = {}
        for item in set_sizes:
            counts[item] = counts.get(item, 0) + 1
        list_set_sizes_per_class.append(counts)
        if print_stats:
            print(f"Prediction Set Sizes Per Class: {counts}")
            print(f"Proportion of Set Size 1: {counts[1]/sum(counts.values())}")
        
        print("\n--- Empirical Conditional Coverage ---")
        # This shows the actual coverage achieved for each specific class
        
        cc_per_class = {}
        for c in sorted(conditional_coverage_total.keys()):
            total = conditional_coverage_total[c]
            correct = conditional_coverage_correct[c]
            if total > 0:
                cond_cov = correct / total
                cc_per_class[c] = cond_cov
                if print_stats:
                    print(f"  Class {c} Coverage (n={total}): {cond_cov:.4f}")
                    
            else:
                if print_stats:
                    print(f"  Class {c} Coverage (n=0): N/A (No test samples for this class)")
        list_cc_per_class.append(cc_per_class)
        
        # Example visualization of the first 5 predictions
        if print_stats:
            print("\nValidation Predictions (True Label | Prediction Set):")
            for i in range(5):
                print(f"  {pred_val_labels[i]} | {prediction_sets[i]}")

    res_metrics = (list_q_hat, list_emp_marginal_coverage, list_avg_pred_set_size, list_set_sizes_per_class, list_cc_per_class)

    return ccl_model, prediction_sets, train_emb, calib_emb, val_emb, train_labels_extracted, pred_calib_labels, pred_val_labels, res_metrics


