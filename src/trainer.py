import torch
import torch.optim as optim
from tqdm import tqdm
from src import (
    config,
    contrastive_loss
)
from sklearn.metrics import f1_score
import torch.nn.functional as F


def train_siamese_network(model, 
                          train_loader, 
                          val_loader, 
                          lr=config.LEARNING_RATE,
                          batch_size=config.BATCH_SIZE,
                          n_epochs=config.NUM_EPOCHS, 
                          margin=config.MARGIN,
                          device=config.DEVICE):

    """
    Trains a Siamese network using vectorized operations to compute loss for the entire batch at once.
    Args:
        model (torch.nn.Module): The Siamese network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (str): Device to train on ('cpu' or 'cuda').
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss, total_pairs = 0.0, 0
        all_preds, all_labels = [], []
        
        for crops, logos in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            crops, logos = crops.to(device), logos.to(device)
            batch_size = crops.size(0)
            
            neg_logos = torch.roll(logos, shifts=1, dims=0)
            
            inputs1 = torch.cat([crops, crops], dim=0)
            inputs2 = torch.cat([logos, neg_logos], dim=0)
            labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)], dim=0).to(device)
            
            emb1, emb2 = model(inputs1, inputs2)
            
            loss = contrastive_loss(emb1, emb2, labels, margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            distances = F.cosine_similarity(emb1, emb2, dim=1)
            preds = (distances > 0.).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            train_loss += loss.item() * 2 * batch_size  # loss is sum, multiply by pairs
            total_pairs += 2 * batch_size
        
        avg_train_loss = train_loss / total_pairs
        train_f1 = f1_score(all_labels, all_preds, average='binary')
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")

        model.eval()
        val_loss, val_pairs = 0.0, 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for crops, logos in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                crops, logos = crops.to(device), logos.to(device)
                batch_size = crops.size(0)
                
                neg_logos = torch.roll(logos, shifts=1, dims=0)
                
                inputs1 = torch.cat([crops, crops], dim=0)
                inputs2 = torch.cat([logos, neg_logos], dim=0)
                labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)], dim=0).to(device)
                
                emb1, emb2 = model(inputs1, inputs2)
                
                loss = contrastive_loss(emb1, emb2, labels, margin)
                distances = F.cosine_similarity(emb1, emb2, dim=1)
                preds = (distances > 0.).long().cpu().numpy()
                
                val_loss += loss.item() * 2 * batch_size
                val_pairs += 2 * batch_size
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / val_pairs
        val_f1 = f1_score(val_labels, val_preds, average='binary')
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}\n")

    return model
