import torch
import torch.optim as optim
from tqdm import tqdm
from src import (
    config,
    contrastive_loss
)
from sklearn.metrics import f1_score


def train_siamese_network(model, train_loader, val_loader, device=config.DEVICE):
    """
    Trains a Siamese network, computes average training and validation loss and F1 score.
    Args:
        model (torch.nn.Module): The Siamese network model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        device (str, optional): Device to use for training ("cpu" or "cuda"). Defaults to config.DEVICE.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    num_epochs = config.NUM_EPOCHS
    for epoch in range(num_epochs):
        # Training Phase
        running_loss = 0.0
        all_labels = []
        all_preds = []
        model.train()  # Set the model to train mode
        for batch_idx, (crop_images, logo_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            
            batch_size = crop_images.shape[0]
            loss = 0
            labels = []
            preds = []
            for i in range(batch_size):
                # Positive pair
                crop, logo = crop_images[i].to(device), logo_images[i].to(device)
                output1, output2 = model(crop.unsqueeze(0), logo.unsqueeze(0))
                loss += contrastive_loss(output1, output2, torch.tensor([0.], device=device))
                labels.append(0) # positive pair
                preds.append(1 if torch.sigmoid(torch.sum(torch.abs(output1 - output2))) > 0.5 else 0)
                
                #Generate negative pair from other image in batch
                neg_idx = (i+1) % batch_size
                neg_logo = logo_images[neg_idx].to(device)
                output1_neg, output2_neg = model(crop.unsqueeze(0), neg_logo.unsqueeze(0))
                loss += contrastive_loss(output1_neg, output2_neg, torch.tensor([1.], device=device))
                labels.append(1) # negative pair
                preds.append(1 if torch.sigmoid(torch.sum(torch.abs(output1_neg - output2_neg))) > 0.5 else 0)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            all_labels.extend(labels)
            all_preds.extend(preds)

        avg_loss = running_loss / (len(train_loader) * batch_size * 2)
        f1 = f1_score(all_labels, all_preds, average='binary')
        print(f"Epoch {epoch+1}, Training Avg loss: {avg_loss:.4f}, Training F1 Score: {f1:.4f}")

        # Validation Phase
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation during validation
            for batch_idx, (crop_images, logo_images) in enumerate(tqdm(val_loader, desc=f"Validation {epoch+1}")):
               batch_size = crop_images.shape[0]
               loss = 0
               labels = []
               preds = []
               for i in range(batch_size):
                    # Positive pair
                    crop, logo = crop_images[i].to(device), logo_images[i].to(device)
                    output1, output2 = model(crop.unsqueeze(0), logo.unsqueeze(0))
                    loss += contrastive_loss(output1, output2, torch.tensor([0.], device=device))
                    labels.append(0) # positive pair
                    preds.append(1 if torch.sigmoid(torch.sum(torch.abs(output1 - output2))) > 0.5 else 0)
                    
                    #Generate negative pair from other image in batch
                    neg_idx = (i+1) % batch_size
                    neg_logo = logo_images[neg_idx].to(device)
                    output1_neg, output2_neg = model(crop.unsqueeze(0), neg_logo.unsqueeze(0))
                    loss += contrastive_loss(output1_neg, output2_neg, torch.tensor([1.], device=device))
                    labels.append(1) # negative pair
                    preds.append(1 if torch.sigmoid(torch.sum(torch.abs(output1_neg - output2_neg))) > 0.5 else 0)
               val_loss += loss.item()
               all_val_labels.extend(labels)
               all_val_preds.extend(preds)

        avg_val_loss = val_loss / (len(val_loader) * batch_size * 2)
        f1_val = f1_score(all_val_labels, all_val_preds, average='binary')
        print(f"Epoch {epoch+1}, Validation Avg Loss: {avg_val_loss:.4f}, Validation F1 Score: {f1_val:.4f}")
