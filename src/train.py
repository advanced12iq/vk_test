import torch
import torch.optim as optim
from tqdm import tqdm
from models.loss import contrastive_loss
from src import config 


def train_siamese_network(model, train_loader, device=config.DEVICE):
    """
    Args:
        model (torch.nn.Module)
        train_loader (torch.utils.data.DataLoader)
        device (str)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    num_epochs = config.NUM_EPOCHS
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (crop_images, logo_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            
            batch_size = crop_images.shape[0]
            loss = 0
            for i in range(batch_size):
                crop, logo = crop_images[i].to(device), logo_images[i].to(device)
                output1, output2 = model(crop.unsqueeze(0), logo.unsqueeze(0))
                loss += contrastive_loss(output1, output2, torch.tensor([0.], device=device))
                
                neg_idx = (i+1) % batch_size
                neg_logo = logo_images[neg_idx].to(device)
                output1_neg, output2_neg = model(crop.unsqueeze(0), neg_logo.unsqueeze(0))
                loss += contrastive_loss(output1_neg, output2_neg, torch.tensor([1.], device=device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / (len(train_loader) * batch_size * 2)
        print(f"Epoch {epoch+1}, Avg loss: {avg_loss:.4f}")