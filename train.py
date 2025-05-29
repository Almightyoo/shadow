import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from model import NeuralNetwork
from dataclasses import dataclass
import pickle

@dataclass
class ModelConfig:
    n_channels = 22
    n_filters = 256
    n_BLOCKS = 5
    SE_channels = 32
    policy_channels = 76



class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.samples = []
        
        for file in self.data_files:
            data = torch.load(file)
            num_samples = len(data['values'])
            self.samples.extend([(file, i) for i in range(num_samples)])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file, sample_idx = self.samples[idx]
        data = torch.load(file)
        return {
            'state': data['states'][sample_idx],
            'policy': data['policies'][sample_idx],
            'value': data['values'][sample_idx]
        }
def load_checkpoint(checkpoint_path, device):
    """Safely load checkpoint with weights_only=False for custom classes"""
    try:
        # First try with weights_only=True (secure mode)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        # Fall back to weights_only=False if needed (only if you trust the source)
        print("Warning: Falling back to weights_only=False - only do this if you trust the checkpoint source")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def train_model(config, data_dir="selfplay_data", checkpoint_dir="checkpoints", 
                resume_checkpoint=None, batch_size=32, num_epochs=10):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = NeuralNetwork(config).to(device)
    
    # Load checkpoint if specified
    start_epoch = 0
    if resume_checkpoint:
        checkpoint = load_checkpoint(resume_checkpoint, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # Rest of your training code remains the same...
    dataset = ChessDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            states = batch['state'].to(device, non_blocking=True)
            policy_targets = batch['policy'].to(device, non_blocking=True)
            value_targets = batch['value'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            value_pred, policy_pred, loss, policy_loss, value_loss = model(
                states,
                targets={
                    'policy': policy_targets,
                    'value': value_targets
                }
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # Save checkpoint (using safe serialization)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, checkpoint_path, _use_new_zipfile_serialization=True)
        
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    config = ModelConfig()
    
    # Example usage:
    # 1. To train from scratch:
    # train_model(config, data_dir="selfplay_data", checkpoint_dir="checkpoints")
    
    # 2. To resume training from a checkpoint:
    train_model(
        config,
        data_dir="selfplay_data",
        checkpoint_dir="checkpoints",
        resume_checkpoint="checkpoints/checkpoint_epoch_20_20250421_211246.pt",
        batch_size=64,
        num_epochs=20
    )