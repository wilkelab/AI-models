# example adapted from: https://lightning.ai/docs/pytorch/stable/starter/introduction.html
import os
import torch
from torch import optim, nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
torch.set_float32_matmul_precision('medium')


# Define a model
class SimpleModel(nn.Module):
    """Simple autoencoder model"""
    def __init__(self, input_dim=28*28, hidden_dim=64, latent_dim=3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Wrap your model in a LightningModule object
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleModel()
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer





if __name__ == "__main__":
    # init the autoencoder
    autoencoder = LitAutoEncoder()

    # setup data
    dataset = MNIST(f'{os.getcwd()}/data/', download=True, transform=ToTensor())
    
    # dataloaders 
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
        )

    val_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
        )

    # trainer object for multi-GPU training
    trainer = pl.Trainer(
        accelerator='gpu',           # Use GPU
        num_nodes=1,
        devices=-1,                   # Use all available GPUs (or specify [0, 1, 2, 3])
        strategy='ddp',              # Distributed Data Parallel strategy
        max_epochs=10,
        logger=False,                # if you want to log, check out the CSVlogger or tensorboard
        enable_checkpointing=False,  # to save or not checkpoints
    )
    
    # Train the model
    trainer.fit(autoencoder, train_loader, val_loader)