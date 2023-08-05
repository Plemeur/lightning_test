import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import DeviceStatsMonitor


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        print(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), - 1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.view(x.size(0), - 1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Basic load data
train_set = MNIST(
    root="MNIST", download=True, transform=transforms.ToTensor(), train=True
)
tsize = int(len(train_set) * 0.8)
vsize = len(train_set) - tsize
print(f"Test Size {tsize}, Validation Size { vsize}") 

seed = torch.Generator().manual_seed(69)
train_set, valid_set = random_split(train_set, [tsize, vsize], generator=seed)
test_set = MNIST(
    root="MNIST", download=True, transform=transforms.ToTensor(), train=False
)

# Basic torch creation of dataloader.
train_loader = DataLoader(train_set, batch_size=1024)
valid_loader = DataLoader(valid_set, batch_size=1024)
test_loader = DataLoader(test_set, batch_size=1024)

# This part is specific to lightning
autoencoder = LitAutoEncoder(Encoder(), Decoder())


trainer = pl.Trainer(max_epochs=2, callbacks=[DeviceStatsMonitor()])
trainer.fit(
    autoencoder, train_loader, valid_loader
)

trainer.test(model=autoencoder, dataloaders=[test_loader])
