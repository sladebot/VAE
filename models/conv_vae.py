from .vae import VAE, Flatten, Stack
import torch.nn as nn
import pytorch_lightning as pl
import torch
import os
import random
from typing import Optional
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CelebA
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        # print(f"Shape: {x.shape}")
        return x

class UnFlatten(nn.Module):
    def forward(self, input, size=4096):
        # print("Unflatteing")
        return input.view(input.size(0), size, 1, 1)


class Flatten(nn.Module):
    def forward(self, input):
        # print("Flattening")
        return input.view(input.size(0), -1)

class Conv_VAE(pl.LightningModule):
    def __init__(self, channels: int, height: int, width: int, lr: int,
                 latent_size: int, hidden_size: int, alpha: int, batch_size: int,
                 dataset: Optional[str] = None,
                 save_images: Optional[bool] = None,
                 save_path: Optional[str] = None, **kwargs):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        if save_images:
            self.save_path = f'{save_path}/{kwargs["model_type"]}_images/'
        self.save_hyperparameters()
        self.save_images = save_images
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha
        self.dataset = dataset
        assert not height % 4 and not width % 4, "Choose height and width to "\
            "be divisible by 4"
        self.channels = channels
        self.height = height
        self.width = width
        self.latent_size = latent_size
        self.save_hyperparameters()

        self.data_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor()
        ])


        self.encoder = nn.Sequential(
            PrintShape(),
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            PrintShape(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            PrintShape(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            PrintShape(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            PrintShape(),
            Flatten(),
            PrintShape(),
        )

        self.fc1 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, self.hidden_size)

        self.decoder = nn.Sequential(
            PrintShape(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # PrintShape(),
            # nn.BatchNorm1d(self.hidden_size),
            UnFlatten(),
            PrintShape(),
            nn.ConvTranspose2d(self.hidden_size, 256, kernel_size=6, stride=2, padding=1),
            PrintShape(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            PrintShape(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            PrintShape(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            PrintShape(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channels),
            PrintShape(),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu, log_var = self.fc1(hidden), self.fc1(hidden)
        # print("Encoded")
        return mu, log_var

    def decode(self, z):
        # print("Decoding")
        # f = nn.Linear(self.latent_size, self.hidden_size)
        z = self.fc2(z)
        # print(f"L: {z.shape}")
        x = self.decoder(z)
        return x
    
    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(sigma)
        return mu + sigma*z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var, x_out = self.forward(x)
        kl_loss = (-0.5*(1+log_var - mu**2 -
                         torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss

        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.shared_eval(batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch)

    def shared_eval(self, batch):
        x, _ = batch
        mu, log_var, x_out = self.forward(x)

        kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                           torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss * self.alpha + kl_loss
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        # print(x.mean(),x_out.mean())
        return x_out, loss

    def validation_epoch_end(self, outputs):
        if not self.save_images:
            return
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        choice = random.choice(outputs)
        output_sample = choice[0]
        output_sample = output_sample.reshape(-1, 1, self.width, self.height)
        # output_sample = self.scale_image(output_sample)
        save_image(
            output_sample,
            f"{self.save_path}/epoch_{self.current_epoch+1}.png",
            # value_range=(-1, 1)
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        lr_scheduler = ReduceLROnPlateau(optimizer,)
        return {
            "optimizer": optimizer, "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"
        }
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        output = self.decode(hidden)
        return mu, log_var, output

    # Functions for dataloading
    def train_dataloader(self):
        if self.dataset == "mnist":
            train_set = MNIST('data/', download=True,
                              train=True, transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            train_set = FashionMNIST(
                'data/', download=True, train=True,
                transform=self.data_transform)
        elif self.dataset == "celeba":
            train_set = CelebA('data/', download=False, split="train", transform=self.data_transform)
        return DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.dataset == "mnist":
            val_set = MNIST('data/', download=True, train=False,
                            transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            val_set = FashionMNIST(
                'data/', download=True, train=False,
                transform=self.data_transform)
        elif self.dataset == "celeba":
            val_set = CelebA('data/', download=False, split="valid", transform=self.data_transform)
        return DataLoader(val_set, batch_size=self.batch_size)
    
    def test_dataloader(self):
        if self.dataset == "mnist":
            val_set = MNIST('data/', download=True, train=False,
                            transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            val_set = FashionMNIST(
                'data/', download=True, train=False,
                transform=self.data_transform)
        elif self.dataset == "celeba":
            val_set = CelebA('data/', download=False, split="test", transform=self.data_transform)
        return DataLoader(val_set, batch_size=self.batch_size)



