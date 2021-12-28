import pytorch_lightning as pl
import torch.nn as nn
from typing import Optional
import torch

from torchvision.datasets import MNIST, FashionMNIST, CelebA
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Conv_VAE(pl.LightningModule):
    def __init__(self, channels: int, height: int, width: int, lr: int,
                 hidden_size: int, alpha: int, batch_size: int,
                 dataset: Optional[str] = None,
                 save_images: Optional[bool] = None,
                 save_path: Optional[str] = None, **kwargs):
        super().__init__(hidden_size, alpha, lr, batch_size,
                         dataset, save_images, save_path, **kwargs)
        # Our code now will look identical to the VAE class except that the
        # encoder and the decoder have been adjusted
        assert not height % 4 and not width % 4, "Choose height and width to "\
            "be divisible by 4"
        self.channels = channels
        self.height = height
        self.width = width
        self.save_hyperparameters()
        final_height = (self.height//4-3)//2+1
        final_width = (self.width//4-3)//2+1
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 8, 3, padding=1), nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # 128*3*3
            Flatten(),
            nn.Linear(128*final_height*final_width,
                      32*final_height*final_width),
            nn.LeakyReLU(), nn.BatchNorm1d(32*final_height*final_width),
            nn.Linear(32*final_height*final_width, self.hidden_size),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 32*final_height * final_width),
            nn.BatchNorm1d(32*final_height * final_width), nn.ReLU(),
            nn.Linear(32*final_height*final_width,
                      128*final_height*final_width),
            nn.BatchNorm1d(128*final_height * final_width), nn.ReLU(),
            Stack(128, 3, 3),
            nn.ConvTranspose2d(128, 64, 3, 2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.ConvTranspose2d(
                32, 16, 2, 2), nn.BatchNorm2d(16), nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 2, 2), nn.BatchNorm2d(8),
            nn.Conv2d(8, self.channels, 3, padding=1), nn.Tanh()
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def forward(self, x):
        mu, log_var = self.encode(x)
        hidden = self.reparameterize(mu, log_var)
        output = self.decoder(hidden)
        return mu, log_var, output

    def training_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var, x_out = self.forward(x)
        kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                           torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        loss = recon_loss * self.alpha + kl_loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var, x_out = self.forward(x)
        kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                           torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        loss = recon_loss * self.alpha + kl_loss
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return x_out, loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var, x_out = self.forward(x)
        kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                           torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        loss = recon_loss * self.alpha + kl_loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return x_out, loss

    def validation_epoch_end(self, outputs):
        if not self.save_images:
            return
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        choice = random.choice(outputs)
        output_sample = choice[0].reshape(-1, 1, 28, 28)
        save_image(
            output_sample,
            f"{self.save_path}/epoch_{self.current_epoch + 1}.png"
        )

    def train_dataloader(self):
        if self.dataset == "mnist":
            train_set = MNIST("data/", download=True, train=True, transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            train_set = FashionMNIST('data/', download=True, train=True,
                                     transform=self.data_transform)
        elif self.dataset == "celeba":
            train_set = CelebA('data/', download=False, split="train", transform=self.celeba_transform)

        return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        if self.dataset == "mnist":
            val_set = MNIST('data/', download=True, train=False,
                            transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            val_set = FashionMNIST(
                'data/', download=True, train=False,
                transform=self.data_transform)
        elif self.dataset == "celeba":
            val_set = CelebA('data/', download=False, split="valid", transform=self.celeba_transform)
        return DataLoader(val_set, batch_size=64, num_workers=4)

    def test_dataloader(self):
        if self.dataset == "mnist":
            test_set = MNIST("data/", download=True, train=False, transform=self.data_transform)
        elif self.dataset == "fashion-mnist":
            test_set = FashionMNIST('data/', download=True, train=False,
                                    transform=self.data_transform)
        elif self.dataset == "celeba":
            test_set = CelebA('data/', download=False, train=False, transform=self.celeba_transform)

        return DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        lr_scheduler = ReduceLROnPlateau(optimizer, )
        return {
            "optimizer": optimizer, "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"
        }

    def interpolate(self, x1, x2):
        assert x1.shape == x2.shape, "Inputs must be of same shape"

        if x1.dim() == 3:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)

        if self.training:
            raise Exception("This function should not be called when model is still "
                            "in training mode. Use model.eval() before calling the "
                            "function")

        mu1, lv1 = self.encode(x1)
        mu2, lv2 = self.encode(x2)
        z1 = self.reparametrize(mu1, lv1)
        z2 = self.reparametrize(mu2, lv2)
        weights = torch.arange(0.1, 0.9, 0.1)
        intermediate = [self.decode(z1)]
        for wt in weights:
            inter = (1. - wt) * z1 + wt * z2
            intermediate.append(self.decode(inter))
        intermediate.append(self.decode(z2))
        out = torch.stack(intermediate, dim=0).squeeze(1)
        return out, (mu1, lv1), (mu2, lv2)