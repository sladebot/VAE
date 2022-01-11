from .vae import VAE, Flatten, Stack
import torch.nn as nn
import torch
from typing import Optional
import torchvision.transforms as transforms

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

class Conv_VAE(VAE):
    def __init__(self, channels: int, height: int, width: int, lr: int,
                 latent_size: int, hidden_size: int, alpha: int, batch_size: int,
                 dataset: Optional[str] = None,
                 save_images: Optional[bool] = None,
                 save_path: Optional[str] = None, **kwargs):
        super().__init__(latent_size, hidden_size, alpha, lr, batch_size,
                         dataset, save_images, save_path, **kwargs)
        # Our code now will look identical to the VAE class except that the
        # encoder and the decoder have been adjusted
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
            nn.LeakyReLU(),
            PrintShape(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            PrintShape(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            PrintShape(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            PrintShape(),
            Flatten(),
            PrintShape(),
        )

        self.fc1 = nn.Linear(self.hidden_size, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size, self.hidden_size)

        self.decoder = nn.Sequential(
            PrintShape(),
            UnFlatten(),
            nn.ConvTranspose2d(self.hidden_size, 256, kernel_size=6, stride=2, padding=1),
            PrintShape(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            PrintShape(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            PrintShape(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            PrintShape(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=4, stride=2, padding=1),
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
        f = nn.Linear(self.latent_size, self.hidden_size)
        z = f(z)
        # print(f"L: {z.shape}")
        x = self.decoder(z)
        return x



