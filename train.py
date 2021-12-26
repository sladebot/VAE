import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from model import VAE
from util import get_device

class VAETrainer:
    def __init__(self, cfg, model, optimizer, criterion) -> None:
        self.epochs = cfg.EPOCHS
        self.LOG_INTERVAL = cfg.LOG_INTERVAL
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.device == 'cuda' else {}

        # Setting data & results path
        self.results_path = cfg.results_dir
        train_data = datasets.FashionMNIST('./data', train=True, download=True,
            transform=transforms.ToTensor())

        VAL_SIZE=0.2
        train_indices, val_indices, _, _ = train_test_split(
            range(len(train_data)),
            train_data.targets,
            stratify=train_data.targets,
            test_size=VAL_SIZE,
        )

        train_split = Subset(train_data, train_indices)
        val_split = Subset(train_data, val_indices)


        test_data = datasets.FashionMNIST('./data', train=False,
                            transform=transforms.ToTensor())
        
        

        self.train_loader = DataLoader(train_split, batch_size=cfg.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_split, batch_size=cfg.BATCH_SIZE, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=cfg.BATCH_SIZE, shuffle=True, **kwargs)

    def save_generated_img(self, image, name, epoch, nrow=8):
        if not os.path.exists('results'):
            os.makedirs('results')

        if epoch % 5 == 0:
            save_path = 'results/'+name+'_'+str(epoch)+'.png'
            save_image(image, save_path, nrow=nrow)
    
    def train(self):
        self.model.train()
        train_loss = 0
        for epoch in range(self.epochs):
            for batch_idx, (data, _) in enumerate(self.train_loader):
                # data: [batch size, 1, 28, 28]
                # label: [batch size] -> we don't use
                self.optimizer.zero_grad()
                data = data.to(self.device)
                recon_data, mu, logvar = self.model(data)
                loss = self.criterion(recon_data, data, mu, logvar)
                loss.backward()
                cur_loss = loss.item()
                train_loss += cur_loss
                self.optimizer.step()

                if batch_idx % self.LOG_INTERVAL == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                            100. * batch_idx / len(self.train_loader),
                            cur_loss / len(data)))
                    
                    validation_loss = self.evaluate(self._get_validation_batch())
                    print('Validation Loss: ({})'.format(validation_loss.item()))

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(self.train_loader.dataset)
            ))
    
    def _get_validation_batch(self):
        val_iter = iter(self.val_loader)
        return next(val_iter)
    
    def evaluate(self, batch):
        data, _ = batch
        data = data.to(self.device)
        recon_data, mu, logvar = self.model(data)
        return self.criterion(recon_data, data, mu, logvar)
        
