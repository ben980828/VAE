import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import imageio
import scipy.misc
import argparse
import glob
import os
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm
from collections import OrderedDict



class Face_Image(Dataset):
    def __init__(self, fileroot, image_root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.fileroot = fileroot
        self.image_root = image_root
        self.transform = transform

        # read filenames
        self.len = len(self.image_root)       
    def __getitem__(self, index):
        """ Get a sample from the dataset """

        image_fn = self.image_root[index]
        image = Image.open(image_fn).convert('RGB')
        image = self.transform(image) 

        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, in_channel=3, lat_channel=1024, z_channel=32):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(lat_channel, z_channel)
        self.fc2 = nn.Linear(lat_channel, z_channel)
        self.fc3 = nn.Linear(z_channel, lat_channel)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(lat_channel, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channel, kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    def making_latent(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return (mu + eps*std)

    def bottle_neck(self, enc_output):
        mu = self.fc1(enc_output)
        logvar = self.fc2(enc_output)
        z = self.making_latent(mu, logvar)
        return z, mu, logvar

    def forward(self, input, mode='test'):
        enc_out = self.encoder(input)
        z, mu, logvar = self.bottle_neck(enc_out)
        z_latchannel = self.fc3(z)
        output = self.decoder(z_latchannel)
        return output, mu, logvar

def main():
    train_root = 'hw3-ben980828/hw3_data/face/train/'
    train_img = []

    train_list = os.listdir(train_root)
    for fn in train_list:
        train_img.append(os.path.join(train_root, fn))
    train_set = Face_Image(fileroot=train_root, 
        image_root=train_img, 
        transform=transforms.ToTensor())
        
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1)


    model = VAE()
    model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max')

    def loss_function(output_img, input_img, mu, logvar, eps=1e-5):
        MSE = F.mse_loss(output_img, input_img)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (MSE + eps*KLD), MSE, KLD


    epoch = 50
    eps = 1e-5
    iteration = 0
    min_loss = 25
    mse_loss_list = []
    KLD_loss_list = []
    iter_list = []
    # training
    for ep in range(1, epoch+1):
        model.train()
        train_loss = 0
        ep_iter = 0
        print('Current training epoch : ', ep)
        train_bar = tqdm(total=len(train_loader), leave=False)
        for data in train_loader:

            data = data.cuda()
            optimizer.zero_grad()
            output, mu, logvar = model(data)
            loss, mse_loss, KLD_loss = loss_function(output, data, mu, logvar, eps)
            loss.backward()
            optimizer.step()
            iteration += 1
            ep_iter += 1
            iter_list.append(iteration)
            
            train_loss += loss.item()
            mse_loss_list.append(mse_loss)
            KLD_loss_list.append(KLD_loss)
            mean_loss = train_loss/ep_iter

            postfix = OrderedDict([
              ('train_loss', mean_loss),
            ])
            train_bar.set_postfix(postfix)
            train_bar.update(1)
        train_bar.close()
        total_mean_loss = train_loss/ep_iter
        if total_mean_loss < min_loss:
            print('Performance improved : ({:.3f} --> {:.3f}). Save model ==> '.format(min_loss, total_mean_loss))
            min_loss = total_mean_loss
            torch.save(model.state_dict(), 'vae_optimal.pth')
        if (iteration >= 20000) and (min_loss >= 0.05):
            print("\nShitty model, try again.")
            break
        #lr_decay.step(mean_loss)
    print('Final min mean_loss : ', min_loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))

    ax1.plot(iter_list, mse_loss_list)
    ax1.set_title('MSE Loss')
    ax1.set(xlabel="iteration", ylabel="Loss Value")


    ax2.plot(iter_list, KLD_loss_list)
    ax2.set_title('KLD Loss')
    ax2.set(xlabel="iteration", ylabel="Loss Value")

    plt.savefig('Loss_Curve.png')
    plt.show()

if __name__ == '__main__':
    main()
