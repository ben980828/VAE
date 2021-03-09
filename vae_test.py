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
import sys
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torchvision.utils as vutils

torch.manual_seed(0)


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
        return input.view(32, size, 1, 1)


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

    def bottle_neck(self):
        mu = torch.FloatTensor(32, 32).fill_(0.0).cuda()
        logvar = torch.FloatTensor(32, 32).fill_(0.0).cuda()
        z = self.making_latent(mu, logvar)
        return z, mu, logvar

    def forward(self, input, mode='test'):
        z, mu, logvar = self.bottle_neck()
        z_latchannel = self.fc3(z)
        output = self.decoder(z_latchannel)
        return output, mu, logvar


def main():
    # pyfile = sys.argv[0]
    # output_file = sys.argv[1]
    output_file = './VAE_generate_img.png'
    model = VAE()
    model = model.cuda()
    print(model)
    generate_img = []

    state = torch.load('vae_optimal.pth')
    model.load_state_dict(state)
    with torch.no_grad():
        model.eval()
        output_img, _, _ = model(input=0, mode='test')
        output_img = output_img.cpu().detach()
        generate_img.append(vutils.make_grid(output_img, padding=2, normalize=True))

    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.savefig(output_file)


if __name__ == '__main__':
    main()

