#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: michael
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import librosa
from torch.utils import data

path = "/media/michael/DATA/Downloads/ADOS/ADOS1002_clips/"

num_epochs = 100
batch_size = 64
learning_rate = 1e-3

counter = 0
specs = list()
for file in os.listdir(path):
    y, sr = librosa.core.load(path + str(file))
    mag = np.abs(librosa.core.stft(y, 1024, int(.75 * 2048), 1024))
    if mag.shape[1] != 29:
        continue
    mag = mag.reshape(1,513,29)
    specs.append(mag)
    del mag
    print(counter)
    counter += 1
    
dataSet = data.DataLoader(specs, batch_size, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DCVAE(nn.Module):
    def __init__(self, num_latent):
        super(DCVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, (1), 1, 0)
        self.norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, (3,1), 2, 0)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, (3,1), 2, 0)
        self.norm3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(65024, 4064)
        self.fc2 = nn.Linear(4064, 1016)
        self.fc3  = nn.Linear(1016, 64)
        self.latent = nn.Linear(64, num_latent)
        self.mean = nn.Linear(64, num_latent)
        self.var = nn.Linear(64, num_latent)
        self.fc4 = nn.Linear(num_latent, 64)
        
        self.deconv1 = nn.ConvTranspose2d(64, 128, (3,1))
        self.deconv2 = nn.ConvTranspose2d(128, 256, (3,1))
        self.deconv3 = nn.ConvTranspose2d(256, 513, (25,1))
        
    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.conv1(x)))
        x = F.leaky_relu(self.norm2(self.conv2(x)))
        x = F.leaky_relu(self.norm3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        logvar = self.var(x)
        x = x.view(x.size(0),64, 1, 1)
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        x = F.sigmoid((self.deconv3(x)))
        return x, mean, logvar
        
model = DCVAE(32).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

### getting negative loss because the spectograms are not normalized
#Tried normalizing but encountered errors and I could not solve it
#Maybe you can try?
#Or maybe an error on my part with the loss calculation
def lossfn(x, target, mean, logvar):
    bce = nn.BCELoss()
    bce_loss = bce(x, target)
    
    scaling_factor = out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]
    
    kl_loss = -.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
    kl_loss /= scaling_factor
    
    return bce_loss + kl_loss

for epoch in range(num_epochs):
    print("Epoch #" + str(epoch))
    for idx, (spec) in enumerate(dataSet):
        optimizer.zero_grad()
        out, mean, logvar = model(spec.cuda())
        if out.shape[0] == batch_size:
            out = out.reshape(batch_size,1,513,29)
        else:
            out = out.reshape(out.shape[0],1,513,29)
        loss = lossfn(out, spec.cuda(), mean, logvar)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    print(loss)

model = model.eval()
recon, _, _ = model(torch.cuda.FloatTensor(specs[1].reshape(1,1,513,29)))
recon= recon.cpu()
recon = recon.detach().numpy()
recon = recon.reshape(1,513,29)
