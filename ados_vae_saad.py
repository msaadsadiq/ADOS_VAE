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

path = "/home/michael/Downloads/ADOS1002_clips/"

num_epochs = 500
batch_size = 32
learning_rate = 1e-5
hop_size = 2048
n_fft = int(512 / 2)

counter = 0
specs = list()
for file in os.listdir(path):
    y, sr = librosa.core.load(path + str(file))
    mag = np.abs(librosa.core.stft(y, n_fft * 2, int(.75 * hop_size), 512))
    if mag.shape[1] != 29:
        continue
    mag = mag.reshape(1, int(n_fft) + 1, 29)
    specs.append(mag)
    del mag
    print(counter)
    counter += 1
    
dataSet = data.DataLoader(specs, batch_size, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DCVAE(nn.Module):
    def __init__(self, num_latent):
        super(DCVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, n_fft + 1, (257,1), 1)
        self.norm1 = nn.BatchNorm2d(n_fft + 1)
        self.conv2 = nn.Conv2d(n_fft + 1, 128, (1,1), 1)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, (1,1), 1)
        self.norm3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(1856, 928)
        self.fc2 = nn.Linear(928, 512)
        self.fc31  = nn.Linear(512, 100)   #64)
		self.fc32  = nn.Linear(512, 100)

        self.fc4 = nn.Linear(num_latent, 100)
        self.fc5 = nn.Linear(100, 512)
        self.fc6 = nn.Linear(512, 928)
        
        self.deconv1 = nn.ConvTranspose2d(32, 64, (1,1), 1)
        self.norm4 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 128, (1,1), 1)
        self.norm5 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, n_fft + 1, (1,1), 1)
        self.norm6 = nn.BatchNorm2d(n_fft + 1)

	def encode(self, x):
        c1 = F.leaky_relu(self.norm1(self.conv1(x)))
        c2 = F.leaky_relu(self.norm2(self.conv2(c1)))
        c3 = F.leaky_relu(self.norm3(self.conv3(c2)))
		ev = c3.view(c3.size(0), -1)
        h1 = F.relu(self.fc1(ev))
		h2 = F.relu(self.fc2(h1))
        return F.relu(self.fc31(h2)), F.relu(self.fc32(h2))
        
    def reparameterize(self, mean, var):
        std = var.mul(0.5).exp_()
        esp = torch.randn(*mean.size())
        std = std.to(device)
        esp = esp.to(device)
        z = mean + std * esp
        return z

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
		h4 = F.relu(self.fc5(h3))
        h5 = F.relu(self.fc6(h4))
		dv = h5.view(-1, 32, 29, 1)			
		dc1 = F.leaky_relu(self.norm4(self.deconv1(dv)))
        dc2 = F.leaky_relu(self.norm5(self.deconv2(dc1)))
        return F.sigmoid(self.norm6(self.deconv3(dc2)))

    def forward(self, x):
        
		mean, var =  self.encode(x)
		z = self.reparameterize(mean, var)
        return self.decode(z), mean, var, z

		
model = DCVAE(16).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

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
        out, mean, logvar, _ = model(spec.cuda())
        if out.shape[0] == batch_size:
            out = out.reshape(batch_size,1,257,29)
        else:
            out = out.reshape(out.shape[0],1,257,29)
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
