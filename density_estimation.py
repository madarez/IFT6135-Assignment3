#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from samplers import *


# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))
plt.show()

############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
 


class Discriminator(nn.Module):

    def __init__(self, layers):
        super().__init__()
        _layers = []
        for i, _ in enumerate(layers[:-1]):
            _layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                _layers.append(nn.ReLU())
            else:
                self._add_last_layer(_layers)
                pass
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self.layers(x)

    def _add_last_layer(self, layers):
        raise NotImplementedError


class JSDiscriminator(Discriminator):

    def loss(self, inputs, outputs, targets):
        value = torch.log(outputs[targets == 1.]).mean() + torch.log(1. - outputs[targets == 0.]).mean()
        return -value, None, None

    def _add_last_layer(self, layers):
        layers.append(nn.Sigmoid())



# Train the discriminator
device = 'cpu'
discriminator = JSDiscriminator([1, 256, 256, 256, 1]).to(device)
optimizer = optim.SGD(discriminator.parameters(), lr=1e-3)
for itr, real_data, fake_data in zip(range(50000), distribution4(), distribution3()):
    data = torch.cat([torch.tensor(real_data, dtype=torch.float), torch.tensor(fake_data, dtype=torch.float)],
                     dim=0).to(device)
    targets = torch.cat(
        [torch.ones(len(real_data), dtype=torch.uint8), torch.zeros(len(fake_data), dtype=torch.uint8)]).to(device)
    outputs = discriminator(data)

    loss = discriminator.loss(data, outputs, targets)
    optimizer.zero_grad()
    loss[0].backward()
    optimizer.step()
    if itr % 500 == 0:
        print('loss: {} outputs: {} itr: {} '.format(loss, (outputs[:10], outputs[-10:]), itr))









############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density


r = discriminator(torch.tensor(xx, device=device, dtype=torch.float).view(-1, 1)).detach().cpu().numpy() # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

temp = N(xx)
estimate = temp.reshape(*r.shape) * r/(1. - r)#np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate
print('d-shape: {} r-shape: {} r-over-shape: {} estimate-shape: {}'.format(temp.shape, r.shape, (r/(1. - r)).shape,
                                                                           (temp.reshape(*r.shape) * r/(1. - r)).shape))
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')
plt.show()











