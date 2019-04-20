import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from samplers import *


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
        value = torch.log(torch.tensor(2., device=device)) + 0.5 * (
                torch.log(outputs[targets == 1.]).mean() + torch.log(1. - outputs[targets == 0.]).mean())
        return -value, None, None

    def _add_last_layer(self, layers):
        layers.append(nn.Sigmoid())


class WDiscriminator(Discriminator):
    _lambda = 10

    def loss(self, inputs, outputs, targets):
        gp = self._compute_gp(inputs, targets)
        value1 = outputs[targets == 1.].mean() - outputs[targets == 0.].mean()
        value2 = WDiscriminator._lambda * ((gp - 1.) ** 2).mean()
        value = value1 - value2
        return -value, -value1, value2

    def _add_last_layer(self, layers):
        pass

    def _compute_gp(self, inputs, targets):
        real_data = inputs[targets == 1.]
        fake_data = inputs[targets == 0.]

        fracs = torch.rand(len(real_data), 1).to(device)
        interpolated = fracs * real_data + (1. - fracs) * fake_data
        interpolated.requires_grad_()
        value = self(interpolated).sum()
        # grads = torch.autograd.grad(value, [interpolated], create_graph=True)
        # grads.norm(p=2, dim=1).backward()
        value.backward(create_graph=True)
        value = interpolated.grad.norm(p=2, dim=1)
        return value



#  Question 1.1
device = 'cpu'
N_ITERS = 10000
# phi = 1.
# variant = 'JSD'

for phi in np.arange(-1., 1.1, 0.1):
    phi = 0.1
    for variant in ['JSD', 'EMD']:
        if variant == 'JSD':
            discriminator = JSDiscriminator([2, 512, 512, 512, 1]).to(device)
        else:
            discriminator = WDiscriminator([2, 512, 512, 512, 1]).to(device)
        optimizer = optim.SGD(discriminator.parameters(), lr=1e-3)
        for itr, real_data, fake_data in zip(range(N_ITERS), distribution1(0.), distribution1(phi)):
            data = torch.cat([torch.tensor(real_data, dtype=torch.float), torch.tensor(fake_data, dtype=torch.float)],
                             dim=0).to(device)
            targets = torch.cat([torch.ones(len(real_data), dtype=torch.uint8), torch.zeros(len(fake_data), dtype=torch.uint8)]).to(device)
            outputs = discriminator(data)

            loss = discriminator.loss(data, outputs, targets)
            if itr + 1 == N_ITERS or itr % 100 == 0:
                # print('itr: {} loss: {} auxiliary1: {} auxiliary2: {}'.format(itr, loss[0].detach().cpu(),
                #                                                               None if not loss[1] else loss[1].detach().cpu(),
                #                                                               None if not loss[2] else loss[2].detach().cpu()))
                if variant == 'JSD':
                    distance = abs(loss[0].detach().cpu())
                else:
                    distance = abs(loss[1].detach().cpu())
                print('{} {} {}'.format(variant, phi, distance))
                with open("a3.out", "a") as myfile:
                    myfile.write('{} {} {}\n'.format(variant, phi, distance))
                print('outputs: ', outputs)

            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()
