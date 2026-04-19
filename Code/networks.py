"""Neural network building blocks used by the PPO traffic-signal agent.

This file contains the modular CNN+recurrent actor and critic used in the main
training script. The models consume a cropped 3-channel occupancy tensor of the
intersection with shape ``[3, 48, 46]``:

- channel 0: vehicle presence
- channel 1: normalized speed
- channel 2: normalized accumulated waiting time

The convolutional stack extracts spatial features, the recurrent layer keeps
short-term traffic history, and the MLP heads produce either action logits or a
scalar state-value estimate.
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical


def _build_recurrent_layer(recurrent_type, input_size, hidden_size):
    if recurrent_type == "lstm":
        return nn.LSTM(input_size, hidden_size, 1, batch_first=True)
    if recurrent_type == "gru":
        return nn.GRU(input_size, hidden_size, 1, batch_first=True)
    raise ValueError(f"Unsupported recurrent_type: {recurrent_type}")


def _recurrent_slots(recurrent_type):
    return 2 if recurrent_type == "lstm" else 1

class ModularActor(nn.Module):
    def __init__(self, num_conv_layers, num_filters, strides, kernels_size, num_mlp_layers, recurrent_units, num_neurons, action_dim, recurrent_type="lstm"):
        """Policy network for the discrete traffic-light action space.

        The architecture is intentionally parameterized so Optuna can search
        over spatial capacity, recurrent memory size/type, and MLP width while
        the input/output contract stays fixed.
        """
        super(ModularActor, self).__init__()
        input_size = np.array((48, 46))
        self.recurrent_type = recurrent_type
        self.hidden_units = recurrent_units
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList([nn.Conv2d(3, num_filters[0], kernel_size=(kernels_size[0], kernels_size[0]), padding='same')])
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool2d((2,2), stride=strides[0]))
        self.out_size = self.compute_output_size(input_size, kernel_size=2, stride=strides[0])
        for i in range(1, num_conv_layers):
            self.conv_layers.append(nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=(kernels_size[i], kernels_size[i]), padding='same'))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d((2,2), stride=strides[i]))
            self.out_size = self.compute_output_size(self.out_size, kernel_size=2, stride=strides[i])      
        
        self.out_features = num_filters[-1]*np.prod(self.out_size)
        # Create recurrent layer
        self.recurrent = _build_recurrent_layer(recurrent_type, self.out_features, recurrent_units)

        '''if mlp_activation == "relu":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "tanh":
            self.mlp_activation = nn.Tanh()
        elif mlp_activation == "sigmoid":
            self.mlp_activation = nn.Sigmoid()
        elif mlp_activation == "elu":
            self.mlp_activation = nn.ELU()
        elif mlp_activation == "leaky_relu":
            self.mlp_activation = nn.LeakyReLU(negative_slope=0.01)'''
        self.mlp_activation = nn.Tanh()

        #Create MLP layers
        self.mlps = nn.ModuleList([nn.Linear(recurrent_units, num_neurons[0])])
        #self.mlps.append(nn.Dropout1d(p=drops_mlp[0]))
        for i in range(1, num_mlp_layers-1):
            self.mlps.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
            #self.mlps.append(nn.Dropout1d(p=drops_mlp[i]))
        self.mlps.append(nn.Linear(num_neurons[-1], action_dim))

        #self.mlp_sequence = nn.Sequential(*self.mlps)

    def compute_output_size(self, input_size, kernel_size, stride, padding=0, dilation=1):
        return ((input_size + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1
    
    def forward(self, x, prev_hidden):
        """Return latent features and the next recurrent hidden state."""
        for operation in self.conv_layers:
            x = operation(x)
        x = x.view(-1, 1, self.out_features)
        x, recurrent_hidden = self.recurrent(x, prev_hidden)
        for operation in self.mlps[:-1]:
            x = self.mlp_activation(operation(x))
        return x, recurrent_hidden
    
    def pi(self, state, hidden, softmax_dim = -1):
        """Return action probabilities for one observation and hidden state."""
        n, recurrent_hidden = self.forward(state, hidden)
        logits = self.mlps[-1](n)#.clamp(-80,80)
        prob = F.softmax(logits, dim=softmax_dim) #logits
        return prob, recurrent_hidden

    def initial_hidden(self, batch_size, device):
        shape = (1, batch_size, self.hidden_units)
        h0 = torch.zeros(shape, dtype=torch.float32, device=device)
        if self.recurrent_type == "lstm":
            return (h0, torch.zeros(shape, dtype=torch.float32, device=device))
        return h0

    def hidden_state_slots(self):
        return _recurrent_slots(self.recurrent_type)

class ModularCritic(nn.Module):
    def __init__(self, num_conv_layers, num_filters, strides, kernels_size, num_mlp_layers, recurrent_units, num_neurons, recurrent_type="lstm"):
        """Value network mirroring the actor backbone but ending in one scalar.

        Keeping actor and critic backbones structurally similar reduces one
        source of experimental variation when comparing hyperparameter trials.
        """
        super(ModularCritic, self).__init__()
        input_size = np.array((48, 46))
        self.recurrent_type = recurrent_type
        self.hidden_units = recurrent_units
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList([nn.Conv2d(3, num_filters[0], kernel_size=(kernels_size[0], kernels_size[0]), padding='same')])
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool2d((2,2), stride=strides[0]))
        self.out_size = self.compute_output_size(input_size, kernel_size=2, stride=strides[0])
        for i in range(1, num_conv_layers):
            self.conv_layers.append(nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=(kernels_size[i], kernels_size[i]), padding='same'))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d((2,2), stride=strides[i]))
            self.out_size = self.compute_output_size(self.out_size, kernel_size=2, stride=strides[i])      
        
        self.out_features = num_filters[-1]*np.prod(self.out_size)
        # Create recurrent layer
        self.recurrent = _build_recurrent_layer(recurrent_type, self.out_features, recurrent_units)

        '''if mlp_activation == "relu":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "tanh":
            self.mlp_activation = nn.Tanh()
        elif mlp_activation == "sigmoid":
            self.mlp_activation = nn.Sigmoid()
        elif mlp_activation == "elu":
            self.mlp_activation = nn.ELU()
        elif mlp_activation == "leaky_relu":
            self.mlp_activation = nn.LeakyReLU(negative_slope=0.01)'''
        
        self.mlp_activation = nn.ReLU()

        #Create MLP layers
        self.mlps = nn.ModuleList([nn.Linear(recurrent_units, num_neurons[0])])
        #self.mlps.append(self.mlp_activation)
        for i in range(1, num_mlp_layers-1):
            self.mlps.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
            #self.mlps.append(self.mlp_activation)
        self.mlps.append(nn.Linear(num_neurons[-1], 1))

        #self.mlp_sequence = nn.Sequential(*self.mlps)

    def compute_output_size(self, input_size, kernel_size, stride, padding=0, dilation=1):
        return ((input_size + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1
    
    def forward(self, x, prev_hidden):
        """Estimate the state value for one observation and hidden state."""
        for operation in self.conv_layers:
            x = operation(x)
        x = x.view(-1, 1, self.out_features)
        x, recurrent_hidden = self.recurrent(x, prev_hidden)
        for operation in self.mlps[:-1]:
            x = self.mlp_activation(operation(x))
        return self.mlps[-1](x)
