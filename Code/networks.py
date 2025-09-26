import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd 
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical



class ModularActor(nn.Module):
    def __init__(self, num_conv_layers, num_filters, strides, kernels_size, num_mlp_layers, lstm_units, num_neurons, mlp_activation, action_dim):
        super(ModularActor, self).__init__()
        input_size = np.array((48, 46))
        
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
        #Create LSTM layer
        self.lstm = nn.LSTM(self.out_features, lstm_units, 1, batch_first=True)

        if mlp_activation == "relu":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "tanh":
            self.mlp_activation = nn.Tanh()
        elif mlp_activation == "sigmoid":
            self.mlp_activation = nn.Sigmoid()
        elif mlp_activation == "elu":
            self.mlp_activation = nn.ELU()
        elif mlp_activation == "leaky_relu":
            self.mlp_activation = nn.LeakyReLU(negative_slope=0.01)

        #Create MLP layers
        self.mlps = nn.ModuleList([nn.Linear(lstm_units, num_neurons[0])])
        #self.mlps.append(nn.Dropout1d(p=drops_mlp[0]))
        for i in range(1, num_mlp_layers-1):
            self.mlps.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
            #self.mlps.append(nn.Dropout1d(p=drops_mlp[i]))
        self.mlps.append(nn.Linear(num_neurons[-1], action_dim))

        #self.mlp_sequence = nn.Sequential(*self.mlps)

    def compute_output_size(self, input_size, kernel_size, stride, padding=0, dilation=1):
        return ((input_size + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1
    
    def forward(self, x, prev_hidden):
        for operation in self.conv_layers:
            x = operation(x)
        x = x.view(-1, 1, self.out_features)
        x, lstm_hidden = self.lstm(x, prev_hidden)
        for operation in self.mlps[:-1]:
            x = self.mlp_activation(operation(x))
        return x, lstm_hidden
    
    def pi(self, state, hidden, softmax_dim = -1):
        n, lstm_hidden = self.forward(state, hidden)
        prob = F.softmax(self.mlps[-1](n), dim=softmax_dim)
        return prob, lstm_hidden

class ModularCritic(nn.Module):
    def __init__(self, num_conv_layers, num_filters, strides, kernels_size, num_mlp_layers, lstm_units, num_neurons, mlp_activation):
        super(ModularCritic, self).__init__()
        input_size = np.array((48, 46))
        
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
        #Create LSTM layer
        self.lstm = nn.LSTM(self.out_features, lstm_units, 1, batch_first=True)

        if mlp_activation == "relu":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "tanh":
            self.mlp_activation = nn.Tanh()
        elif mlp_activation == "sigmoid":
            self.mlp_activation = nn.Sigmoid()
        elif mlp_activation == "elu":
            self.mlp_activation = nn.ELU()
        elif mlp_activation == "leaky_relu":
            self.mlp_activation = nn.LeakyReLU(negative_slope=0.01)

        #Create MLP layers
        self.mlps = nn.ModuleList([nn.Linear(lstm_units, num_neurons[0])])
        #self.mlps.append(self.mlp_activation)
        for i in range(1, num_mlp_layers-1):
            self.mlps.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
            #self.mlps.append(self.mlp_activation)
        self.mlps.append(nn.Linear(num_neurons[-1], 1))

        #self.mlp_sequence = nn.Sequential(*self.mlps)

    def compute_output_size(self, input_size, kernel_size, stride, padding=0, dilation=1):
        return ((input_size + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1
    
    def forward(self, x, prev_hidden):
        for operation in self.conv_layers:
            x = operation(x)
        x = x.view(-1, 1, self.out_features)
        x, lstm_hidden = self.lstm(x, prev_hidden)
        for operation in self.mlps[:-1]:
            x = self.mlp_activation(operation(x))
        return self.mlps[-1](x)