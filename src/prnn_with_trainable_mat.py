
import math 
import torch
import torch.nn as nn
from J2Tensor_model import J2Material
import numpy as np
import matplotlib.pyplot as plt
import copy

class PRNN(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_matpts, **kwargs):
        super(PRNN,self).__init__()

        self.device = kwargs.get('device',torch.device('cpu'))

        self.n_features   = n_features
        self.mat_pts      = n_matpts
        self.n_latents    = self.mat_pts*self.n_features
        self.n_outputs    = n_outputs
                
        # print('------ PRNN model summary ------')
        # print('Input (strain) size', self.n_features)
        # print('Material layer size (points)', self.mat_pts)
        # print('Material layer size (units)', self.n_latents)
        # print('Output (stress) size', self.n_outputs)
        # print('--------------------------------')
        
        self.fc1 = torch.nn.Linear(in_features=self.n_features,
                                   out_features=self.n_latents,
                                   device = self.device,
                                   bias = False)
        self.fc2 = torch.nn.Linear(in_features=self.n_latents,
                                   out_features=self.n_outputs,
                                   device = self.device,
                                   bias = False)

        E0 = kwargs.get('E0', 0.5)
        nu0 = kwargs.get('nu0', 0.5)
        A0 = kwargs.get('A0', 0.5)
        B0 = kwargs.get('B0', 0.5)
        C0 = kwargs.get('C0', 0.5)
        self.E = nn.Parameter((torch.ones(1)*E0).to(self.device))
        self.nu = nn.Parameter((torch.ones(1)*nu0).to(self.device))
        self.A = nn.Parameter((torch.ones(1)*A0).to(self.device))
        self.B = nn.Parameter((torch.ones(1)*B0).to(self.device))
        self.C = nn.Parameter((torch.ones(1)*C0).to(self.device))
        
    def forward(self,x):
        self.J2 = J2Material(E=self.E, nu_=self.nu, A=self.A, B=self.B, C=self.C, device=self.device)
        batch_size, seq_len, _ = x.size()
        x = x.to(self.device)
        output =  x.clone()
        out = torch.zeros(
                [batch_size,seq_len, self.n_outputs]).to(self.device)

        ip_pointsb = batch_size*self.mat_pts
        self.J2.configure(ip_pointsb) 

        for t in range(seq_len):              
            outputt = self.fc1(output[:, t,:])
            outputt = self.J2.update(
                    outputt.view(ip_pointsb,self.n_features))
            self.J2.commit() 
                        
            outputt = self.fc2(outputt.view(batch_size, self.n_latents))
            out[:, t, :] = outputt.view(-1,self.n_outputs)

        output = out.to(self.device)
        return output
