
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
        self.E = nn.Parameter((torch.ones(self.mat_pts)*E0).to(self.device))
        self.nu = nn.Parameter((torch.ones(self.mat_pts)*nu0).to(self.device))
        self.A = nn.Parameter((torch.ones(self.mat_pts)*A0).to(self.device))
        self.B = nn.Parameter((torch.ones(self.mat_pts)*B0).to(self.device))
        self.C = nn.Parameter((torch.ones(self.mat_pts)*C0).to(self.device))
        
    def forward(self,x):
        self.J2s = [J2Material(E=self.E[i], nu_=self.nu[i], A=self.A[i], B=self.B[i], C=self.C[i], device=self.device)
                   for i in range(self.mat_pts)]
        batch_size, seq_len, _ = x.size()
        x = x.to(self.device)
        epsilon =  x.clone()
        out = torch.zeros(
                [batch_size,seq_len, self.n_outputs]).to(self.device)

        ip_pointsb = batch_size*1

        [J2.configure(ip_pointsb) for J2 in self.J2s]

        for t in range(seq_len):              
            epsilon_t = self.fc1(epsilon[:, t,:]).view(batch_size, 1, self.n_features*self.mat_pts)

            epsilon_t_0 = epsilon_t[:,:,0*self.n_features:(0+1)*self.n_features]
            sigma_0 = self.J2s[0].update(
                epsilon_t_0.view(ip_pointsb,self.n_features))
            self.J2s[0].commit() 

            epsilon_t_1 = epsilon_t[:,:,1*self.n_features:(1+1)*self.n_features]
            sigma_1 = self.J2s[1].update(
                epsilon_t_1.view(ip_pointsb,self.n_features))
            self.J2s[1].commit() 

            epsilon_t_2 = epsilon_t[:,:,2*self.n_features:(2+1)*self.n_features]
            sigma_2 = self.J2s[2].update(
                epsilon_t_2.view(ip_pointsb,self.n_features))
            self.J2s[2].commit() 

            # epsilon_t_3 = epsilon_t[:,:,3*self.n_features:(3+1)*self.n_features]
            # sigma_3 = self.J2s[3].update(
            #     epsilon_t_3.view(ip_pointsb,self.n_features))
            # self.J2s[3].commit() 

            # epsilon_t_4 = epsilon_t[:,:,4*self.n_features:(4+1)*self.n_features]
            # sigma_4 = self.J2s[4].update(
            #     epsilon_t_4.view(ip_pointsb,self.n_features))
            # self.J2s[4].commit() 

            sigma_t = torch.cat((sigma_0,sigma_1,sigma_2), axis=1)

            outputt = self.fc2(sigma_t.view(batch_size, self.n_latents))
            out[:, t, :] = outputt.view(-1,self.n_outputs)

        output = out.to(self.device)
        return output
