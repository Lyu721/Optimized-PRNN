#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Physically Recurrent Neural Network (PRNN)

The PRNN is an encoder-decoder architecture that learns tasks akin to
localization and homogenization operations in micromechanics. Meaningful
representations are encouraged by including between encoder and decoder
a material layer with a set of real (physics-based) constitutive models.

    - Encoder (localization): encodes global strains to a set of local
      strains for an adjustable number of 'fictitious' material points.
      The current setup implements a single affine and fully-connected
      layer. Including non-linearity through hidden layers might be 
      beneficial depending on the application.

    - Material layer: converts local strains to local stresses according
      to fixed (non-trainable) and purely physics-based constitutive laws.
      If targets come from microscale simulations, including in the material 
      layer the exact same constitutive models as in the microscale model
      tends to yield the best results. Material properties are in principle
      fixed to their microscale values, although the code can be trivially
      extended to make them trainable. State variable evolution (e.g plastic
      strains, damage) is handled exclusively by the material models and
      are therefore not learned from data

    - Decoder (homogenization): decodes the complete set of local stresses
      to a global homogenized stress. To more closely mimic the usual
      micromodel volume averaging, decoder weights are constrained to be
      positive by activating them with softplus.

The code here also includes a small custom layer class implementing the 
softplus weight activation.
"""

import math 
import torch
import torch.nn as nn
from J2Tensor_vect import J2Material


class PRNN(torch.nn.Module):
    def __init__(self, n_features, n_outputs, n_matpts, **kwargs):
        super(PRNN,self).__init__()

        self.device = kwargs.get('device',torch.device('cpu'))

        self.n_features   = n_features
        self.mat_pts      = n_matpts
        self.n_outputs    = n_outputs
        self.n_latents    = self.mat_pts*self.n_outputs
                
        print('------ PRNN model summary ------')
        print('Input (strain) size', self.n_features)
        print('Material layer size (points)', self.mat_pts)
        print('Material layer size (units)', self.n_latents)
        print('Output (stress) size', self.n_outputs)
        print('--------------------------------')
        
        self.fc1 = torch.nn.Linear(in_features=self.n_features,
                                   out_features=self.n_latents,
                                   device = self.device,
                                   bias = False)
        self.fc2 = torch.nn.Linear(in_features=self.n_latents,
                                   out_features=self.n_outputs,
                                   device = self.device,
                                   bias = False)

        self.E = kwargs.get('E', 3130.0)
        self.nu_ = kwargs.get('nu_', 0.3)
        self.A = kwargs.get('A', 64.8)
        self.B = kwargs.get('B', 33.6)
        self.C = kwargs.get('C', 0.003407)

    def forward(self,x):
        batch_size, seq_len, _ = x.size()
        x = x.to(self.device)
        output =  x.clone()
        out = torch.zeros(
                [batch_size,seq_len, self.n_outputs]).to(self.device)
        material_model = J2Material(E=self.E, nu_=self.nu_,
                                    A=self.A, B=self.B, C=self.C, device=self.device) 
        
        ip_pointsb = batch_size*self.mat_pts
        material_model.configure(ip_pointsb) 
        # Create material model and fictitious integration points
        # Process (batched) strain paths one time step at a time

        for t in range(seq_len):
          # Encoder (localization)
               
            outputt = self.fc1(output[:, t,:])
                
            # Run material model (strain, oldstate -> stress, newstate) 
            
            outputt = material_model.update(
                    outputt.view(ip_pointsb,self.n_features))

            # Store updated material history

            material_model.commit() 

            # Decoder (homogenization)
                    
            outputt = self.fc2(outputt.view(batch_size, self.n_latents))
            out[:, t, :] = outputt.view(-1,self.n_outputs)

        output = out.to(self.device)
        return output

