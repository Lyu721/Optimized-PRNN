import os
import random
import numpy as np
import torch
import glob
import shutil
import zipfile

import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from utils import Trainer
from utils import StressStrainDataset

from prnn import PRNN



torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
 
mode = 'eval' #eval, retrain, train
work_dir = '../'

num_steps = 100
dataset = StressStrainDataset(work_dir+'data/Vf60_rve_ts100_n100_gp_epsilon_le_0.05.txt',
                               [0,1,2], [3,4,5],
                               usage_rate=1.0,
                               seq_length=num_steps, normalize_features=False)

tset, vset = torch.utils.data.random_split(dataset, [0.80, 0.20], generator=torch.Generator().manual_seed(42))

tloader = torch.utils.data.DataLoader(tset, batch_size=len(tset), shuffle=True)
vloader = torch.utils.data.DataLoader(vset, batch_size=len(vset), shuffle=True)

n_matpts=3
model = PRNN(n_features=3, n_outputs=3, n_matpts=n_matpts, device=device,
            E=3130.0, nu_=0.3, A=64.8, B=33.6, C=0.003407)

    
for i in range(1):
    model_name = 'prnn_mat{}_baseline_Vf60'.format(n_matpts)
    if mode == 'train':
        trainer = Trainer(model, optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-1), device=device)
        results = trainer.train(tloader, vloader, epochs=10, patience=100, interval=2, verbose=True, device=device)
        trainer.save(work_dir+'/models/{}.pth'.format(model_name))
        train_loss_log = results.get('train_loss_log')
        valid_loss_log = results.get('valid_loss_log')
        fig,ax = plt.subplots(2,1, figsize=(6,6))
        ax[0].plot(train_loss_log[:,0], train_loss_log[:,1])
        ax[1].plot(valid_loss_log[:,0], valid_loss_log[:,1])
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        plt.show()
        exit()
    elif mode == 'retrain':
        checkpoint = torch.load(work_dir+'/models/{}.pth'.format(model_name))
        model.load_state_dict(checkpoint['best_state_dict'])
        trainer = Trainer(model, optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-3), device=device)
        results = trainer.train(tloader, vloader, epochs=10, patience=20000, interval=1, verbose=True, device=device)
        trainer.save(work_dir+'/models/{}.pth'.format(model_name))
        train_loss_log = results.get('train_loss_log')
        valid_loss_log = results.get('valid_loss_log')
        fig,ax = plt.subplots(2,1, figsize=(6,6))
        ax[0].plot(train_loss_log[:,0], train_loss_log[:,1])
        ax[1].plot(valid_loss_log[:,0], valid_loss_log[:,1])
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        plt.show()
        exit()
        
    elif mode == 'eval':
        checkpoint = torch.load(work_dir+'/models/{}.pth'.format(model_name))
        model.load_state_dict(checkpoint['best_state_dict'],strict=True)
        model.eval()


    idx = 1
    for i in range(10):
        strain = vloader.dataset[i][0]
        stress = vloader.dataset[i][1]
        epsilon = torch.zeros((1,idx,3),dtype=torch.double)
        epsilon[0] = strain[0:idx]
        epsilon.requires_grad_()
        print("epsilon:",epsilon)
        stressnn = model.forward(epsilon)
        DDSDDE = torch.zeros((3,3))
        for i in range(3):
            DDSDDE_ = torch.autograd.grad(
                            inputs=epsilon,
                            outputs=stressnn[:,-1,i],
                            grad_outputs=torch.ones_like(stressnn[:,-1,i]),
                            retain_graph=True,
                            create_graph=True,
                            allow_unused=True)[0]
            DDSDDE[i,:] = DDSDDE_[0,-1,:]
        print("strain:",strain[0:idx+1])
        print("stress:",stress[0:idx+1])
        print("epsilon:",epsilon)
        DDSDDE = DDSDDE.cpu()
        depsilon = strain[idx] - strain[idx-1]
        dsigma = (DDSDDE @ depsilon.reshape(3,1)).reshape(1,-1)
        sigma  = stressnn[0][-1].cpu()+dsigma
        print('depsilon:', depsilon)
        print('dsigma:', dsigma)
        print('sigma:', sigma)
        print('stress:', stress[idx])
        stressnn = (model.forward(torch.unsqueeze(strain, 0))).reshape([num_steps, 3]).cpu().detach().numpy()
        fig, ax = plt.subplots(3,1,figsize=(6,12))
        ax[0].plot(strain[:,0], stress[:,0], '*', label='true')
        ax[0].plot(strain[:,0], stressnn[:,0], label='nn')
        ax[1].plot(strain[:,1], stress[:,1], '*', label='true')
        ax[1].plot(strain[:,1], stressnn[:,1], label='nn')
        ax[2].plot(strain[:,2], stress[:,2], '*', label='true')
        ax[2].plot(strain[:,2], stressnn[:,2], label='nn')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()