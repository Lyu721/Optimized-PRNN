import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import StressStrainDataset
from prnn_with_trainable_mat import PRNN
from prnn import PRNN as PRNN0

from utils import AdamOptimizer, TrainerAdam
import copy
import time

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

mode = 'train' #eval, retrain, train
work_dir = '../'

num_steps = 100
dataset = StressStrainDataset(work_dir+'/data/Vf60_rve_ts100_n100_gp_epsilon_le_0.05.txt',
                               [0,1,2], [3,4,5],
                               usage_rate=1,
                               seq_length=num_steps, normalize_features=False)
tset, vset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

tloader = torch.utils.data.DataLoader(tset, batch_size=len(tset), shuffle=True)
vloader = torch.utils.data.DataLoader(vset, batch_size=len(vset), shuffle=True)

n_matpts = 3

for i in range(1):
    
    model_name = 'prnn_mat{}_adam_{}'.format(n_matpts, i)

    if mode == 'train':
        
        def get_model():
            np.random.seed(i**2+17)
            E0, nu0, A0, B0, C0 = np.random.rand(5)
            model = PRNN(n_features=3, n_outputs=3, n_matpts=n_matpts,
                        E0=E0, nu0=nu0, A0=A0, B0=B0, C0=C0, device=device)
            return model
        model = get_model()
        optimizer = AdamOptimizer(parameters=model.named_parameters(), lr=0.1)
        trainer = TrainerAdam(get_model, optimizer=optimizer,
                            device=device)
        t1 = time.time()
        
        results = trainer.train(tloader, vloader,
                                epochs=10, patience=2000,fixed_mat=False,
                                interval=2, verbose=True, device=device,
                                fine_tune=5000)
        t2 = time.time()
                                
        trainer.save(work_dir+'/models/{}.pth'.format(model_name))
        
        train_loss_log = results.get('train_loss_log')
        valid_loss_log = results.get('valid_loss_log')
        mat_params_log = results.get('mat_params_log')

        mat_params = np.zeros_like(mat_params_log)
        mat_params[:,0] = mat_params_log[:,0]*9000.0+1000.0
        mat_params[:,1] = mat_params_log[:,1]*0.40+0.05
        mat_params[:,2] = mat_params_log[:,2]*100.+0.
        mat_params[:,3] = mat_params_log[:,3]*60.+0.
        mat_params[:,4] = mat_params_log[:,4]*0.0099+0.0001
        fig,ax = plt.subplots(3,1, figsize=(6,6))
        ax[0].plot(train_loss_log[:,0], train_loss_log[:,1])
        ax[1].plot(valid_loss_log[:,0], valid_loss_log[:,1])
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[2].plot(mat_params_log[:,0],label='E')
        ax[2].plot(mat_params_log[:,1],label='nu')
        ax[2].plot(mat_params_log[:,2],label='A')
        ax[2].plot(mat_params_log[:,3],label='B')
        ax[2].plot(mat_params_log[:,4],label='C')
        ax[2].legend()
        plt.show()
        exit()
        
    elif mode == 'retrain':
        def get_model():
            np.random.seed(i**2+17)
            E0, nu0, A0, B0, C0 = np.random.rand(5)
            model = PRNN(n_features=3, n_outputs=3, n_matpts=n_matpts,
                        E0=E0, nu0=nu0, A0=A0, B0=B0, C0=C0, device=device)
            checkpoint = torch.load(work_dir+'/models/{}.pth'.format(model_name))
            model.load_state_dict(checkpoint['best_state_dict'],strict=True)
            return model
        model = get_model()
        optimizer = AdamOptimizer(parameters=model.named_parameters(), lr=0.01)
        trainer = TrainerAdam(get_model, optimizer=optimizer,device=device)
        results =  trainer.train(tloader, vloader, epochs=200, patience=500,fixed_mat=False,
                                                        interval=10, verbose=True, device=device,
                                                        fine_tune=50000)
        trainer.save(work_dir+'/models/{}_retrain.pth'.format(model_name))
        
        train_loss_log = results.get('train_loss_log')
        valid_loss_log = results.get('valid_loss_log')
        mat_params_log = results.get('mat_params_log')
        
        mat_params = np.zeros_like(mat_params_log)
        mat_params[:,0] = mat_params_log[:,0]*9000.0+1000.0
        mat_params[:,1] = mat_params_log[:,1]*0.40+0.05
        mat_params[:,2] = mat_params_log[:,2]*40.+mat_params_log[:,3]*60.
        mat_params[:,3] = mat_params_log[:,3]*60.+0.
        mat_params[:,4] = mat_params_log[:,4]*0.0099+0.0001
        # np.savetxt(output_dir+'/'+'mat_params_{}_retrain.txt'.format(model_name), mat_params)
        fig,ax = plt.subplots(3,1, figsize=(6,6))
        ax[0].plot(train_loss_log[:,0], train_loss_log[:,1])
        ax[1].plot(valid_loss_log[:,0], valid_loss_log[:,1])
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[2].plot(mat_params_log[:,0],label='E')
        ax[2].plot(mat_params_log[:,1],label='nu')
        ax[2].plot(mat_params_log[:,2],label='A')
        ax[2].plot(mat_params_log[:,3],label='B')
        ax[2].plot(mat_params_log[:,4],label='C')
        ax[2].legend()
        plt.show()
        exit()
        continue

    elif mode == 'eval':

        def get_model():
            np.random.seed(i**2+17)
            E0, nu0, A0, B0, C0 = np.random.rand(5)
            return PRNN(n_features=3, n_outputs=3, n_matpts=n_matpts,
                        E0=E0, nu0=nu0, A0=A0, B0=B0, C0=C0, device=device)
        model = get_model()
        checkpoint = torch.load(work_dir+'/models/{}.pth'.format(model_name))
        model.fc1.weight.data = checkpoint['best_state_dict']['fc1.weight']
        model.fc2.weight.data = checkpoint['best_state_dict']['fc2.weight']
        model.E.data = checkpoint['best_state_dict']['E']
        model.nu.data = checkpoint['best_state_dict']['nu']
        model.A.data = checkpoint['best_state_dict']['A']
        model.B.data = checkpoint['best_state_dict']['B']
        model.C.data = checkpoint['best_state_dict']['C']

    model.eval()

    idx = 1
    for i in range(12):
        strain = vloader.dataset[i][0]
        stress = vloader.dataset[i][1]
        epsilon = torch.zeros((1,idx,3),dtype=torch.double)
        epsilon[0] = strain[0:idx]
        epsilon.requires_grad_()
        print("epsilon:",epsilon)
        stressnn = model.forward(epsilon)
        DDSDDE = torch.zeros((3,3))
        for ii in range(3):
            DDSDDE_ = torch.autograd.grad(
                            inputs=epsilon,
                            outputs=stressnn[:,-1,ii],
                            grad_outputs=torch.ones_like(stressnn[:,-1,ii]),
                            retain_graph=True,
                            create_graph=True,
                            allow_unused=True)[0]
            DDSDDE[ii,:] = DDSDDE_[0,-1,:]
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
