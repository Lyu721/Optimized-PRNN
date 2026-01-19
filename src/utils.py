#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes for training RNNs and PRNNs"""


import os
import numpy as np

import torch
import copy
import pandas
from prnn_with_trainable_mat import PRNN
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, MultiStepLR


def get_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # L2 范数
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5  # 总范数

class Normalizer:
    """Normalization for strain features.

    Scales strain data to a [-1,1] interval
    """

    def __init__(self,X=None):
        if X is not None:
            self.min = X.min(axis=0).values
            self.max = X.max(axis=0).values

    def normalize(self,x):
        return 2.0 * ((x - self.min) / (self.max-self.min)) - 1.0
    

class StressStrainDataset(torch.utils.data.Dataset):
    """Custom dataset for handling stress-strain paths.

    Dataset is loaded with pandas and stress-strain pairs are
    split into paths of 'seq_length' time steps. Normalization
    is optional and the normalizer can be inherited from another
    dataset (e.g. to handle an entirely new test set).
    """

    def __init__(self, filename, features, targets, seq_length, **kwargs):
        df = pandas.read_csv(filename, delim_whitespace=True,header=None)

        self.seq_length = seq_length
        usage_rate = kwargs.get('usage_rate', 1.0)
        usage_num = int(df.shape[0]*usage_rate)
        self.X = torch.tensor(df[features].values[0:usage_num], dtype=torch.float64)
        self.T = torch.tensor(df[targets].values[0:usage_num],  dtype=torch.float64)

        self.normalize_features = kwargs.get('normalize_features',False)

        if self.normalize_features:
            self._normalizer = kwargs.get('normalizer',Normalizer(self.X))

    def __len__(self):
        return int(self.X.shape[0]/self.seq_length)

    def __getitem__(self,idx):
        start = idx * self.seq_length
        end   = start + self.seq_length

        x = self.X[start:end,:]
        t = self.T[start:end,:]

        if self.normalize_features:
            return self._normalizer.normalize(x), t
        else:
            return x, t

    def get_normalizer(self):
        if self.normalize_features:
            return self._normalizer
        else:
            return None

class Trainer ():
    """Class for handling standard network training tasks.

    Wraps an existing model inherited from 'torch.nn.Module' and
    performs training and evaluation tasks. Loss function and optimizer 
    are fixed but training and validation can be performed with different
    DataLoaders by calling the 'train()' function multiple times.

    Early stopping is implemented with adjustable patience. At the end
    of training, the network with lowest historical validation error
    is stored.

    Evaluation can be done on a testset DataLoader. The (time) average
    error for each sample (path) is reported, as well as the average
    over the full test set.

    Network and optimizer states can be saved and loaded from files.
    """

    def __init__(self, model, **kwargs):
        self.device = kwargs.get('device',torch.device('cpu'))
        self._model = model.to(self.device)

        self._epoch = 0
        self._criterion = kwargs.get('loss',torch.nn.MSELoss())
        self._optimizer = kwargs.get(
                'optimizer',
                torch.optim.Adam(self._model.parameters()))
        self._lr_scheduler = kwargs.get('lr_scheduler', None)
        total_params = 0
        for name, parameter in self._model.named_parameters():
            if not parameter.requires_grad:
                continue
            total_params += parameter.numel()

        print('Total parameter count:', total_params,'\n')

    def train(self, training_loader, validation_loader, **kwargs):
        self._model.train()
        
        epochs   = kwargs.get('epochs',100)
        patience = kwargs.get('patience',20)
        interval = kwargs.get('interval',1)
        verbose  = kwargs.get('verbose',True)

        stall_iters = 0
        torch.autograd.set_detect_anomaly(False)
        train_loss_log = np.zeros((epochs,2))
        valid_loss_log = np.zeros((int((epochs-1)/interval)+1,2))
        lr_log = []

        # scheduler = MultiStepLR(self._optimizer, milestones=[500, 1000], gamma=0.1)
        for i in range(epochs):
            running_loss = 0

            # 手动调整学习率
            if self._lr_scheduler is not None:
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = self._lr_scheduler(i)

            for x, t in training_loader:
                x = x.to(self.device)
                t = t.to(self.device)
                y = self._model(x).to(self.device)
                loss = self._criterion(y, t)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                running_loss += loss.item()
            # scheduler.step()

            running_loss /= len(training_loader)
            train_loss_log[i,0] = i
            train_loss_log[i,1] = running_loss

            self._epoch += 1

            if i == 0 or i % interval == 0:
                with torch.no_grad():
                    self._model.eval()
                running_loss_val = 0
                for x, t in validation_loader:
                    x = x.to(self.device)
                    t = t.to(self.device)
                    y = self._model(x)
                    loss = self._criterion(y, t)
                    running_loss_val += loss.item()
                running_loss_val /= len(validation_loader)
                self._model.train()

                print('\n'+'*'*20)
                if verbose:
                    print('Epoch',
                          self._epoch,
                          'training loss',
                          running_loss)
                if verbose:
                    print('Epoch',
                          self._epoch,
                          'validation loss',
                          running_loss_val)
                if verbose:
                    print('Epoch',
                          self._epoch,
                          'learning rate',
                          self._optimizer.param_groups[0]['lr'])
                    
                valid_loss_log[int(i/interval),0] = i
                valid_loss_log[int(i/interval),1] = running_loss_val
                lr_log.append(self._optimizer.param_groups[0]['lr'])

                if (self._epoch == 1):
                    self._best_val = running_loss_val

                if (running_loss_val <= self._best_val):
                    self._best_val = running_loss_val
                    self._best_state_dict = copy.deepcopy(self._model.state_dict())
                    stall_iters = 0

                    if verbose:
                        print('The best historical model has been updated.',
                              'Resetting early stop counter.')
                else:
                    if i <= interval:
                        stall_iters += 1
                    else:
                        stall_iters += interval

                if (stall_iters >= patience):
                    if verbose:
                        print('Early stopping criterion reached.')
                    break
        print('End of training.')
        results = {}
        results['train_loss_log'] = train_loss_log
        results['valid_loss_log'] = valid_loss_log
        return results

    def eval(self, test_loader, **kwargs):
        criterion  = kwargs.get('loss',self._criterion)
        verbose    = kwargs.get('verbose',True)

        live_state = copy.deepcopy(self._model.state_dict())

        self._model.load_state_dict(self._best_state_dict)

        combined_loss = 0.0

        self._model.eval()
        for j, (x,t) in enumerate(test_loader):            
            
            y = self._model(x)
            loss = criterion (y, t)
            combined_loss += loss.item()

            if verbose:
                print('Loss for test batch ',
                      j+1, '/', len(test_loader),':',loss.item())
            
        if verbose:
            print('Aggregated test set loss:', 
                  combined_loss/len(test_loader))

        self._model.load_state_dict(live_state)
        
        return combined_loss/len(test_loader)

    def save(self, filename):
        torch.save({
                    'epoch': self._epoch,
                    'best_val': self._best_val,
                    'model_state_dict': self._model.state_dict(),
                    'best_state_dict': self._best_state_dict,
                    'optimizer_state_dict': self._optimizer.state_dict()
                   }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)

        self._epoch = checkpoint['epoch']
        self._best_val = checkpoint['best_val']
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._best_state_dict = checkpoint['best_state_dict']
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        初始化 Adam 优化器。

        参数:
            params: 需要优化的参数（可迭代对象，如 model.parameters()）。
            lr: 学习率（默认 0.001)。
            beta1: 一阶矩衰减率（默认 0.9)。
            beta2: 二阶矩衰减率（默认 0.999)。
            eps: 数值稳定性常数（默认 1e-8)。
        """
        params = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # 时间步
        self.m = [torch.zeros_like(p[1]) for p in params]  # 一阶矩
        self.v = [torch.zeros_like(p[1]) for p in params]  # 二阶矩

    def step(self, parameters):
        """执行一次参数更新。"""
        params = list(parameters)
        self.t += 1
        grads = np.zeros(len(params))
        total_norm = 0.0
        for i, p in enumerate(params):
            name, param = p
            if param.grad is None:
                continue
            grad = param.grad
            if name in ['E', 'nu', 'A', 'B', 'C']:
                # grad = torch.clamp(grad, -0.1, 0.1)
                # 更新一阶矩和二阶矩
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
                # 偏差修正
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                # 更新参数
                param.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
                # grads[i] = grad.item()
            else:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
                # 偏差修正
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
            param_norm = param.grad.data.norm(2)  # L2 范数
            total_norm += param_norm.item() ** 2
        return total_norm

    def zero_grad(self,parameters):
        """清空梯度。"""
        params = list(parameters)
        for param in params:
            if param[1].grad is not None:
                param[1].grad.zero_()


class TrainerAdam:

    def __init__(self, get_model:callable,**kwargs):
        self.device = kwargs.get('device',torch.device('cpu'))
        self.get_model = get_model
        self._epoch = 0
        self._criterion = kwargs.get('loss',torch.nn.MSELoss())
        self._lr_scheduler = kwargs.get('lr_scheduler', None)
        total_params = 0
        model = self.get_model()           
        self._optimizer = kwargs.get(
                'optimizer',
                AdamOptimizer(model.named_parameters()))
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            total_params += parameter.numel()

        print('Total parameter count:', total_params,'\n')

    def train(self, training_loader, validation_loader, fixed_mat=False, **kwargs):
        
        epochs   = kwargs.get('epochs',100)
        patience = kwargs.get('patience',20)
        interval = kwargs.get('interval',1)
        verbose  = kwargs.get('verbose',True)
        fine_tune = kwargs.get('fine_tune',epochs)
        stall_iters = 0
        torch.autograd.set_detect_anomaly(False)
        train_loss_log = np.zeros((epochs,2))
        valid_loss_log = np.zeros((int((epochs-1)/interval)+1,2))
        mat_params_log = np.zeros((epochs,5))
        grads_log  = []
        flag_1 = True
        flag_2 = True
        for epoch in range(epochs):
            running_loss = 0
            model = (self.get_model()).to(self.device)
            
            # At the beginning of each epoch, the parameters of the previous model are imported before optimization.
            if epoch != 0:
                model.load_state_dict(state_dict_old)

            if self._lr_scheduler is not None:
                self._optimizer.lr = self._lr_scheduler(epoch)

            for x, t in training_loader:
                x = x.to(self.device)
                t = t.to(self.device)
                y = model(x).to(self.device)
                loss = self._criterion(y, t)
                self._optimizer.zero_grad(model.named_parameters()) 
                loss.backward()
                total_norm = self._optimizer.step(model.named_parameters())
                running_loss += loss.item()
                grads_log.append(total_norm**0.5)

            running_loss /= len(training_loader)
            train_loss_log[epoch,0] = epoch
            train_loss_log[epoch,1] = running_loss

            # constrain the parameters within the ranges
            model.E.data = torch.clamp(model.E.data, 0.0, 1.0)
            model.nu.data = torch.clamp(model.nu.data, 0.0, 1.0)
            model.A.data = torch.clamp(model.A.data, 0.0, 1.0)
            model.B.data = torch.clamp(model.B.data, 0.0, 1.0)
            model.C.data = torch.clamp(model.C.data, 0.0, 1.0)

            mat_params_log[epoch,0] = model.E.data
            mat_params_log[epoch,1] = model.nu.data
            mat_params_log[epoch,2] = model.A.data
            mat_params_log[epoch,3] = model.B.data
            mat_params_log[epoch,4] = model.C.data

            state_dict_old = copy.deepcopy(model.state_dict())

            self._epoch += 1

            if epoch == 0 or (epoch+1) % interval == 0:
                
                with torch.no_grad():
                    model.eval()
                running_loss_val = 0
                for x, t in validation_loader:
                    x = x.to(self.device)
                    t = t.to(self.device)
                    y = model(x)
                    loss = self._criterion(y, t)
                    running_loss_val += loss.item()
                running_loss_val /= len(validation_loader)
                model.train()

                print('\n'+'*'*20)
                if verbose:
                    print('Epoch',
                          epoch,
                          'training loss',
                          running_loss)
                if verbose:
                    print('Epoch',
                          epoch,
                          'validation loss',
                          running_loss_val)
                print('Iteration:{}, E:{:.4e}, nu:{:.4e}, A:{:.4e}, B:{:.4e}, C:{:.4e}'.
                    format(epoch, model.E.item(), model.nu.item(), model.A.item(), model.B.item(), model.C.item()))
                valid_loss_log[int(epoch/interval),0] = epoch
                valid_loss_log[int(epoch/interval),1] = running_loss_val

                if (self._epoch == 1):
                    self._best_val = running_loss_val

                if (running_loss_val <= self._best_val):
                    self._best_val = running_loss_val
                    self._best_state_dict = copy.deepcopy(model.state_dict())
                    stall_iters = 0

                    if verbose:
                        print('The best historical model has been updated.',
                              'Resetting early stop counter.')
                else:
                    if epoch <= interval:
                        stall_iters += 1
                    else:
                        stall_iters += interval

                if (stall_iters >= patience):
                    if verbose:
                        print('Early stopping criterion reached.')
                    break
        print('End of training.')
        results = {}
        results['train_loss_log'] = train_loss_log
        results['valid_loss_log'] = valid_loss_log
        results['mat_params_log'] = mat_params_log
        results['grads_log'] = grads_log
        return results

    def save(self, filename):
        torch.save({
                    'epoch': self._epoch,
                    'best_val': self._best_val,
                    'best_state_dict': self._best_state_dict,
                   }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self._epoch = checkpoint['epoch']
        self._best_val = checkpoint['best_val']
        self._best_state_dict = checkpoint['best_state_dict']


class TrainerAdamVari:

    def __init__(self, get_model:callable,**kwargs):
        self.device = kwargs.get('device',torch.device('cpu'))
        self.get_model = get_model
        self._epoch = 0
        self._criterion = kwargs.get('loss',torch.nn.MSELoss())
        self._lr_scheduler = kwargs.get('lr_scheduler', None)
        total_params = 0
        model = self.get_model()           
        self._optimizer = kwargs.get(
                'optimizer',
                AdamOptimizer(model.named_parameters()))
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            total_params += parameter.numel()

        print('Total parameter count:', total_params,'\n')

    def train(self, training_loader, validation_loader, fixed_mat=False, **kwargs):
        
        epochs   = kwargs.get('epochs',100)
        patience = kwargs.get('patience',20)
        interval = kwargs.get('interval',1)
        verbose  = kwargs.get('verbose',True)
        fine_tune = kwargs.get('fine_tune',epochs)
        stall_iters = 0
        torch.autograd.set_detect_anomaly(False)
        train_loss_log = np.zeros((epochs,2))
        valid_loss_log = np.zeros((int((epochs-1)/interval)+1,2))
        mat_params_log = np.zeros((epochs,5,3))
        flag_1 = True
        flag_2 = True
        for epoch in range(epochs):
            running_loss = 0
            model = (self.get_model()).to(self.device)
            if epoch != 0:
                model.load_state_dict(state_dict_old)

            if self._lr_scheduler is not None:
                self._optimizer.lr = self._lr_scheduler(epoch)

            for x, t in training_loader:
                x = x.to(self.device)
                t = t.to(self.device)
                y = model(x).to(self.device)
                loss = self._criterion(y, t)
                self._optimizer.zero_grad(model.named_parameters()) 
                loss.backward()
                self._optimizer.step(model.named_parameters())
                running_loss += loss.item()

            running_loss /= len(training_loader)
            train_loss_log[epoch,0] = epoch
            train_loss_log[epoch,1] = running_loss

            model.E.data = torch.clamp(model.E.data, 0.0, 1.0)
            model.nu.data = torch.clamp(model.nu.data, 0.0, 1.0)
            model.A.data = torch.clamp(model.A.data, 0.0, 1.0)
            model.B.data = torch.clamp(model.B.data, 0.0, 1.0)
            model.C.data = torch.clamp(model.C.data, 0.0, 1.0)
            
            mat_params_log[epoch,0,:] = model.E.detach().cpu().numpy()
            mat_params_log[epoch,1,:] = model.nu.detach().cpu().numpy()
            mat_params_log[epoch,2,:] = model.A.detach().cpu().numpy()
            mat_params_log[epoch,3,:] = model.B.detach().cpu().numpy()
            mat_params_log[epoch,4,:] = model.C.detach().cpu().numpy()

            state_dict_old = copy.deepcopy(model.state_dict())

            self._epoch += 1

            if epoch == 0 or (epoch) % interval == 0:
                with torch.no_grad():
                    model.eval()
                    running_loss_val = 0
                    for x, t in validation_loader:
                        x = x.to(self.device)
                        t = t.to(self.device)
                        y = model(x)
                        loss = self._criterion(y, t)
                        running_loss_val += loss.item()
                    running_loss_val /= len(validation_loader)
                    model.train()

                print('\n'+'*'*20)
                if verbose:
                    print('Epoch',
                          epoch,
                          'training loss',
                          running_loss)
                if verbose:
                    print('Epoch',
                          epoch,
                          'validation loss',
                          running_loss_val)
                # print('Iteration:{}, E:{:.4e}, nu:{:.4e}, A:{:.4e}, B:{:.4e}, C:{:.4e}'.
                    # format(epoch, model.E.item(), model.nu.item(), model.A.item(), model.B.item(), model.C.item()))
                print('Iteration:{}'.format(epoch))
                print('E: ', model.E)
                print('nu: ', model.nu)
                print('A: ', model.A)
                print('B: ', model.B)
                print('C: ', model.C)
                valid_loss_log[int(epoch/interval),0] = epoch
                valid_loss_log[int(epoch/interval),1] = running_loss_val

                if (self._epoch == 1):
                    self._best_val = running_loss_val

                if (running_loss_val <= self._best_val):
                    self._best_val = running_loss_val
                    self._best_state_dict = copy.deepcopy(model.state_dict())
                    stall_iters = 0

                    if verbose:
                        print('The best historical model has been updated.',
                              'Resetting early stop counter.')
                else:
                    if epoch <= interval:
                        stall_iters += 1
                    else:
                        stall_iters += interval

                if (stall_iters >= patience):
                    if verbose:
                        print('Early stopping criterion reached.')
                    break

        print('End of training.')
        results = {}
        results['train_loss_log'] = train_loss_log
        results['valid_loss_log'] = valid_loss_log
        results['mat_params_log'] = mat_params_log
        return results

    def save(self, filename):
        torch.save({
                    'epoch': self._epoch,
                    'best_val': self._best_val,
                    'best_state_dict': self._best_state_dict,
                   }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self._epoch = checkpoint['epoch']
        self._best_val = checkpoint['best_val']
        self._best_state_dict = checkpoint['best_state_dict']


