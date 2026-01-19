import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import Trainer
from utils import StressStrainDataset
from prnn import PRNN
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import MinMaxScaler
from UQpy.sampling import LatinHypercubeSampling
from UQpy.distributions import Uniform
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import *
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import time
import sys

work_dir = '../'

start_time = time.time()

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
 
num_steps = 100
dataset = StressStrainDataset(work_dir+'/data/Vf60_rve_ts100_n100_gp_epsilon_le_0.05.txt',
                               [0,1,2], [3,4,5],
                               usage_rate=1,
                               seq_length=num_steps, normalize_features=False)
                               
tset, vset = torch.utils.data.random_split(dataset, [0.80, 0.20], generator=torch.Generator().manual_seed(42))

tloader = torch.utils.data.DataLoader(tset, batch_size=len(tset), shuffle=True)
vloader = torch.utils.data.DataLoader(vset, batch_size=len(vset), shuffle=True)


n_matpts = 3

bounds = {}
bounds['E'] = [1000., 10000.]  # 
bounds['nu'] = [0.05, 0.45]  # 
bounds['A'] = [0., 100.]
bounds['B'] = [0., 60.]  # 
bounds['C'] = [0.0001, 0.01]

bounds = np.array([_ for _ in bounds.values()])
scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(bounds.T)
scaler.min_ = bounds[:, 0]
scaler.scale_ = bounds[:, 1]-bounds[:, 0]
dim_x = len(bounds)
bounds_scaled = np.concatenate((np.zeros((dim_x, 1)), np.ones((dim_x, 1))), axis=1)

n0 = 2
nMax = 5
num_iterations = nMax-n0
output_dir = work_dir+'models/BO'


def objfunc(x):
    num, dim = x.shape
    out = torch.zeros((num, 1))
    x = scaler.transform(x)
    for i in range(num):
        E, nu_, A, B, C = x[i]
        prnn = PRNN(n_features=3, n_outputs=3, n_matpts=n_matpts, device=device,
                    E=E, nu_=nu_, A=A, B=B, C=C)
        trainer = Trainer(prnn, optimizer=torch.optim.Adam(prnn.parameters(), lr=1.0e-1),device=device)
        try:
            results =  trainer.train(tloader, vloader, epochs=5,
                                    patience=100, interval=1, 
                                    verbose=True, device=device)
            out[i,0] = -trainer._best_val
        except:
            out[i,0] = -2000.0
    return out

dists = [Uniform(loc=bounds_scaled[i,0], scale=bounds_scaled[i,1]-bounds_scaled[i,0]) for i in range(dim_x)]
train_X = LatinHypercubeSampling(distributions=dists,
                                random_state=np.random.RandomState(789),
                                criterion=MaxiMin(metric=DistanceMetric.CHEBYSHEV),
                                nsamples=n0)._samples

train_X = torch.from_numpy(train_X)
train_Y = objfunc(train_X)



model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)  # 拟合模型

# 定义获取函数 (Expected Improvement)
best_value = train_Y.max()  # 当前最优值
EI = ExpectedImprovement(model, best_f=best_value)

# 优化获取函数以找到下一个采样点
bounds = torch.tensor([[0.0]*dim_x, [1.0]*dim_x])  # 定义搜索范围
EI_log = np.zeros(num_iterations)
for i in range(num_iterations):
    # 优化获取函数
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds, q=1, num_restarts=20, raw_samples=1024
    )
    EI_log[i] = acq_value.item()
    # 计算新采样点的目标函数值
    new_Y = objfunc(candidate)
    
    # 更新训练数据
    train_X = torch.cat([train_X, candidate])
    train_Y = torch.cat([train_Y, new_Y])
    
    # 重新拟合模型
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    
    # 更新获取函数
    best_value = train_Y.max()
    EI = ExpectedImprovement(model, best_f=best_value)
    
    print('Iteration ',i,': Candidate = ',scaler.transform(candidate))
    print(f", Value = {new_Y.item()}")
    
    np.savetxt(output_dir+'/'+'train_X.txt',train_X.numpy())
    np.savetxt(output_dir+'/'+'train_Y.txt',train_Y.numpy())
    np.savetxt(output_dir+'/'+'EI.txt',EI_log) 
    

best_index = train_Y.argmax()
best_X = train_X[best_index]
best_Y = train_Y[best_index]

best_X = scaler.transform(best_X.reshape(1,-1))
np.savetxt(output_dir+'/'+'best_x.txt',best_X)
np.savetxt(output_dir+'/'+'train_X.txt',train_X.numpy())
np.savetxt(output_dir+'/'+'train_Y.txt',train_Y.numpy())
np.savetxt(output_dir+'/'+'EI.txt',EI_log)

print('Optimal X:', best_X)
print(f"Optimal Y: {best_Y.item()}")


end_time = time.time()

print("Time cost: {:.2f} s".format(end_time-start_time))

fig, ax = plt.subplots(1,1)
ax.plot(EI_log)
plt.show()