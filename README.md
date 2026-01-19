# Optimized-PRNN
Code for the paper "An Optimized Physically Recurrent Neural Network for Multiscale Modeling of Composite Materials"

### Requirements：

The core dependency required to run the code in this repository is **PyTorch**. All experiments have been tested with **PyTorch 2.0.1**, and other compatible versions are expected to work as well.
 In addition, to run the Bayesian optimization module, the following packages are required:

- **BoTorch 0.10.0**
- **UQpy 4.1.1** (used for initial sampling)

It is also worth noting that the PRNN implementation in this repository is developed based on the work of **Maia et al.**, using the code provided at
 https://github.com/SLIMM-Lab/pyprnn.
 The RVE stress–strain loading path data used for training are generated using the **RVESimulator** developed by **Yi and Bessa**, available at
 https://github.com/bessagroup/rvesimulator.



### Code Structure

- **prnn.py**
   Implements the original PRNN model with *non-trainable* material parameters.
- **prnn_with_trainable_mat.py**
   Implements a PRNN model with *trainable* material parameters, where all virtual material points share identical parameters.
- **prnn_with_trainable_mat_vari.py**
   Implements a PRNN model with *trainable* material parameters, where parameters of different virtual material points are independent.
- **train_prnn.py**
   Training script for the PRNN model with non-trainable material parameters.
- **train_prnn_with_trainable_mat.py**
   Training script for the PRNN model with trainable material parameters shared across all virtual material points.
- **train_prnn_with_trainable_mat_vari.py**
   Training script for the PRNN model with trainable and independent material parameters at each virtual material point.
- **train_prnn_bayesian_optimization.py**
   Bayesian optimization of the PRNN model with trainable material parameters shared across virtual material points.
- **J2Tensor_vect**
   PyTorch implementation of a von Mises elastoplastic constitutive model under plane stress conditions.
- **J2Tensor_model**
   PyTorch implementation of a von Mises elastoplastic constitutive model under plane stress conditions, with trainable material parameters.
- **utils.py**
   Utility functions for dataset loading and model training.



### Important Note

Due to the recurrent nature of the PRNN—similar to traditional RNNs that involve looping along the time axis—standard training procedures fail to correctly compute gradients of the loss function with respect to material parameters beyond the first iteration. This issue is likely caused by limitations in PyTorch’s computational graph handling for such recurrent constitutive updates.

To circumvent this problem, we adopt a simple workaround:
 at the end of each training iteration, the model parameters are saved and the model is deleted; before the next iteration, the model is reconstructed and the saved parameters are reloaded. This strategy avoids computational graph conflicts and enables training to proceed.

Admittedly, this is not an elegant solution. If you are aware of a more robust or principled approach, we would greatly appreciate your suggestions.



### Remarks

This repository currently provides the **core implementation for optimizing PRNNs** as presented in the paper
 *“An Optimized Physically Recurrent Neural Network for Multiscale Modeling of Composite Materials”*.
 Further improvements and extensions are ongoing.
 If you have any questions, please feel free to contact me [lyuxl@hhu.edu.cn] or open an issue on GitHub.



