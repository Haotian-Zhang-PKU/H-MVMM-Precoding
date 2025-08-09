# Heterogeneous Multi-Vehicle Multi-Modal Sensing Aided FDD Precoding

 This simulation code package is mainly used to reproduce the results of the following paper [1]:

 __[1] H. Zhang, S. Gao, W. Wen, and X. Cheng, "Synesthesia of Machines (SoM)-Aided FDD Precoding with Sensing Heterogeneity: A Vertical Federated Learning Approach," in 2025 IEEE International Conference on Communications (ICC), Montreal, Canada.__

 __If you use this simulation code package in any way, please cite the original paper [1] above.__

 Copyright reserved by the Pervasive Connectivity and Networked Intelligence Laboratory (led by Dr. Xiang Cheng), School of Electronics, Peking University, Beijing 100871, China. 


# Dependencies:
1) Python 3.9.7 

2) Pytorch 1.10.2

3) NVIDIA GPU with a compatible CUDA toolkit (see [NVIDIA website](https://developer.nvidia.com/cuda-toolkit)).

4) Dataset, which can be downloaded at https://pan.baidu.com/s/1VEd0GVo63duBSleXsKvSmw with extraction code 1fcs. 


# Running the code:

1) Ensure that you have the dataset required for H-MVMM.

2) Set the paths of the training and testing datasets in the script "train_HMVMM_sixcars.py".

3) Run train_HMVMM_sixcars.py to obtain the precoding vectors of each car. The code first performs H-MVMM training and then obtains inference results. It is worth noting that the provided code is for a case where the number of users is 6.

# Dataset Introduction:
The data interface has been written in train_HMVMM_sixcars.py. You only need to select the corresponding .pth file in the dataset we provided, and replace the file path in the code with your own path.

It is noted that there may be some differences in the results of different training processes. 

