# LoGONE - An instance detection network for the automatic retouching of branded media 
LoGONE is an instance detection network for the automatic retouching of branded media, namely still images and videos. When LoGONE detects a brand (logo/reference/etc.), to the best of its ability, it will attempt to crop that brand from the still, generate a suitable replacement using deep learning techniques, and superimpose that replacement back on the original still. To do this, LoGONE has two fundamental components: 

## Instance Detection 
The instance detection network of LoGONE is its bread and butter. Trained off the [LogoDet-3k](https://www.kaggle.com/datasets/lyly99/logodet3k/data?select=LogoDet-3K) dataset, the instance detection network draws bounding boxes around all identified branded logos in a frame before returning these back to the user. The network structure summary is coming soon. 

## Replacement with Stable Diffusion 
The second stage of the LoGONE pipeline is to take the visual content contained in output bounding boxes, and sending this visual representation to a diffusion model for "retouching." By retouching, we seek to take the logo content and perturb it enough so that there is a clear and distinct change in the branding (i.e. no longer copywritable), while simultaneously remaining visually consistent in the image frame. This is challenging for a number of reasons, with the primary being the fact that ground truth bounding boxes for logos may not be aligned with how the logos themselves appear in the frame. An example of this phenomenon is shown below: 

![LogoDet-3k dataset overview](media/LogoDetOverview.png) 
The above image is an overview of the LogoDet-3k dataset, and can be found in the original paper, [cited here](https://arxiv.org/abs/2008.05359). 

## Getting Started
### Environment Setup 
For this project, we use [Anaconda](https://www.anaconda.com/) for environment management. To get started, ensure that [conda is installed](https://docs.anaconda.com/free/anaconda/install/index.html).

Then we can create our conda environment. For this project, we will use Python 3.10: 
```
conda create -n "LoGONE" python=3.10
```

You can now activate this environment with 
```
conda activate LoGONE
```
and deactivate with 
```
conda deactivate
```

Afterwards, we can install the necessary python packages. This first part of this section is divided by whether or not you want to use CUDA / GPU acceleration or not to train and perform inference on models. For the CUDA setup, read below. For the CPU only setup, read "CPU Only" 

#### Using CUDA / GPU 
Since we are working with a machine learning project we want the ability to train models on GPU. To do this, first ensure that your NVIDIA driver is up to date. To do this, type the following in linux:
```
nvidia-smi
```

You should get the output shown below. If you recieved no output, or the command did not recognize, you may need to [install nvidia drivers](https://ubuntu.com/server/docs/nvidia-drivers-installation). 

```
Sun Apr  7 18:54:40 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.154.05             Driver Version: 535.154.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  | 00000000:02:00.0  On |                  N/A |
| 41%   51C    P0              54W / 260W |   2754MiB / 11264MiB |      8%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1835      G   /usr/lib/xorg/Xorg                         1277MiB |
|    0   N/A  N/A      2016      G   /usr/bin/gnome-shell                        361MiB |
|    0   N/A  N/A      2666      G   ...ures=SpareRendererForSitePerProcess       33MiB |
|    0   N/A  N/A      8978      G   ...yOnDemand --variations-seed-version       89MiB |
|    0   N/A  N/A     84204      G   ...seed-version=20240403-050113.230000      390MiB |
|    0   N/A  N/A    194181      C   python                                      282MiB |
|    0   N/A  N/A    358988      G   ...erProcess --variations-seed-version      120MiB |
+---------------------------------------------------------------------------------------+
```

We can see in the top right that this driver supports a CUDA version __up to__ 12.2, which means I can use any CUDA version before this as well. For this project, we seek to use CUDA 11.8, as it's been well tested with Python 3.10, and convenient for this project. 

In the activated anaconda environment (remember, `conda activate LoGONE`), now type the following: 
```
conda install nvidia/label/cuda-11.8.0::cuda
```
Congrats! You now have cuda installed. You can verify this by typing the following in a terminal: 
```
nvcc --version
```

You should get something like the following: 
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

The last thing you'll have to do in this section is to install PyTorch. Since PyTorch versioning is coupled with CUDA versioning, that's why we have to do this all together. To install PyTorch with CUDA 11.8 bindings, do the following (still in conda environment): 
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
See [PyTorch "Getting Started Locally"](https://pytorch.org/get-started/locally/) for more details. 

#### CPU Only 
If you elected to use CPU only, you can install PyTorch by specfiying the CPU Python wheel:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### For both CPU and GPU 
Now we can install the remainder of the packages for this project. 
