# Wiggle-GAN
Base : [[Tensorflow version]](https://github.com/hwalsuklee/tensorflow-generative-model-collections)

This repository is included code for CPU mode Pytorch, but i did not test. I tested only in GPU mode Pytorch.

## Results

| Image (*Input*) | DepthMap (*Input*) | AE (*output*) | Wiggle-GAN (*output*) | Wiggle-GAN noCR (*output*) |
|:---:|:---:|:---:|:---:|:---:|
| <img src="/Images/Input-Test/1.png" width="128" height="128"> | <img src="/Images/Input-Test/1_d.png" width="128" height="128">  |![](/git_images/Solutions/1_ae.gif) | ![](/git_images/Solutions/1_w.gif)|![](/git_images/Solutions/1_ncr.gif) |
| <img src="/Images/Input-Test/2.png" width="128" height="128"> | <img src="/Images/Input-Test/2_d.png" width="128" height="128">  |![](/git_images/Solutions/2_ae.gif) | ![](/git_images/Solutions/2_w.gif)|![](/git_images/Solutions/2_ncr.gif) |
| <img src="/Images/Input-Test/4.png" width="128" height="128"> | <img src="/Images/Input-Test/4_d.png" width="128" height="128">  |![](/git_images/Solutions/4_ae.gif) | ![](/git_images/Solutions/4_w.gif)|![](/git_images/Solutions/4_ncr.gif) |
| <img src="/Images/Input-Test/8.png" width="128" height="128"> | <img src="/Images/Input-Test/8_d.png" width="128" height="128">  |![](/git_images/Solutions/8_ae.gif) | ![](/git_images/Solutions/8_w.gif)|![](/git_images/Solutions/8_ncr.gif) |
| <img src="/Images/Input-Test/9.png" width="128" height="128"> | <img src="/Images/Input-Test/9_d.png" width="128" height="128">  |![](/git_images/Solutions/9_ae.gif) | ![](/git_images/Solutions/9_w.gif)|![](/git_images/Solutions/9_ncr.gif) |
| <img src="/Images/Input-Test/10.png" width="128" height="128"> | <img src="/Images/Input-Test/10_d.png" width="128" height="128">  |![](/git_images/Solutions/10_ae.gif) | ![](/git_images/Solutions/10_w.gif)|![](/git_images/Solutions/10_ncr.gif) |
| <img src="/Images/Input-Test/11.png" width="128" height="128"> | <img src="/Images/Input-Test/11_d.png" width="128" height="128">  |![](/git_images/Solutions/11_ae.gif) | ![](/git_images/Solutions/11_w.gif)|![](/git_images/Solutions/11_ncr.gif) |
| <img src="/Images/Input-Test/12.png" width="128" height="128"> | <img src="/Images/Input-Test/12_d.png" width="128" height="128">  |![](/git_images/Solutions/12_ae.gif) | ![](/git_images/Solutions/12_w.gif)|![](/git_images/Solutions/12_ncr.gif) |


## Changing the λ_influenceL1

### In Wiggle-GAN

| Image (*Input*) | DepthMap (*Input*) | λ_infL1 = 0 | λ_infL1 = 2 | λ_infL1 = 50 |
|:---:|:---:|:---:|:---:|:---:|
| <img src="/Images/Input-Test/1.png" width="128" height="128"> | <img src="/Images/Input-Test/1_d.png" width="128" height="128">  |![](/git_images/L0/1_w.gif) | ![](/git_images/L2/1_w.gif)|![](/git_images/Solutions/1_w.gif) |
| <img src="/Images/Input-Test/2.png" width="128" height="128"> | <img src="/Images/Input-Test/2_d.png" width="128" height="128">  |![](/git_images/L0/2_w.gif) | ![](/git_images/L2/2_w.gif)|![](/git_images/Solutions/2_w.gif) |
| <img src="/Images/Input-Test/4.png" width="128" height="128"> | <img src="/Images/Input-Test/4_d.png" width="128" height="128">  |![](/git_images/L0/4_w.gif) | ![](/git_images/L2/4_w.gif)|![](/git_images/Solutions/4_w.gif) |
| <img src="/Images/Input-Test/8.png" width="128" height="128"> | <img src="/Images/Input-Test/8_d.png" width="128" height="128">  |![](/git_images/L0/8_w.gif) | ![](/git_images/L2/8_w.gif)|![](/git_images/Solutions/8_w.gif) |
| <img src="/Images/Input-Test/9.png" width="128" height="128"> | <img src="/Images/Input-Test/9_d.png" width="128" height="128">  |![](/git_images/L0/9_w.gif) | ![](/git_images/L2/9_w.gif)|![](/git_images/Solutions/9_w.gif) |
| <img src="/Images/Input-Test/10.png" width="128" height="128"> | <img src="/Images/Input-Test/10_d.png" width="128" height="128">  |![](/git_images/L0/10_w.gif) | ![](/git_images/L2/10_w.gif)|![](/git_images/Solutions/10_w.gif) |
| <img src="/Images/Input-Test/11.png" width="128" height="128"> | <img src="/Images/Input-Test/11_d.png" width="128" height="128">  |![](/git_images/L0/11_w.gif) | ![](/git_images/L2/11_w.gif)|![](/git_images/Solutions/11_w.gif) |
| <img src="/Images/Input-Test/12.png" width="128" height="128"> | <img src="/Images/Input-Test/12_d.png" width="128" height="128">  |![](/git_images/L0/12_w.gif) | ![](/git_images/L2/12_w.gif)|![](/git_images/Solutions/12_w.gif) |

### In Wiggle-GAN with no consistency regularization

| Image (*Input*) | DepthMap (*Input*) | λ_infL1 = 0 | λ_infL1 = 2 | λ_infL1 = 50 |
|:---:|:---:|:---:|:---:|:---:|
| <img src="/Images/Input-Test/1.png" width="128" height="128"> | <img src="/Images/Input-Test/1_d.png" width="128" height="128">  |![](/git_images/L0/1_ncr.gif) | ![](/git_images/L2/1_ncr.gif)|![](/git_images/Solutions/1_ncr.gif) |
| <img src="/Images/Input-Test/2.png" width="128" height="128"> | <img src="/Images/Input-Test/2_d.png" width="128" height="128">  |![](/git_images/L0/2_ncr.gif) | ![](/git_images/L2/2_ncr.gif)|![](/git_images/Solutions/2_ncr.gif) |
| <img src="/Images/Input-Test/4.png" width="128" height="128"> | <img src="/Images/Input-Test/4_d.png" width="128" height="128">  |![](/git_images/L0/4_ncr.gif) | ![](/git_images/L2/4_ncr.gif)|![](/git_images/Solutions/4_ncr.gif) |
| <img src="/Images/Input-Test/8.png" width="128" height="128"> | <img src="/Images/Input-Test/8_d.png" width="128" height="128">  |![](/git_images/L0/8_ncr.gif) | ![](/git_images/L2/8_ncr.gif)|![](/git_images/Solutions/8_ncr.gif) |
| <img src="/Images/Input-Test/9.png" width="128" height="128"> | <img src="/Images/Input-Test/9_d.png" width="128" height="128">  |![](/git_images/L0/9_ncr.gif) | ![](/git_images/L2/9_ncr.gif)|![](/git_images/Solutions/9_ncr.gif) |
| <img src="/Images/Input-Test/10.png" width="128" height="128"> | <img src="/Images/Input-Test/10_d.png" width="128" height="128">  |![](/git_images/L0/10_ncr.gif) | ![](/git_images/L2/10_ncr.gif)|![](/git_images/Solutions/10_ncr.gif) |
| <img src="/Images/Input-Test/11.png" width="128" height="128"> | <img src="/Images/Input-Test/11_d.png" width="128" height="128">  |![](/git_images/L0/11_ncr.gif) | ![](/git_images/L2/11_ncr.gif)|![](/git_images/Solutions/11_ncr.gif) |
| <img src="/Images/Input-Test/12.png" width="128" height="128"> | <img src="/Images/Input-Test/12_d.png" width="128" height="128">  |![](/git_images/L0/12_ncr.gif) | ![](/git_images/L2/12_ncr.gif)|![](/git_images/Solutions/12_ncr.gif) |


## Development Environment
* Ubuntu 16.04 LTS
* NVIDIA GTX 1080 ti
* cuda 9.0
* Python 3.5.2
* pytorch 0.4.0
* torchvision 0.2.1
* numpy 1.14.3
* matplotlib 2.2.2
* imageio 2.3.0
* scipy 1.1.0
* certifi==2019.11.28
* chardet==3.0.4
* cycler==0.10.0
* idna==2.8
* imageio==2.5.0
* jsonpatch==1.24
* jsonpointer==2.0
* kiwisolver==1.1.0
* matplotlib==3.1.1
* numpy==1.17.2
* Pillow==6.1.0
* pyparsing==2.4.2
* python-dateutil==2.8.0
* PyYAML==5.1.2
* pyzmq==18.1.1
* requests==2.22.0
* scipy==1.1.0
* six==1.12.0
* torch==1.0.1
* torchfile==0.1.0
* tornado==6.0.3
* urllib3==1.25.7
* visdom==0.1.8.9
* websocket-client==0.56.0

## Acknowledgements
This implementation has been based on [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections) and tested with Pytorch 0.4.0 on Ubuntu 16.04 using GPU.

