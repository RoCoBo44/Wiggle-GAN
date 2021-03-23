# Wiggle-GAN

## What is a Wigglegram?

Wigglegrams are a series of photos of the particular object from a different angle. When the photos are combined in a short animation they become three-dimensional thanks to optical illusion.

Here there are some examples:

<img src="/git_images/Intro/mikewinsdaly.gif" width="256" height="256"/> <img src="/git_images/Intro/wiggle_example.gif" width="256" height="256"/>


## Our solution

In our solution we propose the use of a neural network to generate a new image given a picture, his depth map and a direction. Also, in order to produce new samples the NN outputs the depth map that correlates to the output image:

<img src="/git_images/Architectures/general_solution.png" />

To achieve greater results we implemented a Wasserstein Generative Adversarial Network (WGAN) with Consistency Regularization (CR) and a L1 loss between the false image and expected one. The final loss can be seen in the image below:

<img src="/git_images/Losses/big_loss.png" />

Moreover, our generator is based on the U-Net architecture but we added another decoder: 

<img src="/git_images/Architectures/generator.png" />

The full Wiggle-GAN architecture can be seen in the next image, the black box represents the generator and T() means the transformation made to augmentate the data (for the CR loss)

<img src="/git_images/Architectures/critic.png" />

## Google Colab

Follow this steps if you want to generate your own wigglegrams:

1) Copy the folder with the checkpoints (https://drive.google.com/drive/folders/1NDtN_Eem1NYfibuWHlGY1G9deKNzJrw8?usp=sharing) in your google drive, and be sure to save it in the exact rute (Colab Notebooks/DataWiggle).
2) Go to this Google Colab File (https://colab.research.google.com/drive/1N5HJ1geVM1ymLoE5C2jkLs3F_tQn-s-r?usp=sharing) and execute all the blocks, remember to login with your google account in order to access the checkpoints.

## Results

| id | Image (*Input*) | DepthMap (*Input*) | AE (*output*) | Wiggle-GAN (*output*) | Wiggle-GAN noCR (*output*) |
|:---:|:---:|:---:|:---:|:---:|:---:|
|a| <img src="/Images/Input-Test/12.png" width="128" height="128"> | <img src="/Images/Input-Test/12_d.png" width="128" height="128">  |![](/git_images/Solutions/12_ae.gif) | ![](/git_images/Solutions/12_w.gif)|![](/git_images/Solutions/12_ncr.gif) |
|b| <img src="/Images/Input-Test/9.png" width="128" height="128"> | <img src="/Images/Input-Test/9_d.png" width="128" height="128">  |![](/git_images/Solutions/9_ae.gif) | ![](/git_images/Solutions/9_w.gif)|![](/git_images/Solutions/9_ncr.gif) |
|c| <img src="/Images/Input-Test/11.png" width="128" height="128"> | <img src="/Images/Input-Test/11_d.png" width="128" height="128">  |![](/git_images/Solutions/11_ae.gif) | ![](/git_images/Solutions/11_w.gif)|![](/git_images/Solutions/11_ncr.gif) |
|d| <img src="/Images/Input-Test/1.png" width="128" height="128"> | <img src="/Images/Input-Test/1_d.png" width="128" height="128">  |![](/git_images/Solutions/1_ae.gif) | ![](/git_images/Solutions/1_w.gif)|![](/git_images/Solutions/1_ncr.gif) |
|e| <img src="/Images/Input-Test/8.png" width="128" height="128"> | <img src="/Images/Input-Test/8_d.png" width="128" height="128">  |![](/git_images/Solutions/8_ae.gif) | ![](/git_images/Solutions/8_w.gif)|![](/git_images/Solutions/8_ncr.gif) |
|f| <img src="/Images/Input-Test/2.png" width="128" height="128"> | <img src="/Images/Input-Test/2_d.png" width="128" height="128">  |![](/git_images/Solutions/2_ae.gif) | ![](/git_images/Solutions/2_w.gif)|![](/git_images/Solutions/2_ncr.gif) |
|g| <img src="/Images/Input-Test/10.png" width="128" height="128"> | <img src="/Images/Input-Test/10_d.png" width="128" height="128">  |![](/git_images/Solutions/10_ae.gif) | ![](/git_images/Solutions/10_w.gif)|![](/git_images/Solutions/10_ncr.gif) |
|h| <img src="/Images/Input-Test/4.png" width="128" height="128"> | <img src="/Images/Input-Test/4_d.png" width="128" height="128">  |![](/git_images/Solutions/4_ae.gif) | ![](/git_images/Solutions/4_w.gif)|![](/git_images/Solutions/4_ncr.gif) |


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


## Incrementing the shift value to 10

| AE | Wiggle-GAN | Wiggle-GAN noCR |
|:---:|:---:|:---:|
|![](/git_images/shifts/Corr_AE_0.gif) | ![](/git_images/shifts/Corr_W_0.gif)|![](/git_images/shifts/Corr_WnoCR_0.gif) |
|![](/git_images/shifts/Corr_AE_1.gif) | ![](/git_images/shifts/Corr_W_1.gif)|![](/git_images/shifts/Corr_WnoCR_1.gif) |
|![](/git_images/shifts/Corr_AE_2.gif) | ![](/git_images/shifts/Corr_W_2.gif)|![](/git_images/shifts/Corr_WnoCR_2.gif) |
|![](/git_images/shifts/Corr_AE_3.gif) | ![](/git_images/shifts/Corr_W_3.gif)|![](/git_images/shifts/Corr_WnoCR_3.gif) |
|![](/git_images/shifts/Corr_AE_4.gif) | ![](/git_images/shifts/Corr_W_4.gif)|![](/git_images/shifts/Corr_WnoCR_4.gif) |
|![](/git_images/shifts/Corr_AE_5.gif) | ![](/git_images/shifts/Corr_W_5.gif)|![](/git_images/shifts/Corr_WnoCR_5.gif) |
|![](/git_images/shifts/Corr_AE_6.gif) | ![](/git_images/shifts/Corr_W_6.gif)|![](/git_images/shifts/Corr_WnoCR_6.gif) |
|![](/git_images/shifts/Corr_AE_7.gif) | ![](/git_images/shifts/Corr_W_7.gif)|![](/git_images/shifts/Corr_WnoCR_7.gif) |
|![](/git_images/shifts/Corr_AE_8.gif) | ![](/git_images/shifts/Corr_W_8.gif)|![](/git_images/shifts/Corr_WnoCR_8.gif) |
|![](/git_images/shifts/Corr_AE_9.gif) | ![](/git_images/shifts/Corr_W_9.gif)|![](/git_images/shifts/Corr_WnoCR_9.gif) |
|![](/git_images/shifts/Corr_AE_10.gif) | ![](/git_images/shifts/Corr_W_10.gif)|![](/git_images/shifts/Corr_WnoCR_10.gif) |

## Changing depth map in order to improve the results

| Image (input) | DepthMap (input) | Wiggle-GAN |
|:---:|:---:|:---:|
| <img src="/git_images/depth_map_changes/input.png" width="256" height="256"> | <img src="/git_images/depth_map_changes/1_d.png" width="256" height="256">  |![](/git_images/depth_map_changes/prev.gif) |
| <img src="/git_images/depth_map_changes/input.png" width="256" height="256"> | <img src="/git_images/depth_map_changes/2_d.png" width="256" height="256">  |![](/git_images/depth_map_changes/post.gif) |

## Splitting to work with bigger images

![](/git_images/split/small_complete.gif)<br/>

![](/git_images/split/small_top_left.gif)![](/git_images/split/small_top_right.gif)<br/>
![](/git_images/split/small_down_left.gif)![](/git_images/split/small_down_right.gif)<br/>

![](/git_images/split/big_complete.gif)<br/>


## Development environment
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
