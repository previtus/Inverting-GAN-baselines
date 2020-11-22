# Inverting-GAN-baselines
Exploring methods of inverting GANs (projecting from the data space / image space into the latent representation space)


## Explored papers

| paper                                          | code available | compatability details | notes                                                                                                |
|------------------------------------------------|----------------|-----------------------|------------------------------------------------------------------------------------------------------|
| In-Domain GAN Inversion for Real Image Editing          | available      | TF 1.12, CUDA 9       | encode + optimize, not real-time, focus on semantically meaningful embedding, paper presents 256x256 |
| Image2StyleGAN / Image2StyleGAN++                       | nope           |                       | slow, best reconstructed quality, might not be the most semantically meaningful                      |
| public Encoder to StyleGAN implementations              | multiple       | (depends)             | fast, relatively low quality                                                                         |
| Adversarial Latent Autoencoders                         | available      | TF 1.10 + CUDA 9      | (fast, need to test ...)                                                                             |
| Encoding in Style: A StyleGAN Encoder for Image-to-Image Translation | available   |  Pytorch    | (fast, need to test ...)                                                                             |
| (More to be added)                                      |                |                       |                                                                                                      |


[//]: # (|                                                       |                |                       |                                                 |)


## Preparation

Dependency on different versions of Tensorflow might also require multiple CUDA version installations (namely from around TF 1.13.0 CUDA 10 is needed, while the older versions need CUDA 9). Note that there might be some other custom builds, but my solution was to install all needed CUDA versions and swap between them. When installing the second version, don't override your link in "/usr/local/cuda". Also comment the paths initialization in your .bashrc file (so that by default you have no CUDA version loaded on a freshly opened terminal).

It's possible to switch between CUDA versions depending on which project you run (and which TF it needs).

### CUDA 9 codes

Let's start by loading the CUDA 9 paths (you can also have this in a file cuda-9.0-env and run '''source cuda-9.0-en '''):

'''
export PATH=$PATH:/usr/local/cuda-9.0/bin
export CUDADIR=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
'''

PS: Now you can test '''nvcc --version'''.

Now load your environment with the correct TF version (you might have one for each baseline), I use Anaconda.

PS: Now your should also test with '''python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"'''

### CUDA 10 codes

Same as in the CUDA 9 example, except your paths should point to /usr/local/cuda-10.0 and you would load different Anaconda environments.

'''
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDADIR=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
'''

## Running

(To be continued ... either use each code's instructions, or write your own "runner" ...)