#### &lt;Call for Feedback&gt;

A PyTorch LMS user recently opened an issue asking for community support to integrate LMS into an official version of PyTorch:

https://github.com/pytorch/pytorch/issues/35633

This is a good opportunity to gather any and all user testimonials and success stories to document the value of LMS in a public place.  Please feel free to share your support and any thoughts there in the conversation.

#### &lt;/Call for Feedback&gt;

***

# PyTorch Large Model Support

PyTorch Large Model Support (LMS) is a feature in the PyTorch provided
by [IBM Watson Machine Learning Community Edition](https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/) (WML CE) that allows the
successful training of deep learning models that would otherwise exhaust GPU
memory and abort with "out-of-memory" errors. LMS manages this
oversubscription of GPU memory by temporarily swapping tensors to host memory
when they are not needed.

One or more elements of a deep learning model can lead to GPU memory exhaustion.

These include:

 * Model depth and complexity
 * Base data size (for example, high-resolution images)
 * Batch size

Traditionally, the solution to this problem has been to modify the model until
it fits in GPU memory. This approach, however, can negatively impact
accuracy – especially if concessions are made by reducing data
fidelity or model complexity.

With LMS, deep learning models can scale significantly beyond what was
previously possible and, ultimately, generate more accurate results.

# Installing PyTorch Large Model Support

LMS is built into the `pytorch` conda package so it is installed by
default when you install the GPU enabled PyTorch from WML CE.
The support is currently available in the [WML CE conda channel](https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/#/).
For more information on this channel, how to add channels, and install
frameworks see [this WML CE install documentation](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.7.0/navigation/wmlce_install.htm).


# How to enable LMS

The LMS functionality is disabled by default in PyTorch and needs to be
enabled before your model creates tensors. Enabling LMS is
as simple as calling the enablement API at the start of your program:

```python
import torch
torch.cuda.set_enabled_lms(True)
```

# Examples
The ManyModel.py example, found in the [PyTorch LMS examples](examples/),
uses synthetic random images with multiple models provided by
PyTorch's torchvision to allow users a fast hands-on experience with
LMS. The example allows users to change the image size, explore auto-tuning,
and manually set the LMS tunable parameters on various model architectures.

# Usage tips

## Use NUMA pinning for single GPU use
If you are utilizing a single GPU it is recommended to use NUMA pinning to pin
the process to the CPU and memory that is on the same system socket as the
GPU being used. Pinning the process allows the fastest connection paths between
system memory and GPU memory, which reduces the training or inferencing time.
WML CE includes the numactl utility that can be used to do this pinning. It
can be installed with the `conda install numactl` command. The following
example shows how to specify a single GPU to be used and how to pin the
process to use the CPU cores and memory that are on the same socket
as the specified GPU:

```sh
export CUDA_VISIBLE_DEVICES=0
numactl --cpunodebind=0 --membind=0 python train.py
```

## Use Horovod when using more than one GPU
It is recommended to use Horovod distribution when using more than one GPU
because Horovod creates a separate process per GPU and automatically sets the
process have socket affinity with the GPU which allows the fastest
connection paths between system memory and GPU memory, which reduces the
training or inferencing time.

# Model memory usage analysis with allocator statistics
LMS adds a few statistics to the GPU memory statistics API such as
the distribution of allocation sources (free-list, cudaMalloc, LMS reclaim, etc), the amount
of memory swapped, and more. For more information on the new statistics
and examples of their usage see the [PyTorch LMS examples](examples/).

# Building PyTorch from source with Large Model Support
The [patches](patches/) directory contains git patches for the LMS code.
The file names correspond to tag levels in the
[PyTorch source](https://github.com/pytorch/pytorch/). To build
PyTorch from source with Large Model Support, check out the
specific PyTorch git tag and then apply the corresponding PyTorch Large
Model Support patch file.

For example:
```sh
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.4.0
git am /pytorch-large-model-support/patches/pytorch_v1.4.0_large_model_support.patch
```

# Contribution guidelines

If you want to contribute to PyTorch Large Model Support please read the
[contribution guidelines](CONTRIBUTING.md).
