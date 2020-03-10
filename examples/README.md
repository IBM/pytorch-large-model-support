# PyTorch Large Model Support Examples

This directory contains examples for using the PyTorch
Large Model Support (LMS).

## Adjustable image resolution ResNet, DenseNet, and other models

The [ManyModel.py](ManyModel.py) file (based on PyTorch's
[imagenet example](https://github.com/pytorch/examples/blob/ee964a2/imagenet/main.py))
uses the various models from `torchvision` to demonstrate PyTorch
Large Model Support in models that cannot fit in GPU memory when using
larger resolution data. It provides a convenient way to test out the
capabilities of LMS with various model architectures (ResNet,
DenseNet, Inception, MobileNet, NASNet, etc.). Command line parameters
allow the user to change the size of the input image data, enable or
disable LMS, and log memory allocator statistics.

The ManyModel.py example can be run by adding the `examples` directory to
the PYTHONPATH and running like as shown:

```bash
cd examples
export PYTHONPATH=`pwd`
python ManyModel.py -h
```

## Memory Allocator statistics
PyTorch provides APIs to retrieve statistics from
the GPU memory allocator. These statistics provide a means to
do deeper analysis of a model's memory usage, including how often LMS
reclaims memory and how many bytes of memory are being reclaimed.

The [statistics module](lmsstats.py) provides a working example of how the APIs
can be used in used to log per-iteration and aggregate memory statistics. The
`LMSStatsLogger` and `LMSStatsSummary` classes in this module are used by the ManyModel
example to demonstrate how the statistics APIs can be used in model training.
