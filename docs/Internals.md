## Table of Contents

* [Overview](#overview)
* [Usage](#usage)
* [Example](#example)
* [PyTorch API Extensions](#pytorch-api-extensions)
* [FAQ](#faq)
* [Implementation Details](#implementation-details)

## Overview

Large Model Support (LMS) allows the successful training of deep learning models that would otherwise exhaust GPU memory and abort with “out of memory” errors. LMS manages this oversubscription of GPU memory by temporarily swapping tensors to host memory when they are not needed.

One or more elements of a deep learning model can lead to GPU memory exhaustion. These include:

* Model depth and complexity
* Base data size (for example, high-resolution images)
* Batch size

Traditionally, the solution to this problem has been to modify the model until it fits in GPU memory. This approach, however, can negatively impact accuracy – especially if concessions are made by reducing data fidelity or model complexity.

With LMS, deep learning models can scale significantly beyond what was previously possible and, ultimately, generate more accurate results.

## Usage

A PyTorch program enables Large Model Support by calling `torch.cuda.set_enabled_lms(True)` prior to model creation.

In addition, a pair of tunables is provided to control how GPU memory used for tensors is managed under LMS.

* `torch.cuda.set_limit_lms(limit)`

Defines the soft limit in bytes on GPU memory allocated for tensors (default: 0).

By default, LMS favors GPU memory reuse (moving inactive tensors to host memory) over new allocations. This effectively minimizes GPU memory consumption.

However, when a limit is defined, the algorithm favors allocation of GPU memory up to the limit prior to swapping any tensors out to host memory. This allows the user to control the amount of GPU memory consumed when using LMS.

Tuning this limit to optimize GPU memory utilization, therefore, can reduce data transfers and improve performance. Since the ideal tuning for any given scenario may differ, it is considered a best practice to determine the value experimentally, arriving at the largest value that does not result in an out of memory error.

* `torch.cuda.set_size_lms(size)`

Defines the minimum tensor size in bytes that is eligible for LMS swapping (default: 1 MB).

Any tensor smaller than this value is exempt from LMS reuse and persists in GPU memory.

## Example

The PyTorch imagenet example provides a simple illustration of Large Model Support in action. ResNet-152 is a deep residual network that requires a significant amount of GPU memory.

On a system with a single 16 GB GPU, without LMS enabled, a training attempt with the default batch size of 256 will fail with insufficient GPU memory:

```
python main.py -a resnet152 -b 256 [imagenet-folder with train and val folders]
=> creating model 'resnet152'
[...]
RuntimeError: CUDA error: out of memory
```

After enabling LMS, the training proceeds without issue:

```
git diff
--- a/imagenet/main.py
+++ b/imagenet/main.py
@@ -90,6 +90,7 @@ def main():
                      world_size=args.world_size)
 # create model
 + torch.cuda.set_enabled_lms(True)
   if args.pretrained:
      print("=> using pre-trained model '{}'".format(args.arch))
      model = models.__dict__[args.arch](pretrained=True)
python main.py -a resnet152 -b 256 [imagenet-folder with train and val folders]
=> creating model 'resnet152'
Epoch: [0][0/5005] [...]
Epoch: [0][10/5005] [...]
Epoch: [0][20/5005] [...]
Epoch: [0][30/5005] [...]
Epoch: [0][40/5005] [...]
Epoch: [0][50/5005] [...]
Epoch: [0][60/5005] [...]
[...]
```

## PyTorch API Extensions

Large Model Support extends the `torch.cuda` package to provide the following control and tuning interfaces.

`torch.cuda.set_enabled_lms(enable)`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Enable/disable Large Model Support.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: enable (bool): desired LMS setting.

`torch.cuda.get_enabled_lms()`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Returns a bool indicating whether Large Model Support is currently enabled.

`torch.cuda.set_limit_lms(limit)`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sets the allocation limit (in bytes) for LMS.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: limit (int): soft limit on GPU memory allocated for tensors.

`torch.cuda.get_limit_lms()`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Returns the allocation limit (in bytes) for LMS.

`torch.cuda.set_size_lms(size)`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sets the minimum size (in bytes) for LMS.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters: size (int): any tensor smaller than this value is exempt from LMS reuse and persists in GPU memory.

`torch.cuda.get_size_lms()`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Returns the minimum size (in bytes) for LMS.

## FAQ

**Q1. What do you swap between CPU and GPU memory? My current understanding is, since almost all data are tensors in PyTorch, so you can swap everything including intermediate results, feature maps, weights, and biases?**

Correct.  Any inactive tensor allocated via PyTorch's CUDA Caching Allocator is eligible to be swapped out.  This includes all of the data types mentioned above.

***

**Q2. What is the swapping mechanism? When and how do you trigger the swapping? Some phase diagram illustration would be appreciated**

![LMS phase diagram](/docs/images/LMS_Flow.png)

A newly constructed tensor starts in the Allocated state.  Tensors may be destroyed in any state other than Active.<br>
Any transition to/from the Reclaimed state (crossing the dotted line in the figure above) requires a transfer of the tensor's data between GPU and host memory.

Actions/Triggers:
* _Pin_: Performed on all input tensors to a given operation (e.g. network layer) prior to data access and computation.
* _Unpin_: Reverse of the pin operation.  Performed after operation completes.
* _Reclaim_: Performed by the CUDA Caching Allocator as needed to satisfy new allocation requests.  This is done only when the free list (cache) contains no suitable allocations and the soft allocation limit has been met.  The operation is performed on a minimal subset of inactive tensors in order to satisfy the allocation request.
* _Access_: This represents a request to access the tensor data outside of a pin operation. This is rare, but may occur in some cases when accessing the output of an operation. It is even less likely to occur for a tensor that has already been reclaimed (due to the general FIFO management policy for the inactive tensor list and the recent use of the output tensor).

***

**Q3. Any experimental results about the benefits in terms of batch size increasing, larger data support, etc.?**

See [Sam Matzek's blog post](https://developer.ibm.com/linuxonpower/2018/07/27/tensorflow-large-model-support-case-study-3d-image-segmentation/) on the benefits of larger data support.  While specifically about Tensorflow, the general points are valid regardless of the framework used.

This [article](https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e) investigates the effects of batch size on training.  While it doesn't come to any universal conclusion, it shows that the ability to support larger batch sizes will provide value in some scenarios:
> It is generally accepted that there is some “sweet spot” for batch size between 1 and the entire training dataset that will provide the best generalization.

## Implementation Details

CUDA Caching Allocator (`c10::cuda::CudaCachingAllocator` et al.)
* Add per-device allocator object (`DeviceCachingAllocator`) to reduce lock contention and `BlockPool` management costs. The new device-specific allocator manages BlockPools (free blocks) and the set of currently inactive tensors for each GPU (`reclaim_list`).
* Add management of LMS settings (enabled, allocation limit, size threshold) to `THCCachingAllocator` (`lms_settings`).
* Provide CUDA-specific implementation of `LMSImpl` (`CudaLMSImpl`).  This defines the low level Tensor operations required for LMS (page-out, page-in, reclaim-list-add/remove).
* Provide CUDA-specific implementation of `Allocator::lms()` (`CudaCachingAllocator::lms()`).  When LMS is enabled, this supplies a new `CudaLMSImpl` instance during `StorageImpl` construction -- effectively enabling LMS for any associated Tensor.
* Add ability to reclaim GPU memory from suitable inactive tensors in order to satisfy new allocation requests (`try_lms_reclaim`).
* Add new statistic, `amount_active`, equal to the amount of allocated memory less the set inactive tensors.

Allocator (`c10::Allocator`)
* Add `lms()` virtual function with default implementation that simply returns `nullptr`. LMS is not enabled/supported by default.  Subclasses must explicitly implement and advertise support.

LMSImpl (`c10::LMSImpl`)
* This new abstract class represents an LMS implementation.
* It defines operations required for LMS (pin, unpin, page-out, page-in, reclaim-list-add/remove) -- providing common logic (applicable across different implementations) and calling out to the low-level methods implemented in the allocator's derived class otherwise.

LMS (`c10::LMS`)
* This new lightweight object is embedded within `StorageImpl` and provides access to the underlying LMS implementation (if any) specified during construction.
* It defines operations required for LMS (pin, unpin, page-out, page-in, reclaim-list-add/remove).  These are, for the most part, simply pass-throughs to the the underlying LMS implementation.

StorageImpl (`c10::StorageImpl`)
* Add member, `LMS lms_`.  This provides access to the underlying LMS implementation (if any) specified by the allocator during construction.
* Add high level entry points for operations required for LMS (pin, unpin, data-access, page-out). These are, for the most part, simply pass-throughs to the underlying LMS object.

IntrusiveList and IntrusiveListHook (`c10::IntrusiveList`, `c10::IntrusiveListHook`)
* These new classes are used to manage the set of inactive tensors.
* Element objects embed the `IntrustiveListHook`, which provides the following properties:
  * Insertion and removal operations are O(1) and require no memory allocation or deletion.
  * Element destruction is valid and can be performed safely regardless of list membership.

TensorGuard (`at::TensorGuard`)
* This new class ensures that a tensor's storage is pinned during an operation in which its data may be accessed.
* This is analogous to the existing `DeviceGuard`. Like `DeviceGuard`, these objects are instantiated in the operation-specific generated code (see `function_wrapper.py`) and leverage C++ scoping to pin/unpin the storage corresponding to the set of tensors involved in the given operation.

PyTorch Python API (`torch.cuda`)
* Add LMS control and tuning services (get/set-enabled, get/set-allocation-limit, get/set-size-threshold)
* Add access to active memory statistic (get-current, get-max, reset)

