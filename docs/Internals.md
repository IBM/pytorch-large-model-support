## PyTorch Large Model Support Internals

The implemenation of Large Model Support introduces some basic tensor states.  Understanding these states and the actions that trigger state transitions provides a good overview of how LMS works.

![LMS phase diagram](/docs/images/LMS_Flow.png)

A newly constructed tensor starts in the Allocated state.  Tensors may be destroyed in any state other than Active.<br>
Any transition to/from the Reclaimed state (crossing the dotted line in the figure above) requires a transfer of the tensor's data between GPU and host memory.

Actions/Triggers:
* _Pin_: Performed on all input tensors to a given operation (e.g. network layer) prior to data access and computation.
* _Unpin_: Reverse of the pin operation.  Performed after operation completes.
* _Reclaim_: Performed by the CUDA Caching Allocator as needed to satisfy new allocation requests.  This is done only when the free list (cache) contains no suitable allocations and the allocation limit has been met.  The operation is performed on a minimal subset of inactive tensors in order to satisfy the allocation request.
* _Access_: This represents a request to access the data of an unpinned tensor. This is rare.

## Implementation Details

CUDA Caching Allocator (`c10::cuda::CudaCachingAllocator` et al.)
* Add per-device allocator object (`DeviceCachingAllocator`) to reduce lock contention and `BlockPool` management costs. The new device-specific allocator manages BlockPools (free blocks) and the set of currently inactive tensors (`reclaim_list`) for each GPU.
* Add management of LMS settings (enabled, allocation limit) to `THCCachingAllocator` (`lms_settings`).
* Provide CUDA-specific implementation of `LmsStorageImpl` (`CudaLmsStorageImpl`).  This defines the low level Tensor operations required for LMS (page-out, page-in, reclaim-list-add/remove).
* Provide CUDA-specific implementation of `Allocator::AsLmsStorage()` (`CudaCachingAllocator::AsLmsStorage()`).  When LMS is enabled, this supplies a new `CudaLmsStorageImpl` instance during `StorageImpl` construction -- effectively enabling LMS for any associated Tensor.
* Add ability to reclaim GPU memory from suitable inactive tensors in order to satisfy new allocation requests (`reclaim_block`).
* Add speculative page-out mechanism.  This predicts which tensors will be reclaimed and triggers early page-out (concurrent with the compute stream) to reduce the swapping latency (`predict_reclaim()`, `record_reclaim()`)
* Add new statistics (pinned, reclaimed, allocation distribution).

Allocator (`c10::Allocator`)
* Add `AsLmsStorage()` virtual function with default implementation that simply returns `nullptr`. LMS is not enabled/supported by default.  Subclasses must explicitly implement and advertise support.

LmsStorageImpl (`c10::LmsStorageImpl`)
* This new abstract class represents an LMS implementation.
* It defines operations required for LMS (pin, unpin, page-out, page-in, reclaim-list-add/remove) -- providing common logic (applicable across different implementations) and calling out to the low-level methods implemented in the allocator's derived class otherwise.

StorageImpl (`c10::StorageImpl`)
* Add member, `std::unique_ptr<LmsStorageImpl> lms_`.  This provides access to the underlying LMS implementation (if any) specified by the allocator during construction.
* Add high level entry points for operations required for LMS (pin, unpin, data-access). These are simply pass-throughs to the underlying LMS object.

IntrusiveList and IntrusiveListHook (`c10::IntrusiveList`, `c10::IntrusiveListHook`)
* These new classes are used to manage the set of inactive tensors.
* Element objects embed the `IntrustiveListHook`, which provides the following properties:
  * Insertion and removal operations are O(1) and require no memory allocation or deletion.
  * Element destruction is valid and can be performed safely regardless of list membership.

TensorGuard (`at::TensorGuard`)
* This new class ensures that a tensor's storage is pinned during an operation in which its data may be accessed.
* This is analogous to the existing `DeviceGuard`. Like `DeviceGuard`, these objects are instantiated in the operation-specific generated code (see `function_wrapper.py`) and leverage C++ scoping to pin/unpin the storage corresponding to the set of tensors involved in the given operation.

PyTorch Python API (`torch.cuda`)
* Add LMS control and tuning services (enable, allocation limit).
* Add LMS statistics to cuda `memory_stats` API (pinned, reclaimed, allocation distribution).

Unit Tests (`test_cuda.py`)
* Add `test_large_model_support`.
