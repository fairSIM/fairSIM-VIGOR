
<link href="style.css" type="text/css" rel="stylesheet"></link>

# linalg with JNI and CUDA

Notes concerning the concept of linking the fairSIM JAVA code
to CUDA. This might be helpful for people wanting to read the
actual code and/or to debug errors. Most of the code is also
commented, but this has the high level concept.

## Vectors in JAVA and CUDA

* `org.fairsim.accel` package handles "accelerated" vectors
* `AccelVectorReal/Cplx` in that package call into the `cuda_????.cu` files here via JNI
* `AccelVectorFactory` handles vector creation (in Java and on GPU) and such
* Each vector has a struct (see `cuda_common.h`), allocated native code, with a pointer to it held in JAVA 

## Memory, vector allocation

Concerning memory:

* All vectors occupy CUDA device memory (obviously)
* All vectors occupy some (0.01 to 0.02 of size) additional buffers for reduce operations (scalar product, norm) on host and device
* All vectors occupy permanent host-side copy in Java / the JVM (thus, float [])
* The CPU-side copy could be omitted by slight changes on the Java side, creating a CPU-side copy only when needed (but I do not see a downside to having it permanently)

Concerning constructors and finalize:

* Constructors of `AccelVectorReal/Cplx` call into native code to allocate vectors
* `finalize` is overloaded, so GC on these vectors cleans them up on the GPU side
* Various vector functions (`add(...)`, `fft2d(...)` ) are overloaded to call into GPU code
* Operations not yet implemented in GPU code semi-automatically fall back to JAVA implementation
* JAVA<->CUDA copies only occur when needed: Vectors (their JAVA code) keeps track if most recent version is on CPU or GPU
* This allows for efficient `get(x,y)`, `set(x,y)` in loops, as only the first call updates the CPU-sided copy

## Copy between JAVA<->CUDA 

There are now 3 versions of host<->device copy implemented (see also `org.fairsim.accel.TestAccel` to run
a performance test):

0.  standard `cudaMemcpy( .. )`
1.  async `cudaMemcpyAsync( .. )` with temporally pinned host memory
2.  async `cudaMemcpyAsync( .. )` through a pinned buffer in native host memory

All methods first obtain a pointer to the vectors `float []` array in
native / JNI code, through `GetPrimitiveArrayCritical()`.

### Method 0

simply copies from the Java-side pointer to the GPU with 
standard synchronous `cudaMemcpy(..)`, releases the critical array, returns.  Pro / contra:

* This should always work (no allocation, buffers, etc.), so its a good fallback
* The host memory is not pinned, so performance might be degraded
* More importantly, the call is completely blocking (as in, `cudaSyncDevice()`-like blocking), so no overlap with kernel execution.
* Current CUDA does "separate default stream per host thread" (compile flag), but I am not sure if this helps here:
* Memory is still not pinned, so I would guess the copy call still blocks the full device?

### Method 1

uses `cudaMemcpyAsync( .. )`. To do so, the pointer obtained from
JNI (to Javas float []) is page-locked (`cudaHostRegister( .. )`), the memcpy
is issued to the vectors stream (see next section), the host code waits for completion
of the copy, de-registers the pinning and releases the critical pointer. Pro / contra: 

* `cudaHostRegister` has lost of overhead, so much as to degrade performance compared to both other methods
* We still need to block the host thread (`cudaSyncStream( .. )`), as we have to wait for copy to complete to free the critical pointer
* We cannot release the array through a CUDA stream callback to run fully async:<br /> 
  copy is still in progress AND Java is not allowed to access it while we hold a critical pointer
* The advantage, however, is that we do not block the GPU *for other threads*. This means, when
  method 0 blocks (even with the automatic per-thread streams in current CUDA), this might be faster.

### Method 2

also uses `cudaMemcpyAsync( .. )`, but through a two-step process, copying into a fixed, pinned, host-sided
buffer first: A pool of pre-allocated, pinned (`cudaAllocHost(..)`) buffers is managed in Java. The copy
command obtains a buffer (thread-save), copies the vector from the Java pointer (plain `memcpy`), and 
releases the critical array. It then issues a `cudaMemcpyAsync(..)` from that buffer to device memory,
in the vectors stream (see below). A callback function is issued into the stream, which
returns the host-side buffer into the pool as soon as the copy is completed. Pro / contra:

* Despite the double-copy (Java->Host, Host->Device), it benchmarks as (among the) fastest way to do the copy
* It occupies some CPU memory for the host-side buffer (but on typical systems, CPU-mem >> GPU-mem)
* This runs fully async: The first copy (Java->Host) only blocks the CPU thread, the second (Host->Device)
  is async in the vectors stream, so it can be interleaved with kernel execution.

## Vector synchronization

To facilitate parallel execution and memory copies, without (necessary) relying on multiple
CPU threads, the CUDA interface makes use of CUDA streams.

* Each vector generates its own stream on allocation
* Copy operations (see above) sync to that stream
* Mathematical operations that only involve the vector itself (`fft2d(..)`, `scal(0.1)`, ...) 
  also sync to the stream. If they return results (`norm2()`), they of course implicitly block
  the CPU thread.
* Mathematical operations that involve a right-hand side (`add( Vec a )`, `axpy( Vec a, ... )`) 
  wait for all operations in the stream of the right-hand side vector to finish, and then
  block the right-hand side stream until the operation issued on the
  left-hand side has completed.
    * This scheme should ensure the code is correct :) 
    * The synchronization is achieved completely through CUDA events, 
      so the CPU-side thread does not have to block

I think this scheme allows a good compromise between performance (overlap of memory copies
and computation kernels), without introducing too much additional low-level management
into the actual algorithms. At most, some clever reordering of loops (when to copy and when to
compute) should be everything thats needed to get a good speedup of the GPU code.


## Pitfalls concerning thread safety

Please keep in mind that the synchronization scheme laid out
above allows to introduce deadlocks when using the same vectors from multiple threads
without additional synchronization:

    Thread 1: a.add(b);
    Thread 2: b.add(c);
    Thread 3: c.add(a);

where now a waits for b, b waits for c, c waits for a. 

These problems should only occur in code thats already incorrect in the first place: The
the results of the example above will for example change depending on the 
random thread execution order. However, while on the CPU this example only introduces
wrong results, it now potentially also introduces dead-locks.

## My notes on CUDA event interface

`cudaStreamWaitEvent( evt )` has some requirements and features concerning the host-side of the code:

* The record call of the `evt` must have returned before we can wait on it. This has
  to be ensured by thread sync or simply by the correct order when running a single thread.
* If `evt` is recorded multiple times to the stream, it will wait for the last recorded position
* After the call is complete, the event may be released!

























