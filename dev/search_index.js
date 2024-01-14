var documenterSearchIndex = {"docs":
[{"location":"functionality/device/#AbstractDeviceArray","page":"AbstractDeviceArray","title":"AbstractDeviceArray","text":"","category":"section"},{"location":"functionality/device/","page":"AbstractDeviceArray","title":"AbstractDeviceArray","text":"TODO: describe functionality","category":"page"},{"location":"functionality/host/#AbstractGPUArray","page":"AbstractGPUArray","title":"AbstractGPUArray","text":"","category":"section"},{"location":"functionality/host/","page":"AbstractGPUArray","title":"AbstractGPUArray","text":"TODO: describe functionality","category":"page"},{"location":"interface/#Interface","page":"Interface","title":"Interface","text":"","category":"section"},{"location":"interface/","page":"Interface","title":"Interface","text":"To extend the above functionality to a new array type, you should use the types and implement the interfaces listed on this page. GPUArrays is design around having two different array types to represent a GPU array: one that only ever lives on the host, and one that actually can be instantiated on the device (i.e. in kernels).","category":"page"},{"location":"interface/#Device-functionality","page":"Interface","title":"Device functionality","text":"","category":"section"},{"location":"interface/","page":"Interface","title":"Interface","text":"Several types and interfaces are related to the device and execution of code on it. First of all, you need to provide a type that represents your execution back-end and a way to call kernels:","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"GPUArrays.AbstractGPUBackend\nGPUArrays.AbstractKernelContext\nGPUArrays.gpu_call\nGPUArrays.thread_block_heuristic","category":"page"},{"location":"interface/#GPUArrays.gpu_call","page":"Interface","title":"GPUArrays.gpu_call","text":"gpu_call(kernel::Function, arg0, args...; kwargs...)\n\nExecutes kernel on the device that backs arg (see backend), passing along any arguments args. Additionally, the kernel will be passed the kernel execution context (see [AbstractKernelContext]), so its signature should be (ctx::AbstractKernelContext, arg0, args...).\n\nThe keyword arguments kwargs are not passed to the function, but are interpreted on the host to influence how the kernel is executed. The following keyword arguments are supported:\n\ntarget::AbstractArray: specify which array object to use for determining execution properties (defaults to the first argument arg0).\nelements::Int: how many elements will be processed by this kernel. In most circumstances, this will correspond to the total number of threads that needs to be launched, unless the kernel supports a variable number of elements to process per iteration. Defaults to the length of arg0 if no other keyword arguments that influence the launch configuration are specified.\nthreads::Int and blocks::Int: configure exactly how many threads and blocks are launched. This cannot be used in combination with the elements argument.\nname::String: inform the back end about the name of the kernel to be executed. This can be used to emit better diagnostics, and is useful with anonymous kernels.\n\n\n\n\n\n","category":"function"},{"location":"interface/","page":"Interface","title":"Interface","text":"You then need to provide implementations of certain methods that will be executed on the device itself:","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"GPUArrays.AbstractDeviceArray\nGPUArrays.LocalMemory\nGPUArrays.synchronize_threads\nGPUArrays.blockidx\nGPUArrays.blockdim\nGPUArrays.threadidx\nGPUArrays.griddim","category":"page"},{"location":"interface/#GPUArrays.AbstractDeviceArray","page":"Interface","title":"GPUArrays.AbstractDeviceArray","text":"AbstractDeviceArray{T, N} <: DenseArray{T, N}\n\nSupertype for N-dimensional GPU arrays (or array-like types) with elements of type T. Instances of this type are expected to live on the device, see AbstractGPUArray for host-side objects.\n\n\n\n\n\n","category":"type"},{"location":"interface/#GPUArrays.LocalMemory","page":"Interface","title":"GPUArrays.LocalMemory","text":"Creates a block local array pointer with T being the element type and N the length. Both T and N need to be static! C is a counter for approriately get the correct Local mem id in CUDAnative. This is an internal method which needs to be overloaded by the GPU Array backends\n\n\n\n\n\n","category":"function"},{"location":"interface/#GPUArrays.synchronize_threads","page":"Interface","title":"GPUArrays.synchronize_threads","text":" synchronize_threads(ctx::AbstractKernelContext)\n\nin CUDA terms __synchronize in OpenCL terms: barrier(CLK_LOCAL_MEM_FENCE)\n\n\n\n\n\n","category":"function"},{"location":"interface/#Host-abstractions","page":"Interface","title":"Host abstractions","text":"","category":"section"},{"location":"interface/","page":"Interface","title":"Interface","text":"You should provide an array type that builds on the AbstractGPUArray supertype:","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"AbstractGPUArray","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"First of all, you should implement operations that are expected to be defined for any AbstractArray type. Refer to the Julia manual for more details, or look at the JLArray reference implementation.","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"To be able to actually use the functionality that is defined for AbstractGPUArrays, you should provide implementations of the following interfaces:","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"GPUArrays.backend","category":"page"},{"location":"testsuite/#Test-suite","page":"Test suite","title":"Test suite","text":"","category":"section"},{"location":"testsuite/","page":"Test suite","title":"Test suite","text":"GPUArrays provides an extensive test suite that covers all of the functionality that should be available after implementing the required interfaces. This test suite is part of this package, but for dependency reasons it is not available when importing the package. Instead, you should include the code from your runtests.jl as follows:","category":"page"},{"location":"testsuite/","page":"Test suite","title":"Test suite","text":"import GPUArrays\ngpuarrays = pathof(GPUArrays)\ngpuarrays_root = dirname(dirname(gpuarrays))\ninclude(joinpath(gpuarrays_root, \"test\", \"testsuite.jl\"))","category":"page"},{"location":"testsuite/","page":"Test suite","title":"Test suite","text":"With this set-up, you can run the test suite like this:","category":"page"},{"location":"testsuite/","page":"Test suite","title":"Test suite","text":"TestSuite.test(MyGPUArrayType)","category":"page"},{"location":"testsuite/","page":"Test suite","title":"Test suite","text":"If you don't want to run the whole suite, you can also run parts of it:","category":"page"},{"location":"testsuite/","page":"Test suite","title":"Test suite","text":"T = JLArray\nGPUArrays.allowscalar(false) # fail tests when slow indexing path into Array type is used.\n\nTestSuite.test_gpuinterface(T) # interface functions like gpu_call, threadidx, etc\nTestSuite.test_base(T) # basic functionality like launching a kernel on the GPU and Base operations\nTestSuite.test_blas(T) # tests the blas interface\nTestSuite.test_broadcasting(T) # tests the broadcasting implementation\nTestSuite.test_construction(T) # tests all kinds of different ways of constructing the array\nTestSuite.test_linalg(T) # linalg function tests\nTestSuite.test_mapreduce(T) # mapreduce sum, etc\nTestSuite.test_indexing(T) # indexing tests\nTestSuite.test_random(T) # randomly constructed arrays\nTestSuite.test_io(T)","category":"page"},{"location":"#GPUArrays.jl","page":"Home","title":"GPUArrays.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GPUArrays is a package that provides reusable GPU array functionality for Julia's various GPU backends. Think of it as the AbstractArray interface from Base, but for GPU array types. It allows you to write generic julia code for all GPU platforms and implements common algorithms for the GPU. Like Julia Base, this includes BLAS wrapper, FFTs, maps, broadcasts and mapreduces. So when you inherit from GPUArrays and overload the interface correctly, you will get a lot of functionality for free. This will allow to have multiple GPUArray implementation for different purposes, while maximizing the ability to share code.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package is not intended for end users! Instead, you should use one of the packages that builds on GPUArrays.jl. There is currently only a single package that actively builds on these interfaces, namely CuArrays.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"In this documentation, you will find more information on the interface that you are expected to implement, the functionality you gain by doing so, and the test suite that is available to verify your implementation. GPUArrays.jl also provides a reference implementation of these interfaces on the CPU: The JLArray array type uses Julia's parallel programming functionality to simulate GPU execution, and will be used throughout this documentation to illustrate functionality.","category":"page"}]
}
