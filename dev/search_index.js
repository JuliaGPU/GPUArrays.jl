var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#GPUArrays-Documentation-1",
    "page": "Home",
    "title": "GPUArrays Documentation",
    "category": "section",
    "text": "GPUArrays is an abstract interface for GPU computations. Think of it as the AbstractArray interface in Julia Base but for GPUs. It allows you to write generic julia code for all GPU platforms and implements common algorithms for the GPU. Like Julia Base, this includes BLAS wrapper, FFTs, maps, broadcasts and mapreduces. So when you inherit from GPUArrays and overload the interface correctly, you will get a lot of functionality for free. This will allow to have multiple GPUArray implementation for different purposes, while maximizing the ability to share code. Currently there are two packages implementing the interface namely CLArrays and CuArrays. As the name suggests, the first implements the interface using OpenCL and the latter uses CUDA."
},

{
    "location": "#GPUArrays.gpu_call",
    "page": "Home",
    "title": "GPUArrays.gpu_call",
    "category": "function",
    "text": "gpu_call(kernel::Function, A::GPUArray, args::Tuple, configuration = length(A))\n\nCalls function kernel on the GPU. A must be an GPUArray and will help to dispatch to the correct GPU backend and supplies queues and contexts. Calls the kernel function with kernel(state, args...), where state is dependant on the backend and can be used for getting an index into A with linear_index(state). Optionally, a launch configuration can be supplied in the following way:\n\n1) A single integer, indicating how many work items (total number of threads) you want to launch.\n    in this case `linear_index(state)` will be a number in the range `1:configuration`\n2) Pass a tuple of integer tuples to define blocks and threads per blocks!\n\n\n\n\n\n"
},

{
    "location": "#GPUArrays.linear_index-Tuple{Any}",
    "page": "Home",
    "title": "GPUArrays.linear_index",
    "category": "method",
    "text": "linear_index(state)\n\nlinear index corresponding to each kernel launch (in OpenCL equal to getglobalid).\n\n\n\n\n\n"
},

{
    "location": "#GPUArrays.global_size-Tuple{Any}",
    "page": "Home",
    "title": "GPUArrays.global_size",
    "category": "method",
    "text": "global_size(state)\n\nGlobal size == blockdim * griddim == total number of kernel execution\n\n\n\n\n\n"
},

{
    "location": "#GPUArrays.@linearidx-Tuple{Any,Any}",
    "page": "Home",
    "title": "GPUArrays.@linearidx",
    "category": "macro",
    "text": "linearidx(A, statesym = :state)\n\nMacro form of linear_index, which calls return when out of bounds. So it can be used like this:\n\n```julia\nfunction kernel(state, A)\n    idx = @linear_index A state\n    # from here on it\'s save to index into A with idx\n    @inbounds begin\n        A[idx] = ...\n    end\nend\n```\n\n\n\n\n\n"
},

{
    "location": "#GPUArrays.@cartesianidx-Tuple{Any,Any}",
    "page": "Home",
    "title": "GPUArrays.@cartesianidx",
    "category": "macro",
    "text": "cartesianidx(A, statesym = :state)\n\nLike @linearidx(A, statesym = :state), but returns an N-dimensional NTuple{ndim(A), Int} as index\n\n\n\n\n\n"
},

{
    "location": "#GPUArrays.synchronize_threads-Tuple{Any}",
    "page": "Home",
    "title": "GPUArrays.synchronize_threads",
    "category": "method",
    "text": " synchronize_threads(state)\n\nin CUDA terms __synchronize in OpenCL terms: barrier(CLK_LOCAL_MEM_FENCE)\n\n\n\n\n\n"
},

{
    "location": "#GPUArrays.device-Tuple{AbstractArray}",
    "page": "Home",
    "title": "GPUArrays.device",
    "category": "method",
    "text": "device(A::AbstractArray)\n\nGets the device associated to the Array A\n\n\n\n\n\n"
},

{
    "location": "#GPUArrays.synchronize-Tuple{AbstractArray}",
    "page": "Home",
    "title": "GPUArrays.synchronize",
    "category": "method",
    "text": "synchronize(A::AbstractArray)\n\nBlocks until all operations are finished on A\n\n\n\n\n\n"
},

{
    "location": "#GPUArrays.@LocalMemory-Tuple{Any,Any,Any}",
    "page": "Home",
    "title": "GPUArrays.@LocalMemory",
    "category": "macro",
    "text": "Creates a local static memory shared inside one block. Equivalent to __local of OpenCL or __shared__ (<variable>) of CUDA.\n\n\n\n\n\n"
},

{
    "location": "#The-Abstract-GPU-interface-1",
    "page": "Home",
    "title": "The Abstract GPU interface",
    "category": "section",
    "text": "Different GPU computation frameworks like CUDA and OpenCL, have different names for accessing the same hardware functionality. E.g. how to launch a GPU Kernel, how to get the thread index and so forth. GPUArrays offers a unified abstract interface for these functions. This makes it possible to write generic code that can be run on all hardware. GPUArrays itself even contains a pure Julia implementation of this interface. The julia reference implementation is a great way to debug your GPU code, since it offers more informative errors and debugging information compared to the GPU backends - which mostly silently error or give cryptic errors (so far).You can use the reference implementation by using the GPUArrays.JLArray type.The functions that are currently part of the interface:The low level dim + idx function, with a similar naming scheme as in CUDA:# with * being either of x, y or z\nblockidx_*(state), blockdim_*(state), threadidx_*(state), griddim_*(state)\n# Known in OpenCL as:\nget_group_id,      get_local_size,    get_local_id,       get_num_groupsHigher level functionality:gpu_call(f, A::GPUArray, args::Tuple, configuration = length(A))\n\nlinear_index(state)\n\nglobal_size(state)\n\n@linearidx(A, statesym = :state)\n\n@cartesianidx(A, statesym = :state)\n\nsynchronize_threads(state)\n\ndevice(A::AbstractArray)\n\nsynchronize(A::AbstractArray)\n\n@LocalMemory(state, T, N)"
},

{
    "location": "#The-abstract-TestSuite-1",
    "page": "Home",
    "title": "The abstract TestSuite",
    "category": "section",
    "text": "Since all array packages inheriting from GPUArrays need to offer the same functionality and interface, it makes sense to test them in the same way. This is why GPUArrays contains a test suite which can be called with the array type you want to test.You can run the test suite like this:using GPUArrays, GPUArrays.TestSuite\nTestSuite.run_tests(MyGPUArrayType)If you don\'t want to run the whole suite, you can also run parts of it:Typ = JLArray\nGPUArrays.allowslow(false) # fail tests when slow indexing path into Array type is used.\n\nTestSuite.run_gpuinterface(Typ) # interface functions like gpu_call, threadidx, etc\nTestSuite.run_base(Typ) # basic functionality like launching a kernel on the GPU and Base operations\nTestSuite.run_blas(Typ) # tests the blas interface\nTestSuite.run_broadcasting(Typ) # tests the broadcasting implementation\nTestSuite.run_construction(Typ) # tests all kinds of different ways of constructing the array\nTestSuite.run_fft(Typ) # fft tests\nTestSuite.run_linalg(Typ) # linalg function tests\nTestSuite.run_mapreduce(Typ) # mapreduce sum, etc\nTestSuite.run_indexing(Typ) # indexing tests"
},

]}
