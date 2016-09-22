

@target ptx function reduce_kernel{F<:Function, T, N, BlockSize}(
        op::F, g_idata::AbstractArray{T,N}, g_odata::AbstractArray{T,N},
        n, Val{BlockSize}
    )
    #extern __shared__ int sdata[];
    sdata = @cuStaticSharedMem(T, BlockSize)
    tid = threadIdx().x;
    i = blockIdx().x*(blockSize*2) + tid;
    gridSize = blockSize*2*gridDim().x;
    sdata[tid] = 0;
    while (i < n)
        sdata[tid] = op(sdata[tid], op(g_idata[i], g_idata[i+blockSize]))
        i += gridSize
    end
    sync_threads()
    if (blockSize >= 512
        (tid < 256) && (sdata[tid] = op(sdata[tid], sdata[tid + 256]))
        sync_threads()
    end
    if (blockSize >= 256
        (tid < 128) && (sdata[tid] = op(sdata[tid], sdata[tid + 128]))
        sync_threads()
    end
    if (blockSize >= 128
        (tid < 64) && (sdata[tid] = op(sdata[tid], sdata[tid + 64]))
        sync_threads()
    end
    if tid < 32
        (blockSize >= 64) && (sdata[tid] = op(sdata[tid], sdata[tid + 32]))
        (blockSize >= 32) && (sdata[tid] = op(sdata[tid], sdata[tid + 16]))
        (blockSize >= 16) && (sdata[tid] = op(sdata[tid], sdata[tid + 8]))
        (blockSize >= 8) && (sdata[tid] = op(sdata[tid], sdata[tid + 4]))
        (blockSize >= 4) && (sdata[tid] = op(sdata[tid], sdata[tid + 2]))
        (blockSize >= 2) && (sdata[tid] = op(sdata[tid], sdata[tid + 1]))
    end
    (tid == 0) && (g_odata[blockIdx.x] = sdata[0])
    nothing
end

using CUDAnative, CUDAdrv
const cu = CUDAnative
dev = CUDAdrv.CuDevice(0)
ctx = CUDAdrv.CuContext(dev)
