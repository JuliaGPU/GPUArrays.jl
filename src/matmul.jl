import CUDAnative

function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize) where T
    ilin = linear_index(state)
    idx = GPUArrays.gpu_ind2sub(outSize, ilin)
    if (idx[1] > outSize[1] || idx[2] > outSize[2])
        return
    end

    # const int globalRow = get_global_id(0); // Row ID of C (0..M)
    # const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    # // Compute a single element (loop over K)
    acc = 0.0
    for k in 1:Asize[2]
        acc += A[(k - 1) * Asize[1] + (idx[1] - 1) + 1] * B[(idx[2] - 1) *Asize[2] + (k - 1) + 1];
    end
    out[(idx[2] - 1) * Asize[1] + (idx[1] - 1) + 1] = acc
    return
end


function matmul(a, b)
    Asize = size(a)
    Bsize = size(b)
    out = similar(a, Asize[1], Bsize[2])
    outSize = size(out)
    Asize = UInt32.(Asize)
    gpu_call(matmul_kernel, out, (a,b, out, UInt32.(Asize), UInt32.(Bsize), UInt32.(outSize)))
    out
end


