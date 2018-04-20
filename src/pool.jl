function maxpool2d_kernel(state, A::AbstractArray{T}, out, Asize, pool, stride, outSize) where T
    ilin = linear_index(state)
    idx = GPUArrays.gpu_ind2sub(Asize, ilin)
    if (idx[1] > outSize[1] || idx[2] > outSize[2] || idx[3] > outSize[3] || idx[4] > outSize[4])
        return
    end

    temp_max = A[((idx[1] - 1) * stride) + Asize[1] * (idx[2] - 1) * stride + (Asize[1] * Asize[2]) * (idx[3] - 1) + (Asize[1] * Asize[2] * Asize[3]) * (idx[4] - 1) + 1]
    max_pos = ((idx[1] - 1) * stride) + Asize[1] * (idx[2] - 1) * stride + (Asize[1] * Asize[2]) * (idx[3] - 1) + (Asize[1] * Asize[2] * Asize[3]) * (idx[4] - 1) + 1
    curr_pos = ((idx[1] - 1) * stride) + Asize[1] * (idx[2] - 1) * stride + (Asize[1] * Asize[2]) * (idx[3] - 1) + (Asize[1] * Asize[2] * Asize[3]) * (idx[4] - 1) + 1

    for p in 1:pool
        for p in 1:pool
            m = A[curr_pos]
            if (m > temp_max)
                    temp_max = m
                    max_pos = curr_pos
            end
            curr_pos += 1
        end
        curr_pos += Asize[1] - pool
    end
    out[(idx[1] - 1) + outSize[1] * (idx[2] - 1) + (outSize[1] * outSize[2]) * (idx[3] - 1) + (outSize[1] * outSize[2] * outSize[3]) * (idx[4] - 1) + 1] = temp_max
    return
end


function maxpool2d{T <: Integer}(a, pool::T; stride = pool, pad = 0)
    b = zeros(typeof(a), size(a,1) + pad * 2, size(a,2) + pad * 2, size(a,3), size(a,4))
    b[pad + 1 : pad + size(a,1), pad + 1 : pad + size(a,2), :, :] = a
    Asize = UInt32.(size(b))
    pool = UInt32(pool)
    stride = UInt32(stride)
    outSize = [i for i in size(b)]
    outSize[1:2] = [div(Asize[1] - pool, stride) + 1, div(Asize[2] - pool, stride) + 1]
    out = similar(b, outSize...)
    outSize = UInt32.(tuple(outSize...))
    gpu_call(maxpool2d_kernel, b, (b, out, Asize, pool, stride, outSize))
    GPUArrays.synchronize(out)
    out
end


