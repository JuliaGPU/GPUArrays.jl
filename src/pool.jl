import CUDAnative

function maxpool2d_kernel(state, A::AbstractArray{T}, out, Asize, pool, stride_, outSize) where T
    ilin = linear_index(state)
    idx = GPUArrays.gpu_ind2sub(Asize, ilin)
    if (idx[1] > outSize[1] || idx[2] > outSize[2] || idx[3] > outSize[3] || idx[4] > outSize[4])
        return
    end

    temp_max = A[((idx[1] - 1) * stride_) + Asize[1] * (idx[2] - 1) * stride_ + (Asize[1] * Asize[2]) * (idx[3] - 1) + (Asize[1] * Asize[2] * Asize[3]) * (idx[4] - 1) + 1]
    max_pos = ((idx[1] - 1) * stride_) + Asize[1] * (idx[2] - 1) * stride_ + (Asize[1] * Asize[2]) * (idx[3] - 1) + (Asize[1] * Asize[2] * Asize[3]) * (idx[4] - 1) + 1
    curr_pos = ((idx[1] - 1) * stride_) + Asize[1] * (idx[2] - 1) * stride_ + (Asize[1] * Asize[2]) * (idx[3] - 1) + (Asize[1] * Asize[2] * Asize[3]) * (idx[4] - 1) + 1

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


function maxpool2d(a, pool; stride_ = 1)
    Asize = UInt32.(size(a))
    pool = UInt32(pool)
    stride_ = UInt32(stride_)
    out = similar(a)
    out = out[1:(div(Asize[1] - pool, stride_) + 1), 1:(div(Asize[2] - pool, stride_) + 1), :, :]
    outSize = UInt32.(size(out))
    gpu_call(maxpool2d_kernel, a, (a, out, Asize, pool, stride_, outSize))
    GPUArrays.synchronize(out)
    out
end


