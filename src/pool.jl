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
    a2 = size.((a,), (1, 2))
    b = zeros(typeof(a), (a2 .+ 2pad)..., size(a, 3), size(a, 4))
    apad = a2 .+ pad
    b[pad + 1 : apad[1], pad + 1 : apad[2], :, :] = a
    as = ((size(b) .- pool) .÷ stride) .+ 1
    out = similar(b, (as[1], as[2], size(b, 3), size(b, 4)))
    sizes = map(x-> UInt32.(x), (size(b), pool, stride, size(out)))
    gpu_call(maxpool2d_kernel, b, (b, out, sizes...))
    out
end


