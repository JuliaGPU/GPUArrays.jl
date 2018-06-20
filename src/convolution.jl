# function convolution_kernel(
#         state,
#         imgSrc::AbstractArray{T},
#         kernelValues,
#         kernel_width,
#         imgConvolved,
#         ::Val{BLOCK_SIZE},
#         ::Val{LOCAL_WIDTH}
#     ) where {T, BLOCK_SIZE, LOCAL_WIDTH}
#     ui1 = UInt32(1); ui0 = UInt32(0)
#     w = kernel_width
#     wBy2 = w >> ui1 #w divided by 2
#     #Goes up to 15x15 filters
#     ptr = LocalMemory(state, T, LOCAL_WIDTH) # local width need to be static, so calculating it from block size won't cut it
#     P = CLArrays.LocalArray{T, 2}(ptr, (LOCAL_WIDTH, LOCAL_WIDTH))
#
#     i = blockidx_x(state)
#     j = blockidx_y(state) #Identification of work-item
#     idX = threadidx_x(state)
#     idY = threadidx_y(state)
#
#     ii = i*BLOCK_SIZE + idX; # == get_global_id(0);
#     jj = j*BLOCK_SIZE + idY; # == get_global_id(1);
#     #Reads pixels
#     P[idX, idY] = imgSrc[ii, jj]
#     #Needs to read extra elements for the filter in the borders
#     if (idX < w)
#         P[idX + BLOCK_SIZE, idY] = imgSrc[ii + BLOCK_SIZE, jj]
#     end
#     if (idY < w)
#         P[idX, idY + BLOCK_SIZE] = imgSrc[ii, jj + BLOCK_SIZE]
#     end
#     synchronize_threads(state)
#     ##############
#     convPix = zero(T);
#     for ix = ui0:(w - ui1)
#         for jy = ui0:(w - ui1)
#             temp = P[ix, jy]
#             convPix += temp * kernelValues[ix + w*jy]
#         end
#     end
#     ##############
#     synchronize_threads(state)
#     imgConvolved[ii + wBy2, jj + wBy2] = P[idX + wBy2, idY + wBy2]
#     return
# end


function convolution_kernel(state, A::AbstractArray{T}, out, K, Asize, Ksize) where T
    ilin = linear_index(state)
    idx = GPUArrays.gpu_ind2sub(Asize, ilin)
    if idx[1] >= Asize[1] - Ksize[1] || idx[2] >= Asize[2] - Ksize[2]
        return
    end
    accum = zero(T)
    kw, kh = Ksize[1], Ksize[2]
    for ix = UInt32(0):(kw - UInt32(1))
        for jy = UInt32(0):(kh - UInt32(1))
            temp = A[gpu_sub2ind(Asize, idx .+ (ix, jy))]
            accum += temp * K[ix + kw*jy + 1]
        end
    end
    out[ilin] = accum
    return
end


function convolution!(a, out, k)
    gpu_call(convolution_kernel, a, (a, out, k, UInt32.(size(a)), UInt32.(size(k))))
    GPUArrays.synchronize(out)
    out
end

struct FFTKernel{T}
    kernel::T
    irfftplan
    rfftplan
end

function fftkernel(A, kernel)
    plan_rfft!(A)

end

function convolution_fft!(a, out, k)
    irfft(rfft(A).*conj(rfft(krn)), length(axes(A,1)))
    out
end
