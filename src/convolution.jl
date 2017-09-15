function convolution_kernel(
        imgSrc::AbstractArray{T},
        kernelValues,
        kernelSize,
        imgConvolved
    ) where T

    w = kernelSize[1]
    wBy2 = w >> 1; #w divided by 2
    #Goes up to 15x15 filters
    p = LocalMemory(T, BLOCK_SIZE + 14, BLOCK_SIZE + 14) #Identification of this workgroup
    i = get_group_id(0);
    j = get_group_id(1); #Identification of work-item
    idX = get_local_id(0);
    idY = get_local_id(1);

    ii = i*BLOCK_SIZE + idX; # == get_global_id(0);
    jj = j*BLOCK_SIZE + idY; # == get_global_id(1);
    coords = (ii, jj)
    #Reads pixels
    P[idX][idY] = imgSrc[gpu_ind2sub(sizeA, (ii, jj))]
    #Needs to read extra elements for the filter in the borders
    if (idX < w)
        P[idX + BLOCK_SIZE][idY] = imgSrc[gpu_ind2sub(sizeA, (ii + BLOCK_SIZE, jj))]
    end
    if (idY < w)
        P[idX][idY + BLOCK_SIZE] = imgSrc[gpu_ind2sub(sizeA, (ii, jj + BLOCK_SIZE))]
    end
    barrier(CLK_LOCAL_MEM_FENCE)
    ##############
    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 temp;
    for (int ix = 0; ix < w; ix++)
        for (int jy = 0; jy < w; jy++)
            temp = (float4)((float)P[ix][jy].x,
                            (float)P[ix][jy].y,
                            (float)P[ix][jy].z,
                            (float)P[ix][jy].w);
            convPix += temp * kernelValues[ix + w*jy];
        end
    end
    ##############
    barrier(CLK_LOCAL_MEM_FENCE);
    imgConvolved[ii+wBy2, jj+wBy2] = P[idX+wBy2][idY+wBy2]
end


function convolution_kernel(state, A::AbstractArray{T}, out, K, Asize, Ksize) where T
    ilin = linear_index(state)
    idx = gpu_ind2sub(Asize, ilin)
    if idx[1] >= Asize[1] - Ksize[1] || idx[2] >= Asize[2] - Ksize[2]
        return
    end
    accum = zero(T)
    kw, kh = Ksize[1], Ksize[2]
    for ix = Cuint(0):(kw - Cuint(1))
        for jy = Cuint(0):(kh - Cuint(1))
            temp = A[gpu_sub2ind(Asize, idx .+ (ix, jy))]
            accum += temp * K[ix + kw*jy + 1]
        end
    end
    out[ilin] = accum
    return
end


function conv!(a, out, k)
    gpu_call(convolution_kernel, a, (a, out, k, Cuint.(size(a)), Cuint.(size(k))))
    GPUArrays.synchronize(out)
    out
end
