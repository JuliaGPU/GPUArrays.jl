function transpose_kernel{BLOCK_DIM}(
        odata, idata, offset, width, height, block, ::Val{BLOCK_DIM}
    )
    # read the matrix tile into shared memory
    xIndex = get_global_id(0)
    yIndex = get_global_id(1)

    if((xIndex + offset < width) && (yIndex < height))
        index_in = yIndex * width + xIndex + offset
        block[get_local_id(1)*(BLOCK_DIM + 1) + get_local_id(0)] = idata[index_in]
    end

    barrier(CLK_LOCAL_MEM_FENCE)

    # write the transposed matrix tile to global memory
    xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0)
    yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1)
    if((xIndex < height) && (yIndex + offset < width))
        index_out = yIndex * height + xIndex
        odata[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)]
    end
    return
end
