function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize, ::Val{TS}, ::Val{TS²}, ::Val{IntTS²}) where {T, TS, TS², IntTS²}
    # Thread identifiers
    row = threadidx_x(state) # Local row ID (max: TS)
    col = threadidx_y(state)
    # col = row


    groups_1 = blockidx_x(state)
    groups_2 = blockidx_y(state)

    globalRow = TS * (groups_1 - 1) + (row - 1) +1# Row ID of C (0..M)
    globalCol = TS * (groups_2 - 1) + (col - 1) +1# Col ID of C (0..N)

    @inbounds begin
        if globalRow > Asize[1] || globalCol > Bsize[2]
            return
        end
    end

    # @show globalRow
    # @show globalCol

    # Local memory to fit a tile of TS*TS elements of A and B
    Asub = @LocalMemory(state, T, IntTS²)
    Bsub = @LocalMemory(state, T, IntTS²)

    # Initialise the accumulation register
    acc = T(0.0)

    # Loop over all tiles
    numTiles = div(Asize[2], TS)
    for t in UInt32(1):UInt32(numTiles)

        # Load one tile of A and B into local memory
        @inbounds tiledRow = TS * (t - 1) + (row - 1) + 1
        @inbounds tiledCol = TS * (t - 1) + (col - 1) + 1
        @inbounds Asub[(col - 1) * TS + row] = A[(tiledCol - 1) * Asize[1] + globalRow]
        @inbounds Bsub[(col - 1) * TS + row] = B[(globalCol - 1) * Asize[2] + tiledRow]

        # Synchronise to make sure the tile is loaded
        synchronize_threads(state)

        # Perform the computation for a single tile
        for k in UInt32(1):UInt32(TS)
            @inbounds acc += Asub[(k - 1)*TS + (row - 1 ) + 1] * Bsub[(col - 1) * TS + (k - 1) + 1]
        end
        # Synchronise before loading the next tile
        synchronize_threads(state)
    end

    # Store the final result in out
    @inbounds out[(globalCol - 1) * Asize[1] + (globalRow - 1) + 1] = acc

    return

end


function matmul!(dest::GPUArray, a::GPUArray{T, 2}, b::GPUArray{T, 2}) where T
    Asize = size(a)
    Bsize = size(b)
    device = GPUArrays.device(a)
    thr = GPUArrays.threads(device)
    TS = Int(sqrt(thr))
    outSize = UInt32.(size(dest))
    Asize = UInt32.(Asize)
    Bsize = UInt32.(Bsize)
    config = ((div(Asize[1], TS), div(Bsize[2], TS)), (TS, TS))
    gpu_call(matmul_kernel, dest, (a,b, dest, Asize, Bsize, outSize, Val{UInt32(TS)}(), Val{UInt32(TS^2)}(), Val{TS^2}()), config)
    dest
end
#
# A = JLArray(rand(10, 10))
# B = JLArray(rand(10, 10))
# out = JLArray(zeros(size(A, 1), size(B, 2)))
# matmul!(out, A, B)
