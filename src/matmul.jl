function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize, ::Val{TS}, ::Val{TS²}, ::Val{IntTS²}, ::Val{WPT}, ::Val{numTiles}, ::Val{RTS}) where {T, TS, TS², IntTS², WPT, numTiles, RTS}
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

    # Local memory to fit a tile of TS*TS elements of A and B
    Asub = @LocalMemory(state, T, IntTS²)
    Bsub = @LocalMemory(state, T, IntTS²)

    # Initialise the accumulation register
    acc = zeros(T, WPT)

    # Loop over all tiles
    for t in UInt32(1):UInt32(numTiles)
        # Load one tile of A and B into local memory
        for w in UInt32(1):WPT
            @inbounds tiledRow = TS * (t - 1) + (row - 1) + 1
            @inbounds tiledCol = TS * (t - 1) + (col - 1) + 1
            @inbounds Asub[(col - 1 + (w - 1)*RTS) * TS + row] = A[(tiledCol - 1 + (w - 1)*RTS) * Asize[1] + globalRow]
            @inbounds Bsub[(col - 1 + (w - 1)*RTS) * TS + row] = B[(globalCol - 1 + (w - 1)*RTS) * Asize[2] + tiledRow]
        end

        # Synchronise to make sure the tile is loaded
        synchronize_threads(state)

        # Perform the computation for a single tile
        for k in UInt32(1):UInt32(TS)
            for w in UInt32(1):WPT
                @inbounds acc[w] += Asub[(k - 1)*TS + (row - 1 ) + 1] * Bsub[(col - 1) * TS + (k - 1) + 1]
            end
        end
        # Synchronise before loading the next tile
        synchronize_threads(state)
    end

    # Store the final result in out
    for w in UInt32(1):WPT
        @inbounds out[(globalCol - 1 + (w - 1)*RTS) * Asize[1] + (globalRow - 1) + 1] = acc[w]
    end
    return

end


function matmul!(dest::GPUArray, a::GPUArray{T, 2}, b::GPUArray{T, 2}) where T
    Asize = size(a)
    Bsize = size(b)
    device = GPUArrays.device(a)
    thr = GPUArrays.threads(device)
    TS = Int(sqrt(thr))
    WPT = 8
    outSize = UInt32.(size(dest))
    Asize = UInt32.(Asize)
    Bsize = UInt32.(Bsize)
    config = ((div(Asize[1], TS), div(Bsize[2], TS)), (TS, div(TS, WPT)))
    gpu_call(matmul_kernel, dest, (a,b, dest, Asize, Bsize, outSize, Val{UInt32(TS)}(), Val{UInt32(TS^2)}(), Val{TS^2}(), Val{UInt32(WPT)}(), Val{UInt32(div(Asize[2], TS))}(), Val{UInt32(div(TS, WPT))}()), config)
    dest
end
#
# A = JLArray(rand(10, 10))
# B = JLArray(rand(10, 10))
# out = JLArray(zeros(size(A, 1), size(B, 2)))
# matmul!(out, A, B)
