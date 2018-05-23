function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize, ::Val{TS}, ::Val{TS²}) where {T, TS, TS²}
    # Thread identifiers
    row = threadidx_x(state) # Local row ID (max: TS)
    col = threadidx_y(state)
    # col = row


    groups_1 = blockidx_x(state)
    groups_2 = blockidx_y(state)

    globalRow = TS * (groups_1 - 1) + (row - 1) + 1 # Row ID of C (0..M)
    globalCol = TS * (groups_2 - 1) + (col - 1) + 1 # Col ID of C (0..N)

    @inbounds begin
        if globalRow > Asize[1] || globalCol > Bsize[2]
            return
        end
    end

    # println("groups_1: ", groups_1, " groups_2: ", groups_2, " row: ", row, " col: ", col)
    # @show globalRow
    # @show globalCol

    # Local memory to fit a tile of TS*TS elements of A and B
    Asub = @LocalMemory(state, T, TS²)
    Bsub = @LocalMemory(state, T, TS²)

    # Initialise the accumulation register
    acc = T(0.0)

    # Loop over all tiles
    numTiles = div(Asize[2], TS)
    for t in UInt32(1):UInt32(numTiles)

        # Load one tile of A and B into local memory
        tiledRow = TS * (t - 1) + (row - 1) + 1
        tiledCol = TS * (t - 1) + (col - 1) + 1
        @show tiledRow, tiledCol
        @show globalRow, globalCol
        println()
        Asub[(col - 1) * TS + (row - 1) + 1] = A[(tiledCol - 1) * Asize[1] + (globalRow - 1) + 1]
        Bsub[(col - 1) * TS + (row - 1) + 1] = B[(globalCol - 1) * Asize[2] + (tiledRow - 1) + 1]

        # Synchronise to make sure the tile is loaded
        synchronize_threads(state)

        # Perform the computation for a single tile
        for k in UInt32(1):UInt32(TS)
            acc += Asub[(k - 1)*TS + (row - 1 ) + 1] * Bsub[(col - 1) * TS + (k - 1) + 1]
        end

        # Synchronise before loading the next tile
        synchronize_threads(state)
    end

    # Store the final result in C
    out[(globalCol - 1) * Asize[1] + (globalRow - 1) + 1] = acc

    return

end


function matmul!(dest::GPUArray, a::GPUArray{T, 2}, b::GPUArray{T, 2}) where T
    Asize = size(a)
    Bsize = size(b)
    TS = UInt32(4)
    outSize = UInt32.(size(dest))
    Asize = UInt32.(Asize)
    Bsize = UInt32.(Bsize)
    config = ((Asize[1] - TS + UInt32(1), Bsize[2] - TS + UInt32(1)), (TS, TS))
    println("config: ",config)
    gpu_call(matmul_kernel, dest, (a,b, dest, Asize, Bsize, outSize, Val{UInt32(4)}(), Val{UInt32(16)}()), config)
    dest
end
#
# A = JLArray(rand(10, 10))
# B = JLArray(rand(10, 10))
# out = JLArray(zeros(size(A, 1), size(B, 2)))
# matmul!(out, A, B)
