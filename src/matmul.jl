function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize) where {T}
    # Thread identifiers
    TS = UInt32(2)
    row = threadidx_x(state) # Local row ID (max: TS)
    col = threadidx_y(state)
    # col = row

    groups_1 = blockidx_x(state)
    groups_2 = blockidx_y(state)

    # println("curr: ", row)
    # CUDAnative.@cuprintf("curr: %ld\n", convert(Int64, row))

    globalRow = TS * (groups_1[1] - 1) + (row[1] - 1) + 1 # Row ID of C (0..M)
    globalCol = TS * (groups_2[1] - 1) + (col[1] - 1) + 1 # Col ID of C (0..N)
        
    @inbounds begin
        if globalRow > (Asize[1] - TS) || globalCol > (Bsize[2] - TS)
            return
        end
    end
     
    # Local memory to fit a tile of TS*TS elements of A and B
    Asub = zeros(Float32, TS * TS)
    Bsub = zeros(Float32, TS * TS)
 
    # Initialise the accumulation register
    acc = Float32(0.0)
    
    # Loop over all tiles
    numTiles = Asize[2]/TS
    for t in UInt32(1):UInt32(numTiles)
 
        # Load one tile of A and B into local memory
        @inbounds tiledRow = TS * (t - 1) + (row[1] - 1) + 1
        @inbounds tiledCol = TS * (t - 1) + (col[1] - 1) + 1
        @inbounds Asub[(col[1] - 1) * TS + (row[1] - 1) + 1] = A[(tiledCol - 1) * Asize[1] + (globalRow - 1) + 1]
        @inbounds Bsub[(col[1] - 1) * TS + (row[1] - 1) + 1] = B[(globalCol - 1) * Asize[2] + (tiledRow - 1) + 1]
 
        # Synchronise to make sure the tile is loaded
        synchronize_threads(state)
 
        # Perform the computation for a single tile
        for k in UInt32(1):UInt32(TS)
            acc += Asub[(k - 1)*TS + (row[1] - 1 ) + 1] * Bsub[(col[1] - 1) * TS + (k - 1) + 1]
        end
 
        # Synchronise before loading the next tile
        synchronize_threads(state)
    end
 
    # Store the final result in C
    @inbounds out[(globalCol - 1) * Asize[1] + (globalRow - 1) + 1] = acc

    return nothing
    
end


function matmul(a::GPUArray{T, 2}, b::GPUArray{T, 2}) where T
    Asize = size(a)
    Bsize = size(b)
    out = similar(a, Asize[1], Bsize[2])
    outSize = size(out)
    Asize = UInt32.(Asize)
    config = (ceil.(Int, size(out) ./ (UInt32(2), UInt32(2))), (UInt32(2), UInt32(2)))
    # println("config: ",config)
    gpu_call(matmul_kernel, out, (a,b, out, UInt32.(Asize), UInt32.(Bsize), UInt32.(outSize)), config)
    out
end


