@generated function ntuple_args(f, ::Val{N}, args::Vararg{<: Any, Nargs}) where {N, Nargs}
    expr = Expr(:tuple)
    for i = 1:N
        call = Expr(:call, :f, i)
        for j = 1:Nargs
            push!(call.args, :(args[$j]))
        end
        push!(expr.args, call)
    end
    quote
        Base.@_inline_meta
        $expr
    end
end

function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize, ::Val{TS}, ::Val{TS²}, ::Val{WPT}, ::Val{numTiles}, ::Val{RTS}) where {T, TS, TS², WPT, numTiles, RTS}
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
    Asub = @LocalMemory(state, T, TS²)
    Bsub = @LocalMemory(state, T, TS²)

    acc = ntuple(Val{WPT}) do i 
     0.0f0
    end

    # Loop over all tiles
    for t in UInt32(1):UInt32(numTiles)
        # Load one tile of A and B into local memory
        for w in UInt32(1):UInt32(WPT)
            @inbounds tiledRow = UInt32(TS) * (t - 1) + (row - 1) + 1
            @inbounds tiledCol = UInt32(TS) * (t - 1) + (col - 1) + 1
            # save_print("gpu_sub2ind(UInt32.((TS, TS)), UInt32.((row, col + w*RTS))): ", gpu_sub2ind(UInt32.((TS, TS)), UInt32.((row, col + w*RTS))))
            @inbounds Asub[gpu_sub2ind(UInt32.((TS, TS)), UInt32.((row, (col - 1) + (w - 1)*RTS + 1)))] = A[(tiledCol - 1 + (w - 1)*UInt32(RTS)) * Asize[1] + globalRow]
            @inbounds Bsub[gpu_sub2ind(UInt32.((TS, TS)), UInt32.((row, (col - 1) + (w - 1)*RTS + 1)))] = B[(globalCol - 1 + (w - 1)*UInt32(RTS)) * Asize[2] + tiledRow]
        end

        # Synchronise to make sure the tile is loaded
        synchronize_threads(state)

        # Perform the computation for a single tile
        for k in UInt32(1):UInt32(TS)
            acc = ntuple_args(Val{WPT}(), acc, Asub, Bsub, TS, RTS, row, col, k) do w, acc, Asub, Bsub, TS, RTS, row, col, k
             @inbounds return (acc[w] +  Asub[gpu_sub2ind(UInt32.((TS, TS)), UInt32.((row, k)))] * Bsub[gpu_sub2ind(UInt32.((TS, TS)), UInt32.((k, (col - 1) + (w - 1)*RTS + 1)))])
            end
        end
        # Synchronise before loading the next tile
        synchronize_threads(state)
    end

    # Store the final result in out
    for w in UInt32(1):UInt32(WPT)
        @inbounds out[(globalCol - 1 + (w - 1)*UInt32(RTS)) * Asize[1] + (globalRow - 1) + 1] = acc[w]
    end
    return

end


function matmul!(dest::GPUArray, a::GPUArray{T, 2}, b::GPUArray{T, 2}) where T
    Asize = size(a)
    Bsize = size(b)
    device = GPUArrays.device(a)
    thr = GPUArrays.threads(device)
    TS = 2 * ceil(Int,sqrt(thr))
    WPT = 8
    outSize = UInt32.(size(dest))
    Asize = UInt32.(Asize)
    Bsize = UInt32.(Bsize)
    acc = zeros(typeof(a), WPT)
    config = (UInt32.((div(Asize[1], TS), div(Bsize[2], TS))), UInt32.((TS, div(TS, WPT))))
    numTiles = div(Asize[2], TS)
        gpu_call(matmul_kernel, dest, (a,b, dest, Asize, Bsize, outSize, Val{TS}(), Val{TS^2}(), Val{WPT}(), Val{numTiles}(), Val{div(TS, WPT)}()), config)
    # end
    # CUDAdrv.Profile.stop()
    dest
end
#
# A = JLArray(rand(10, 10))
# B = JLArray(rand(10, 10))
# out = JLArray(zeros(size(A, 1), size(B, 2)))
# matmul!(out, A, B)
