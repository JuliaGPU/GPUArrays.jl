
function block_matrix_product(state, A, B, C) where {ThreadItemsY, ThreadItemsX}

    # Fragments used to store data fetched from SMEM
    frag_a = @LocalMemory(state, T, ThreadItemsY)
    frag_b = @LocalMemory(state, T, ThreadItemsX)

    # Accumulator storage
    accumulator = @LocalMemory(state, T, ThreadItemsX, ThreadItemsY)

    # GEMM Mainloop - iterates over the entire K dimension - not unrolled
    for kblock in Int32(1):BlockItemsK:K_dim
        # Load A and B tiles from global memory and store to SMEM
        #
        # (not shown for brevity - see the CUTLASS source for more detail)

        synchronize_threads(state)
        # Warp tile structure - iterates over the Thread Block tile
        #pragma unroll
        for warp_k in Int32(1):WarpItemsK:BlockItemsK
            # Fetch frag_a and frag_b from SMEM corresponding to k-index
            #
            # (not shown for brevity - see CUTLASS source for more detail)

            # Thread tile structure - accumulate an outer product
            #pragma unroll
            for thread_x in Int32(1):ThreadItemsX
                #pragma unroll
                for thread_y in Int32(1):ThreadItemsY
                    accumulator[thread_x, thread_y] += frag_a[y] * frag_b[x]
                end
            end
        end
        synchronize_threads(state)
    end
end
