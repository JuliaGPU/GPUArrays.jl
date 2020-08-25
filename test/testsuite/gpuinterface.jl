@testsuite "interface" AT->begin
    AT <: AbstractGPUArray || return

    N = 10
    x = AT(Vector{Int}(undef, N))
    x .= 0
    gpu_call(x) do ctx, x
        x[linear_index(ctx)] = 2
        return
    end
    @test all(x-> x == 2, Array(x))

    gpu_call(x; total_threads=N) do ctx, x
        x[linear_index(ctx)] = 2
        return
    end
    @test all(x-> x == 2, Array(x))
    gpu_call(x; threads=2, blocks=(N ÷ 2)) do ctx, x
        x[linear_index(ctx)] = threadidx(ctx)
        return
    end
    @test Array(x) == [1,2,1,2,1,2,1,2,1,2]

    gpu_call(x; threads=2, blocks=(N ÷ 2)) do ctx, x
        x[linear_index(ctx)] = blockidx(ctx)
        return
    end
    @test Array(x) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    x2 = AT([0])
    gpu_call(x2; threads=2, blocks=(N ÷ 2), target=x) do ctx, x
        x[1] = blockdim(ctx)
        return
    end
    @test Array(x2) == [2]

    gpu_call(x2; threads=2, blocks=(N ÷ 2), target=x) do ctx, x
        x[1] = griddim(ctx)
        return
    end
    @test Array(x2) == [5]

    gpu_call(x2; threads=2, blocks=(N ÷ 2), target=x) do ctx, x
        x[1] = global_size(ctx)
        return
    end
    @test Array(x2) == [10]
end
