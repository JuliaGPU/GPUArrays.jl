function run_gpuinterface(Typ)
    @testset "parallel execution interface" begin
        N = 10
        x = Typ(Vector{Int}(N))
        x .= 0
        gpu_call(x, (x,)) do state, x
            x[linear_index(state)] = 2
            return
        end
        @test all(x-> x == 2, Array(x))

        gpu_call(x, (x,), N) do state, x
            x[linear_index(state)] = 2
            return
        end
        @test all(x-> x == 2, Array(x))
        configuration = ((N ÷ 2,), (2,))
        gpu_call(x, (x,), configuration) do state, x
            x[linear_index(state)] = threadidx_x(state)
            return
        end
        @test Array(x) == [1,2,1,2,1,2,1,2,1,2]

        gpu_call(x, (x,), configuration) do state, x
            x[linear_index(state)] = blockidx_x(state)
            return
        end
        @test Array(x) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        x2 = Typ([0])
        gpu_call(x, (x2,), configuration) do state, x
            x[1] = blockdim_x(state)
            return
        end
        @test Array(x2) == [2]

        gpu_call(x, (x2,), configuration) do state, x
            x[1] = griddim_x(state)
            return
        end
        @test Array(x2) == [5]

        gpu_call(x, (x2,), configuration) do state, x
            x[1] = global_size(state)
            return
        end
        @test Array(x2) == [10]
    end
end
