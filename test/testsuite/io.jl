@testsuite "input output" AT->begin
    @testset "showing" begin
        io = IOBuffer()
        A = AT(Int64[1])
        B = AT(Int64[1 2;3 4]) # vectors and non-vector arrays showing
                               # are handled differently in base/arrayshow.jl

        show(io, MIME("text/plain"), A)
        seekstart(io)
        @test occursin(Regex("^1-element $AT{Int64,1.*}:\n 1\$"), String(take!(io)))

        show(io, A)
        seekstart(io)
        msg = String(take!(io)) # result of e.g. `print` differs on 32bit and 64bit machines
        # due to different definition of `Int` type
        # print([1]) shows as [1] on 64bit but Int64[1] on 32bit
        @test msg == "[1]" || msg == "Int64[1]"

        show(io, MIME("text/plain"), B)
        seekstart(io)
        @test occursin(Regex("^2×2 $AT{Int64,2.*}:\n 1  2\n 3  4\$"), String(take!(io)))

        show(io, B)
        seekstart(io)
        msg = String(take!(io))
        @test msg == "[1 2; 3 4]" || msg == "Int64[1 2; 3 4]"

        show(io, MIME("text/plain"), A')
        seekstart(io)
        msg = String(take!(io)) # the printing of Adjoint depends on global state
        @test occursin(Regex("^1×1 Adjoint{Int64,$AT{Int64,1.*}}:\n 1\$"), msg) ||
            occursin(Regex("^1×1 LinearAlgebra.Adjoint{Int64,$AT{Int64,1.*}}:\n 1\$"), msg)
    end
end
