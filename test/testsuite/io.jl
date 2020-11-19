@testsuite "input output" AT->begin

    # compact=false to avoid type aliases
    replstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :compact => false, :limit => true, :displaysize => (24, 80), kv...), MIME("text/plain"), x), x)
    showstr(x, kv::Pair...) = sprint((io,x) -> show(IOContext(io, :limit => true, :displaysize => (24, 80), kv...), x), x)

    @testset "showing" begin
        # vectors and non-vector arrays showing
        # are handled differently in base/arrayshow.jl
        A = AT(Int64[1])
        B = AT(Int64[1 2;3 4])

        msg = replstr(A)
        @test occursin(Regex("^1-element $AT{Int64,\\s?1}:\n 1\$"), msg)

        # # result of e.g. `print` differs on 32bit and 64bit machines
        # due to different definition of `Int` type
        # print([1]) shows as [1] on 64bit but Int64[1] on 32bit
        msg = showstr(A)
        @test msg == "[1]" || msg == "Int64[1]"

        msg = replstr(B)
        @test occursin(Regex("^2×2 $AT{Int64,\\s?2.*}:\n 1  2\n 3  4\$"), msg)

        msg = showstr(B)
        @test msg == "[1 2; 3 4]" || msg == "Int64[1 2; 3 4]"

        # the printing of Adjoint depends on global state
        msg = replstr(A')
        @test occursin(Regex("^1×1 Adjoint{Int64,\\s?$AT{Int64,\\s?1}}:\n 1\$"), msg) ||
            occursin(Regex("^1×1 LinearAlgebra.Adjoint{Int64,\\s?$AT{Int64,\\s?1}}:\n 1\$"), msg) ||
            occursin(Regex("^1×1 adjoint\\(::$AT{Int64,\\s?1}\\) with eltype Int64:\n 1\$"), msg)
    end
end
