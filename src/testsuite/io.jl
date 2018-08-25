function test_io(AT)
    @testset "input/output" begin
        @testset "showing" begin
          io = IOBuffer()
          A = AT(Int64[1])
          B = AT(Int64[1 2;3 4]) # vectors and non-vector arrays showing
                                 # are handled differently in base/arrayshow.jl

          show(io, MIME("text/plain"), A)
          seekstart(io)
          @test String(take!(io)) == "1-element $AT{Int64,1}:\n 1"

          show(io, A)
          seekstart(io)
          @test String(take!(io)) == "[1]"

          show(io, MIME("text/plain"), B)
          seekstart(io)
          @test String(take!(io)) == "2×2 $AT{Int64,2}:\n 1  2\n 3  4"

          show(io, B)
          seekstart(io)
          @test String(take!(io)) == "[1 2; 3 4]"

          show(io, MIME("text/plain"), A')
          seekstart(io)
          msg = String(take!(io)) # the printing of Adjoint depends on global state
          @test msg == "1×1 Adjoint{Int64,$AT{Int64,1}}:\n 1" ||
                msg == "1×1 LinearAlgebra.Adjoint{Int64,$AT{Int64,1}}:\n 1"
        end
    end
end
