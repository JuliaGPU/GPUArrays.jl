function test_io(AT)
    @testset "input/output" begin
        @testset "showing" begin
          io = IOBuffer()
          A = AT([1])

          show(io, MIME("text/plain"), A)
          seekstart(io)
          @test String(take!(io)) == "1-element $AT{Int64,1}:\n 1"

          show(io, MIME("text/plain"), A')
          seekstart(io)
          @test String(take!(io)) == "1×1 LinearAlgebra.Adjoint{Int64,$AT{Int64,1}}:\n 1"
        end
    end
end
