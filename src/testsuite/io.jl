function test_io(Typ)
    @testset "input/output" begin
        @testset "showing" begin
          io = IOBuffer()
          A = Typ([1])

          show(io, MIME("text/plain"), A)
          seekstart(io)
          @test String(take!(io)) == "1-element $Typ{Int64,1}:\n 1"

          show(io, MIME("text/plain"), A')
          seekstart(io)
          @test String(take!(io)) == "1×1 LinearAlgebra.Adjoint{Int64,$Typ{Int64,1}}:\n 1"
        end
    end
end
