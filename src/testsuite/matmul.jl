using GPUArrays.TestSuite, Base.Test

function run_matmul(Typ)
  for ET in supported_eltypes()
    T = Typ{ET}
    if (ET == Complex{Float32} || ET == Complex{Float64})
      continue
    end
    @testset "$ET" begin
      @testset "matmul" begin
        a = rand(ET, 1024,1024)
        b = rand(ET, 1024,1024)
        out1 = a * b

        a = T(a)
        b = T(b)
        out2 = similar(T(out1))
        GPUArrays.allowslow(true)
        out2 = GPUArrays.matmul!(out2, a, b)
        
        @test out1 ≈ out2
      end
    end
  end
end