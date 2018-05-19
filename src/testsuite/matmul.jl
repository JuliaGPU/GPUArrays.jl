using GPUArrays.TestSuite, Base.Test

function run_matmul(Typ)
  for ET in supported_eltypes()
    T = Typ{ET}
    if (ET == Complex{Float32} || ET == Complex{Float64})
      continue
    end
    @testset "$ET" begin
      @testset "matmul" begin
        a = rand(ET, 4,4)
        b = rand(ET, 4,4)
        out1 = a * b

        a = T(a)
        b = T(b)
        GPUArrays.allowslow(true)
        out2 = GPUArrays.matmul(a, b)
        
        @test out1 ≈ out2
      end
    end
  end
end