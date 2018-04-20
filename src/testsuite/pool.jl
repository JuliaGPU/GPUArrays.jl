using GPUArrays.TestSuite, Base.Test, Flux

function run_pool(Typ)
  for ET in supported_eltypes()
    T = Typ{ET}
    if (ET == Complex{Float32} || ET == Complex{Float64})
      continue
    end
    @testset "$ET" begin
      @testset "maxpool with padding" begin
        pool = 3
        stride = 3
        pad = 3

        a = rand(ET, 9,9,3,1)
        b = zeros(eltype(a), size(a,1) + pad * 2, size(a,2) + pad * 2, size(a,3), size(a,4))
        b[pad + 1 : pad + size(a,1), pad + 1 : pad + size(a,2), :, :] = a
        out1 = maxpool(b, (3, 3))

        a = T(a)
        GPUArrays.allowslow(true)
        out2 = GPUArrays.maxpool2d(a, pool, pad = pad)
        
        @test out1 ≈ out2
      end

      @testset "maxpool without padding" begin
        pool = 3
        stride = 3

        a = rand(ET, 9,9,3,1)
        out1 = maxpool(a, (3, 3))

        a = T(a)
        GPUArrays.allowslow(true)
        out2 = GPUArrays.maxpool2d(a, pool)
        
        @test out1 ≈ out2
      end


      @testset "maxpool with full kernel" begin
        pool = 9
        stride = 1

        a = rand(ET, 9,9,3,1)
        out1 = maxpool(a, (9, 9))

        a = T(a)
        GPUArrays.allowslow(true)
        out2 = GPUArrays.maxpool2d(a, pool, stride = stride)
        
        @test out1 ≈ out2
      end
    end
  end
end
