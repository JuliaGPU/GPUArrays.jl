function test_fft(Typ)
    for n = 1:3
        @testset "FFT with ND = $n" begin
            dims = ntuple(i -> 40, n)
            @test compare(fft!, Typ, rand(ComplexF32, dims))
            @test compare(ifft!, Typ, rand(ComplexF32, dims))

            @test compare(fft, Typ, rand(ComplexF32, dims))
            @test compare(ifft, Typ, rand(ComplexF32, dims))
        end
    end
end
