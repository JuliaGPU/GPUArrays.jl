function test_fft(AT)
    for n = 1:3
        @testset "FFT with ND = $n" begin
            dims = ntuple(i -> 40, n)
            @test compare(fft!, AT, rand(ComplexF32, dims))
            @test compare(ifft!, AT, rand(ComplexF32, dims))

            @test compare(fft, AT, rand(ComplexF32, dims))
            @test compare(ifft, AT, rand(ComplexF32, dims))
        end
    end
end
