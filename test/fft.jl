# TODO add many more tests
for n = 1:3
    @allbackends "FFT with ND = $n" backend begin
        a = rand(Complex64, ntuple(i-> 40, n))
        A = GPUArray(a)
        fft!(A)
        fft!(a)
        @test Array(A) ≈ a
        ifft!(A)
        ifft!(a)
        @test Array(A) ≈ a
    end
end
