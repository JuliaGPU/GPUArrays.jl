using GPUArrays
using GPUArrays.TestSuite
using Base.Test

function run_fft(Typ)
    T = Typ{Complex64}
    for n = 1:3
        @testset "FFT with ND = $n" begin
            dims = ntuple(i-> 40, n)
            against_base(fft!, T, dims)
            against_base(ifft!, T, dims)

            against_base(fft, T, dims)
            against_base(ifft, T, dims)
        end
    end
end
