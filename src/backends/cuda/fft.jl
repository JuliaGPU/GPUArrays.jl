using CUDArt, CUFFT


    A = rand(7,6)
    # Move data to GPU
    G = CudaArray(A)
    # Allocate space for the output (transformed array)
    GFFT = CudaArray(Complex{eltype(A)}, div(size(G,1),2)+1, size(G,2))
    # Compute the FFT
    pl! = plan(GFFT, G)
    pl!(GFFT, G, true)
    # Copy the result to main memory
    AFFTG = to_host(GFFT)
    # Compare against Julia's rfft
    AFFT = rfft(A)
    @test_approx_eq AFFTG AFFT
    # Now compute the inverse transform
    pli! = plan(G,GFFT)
    pli!(G, GFFT, false)
    A2 = to_host(G)
    @test_approx_eq A A2/length(A)
end

function Base.fft!(A::CUArray)
    G, GFFT = CUFFT.RCpair(A)
    fft!(G, GFFT)
end
function Base.fft!(out::CUArray, A::CUArray)
    plan(out, A)(out, A, true)
end

function Base.ifft!(A::CUArray)
    G, GFFT = CUFFT.RCpair(A)
    ifft!(G, GFFT)
end
function Base.ifft!(out::CUArray, A::CUArray)
    plan(out, A)(out, A, false)
end
