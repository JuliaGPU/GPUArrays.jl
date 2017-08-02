using GPUArrays
# an optimised implementation of CH_vectorised!
tocomplex64(x) = Complex64(x) # cudanative doesn't like Complex64.(gpu_array)

# source: http://nbviewer.jupyter.org/url/homepages.warwick.ac.uk/staff/C.Ortner/julia/PlaneWaves.ipynb
function CH_memory!(nruns, N, AT)

    # initialisations
    h = Float32(2*π/N); epsn = Float32(h * 3); C = Float32(2/epsn); tau = Float32(epsn * h)
    k = [0:N/2; -N/2+1:-1]
    Â = AT((Float32.(kron(k.^2, ones(1,N)) + kron(ones(N), k'.^2))))
    u = AT(Float32.((2*(rand(N, N)-0.5))))


    # ============= ACTUAL CODE THAT IS BEING TESTED ======================
    # allocate arrays and define constants
    w = tocomplex64.(u)
    v = copy(w)
    c1 = (C*tau+tau/epsn)
    c2 = (tau/epsn)
    c3 = (epsn*tau)
    c4 = (C*tau)
    planv = plan_fft!(v)
    planw = plan_fft!(w)
    planwi = plan_ifft!(w)
    tic()
    for n = 1:nruns
        v .= Complex64.(u .* u .* u)
        planv * v
        planw * w
        w .= (( (1f0 + c1) .* Â) .* w .- (c2 .* Â) .* v) ./ ((1f0 + c3 .* Â .+ c4) .* Â)
        planwi * w
        u .= real.(w)
    end
    GPUArrays.synchronize(u)
    toc()
    # ======================================================================
    u
end
CUBackend.init()

x = CH_memory!(100, 2^9, GPUArray)

using GPUArrays

# source: https://github.com/johnfgibson/julia-pde-benchmark/blob/master/1-Kuramoto-Sivashinksy-benchmark.ipynb
function ksintegrate(u, Lx, dt, Nt, AT)
    T = eltype(u)
    u = AT((T(1)+T(0)im)*u)                 # force u to be complex
    Nx = length(u)                      # number of gridpoints
    kx = T.(vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1))# integer wavenumbers: exp(2*pi*kx*x/L)
    alpha = T(2)*pi*kx/Lx                  # real wavenumbers:    exp(alpha*x)

    D = T(1)im*alpha                       # spectral D = d/dx operator

    L = alpha.^2 .- alpha.^4            # spectral L = -D^2 - D^4 operator

    G = AT(T(-0.5) .* D)               # spectral -1/2 D operator, to eval -u u_x = 1/2 d/dx u^2

    # convenience variables
    dt2  = T(dt/2)
    dt32 = T(3*dt/2)
    A_inv = AT((ones(T, Nx) - dt2*L).^(-1))
    B = AT(ones(T, Nx) + dt2*L)

    # compute in-place FFTW plans
    FFT! = plan_fft!(u)
    IFFT! = plan_ifft!(u)

    # compute nonlinear term Nn == -u u_x
    powed = u .* u
    Nn = G .* fft(powed);    # Nn == -1/2 d/dx (u^2) = -u u_x
    Nn1 = copy(Nn);        # Nn1 = Nn at first time step
    FFT! * u;

    # timestepping loop
    tic()
    for n = 1:Nt

        Nn1 .= Nn       # shift nonlinear term in time
        Nn .= u         # put u into Nn in prep for comp of nonlinear term

        @time begin
            IFFT! * Nn
            GPUArrays.synchronize(Nn)
        end
            # transform Nn to gridpt values, in place
        Nn .= Nn .* Nn   # collocation calculation of u^2
        FFT!*Nn        # transform Nn back to spectral coeffs, in place

        Nn .= G .* Nn    # compute Nn == -1/2 d/dx (u^2) = -u u_x

        # loop fusion! Julia translates the folling line of code to a single for loop.
        u .= A_inv .* (B .* u .+ dt32 .* Nn .- dt2 .* Nn1)
    end
    GPUArrays.synchronize(u)
    toc()
    u
end

Lx = Float32(64*pi)
Nx = 10^7
dt = 1/16
Nt = 100

x = Lx*(0:Nx-1)/Nx
u = Float32.(cos.(x) + 0.1*sin.(x/8) + 0.01*cos.((2*pi/Lx)*x))
CLBackend.init()
x = ksintegrate(u, Lx, dt, 3, GPUArray)
GPUArrays.free(x); gc()
