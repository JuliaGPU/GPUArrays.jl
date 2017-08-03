using GPUArrays, StaticArrays, FileIO

# Original poincare implementation by https://github.com/RainerEngelken
# GPU version by Simon Danisch


function poincare_inner{N}(rv, result, c, π, ::Val{N}, n)
    # find next spiking neuron
    ϕ₁, ϕ₂, ϕ₃ = rv[1], rv[2], rv[3]
    πh = π / 2f0
    π2 = π * 2f0
    for unused = 1:N
        if ϕ₁ > ϕ₂
            if ϕ₁ > ϕ₃
                # first neuron is spiking
                dt = πh - ϕ₁
                # evolve phases till next spike time
                ϕ₁ = -πh
                ϕ₂ = atan(tan(ϕ₂ + dt) - c)
                ϕ₃ += dt
                # save state of neuron 2 and 3
                x = Cuint(max(round(((ϕ₂ + πh) / π) * (Float32(n) - 1f0)) + 1f0, 1f0))
                y = Cuint(max(round(((ϕ₃ + πh) / π) * (Float32(n) - 1f0)) + 1f0, 1f0))
                i1d = GPUArrays.gpu_sub2ind((n, n), (x, y)) # convert to linear index
                accum = result[i1d]
                # this is unsafe, since it could read + write from different threads, but good enough for the stochastic kind of process we're doing
                result[i1d] = accum + 1f0
                continue
            end
        else
            if ϕ₂ > ϕ₃
                # second neuron is spiking
                dt = πh - ϕ₂
                # evolve phases till next spike time
                ϕ₁ += dt
                ϕ₂ = -πh
                ϕ₃ = atan(tan(ϕ₃ + dt) - c)
                continue
            end
        end
        # third neuron is spikinga
        dt = πh - ϕ₃
        # evolve phases till next spike time
        ϕ₁ += dt
        ϕ₂ = atan(tan(ϕ₂ + dt) - c)
        ϕ₃ = -πh
    end
    return
end

function poincare_gpu(iterations, c = 1f0, divisor = 256)
    srand(2)
    ND = Cuint(1024)
    result = GPUArray(zeros(Float32, ND, ND))
    N = div(iterations, divisor)
    seeds = GPUArray([ntuple(i-> rand(Float32), Val{3}) for x in 1:divisor])
    tic()
    foreach(poincare_inner, seeds, Base.RefValue(result), c, Float32(pi), Val{N}(), ND)
    GPUArrays.synchronize(result) # synchronize for the benchmark
    toc()
    result
end

backend = CLBackend.init() # try different backends, e.g. CUBackend.init()
result = poincare_gpu(10^9, 1f0, 2^11);

res2 = Array(result) ./ 2000f0
img = clamp.(res2, 0f0, 1f0);
#save as an image
save(homedir()*"/Desktop/testcl.png", img)


function poincare_cpu_original(n,c)
    srand(2)
    ϕ₁,ϕ₂,ϕ₃ = rand(3)
    𝚽 = Point2f0[]
    tic()
    for s = 1:n
        # find next spiking neuron
        if ϕ₁ > ϕ₂
            if ϕ₁ > ϕ₃
                # first neuron is spiking
                dt = π/2 - ϕ₁
                # evolve phases till next spike time
                ϕ₁ = -π/2
                ϕ₂ = atan(tan(ϕ₂ + dt) - c)
                ϕ₃ += dt
                # save state of neuron 2 and 3
                push!(𝚽, Point2f0(ϕ₂,ϕ₃))

                continue
            end
        elseif ϕ₂ > ϕ₃
            # second neuron is spiking
            dt = π/2 - ϕ₂
            # evolve phases till next spike time
            ϕ₁ += dt
            ϕ₂ = -π/2
            ϕ₃ = atan(tan(ϕ₃ + dt) - c)
            continue
        end
        # third neuron is spiking
        dt = π/2 - ϕ₃
        # evolve phases till next spike time
        ϕ₁ += dt
        ϕ₂ = atan(tan(ϕ₂ + dt) - c)
        ϕ₃ = -π/2
    end
    toc()
    𝚽
end

poincareFast(10^8, 1f0);
