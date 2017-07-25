using Transpiler.cli: mem_fence, CLK_GLOBAL_MEM_FENCE

# Base.@code_warntype poincare_inner(Vec3f0(0), rand(Float32, 10, 10), 1f0, Float32(π), Val{1}(), Cuint(1))

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
                x = Cuint(round(((ϕ₂ + πh) / π) * Float32(n) - 1f0))
                y = Cuint(round(((ϕ₃ + πh) / π) * Float32(n) - 1f0))
                i1d = GPUArrays.gpu_sub2ind((n, n), (x, y))
                @inbounds if i1d <= Cuint(n * n) && i1d > Cuint(0)
                    accum = result[i1d]
                    result[i1d] = accum + 1f0 # this is unsafe, since it could read + write from different threads, but good enough for the stochastic kind of process we're doing
                end
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

function poincareFast(iterations, c = 1f0, divisor = 256)
    srand(2)
    ND = Cuint(1024)
    result = GPUArray(zeros(Float32, ND, ND))
    N = div(iterations, divisor)
    seeds = GPUArray(rand(Vec3f0, divisor))
    tic()
    foreach(poincare_inner, seeds, Base.RefValue(result), c, Float32(pi), Val{N}(), ND)
    GPUArrays.synchronize(result)
    toc()
    result
end

div(2048, 256)

using GPUArrays, FileIO
using GeometryTypes
backend = CLBackend.init()

result = poincareFast(10^10, 1f0, 2048);

res2 = Array(result) ./ 2000f0
img = clamp.(res2, 0f0, 1f0);
save(homedir()*"/Desktop/testcl.png", img)

rand_idx = calc_idx()
accum = result[rand_idx]
result[rand_idx] = accum + 1f0
mem_fence(CLK_GLOBAL_MEM_FENCE)

function poincareFast(n,c)
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

using GeometryTypes
poincareFast(10^8, 1f0);
