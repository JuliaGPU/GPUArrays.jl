function haversine{T <: Float32}(lat1::T, lon1::T, lat2::T, lon2::T, radius::T)
    c1 = cospi(lat1 / 180.0f0)
    c2 = cospi(lat2 / 180.0f0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    d1 = sinpi(dlat / 360.0f0)
    d2 = sinpi(dlon / 360.0f0)
    t = d2 * d2 * c1 * c2
    a = d1 * d1 + t
    c = 2.0f0 * asin(min(1.0f0, sqrt(a)))
    return radius * c
end


using GPUArrays

function benchmark(a, b, c)
    N = Int32(length(a))
    mapidx(pairwise_dist4, a, (b, c, Int32(N)))
    Array(a)
end

N = 10000
ABC = rand(Float32, N), rand(Float32, N), rand(Float32, N, N);
abc = map(GPUArray, ABC);

@time benchmark(abc...);
@time pairwise_dist_cpu(ABC..., N);

using GPUArrays
CLBackend.init()
function haversine_gpu(lat1, lon1, lat2, lon2, radius)
    c1 = cospi(lat1 / 180.0f0)
    c2 = cospi(lat2 / 180.0f0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    d1 = sinpi(dlat / 360.0f0)
    d2 = sinpi(dlon / 360.0f0)
    t = d2 * d2 * c1 * c2
    a = d1 * d1 + t
    c = 2.0f0 * asin(min(1.0f0, sqrt(a)))
    return radius * c
end

testdata = ntuple(i-> rand(Float32, 10^7), 4)
gpu_testdata = map(GPUArray, testdata)
result = similar(first(testdata))
gpu_result = GPUArray(result)

function bench(result, data)
    result .= haversine_gpu.(data..., 4f0)
    GPUArrays.synchronize(result)
end

bcpu = @elapsed bench(result, testdata)
bgpu = @elapsed bench(gpu_result, gpu_testdata)
@assert Array(gpu_result) ≈ result
bcpu / bgpu
