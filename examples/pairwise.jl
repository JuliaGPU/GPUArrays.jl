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

function pairwise_dist4(i, lat, lon, rowresult, n)
    for j in 1:n
        @inbounds rowresult[i, j] = haversine(lat[i], lon[i], lat[j], lon[j] , 6372.8f0)
    end
end
function pairwise_dist_cpu(lat, lon, rowresult, n)
    for i = 1:n, j in 1:n
        @inbounds rowresult[i, j] = haversine(lat[i], lon[i], lat[j], lon[j] , 6372.8f0)
    end
end

using GPUArrays

function benchmark(a, b, c)
    N = Int32(length(a))
    mapidx(pairwise_dist4, a, (b, c, Int32(N)))
    GPUArrays.synchronize(a)
end

N = 2048
ABC = rand(Float32, N), rand(Float32, N), rand(Float32, N, N);
abc = map(GPUArray, ABC);

@time benchmark(abc...)
@time pairwise_dist_cpu(ABC..., N)
