using Colors, ColorVectorSpace, FileIO
using GPUArrays
import GPUArrays: mapidx, AbstractAccArray

@inline euclidian(a::AbstractFloat, b::AbstractFloat) = abs(a - b)
@inline function euclidian(a::Colorant, b::Colorant)
    abs(comp1(a) - comp1(b)) +
    abs(comp2(a) - comp2(b)) +
    abs(comp3(a) - comp3(b))
end
function labeldata{T}(val, means, k::T)
    label = T(0); mindist = Float32(Inf)
    @inbounds for ki = T(1):k
        dist = euclidian(means[ki], val)
        if dist < mindist
            mindist = dist
            label = ki
        end
    end
    return UInt8(label)
end

@inline function meancount{T}(v::T, label, k)
    count = UInt32(label == k)
    countf = Float32(count)
    (v * countf, count)
end
dotplus(ab0, ab1) = (ab1[1] + ab0[1], ab1[2] + ab0[2]) # could be .+ on 0.6

function shiftmeans!(tmp, data, means, labels)
    T = eltype(means)
    @inbounds for k = 1:length(means)
        sum, len = mapreduce(
            meancount, dotplus,
            (zero(T), UInt32(0)), data, labels, UInt32(k)
        )
        tmp[k] = sum / (len + 1e-5)
    end
    copy!(means, tmp)
    nothing
end
function clustersmoved(prev_means, thrr_means)
    mapreduce(
        (a, b)-> UInt32(a != b), +,
        UInt32(0), prev_means, thrr_means
    )
end
function kmeans{T <: AbstractAccArray}(A::T, initialmeans, iter = 10)
    means = identity.(initialmeans)
    n = length(A); CT = eltype(T); ET = eltype(CT); k = length(means)
    labels = map(x-> UInt8(0), A)
    prevlabels = identity.(labels)
    tmp = zeros(eltype(means), length(means))
    for i = 1:iter
        copy!(prevlabels, labels)
        labels .= labeldata.(A, Base.RefValue(means), Int(k))
        num_changed = clustersmoved(prevlabels, labels)
        if num_changed < (n / 1000) + 1
            return means, labels
        end
        shiftmeans!(tmp, A, means, labels)
    end
    means, labels
end

impath = homedir()*"/juliastuff/HurricaneKatrina/images/1.jpg"
clusters = [
    RGB{Float32}(0, 0, 0), RGB{Float32}(1, 0, 0),
    RGB{Float32}(0, 1, 0), RGB{Float32}(0, 0, 1),
    RGB{Float32}(1, 1, 0), RGB{Float32}(0, 1, 1),
]

imgs = RGB{Float32}.(load(impath)) # scale and convert image
imgvec = vec(imgs)
clusters = rand(eltype(imgvec), 32)
CLBackend.init()

thimgvec, thclusters = GPUArray(imgvec), GPUArray(clusters)
eltype(thimgvec)
@time begin
    x,l = kmeans(thimgvec, thclusters)
    GPUArrays.synchronize(x)
end

tht1 = @elapsed kmeans(thimgvec, thclusters)
tht2 = @elapsed kmeans(thimgvec, thclusters)
