import CUFFT

# figure out a gc safe way to store plans.
# weak refs won't work, since the caching should keep them alive.
# But at the end, we need to free all of these, otherwise CUFFT will crash
# at closing time.
# An atexit hook here, which will empty the dictionary seems to introduce racing
# conditions.
#const plan_dict = Dict()
import Base: *, plan_ifft!, plan_fft!, plan_fft, plan_ifft, size, plan_bfft, plan_bfft!

immutable CUFFTPlan{Direction, Inplace, T, N, P <: CUFFT.Plan} <: Base.FFTW.FFTWPlan{T, Direction, Inplace}
    plan::P
    function (::Type{CUFFTPlan{Direction, Inplace}})(dest::CUArray{T, N}, src = dest) where {T, N, Direction, Inplace}
        ctx = context(dest)
        p = Cint[0]
        sz = CUFFT.plan_size(dest, src)
        inembed = Cint[reverse(size(src))...]
        onembed = Cint[reverse(size(dest))...]
        rtsrc = CUBackend.to_cudart(src)
        rtdest = CUBackend.to_cudart(dest)
        inembed[end] = CUFFT.pitchel(rtsrc)
        onembed[end] = CUFFT.pitchel(rtdest)
        plantype = CUFFT.plan_dict[(eltype(src), eltype(dest))]
        CUFFT.lib.cufftPlanMany(p, ndims(dest), sz, inembed, 1, 1, onembed, 1, 1, plantype, 1)
        pl = CUFFT.Plan{eltype(src), eltype(dest), ndims(dest)}(p[1])
        str = CUDArt.Stream(CuDefaultStream(), CUDArt.AsyncCondition())
        CUFFT.tie(pl, str)
        new{Direction, Inplace, T, N, typeof(pl)}(pl)
    end
end
size(x::CUFFTPlan) = (CUFFT.lengths(x.plan)...,)

# ignore flags, but have them to make it base compatible.
# TODO can we actually implement the flags?
function plan_fft(A::CUArray; flags = nothing, timelimit = Inf)
    CUFFTPlan{:forward, false}(A)
end
function plan_bfft(A::CUArray, region; flags = nothing, timelimit = Inf)
    CUFFTPlan{:backward, false}(A)
end
function plan_bfft!(A::CUArray, region; flags = nothing, timelimit = Inf)
    CUFFTPlan{:backward, true}(A)
end
function plan_fft!(A::CUArray; flags = nothing, timelimit = Inf)
    CUFFTPlan{:forward, true}(A)
end

function *(plan::CUFFTPlan{Direction, true, T, N}, A::CUArray{T, N}) where {T, N, Direction}
    a_rt = to_cudart(A)
    CUFFT.exec!(plan.plan, a_rt, a_rt, Direction == :forward)
    A
end
function *(plan::CUFFTPlan{Direction, false, T, N}, A::CUArray{T, N}) where {T, N, Direction}
    dest = similar(A)
    CUFFT.exec!(plan.plan, to_cudart(A), to_cudart(dest), Direction == :forward)
    dest
end
