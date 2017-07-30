import CUFFT

# figure out a gc safe way to store plans.
# weak refs won't work, since the caching should keep them alive.
# But at the end, we need to free all of these, otherwise CUFFT will crash
# at closing time.
# An atexit hook here, which will empty the dictionary seems to introduce racing
# conditions.
#const plan_dict = Dict()
import Base: *, plan_ifft!, plan_fft!, plan_fft, plan_ifft, size, plan_bfft

immutable CUFFTPlan{Direction, Inplace, T, N} <: Base.FFTW.FFTWPlan{T, Direction, Inplace}
    plan::CUFFT.Plan{T}
    function (::Type{CUFFTPlan{Direction, Inplace}})(A::CLArray{T, N}) where {T, N, Direction, Inplace}
        ctx = context(A)
        p = Cint[0]
        sz = CUFFT.plan_size(A, A)
        inembed = reverse(Cint[size(src)...])
        onembed = reverse(Cint[size(dest)...])
        inembed[end] = CUFFT.pitchel(src)
        onembed[end] = CUFFT.pitchel(dest)
        plantype = CUFFT.plan_dict[(eltype(src), eltype(dest))]
        lib.cufftPlanMany(p, ndims(dest), sz, inembed, 1, 1, onembed, 1, 1, plantype, 1)
        CUFFT.pl = Plan{eltype(src),eltype(dest),ndims(dest)}(p[1])
        tie(pl, stream)
        new{Direction, Inplace, T, N}(p)
    end
end
size(x::CUFFTPlan) = (CUFFT.lengths(x.plan)...,)

# ignore flags, but have them to make it base compatible.
# TODO can we actually implement the flags?
function plan_fft(A::CLArray; flags = nothing, timelimit = Inf)
    CUFFTPlan{:forward, false}(A)
end
function plan_bfft(A::CLArray, region; flags = nothing, timelimit = Inf)
    CUFFTPlan{:forward, false}(A)
end
function plan_fft!(A::CLArray; flags = nothing, timelimit = Inf)
    CUFFTPlan{:forward, true}(A)
end

function plan_ifft!(A::CLArray; flags = nothing, timelimit = Inf)
    CUFFTPlan{:backward, true}(A)
end

function *(p::Base.DFT.ScaledPlan{T, CUFFTPlan{Direction, Inplace, T, N}}, x::CLArray{T, N}) where {T, N, Inplace, Direction}
    CUFFT.set_scaling_factor!(p.p.plan, Direction, p.scale)
    p.p * x
end

const _queue_ref = Vector{cl.CmdQueue}(1)
function *(plan::CUFFTPlan{Direction, true, T, N}, A::CLArray{T, N}) where {T, N, Direction}
    _queue_ref[] = context(A).queue
    CUFFT.enqueue_transform(plan.plan, Direction, _queue_ref, buffer(A), nothing)
    A
end
function *(plan::CUFFTPlan{Direction, false, T, N}, A::CLArray{T, N}) where {T, N, Direction}
    _queue_ref[] = context(A).queue
    y = typeof(A)(size(plan))
    CUFFT.enqueue_transform(plan.plan, Direction, _queue_ref, buffer(A), buffer(y))
    y
end
