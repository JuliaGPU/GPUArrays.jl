import CLFFT

# figure out a gc safe way to store plans.
# weak refs won't work, since the caching should keep them alive.
# But at the end, we need to free all of these, otherwise CLFFT will crash
# at closing time.
# An atexit hook here, which will empty the dictionary seems to introduce racing
# conditions.
#const plan_dict = Dict()
import Base: *, plan_ifft!, plan_fft!, plan_fft, plan_ifft, size, plan_bfft, plan_bfft!

struct CLFFTPlan{Direction, Inplace, T, N} <: Base.FFTW.FFTWPlan{T, Direction, Inplace}
    plan::CLFFT.Plan{T}
    function CLFFTPlan{Direction, Inplace}(A::CLArray{T, N}) where {T, N, Direction, Inplace}
        ctx = context(A)
        p = CLFFT.Plan(T, ctx.context, size(A))
        CLFFT.set_layout!(p, :interleaved, :interleaved)
        if Inplace
            CLFFT.set_result!(p, :inplace)
        else
            CLFFT.set_result!(p, :outofplace)
        end
        CLFFT.set_scaling_factor!(p, Direction, 1f0)
        CLFFT.bake!(p, ctx.queue)
        new{Direction, Inplace, T, N}(p)
    end
end
size(x::CLFFTPlan) = (CLFFT.lengths(x.plan)...,)

# ignore flags, but have them to make it base compatible.
# TODO can we actually implement the flags?
function plan_fft(A::CLArray; flags = nothing, timelimit = Inf)
    CLFFTPlan{:forward, false}(A)
end
function plan_fft!(A::CLArray; flags = nothing, timelimit = Inf)
    CLFFTPlan{:forward, true}(A)
end
function plan_bfft(A::CLArray, region; flags = nothing, timelimit = Inf)
    CLFFTPlan{:backward, false}(A)
end
function plan_bfft!(A::CLArray, region; flags = nothing, timelimit = Inf)
    CLFFTPlan{:backward, true}(A)
end

const _queue_ref = Vector{cl.CmdQueue}(1)
function *(plan::CLFFTPlan{Direction, true, T, N}, A::CLArray{T, N}) where {T, N, Direction}
    _queue_ref[] = context(A).queue
    CLFFT.enqueue_transform(plan.plan, Direction, _queue_ref, buffer(A), nothing)
    A
end
function *(plan::CLFFTPlan{Direction, false, T, N}, A::CLArray{T, N}) where {T, N, Direction}
    _queue_ref[] = context(A).queue
    y = typeof(A)(size(plan))
    CLFFT.enqueue_transform(plan.plan, Direction, _queue_ref, buffer(A), buffer(y))
    y
end
