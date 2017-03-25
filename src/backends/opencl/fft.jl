import CLFFT

const plan_dict = Dict()

function getplan!{T, N}(A::CLArray{T, N})
    if haskey(plan_dict, size(A))
        return plan_dict[size(A)]
    else
        ctx = context(A)
        p = CLFFT.Plan(T, ctx.context, size(A))
        CLFFT.set_layout!(p, :interleaved, :interleaved)
        CLFFT.set_result!(p, :inplace)
        CLFFT.bake!(p, ctx.queue)
        plan_dict[size(A)] = p
        p
    end
end

function Base.fft!{T, N}(A::CLArray{T, N})
    qref = [context(A).queue]
    CLFFT.enqueue_transform(getplan!(A), :forward, qref, buffer(A), nothing)
end
function Base.ifft!{T, N}(A::CLArray{T, N})
    qref = [context(A).queue]
    CLFFT.enqueue_transform(getplan!(A), :backward, qref, buffer(A), nothing)
end
