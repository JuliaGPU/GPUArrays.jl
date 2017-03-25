const plan_dict = Dict{NTuple{3, Int}, clfft.Plan}()

function getplan!{T, N}(A::CLArray{T, N})
    get!(plan_dict, size(A)) do
        p = clfft.Plan(T, ctx, size(A))
        clfft.set_layout!(p, :interleaved, :interleaved)
        clfft.set_result!(p, :inplace)
        clfft.bake!(p, queue)
        p
    end
end

function Base.fft!(A::cl.CLArray)
    qref = [queue]
    clfft.enqueue_transform(getplan!(A), :forward, qref, A.buffer, nothing)
end
function Base.ifft!(A::cl.CLArray)
    qref = [queue]
    clfft.enqueue_transform(getplan!(A), :backward, qref, A.buffer, nothing)
end
