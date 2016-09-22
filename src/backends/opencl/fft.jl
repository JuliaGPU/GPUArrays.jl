const plan_dict = Dict{NTuple{3, Int}, clfft.Plan}()

function getplan!(A)
    get!(plan_dict, size(A)) do
        p = clfft.Plan(Complex64, ctx, size(A))
        clfft.set_layout!(p, :interleaved, :interleaved)
        clfft.set_result!(p, :inplace)
        clfft.bake!(p, queue)
        p
    end
end

function Base.fft!(A::cl.CLArray)
    clfft.enqueue_transform(getplan!(A), :forward, [queue], A.buffer, nothing)
end
function Base.ifft!(A::cl.CLArray)
    clfft.enqueue_transform(getplan!(A), :backward, [queue], A.buffer, nothing)
end
