# reductions

struct ArrayNoCopy end
Adapt.adapt_storage(::ArrayNoCopy, x::JLArray) = typed_data(x)

function GPUArrays.mapreducedim!(f, op, R::AnyJLArray, A::Union{AbstractArray,Broadcast.Broadcasted};
                                 init=nothing)
    if init !== nothing
        fill!(R, init)
    end
    @allowscalar Base.reducedim!(op, adapt(ArrayNoCopy(), R), map(f, A))
    R
end
