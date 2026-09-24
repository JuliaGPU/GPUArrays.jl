# findall and predicates

# As for sorting: Base, applied to the storage, with the method shapes of GPUArrays.

Base.findall(bools::AnyJLArray{Bool}) = JLArray(findall(_host(bools)))
Base.findall(f::Function, A::AnyJLArray) = JLArray(findall(f, _host(A)))
Base.findall(f::Base.Fix2{typeof(in)}, A::AnyJLArray) = JLArray(findall(f, _host(A)))
Base.getindex(A::JLArray, mask::AnyJLArray{Bool}) = JLArray(_host(A)[_host(mask)])

for fname in (:any, :all)
    @eval Base.$fname(f::Function, A::AnyJLArray; dims=:) =
        dims === Colon() ? $fname(f, _host(A)) :
                           invoke($fname, Tuple{Function, GPUArrays.AnyGPUArray}, f, A; dims)
end
