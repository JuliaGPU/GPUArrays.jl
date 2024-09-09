# compatibility with EnzymeCore

module EnzymeCoreExt

using GPUArraysCore

if isdefined(Base, :get_extension)
    using EnzymeCore
    using EnzymeCore.EnzymeRules
else
    using ..EnzymeCore
    using ..EnzymeCore.EnzymeRules
end

function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(GPUArraysCore.default_scalar_indexing), args...)
    return nothing
end

function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(GPUArraysCore.assertscalar), args...)
    return nothing
end

function EnzymeCore.EnzymeRules.inactive_noinl(::typeof(GPUArraysCore.allowscalar), args...)
    return nothing
end

end # module
