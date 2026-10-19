# on-device array type and kernel argument conversion

struct Adaptor end
jlconvert(arg) = adapt(Adaptor(), arg)

# FIXME: add Ref to Adapt.jl (but make sure it doesn't cause ambiguities with CUDAnative's)
struct JlRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::JlRefValue) = r.x
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = JlRefValue(adapt(to, r[]))
