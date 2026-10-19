# on-device array type and kernel argument conversion

struct Adaptor end
jlconvert(arg) = adapt(Adaptor(), arg)

# convert what a `Ref` holds, but leave the `Ref` itself to POCL, which also handles `Ref`s
# of types (as passed by `convert.(T, A)`)
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = Ref(adapt(to, r[]))
