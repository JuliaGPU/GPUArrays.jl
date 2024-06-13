module EnzymeExt

using JLArrays

if isdefined(Base, :get_extension)
    using Enzyme
else
    using ..Enzyme
end

# Override default type tree. This is because JLArray stores data as Vector{UInt8}, causing issues for
# type analysis not determining the proper element type (instead determining the memory is of type UInt8).
function Enzyme.typetree_inner(::Type{JLT}, ctx, dl, seen::Enzyme.TypeTreeTable) where {JLT<:JLArray}
    if T isa UnionAll || T isa Union || T == Union{} || Base.isabstracttype(T)
        return TypeTree()
    end

    if !Base.isconcretetype(T)
        return Enzyme.TypeTree(Enzyme.API.DT_Pointer, -1, ctx)
    end

    elT = eltype(JLT)

    fieldTypes = [DataRef{Vector{elT}}, Int, Dims{length(size(JLT))}]

    tt = Enzyme.TypeTree()
    for f in 1:fieldcount(T)
        offset = fieldoffset(T, f)
        subT = fieldTypes[f]
        subtree = copy(Enzyme.typetree(subT, ctx, dl, seen))

        if subT isa UnionAll || subT isa Union || subT == Union{}
            # FIXME: Handle union
            continue
        end

        # Allocated inline so adjust first path
        if allocatedinline(subT)
            Enzyme.shift!(subtree, dl, 0, sizeof(subT), offset)
        else
            Enzyme.merge!(subtree, TypeTree(API.DT_Pointer, ctx))
            Enzyme.only!(subtree, offset)
        end

        Enzyme.merge!(tt, subtree)
    end
    Enzyme.canonicalize!(tt, sizeof(T), dl)
    return tt
end

end # module
