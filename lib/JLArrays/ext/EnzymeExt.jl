module EnzymeExt

using JLArrays

using GPUArrays

if isdefined(Base, :get_extension)
    using Enzyme
else
    using ..Enzyme
end


# Override default type tree. This is because JLArray stores data as Vector{UInt8}, causing issues for
# type analysis not determining the proper element type (instead determining the memory is of type UInt8).
function Enzyme.typetree_inner(::Type{JLT}, ctx, dl, seen::Enzyme.TypeTreeTable) where {JLT<:JLArray}
    if JLT isa UnionAll || JLT isa Union || JLT == Union{} || Base.isabstracttype(JLT)
        return Enzyme.TypeTree()
    end

    if !Base.isconcretetype(JLT)
        return Enzyme.TypeTree(Enzyme.API.DT_Pointer, -1, ctx)
    end

    elT = eltype(JLT)

    fieldTypes = [DataRef{Vector{elT}}, Int, Dims{ndims(JLT)}]

    tt = Enzyme.TypeTree()
    for f in 1:fieldcount(JLT)
        offset = fieldoffset(JLT, f)
        subT = fieldTypes[f]
        subtree = copy(Enzyme.typetree(subT, ctx, dl, seen))

        if subT isa UnionAll || subT isa Union || subT == Union{}
            # FIXME: Handle union
            continue
        end

        # Allocated inline so adjust first path
        if Enzyme.allocatedinline(subT)
            Enzyme.shift!(subtree, dl, 0, sizeof(subT), offset)
        else
            Enzyme.merge!(subtree, Enzyme.TypeTree(Enzyme.API.DT_Pointer, ctx))
            Enzyme.only!(subtree, offset)
        end

        Enzyme.merge!(tt, subtree)
    end
    Enzyme.canonicalize!(tt, sizeof(JLT), dl)
    return tt
end

end # module
