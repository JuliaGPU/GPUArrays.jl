# broadcasting

import Base.Broadcast: BroadcastStyle, Broadcasted

struct JLArrayStyle{N} <: AbstractGPUArrayStyle{N} end
JLArrayStyle{M}(::Val{N}) where {N,M} = JLArrayStyle{N}()

# identify the broadcast style of a (wrapped) array
BroadcastStyle(::Type{<:JLArray{T,N}}) where {T,N} = JLArrayStyle{N}()
BroadcastStyle(::Type{<:AnyJLArray{T,N}}) where {T,N} = JLArrayStyle{N}()

# allocation of output arrays
Base.similar(bc::Broadcasted{JLArrayStyle{N}}, ::Type{T}, dims) where {T,N} =
    similar(JLArray{T}, dims)
