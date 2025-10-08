abstract type AbstractGPUSparseArray{Tv, Ti, N} <: AbstractSparseArray{Tv, Ti, N} end
const AbstractGPUSparseVector{Tv, Ti} = AbstractGPUSparseArray{Tv, Ti, 1}
const AbstractGPUSparseMatrix{Tv, Ti} = AbstractGPUSparseArray{Tv, Ti, 2}

abstract type AbstractGPUSparseMatrixCSC{Tv, Ti} <: AbstractGPUSparseArray{Tv, Ti, 2} end
abstract type AbstractGPUSparseMatrixCSR{Tv, Ti} <: AbstractGPUSparseArray{Tv, Ti, 2} end
abstract type AbstractGPUSparseMatrixCOO{Tv, Ti} <: AbstractGPUSparseArray{Tv, Ti, 2} end
abstract type AbstractGPUSparseMatrixBSR{Tv, Ti} <: AbstractGPUSparseArray{Tv, Ti, 2} end

const AbstractGPUSparseVecOrMat = Union{AbstractGPUSparseVector,AbstractGPUSparseMatrix}

SparseArrays.nnz(g::T) where {T<:AbstractGPUSparseArray} = g.nnz
SparseArrays.nonzeros(g::T) where {T<:AbstractGPUSparseArray} = g.nzVal

SparseArrays.nonzeroinds(g::T) where {T<:AbstractGPUSparseVector} = g.iPtr
SparseArrays.rowvals(g::T) where {T<:AbstractGPUSparseVector} = nonzeroinds(g)

SparseArrays.rowvals(g::AbstractGPUSparseMatrixCSC) = g.rowVal
SparseArrays.getcolptr(S::AbstractGPUSparseMatrixCSC) = S.colPtr

Base.convert(T::Type{<:AbstractGPUSparseArray}, m::AbstractArray) = m isa T ? m : T(m)

_dense_array_type(sa::SparseVector)     = SparseVector
_dense_array_type(::Type{SparseVector}) = SparseVector
_sparse_array_type(sa::SparseVector) = SparseVector
_dense_vector_type(sa::AbstractSparseArray) = Vector
_dense_vector_type(sa::AbstractArray)       = Vector
_dense_vector_type(::Type{<:AbstractSparseArray}) = Vector
_dense_vector_type(::Type{<:AbstractArray})       = Vector
_dense_array_type(sa::SparseMatrixCSC)     = SparseMatrixCSC
_dense_array_type(::Type{SparseMatrixCSC}) = SparseMatrixCSC
_sparse_array_type(sa::SparseMatrixCSC)    = SparseMatrixCSC

function _sparse_array_type(sa::AbstractGPUSparseArray) end
function _dense_array_type(sa::AbstractGPUSparseArray) end

### BROADCAST

# broadcast container type promotion for combinations of sparse arrays and other types
struct GPUSparseVecStyle <: Broadcast.AbstractArrayStyle{1} end
struct GPUSparseMatStyle <: Broadcast.AbstractArrayStyle{2} end
Broadcast.BroadcastStyle(::Type{<:AbstractGPUSparseVector}) = GPUSparseVecStyle()
Broadcast.BroadcastStyle(::Type{<:AbstractGPUSparseMatrix}) = GPUSparseMatStyle()
const SPVM = Union{GPUSparseVecStyle,GPUSparseMatStyle}

# GPUSparseVecStyle handles 0-1 dimensions, GPUSparseMatStyle 0-2 dimensions.
# GPUSparseVecStyle promotes to GPUSparseMatStyle for 2 dimensions.
# Fall back to DefaultArrayStyle for higher dimensionality.
GPUSparseVecStyle(::Val{0}) = GPUSparseVecStyle()
GPUSparseVecStyle(::Val{1}) = GPUSparseVecStyle()
GPUSparseVecStyle(::Val{2}) = GPUSparseMatStyle()
GPUSparseVecStyle(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()
GPUSparseMatStyle(::Val{0}) = GPUSparseMatStyle()
GPUSparseMatStyle(::Val{1}) = GPUSparseMatStyle()
GPUSparseMatStyle(::Val{2}) = GPUSparseMatStyle()
GPUSparseMatStyle(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()

Broadcast.BroadcastStyle(::GPUSparseVecStyle, ::AbstractGPUArrayStyle{1}) = GPUSparseVecStyle()
Broadcast.BroadcastStyle(::GPUSparseVecStyle, ::AbstractGPUArrayStyle{2}) = GPUSparseMatStyle()
Broadcast.BroadcastStyle(::GPUSparseMatStyle, ::AbstractGPUArrayStyle{2}) = GPUSparseMatStyle()

# don't wrap sparse arrays with Extruded
Broadcast.extrude(x::AbstractGPUSparseVecOrMat) = x

## detection of zero-preserving functions

# modified from SparseArrays.jl

# capturescalars takes a function (f) and a tuple of broadcast arguments, and returns a
# partially-evaluated function and a reduced argument tuple where all scalar operations have
# been applied already.
@inline function capturescalars(f, mixedargs)
    let (passedsrcargstup, makeargs) = _capturescalars(mixedargs...)
        parevalf = (passed...) -> f(makeargs(passed...)...)
        return (parevalf, passedsrcargstup)
    end
end

## sparse broadcast style

# Work around losing Type{T}s as DataTypes within the tuple that makeargs creates
@inline capturescalars(f, mixedargs::Tuple{Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((args...)->f(T, args...), Base.tail(mixedargs))
@inline capturescalars(f, mixedargs::Tuple{Ref{Type{T}}, Ref{Type{S}}, Vararg{Any}}) where {T, S} =
    # This definition is identical to the one above and necessary only for
    # avoiding method ambiguity.
    capturescalars((args...)->f(T, args...), Base.tail(mixedargs))
@inline capturescalars(f, mixedargs::Tuple{AbstractGPUSparseVecOrMat, Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((a1, args...)->f(a1, T, args...), (mixedargs[1], Base.tail(Base.tail(mixedargs))...))
@inline capturescalars(f, mixedargs::Tuple{Union{Ref,AbstractArray{<:Any,0}}, Ref{Type{T}}, Vararg{Any}}) where {T} =
    capturescalars((args...)->f(mixedargs[1], T, args...), Base.tail(Base.tail(mixedargs)))

scalararg(::Number) = true
scalararg(::Any) = false
scalarwrappedarg(::Union{AbstractArray{<:Any,0},Ref}) = true
scalarwrappedarg(::Any) = false

@inline function _capturescalars()
    return (), () -> ()
end
@inline function _capturescalars(arg, mixedargs...)
    let (rest, f) = _capturescalars(mixedargs...)
        if scalararg(arg)
            return rest, @inline function(tail...)
                (arg, f(tail...)...)
            end # add back scalararg after (in makeargs)
        elseif scalarwrappedarg(arg)
            return rest, @inline function(tail...)
                (arg[], f(tail...)...) # TODO: This can put a Type{T} in a tuple
            end # unwrap and add back scalararg after (in makeargs)
        else
            return (arg, rest...), @inline function(head, tail...)
                (head, f(tail...)...)
            end # pass-through to broadcast
        end
    end
end
@inline function _capturescalars(arg) # this definition is just an optimization (to bottom out the recursion slightly sooner)
    if scalararg(arg)
        return (), () -> (arg,) # add scalararg
    elseif scalarwrappedarg(arg)
        return (), () -> (arg[],) # unwrap
    else
        return (arg,), (head,) -> (head,) # pass-through
    end
end

@inline _iszero(x) = x == 0
@inline _iszero(x::Number) = Base.iszero(x)
@inline _iszero(x::AbstractArray) = Base.iszero(x)
@inline _zeros_eltypes(A) = (zero(eltype(A)),)
@inline _zeros_eltypes(A, Bs...) = (zero(eltype(A)), _zeros_eltypes(Bs...)...)

## COV_EXCL_START
## iteration helpers

"""
    CSRIterator{Ti}(row, args...)

A GPU-compatible iterator for accessing the elements of a single row `row` of several CSR
matrices `args` in one go. The row should be in-bounds for every sparse argument. Each
iteration returns a 2-element tuple: The current column, and each arguments' pointer index
(or 0 if that input didn't have an element at that column). The pointers can then be used to
access the elements themselves.

For convenience, this iterator can be passed non-sparse arguments as well, which will be
ignored (with the returned `col`/`ptr` values set to 0).
"""
struct CSRIterator{Ti,N,ATs}
    row::Ti
    col_ends::NTuple{N, Ti}
    args::ATs
end

function CSRIterator{Ti}(row, args::Vararg{Any, N}) where {Ti,N}
    # check that `row` is valid for all arguments
    @boundscheck begin
        ntuple(Val(N)) do i
            arg = @inbounds args[i]
            arg isa GPUSparseDeviceMatrixCSR && checkbounds(arg, row, 1)
        end
    end

    col_ends = ntuple(Val(N)) do i
        arg = @inbounds args[i]
        if arg isa GPUSparseDeviceMatrixCSR
            @inbounds(arg.rowPtr[row+1])
        else
            zero(Ti)
        end
    end

    CSRIterator{Ti, N, typeof(args)}(row, col_ends, args)
end

@inline function Base.iterate(iter::CSRIterator{Ti,N}, state=nothing) where {Ti,N}
    # helper function to get the column of a sparse array at a specific pointer
    @inline function get_col(i, ptr)
        arg = @inbounds iter.args[i]
        if arg isa GPUSparseDeviceMatrixCSR
            col_end = @inbounds iter.col_ends[i]
            if ptr < col_end
                return @inbounds arg.colVal[ptr] % Ti
            end
        end
        typemax(Ti)
    end

    # initialize the state
    # - ptr: the current index into the colVal/nzVal arrays
    # - col: the current column index (cached so that we don't have to re-read each time)
    state = something(state,
        ntuple(Val(N)) do i
            arg = @inbounds iter.args[i]
            if arg isa GPUSparseDeviceMatrixCSR
                ptr = @inbounds iter.args[i].rowPtr[iter.row] % Ti
                col = @inbounds get_col(i, ptr)
            else
                ptr = typemax(Ti)
                col = typemax(Ti)
            end
            (; ptr, col)
        end
    )

    # determine the column we're currently processing
    cols = ntuple(i -> @inbounds(state[i].col), Val(N))
    cur_col = min(cols...)
    cur_col == typemax(Ti) && return

    # fetch the pointers (we don't look up the values, as the caller might want to index
    # the sparse array directly, e.g., to mutate it). we don't return `ptrs` from the state
    # directly, but first convert the `typemax(Ti)` to a more convenient zero value.
    # NOTE: these values may end up unused by the caller (e.g. in the count_nnzs kernels),
    #       but LLVM appears smart enough to filter them away.
    ptrs = ntuple(Val(N)) do i
        ptr, col = @inbounds state[i]
        col == cur_col ? ptr : zero(Ti)
    end

    # advance the state
    new_state = ntuple(Val(N)) do i
        ptr, col = @inbounds state[i]
        if col == cur_col
            ptr += one(Ti)
            col = get_col(i, ptr)
        end
        (; ptr, col)
    end

    return (cur_col, ptrs), new_state
end

struct CSCIterator{Ti,N,ATs}
    col::Ti
    row_ends::NTuple{N, Ti}
    args::ATs
end

function CSCIterator{Ti}(col, args::Vararg{Any, N}) where {Ti,N}
    # check that `col` is valid for all arguments
    @boundscheck begin
        ntuple(Val(N)) do i
            arg = @inbounds args[i]
            arg isa GPUSparseDeviceMatrixCSR && checkbounds(arg, 1, col)
        end
    end

    row_ends = ntuple(Val(N)) do i
        arg = @inbounds args[i]
        x = if arg isa GPUSparseDeviceMatrixCSC
            @inbounds(arg.colPtr[col+1])
        else
            zero(Ti)
        end
        x
    end

    CSCIterator{Ti, N, typeof(args)}(col, row_ends, args)
end

@inline function Base.iterate(iter::CSCIterator{Ti,N}, state=nothing) where {Ti,N}
    # helper function to get the column of a sparse array at a specific pointer
    @inline function get_col(i, ptr)
        arg = @inbounds iter.args[i]
        if arg isa GPUSparseDeviceMatrixCSC
            col_end = @inbounds iter.row_ends[i]
            if ptr < col_end
                return @inbounds arg.rowVal[ptr] % Ti
            end
        end
        typemax(Ti)
    end

    # initialize the state
    # - ptr: the current index into the rowVal/nzVal arrays
    # - row: the current row index (cached so that we don't have to re-read each time)
    state = something(state,
        ntuple(Val(N)) do i
            arg = @inbounds iter.args[i]
            if arg isa GPUSparseDeviceMatrixCSC
                ptr = @inbounds iter.args[i].colPtr[iter.col] % Ti
                row = @inbounds get_col(i, ptr)
            else
                ptr = typemax(Ti)
                row = typemax(Ti)
            end
            (; ptr, row)
        end
    )

    # determine the row we're currently processing
    rows = ntuple(i -> @inbounds(state[i].row), Val(N))
    cur_row = min(rows...)
    cur_row == typemax(Ti) && return

    # fetch the pointers (we don't look up the values, as the caller might want to index
    # the sparse array directly, e.g., to mutate it). we don't return `ptrs` from the state
    # directly, but first convert the `typemax(Ti)` to a more convenient zero value.
    # NOTE: these values may end up unused by the caller (e.g. in the count_nnzs kernels),
    #       but LLVM appears smart enough to filter them away.
    ptrs = ntuple(Val(N)) do i
        ptr, row = @inbounds state[i]
        row == cur_row ? ptr : zero(Ti)
    end

    # advance the state
    new_state = ntuple(Val(N)) do i
        ptr, row = @inbounds state[i]
        if row == cur_row
            ptr += one(Ti)
            row = get_col(i, ptr)
        end
        (; ptr, row)
    end

    return (cur_row, ptrs), new_state
end

# helpers to index a sparse or dense array
function _getindex(arg::Union{<:GPUSparseDeviceMatrixCSR,GPUSparseDeviceMatrixCSC}, I, ptr)
    if ptr == 0
        zero(eltype(arg))
    else
        @inbounds arg.nzVal[ptr]
    end
end
@inline function _getindex(arg::AbstractDeviceArray{Tv}, I, ptr)::Tv where {Tv}
    return @inbounds arg[I]::Tv
end
@inline _getindex(arg, I, ptr) = Broadcast._broadcast_getindex(arg, I)

## sparse broadcast implementation
iter_type(::Type{<:AbstractGPUSparseMatrixCSC}, ::Type{Ti}) where {Ti} = CSCIterator{Ti}
iter_type(::Type{<:AbstractGPUSparseMatrixCSR}, ::Type{Ti}) where {Ti} = CSRIterator{Ti}
iter_type(::Type{<:GPUSparseDeviceMatrixCSC}, ::Type{Ti}) where {Ti} = CSCIterator{Ti}
iter_type(::Type{<:GPUSparseDeviceMatrixCSR}, ::Type{Ti}) where {Ti} = CSRIterator{Ti}

_has_row(A, offsets, row, fpreszeros::Bool) = fpreszeros ? 0 : row
_has_row(A::AbstractDeviceArray, offsets, row, ::Bool) = row
function _has_row(A::GPUSparseDeviceVector, offsets, row, ::Bool)
    for row_ix in 1:length(A.iPtr)
        arg_row = @inbounds A.iPtr[row_ix]
        arg_row == row && return row_ix
        arg_row > row && break
    end
    return 0
end

@kernel function compute_offsets_kernel(::Type{<:AbstractGPUSparseVector}, first_row::Ti, last_row::Ti,
                                        fpreszeros::Bool, offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                        args...) where {Ti, N}
    my_ix = @index(Global, Linear)
    row = my_ix + first_row - one(eltype(my_ix))
    if row ≤ last_row
        # TODO load arg.iPtr slices into shared memory
        arg_row_is_nnz = ntuple(Val(N)) do i
            arg = @inbounds args[i]
            _has_row(arg, offsets, row, fpreszeros)
        end
        row_is_nnz = 0
        for i in 1:N
            row_is_nnz |= @inbounds arg_row_is_nnz[i]
        end
        key = (row_is_nnz == 0) ? typemax(Ti) : row
        @inbounds offsets[my_ix] = key => arg_row_is_nnz
    end
end

# kernel to count the number of non-zeros in a row, to determine the row offsets
@kernel function compute_offsets_kernel(T::Type{<:Union{AbstractGPUSparseMatrixCSR, AbstractGPUSparseMatrixCSC}},
                                        offsets::AbstractVector{Ti}, args...) where Ti
    # every thread processes an entire row
    leading_dim = @index(Global, Linear)
    if leading_dim ≤ length(offsets)-1 
        iter = @inbounds iter_type(T, Ti)(leading_dim, args...)

        # count the nonzero leading_dims of all inputs
        accum = zero(Ti)
        for (leading_dim, vals) in iter
            accum += one(Ti)
        end

        # the way we write the nnz counts is a bit strange, but done so that the result
        # after accumulation can be directly used as the rowPtr/colPtr array of a CSR/CSC matrix.
        @inbounds begin
            if leading_dim == 1
                offsets[1] = 1
            end
            offsets[leading_dim+1] = accum
        end
    end
end

@kernel function sparse_to_sparse_broadcast_kernel(f::F, output::GPUSparseDeviceVector{Tv,Ti},
                                                   offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                                   args...) where {Tv, Ti, N, F}
    row_ix = @index(Global, Linear)
    if row_ix ≤ output.nnz
        row_and_ptrs = @inbounds offsets[row_ix]
        row          = @inbounds row_and_ptrs[1]
        arg_ptrs     = @inbounds row_and_ptrs[2]
        vals = ntuple(Val(N)) do i
            @inline
            arg = @inbounds args[i]
            # ptr is 0 if the sparse vector doesn't have an element at this row
            # ptr is 0 if the arg is a scalar AND f preserves zeros
            ptr = @inbounds arg_ptrs[i]
            _getindex(arg, row, ptr)
        end
        output_val = f(vals...)
        @inbounds output.iPtr[row_ix]  = row
        @inbounds output.nzVal[row_ix] = output_val
    end
end

@kernel function sparse_to_sparse_broadcast_kernel(f, output::T, offsets::Union{<:AbstractArray,Nothing},
                                                   args...) where {Ti, T<:Union{GPUSparseDeviceMatrixCSR{<:Any,Ti},
                                                                                GPUSparseDeviceMatrixCSC{<:Any,Ti}}}
    # every thread processes an entire row
    leading_dim = @index(Global, Linear)
    leading_dim_size = output isa GPUSparseDeviceMatrixCSR ? size(output, 1) : size(output, 2)
    if leading_dim ≤ leading_dim_size
        iter = @inbounds iter_type(T, Ti)(leading_dim, args...)


        output_ptrs  = output isa GPUSparseDeviceMatrixCSR ? output.rowPtr : output.colPtr
        output_ivals = output isa GPUSparseDeviceMatrixCSR ? output.colVal : output.rowVal
        # fetch the row offset, and write it to the output
        @inbounds begin
            output_ptr = output_ptrs[leading_dim] = offsets[leading_dim]
            if leading_dim == leading_dim_size
                output_ptrs[leading_dim+one(eltype(leading_dim))] = offsets[leading_dim+one(eltype(leading_dim))]
            end
        end

        # set the values for this row
        for (sub_leading_dim, ptrs) in iter
            index_first  = output isa GPUSparseDeviceMatrixCSR ? leading_dim : sub_leading_dim
            index_second = output isa GPUSparseDeviceMatrixCSR ? sub_leading_dim : leading_dim
            I = CartesianIndex(index_first, index_second)
            vals = ntuple(Val(length(args))) do i
                arg = @inbounds args[i]
                ptr = @inbounds ptrs[i]
                _getindex(arg, I, ptr)
            end

            @inbounds output_ivals[output_ptr] = sub_leading_dim
            @inbounds output.nzVal[output_ptr] = f(vals...)
            output_ptr += one(Ti)
        end
    end
end
@kernel function sparse_to_dense_broadcast_kernel(T::Type{<:Union{AbstractGPUSparseMatrixCSR{Tv, Ti},
                                                                  AbstractGPUSparseMatrixCSC{Tv, Ti}}},
                                                  f, output::AbstractDeviceArray, args...) where {Tv, Ti}
    # every thread processes an entire row
    leading_dim = @index(Global, Linear)
    leading_dim_size = T <: AbstractGPUSparseMatrixCSR ? size(output, 1) : size(output, 2)
    if leading_dim ≤ leading_dim_size
        iter = @inbounds iter_type(T, Ti)(leading_dim, args...)

        # set the values for this row
        for (sub_leading_dim, ptrs) in iter
            index_first  = T <: AbstractGPUSparseMatrixCSR ? leading_dim : sub_leading_dim
            index_second = T <: AbstractGPUSparseMatrixCSR ? sub_leading_dim : leading_dim
            I = CartesianIndex(index_first, index_second)
            vals = ntuple(Val(length(args))) do i
                arg = @inbounds args[i]
                ptr = @inbounds ptrs[i]
                _getindex(arg, I, ptr)
            end

            @inbounds output[I] = f(vals...)
        end
    end
end

@kernel function sparse_to_dense_broadcast_kernel(::Type{<:AbstractGPUSparseVector}, f::F,
                                                  output::AbstractDeviceArray{Tv},
                                                  offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                                  args...) where {Tv, F, N, Ti}
    # every thread processes an entire row
    row_ix = @index(Global, Linear)
    if row_ix ≤ length(output)
        row_and_ptrs = @inbounds offsets[row_ix]
        row          = @inbounds row_and_ptrs[1]
        arg_ptrs     = @inbounds row_and_ptrs[2]
        vals = ntuple(Val(length(args))) do i
            @inline
            arg = @inbounds args[i]
            # ptr is 0 if the sparse vector doesn't have an element at this row
            # ptr is row if the arg is dense OR a scalar with non-zero-preserving f
            # ptr is 0 if the arg is a scalar AND f preserves zeros
            ptr = @inbounds arg_ptrs[i]
            _getindex(arg, row, ptr)
        end
        @inbounds output[row] = f(vals...)
    end
end
## COV_EXCL_STOP

function Broadcast.copy(bc::Broadcasted{<:Union{GPUSparseVecStyle,GPUSparseMatStyle}})
    # find the sparse inputs
    bc = Broadcast.flatten(bc)
    sparse_args = findall(bc.args) do arg
        arg isa AbstractGPUSparseArray
    end
    sparse_types = unique(map(i->nameof(typeof(bc.args[i])), sparse_args))
    if length(sparse_types) > 1
        error("broadcast with multiple types of sparse arrays ($(join(sparse_types, ", "))) is not supported")
    end
    sparse_typ = typeof(bc.args[first(sparse_args)])
    sparse_typ <: Union{AbstractGPUSparseMatrixCSR,AbstractGPUSparseMatrixCSC,AbstractGPUSparseVector} ||
        error("broadcast with sparse arrays is currently only implemented for vectors and CSR and CSC matrices")
    Ti = if sparse_typ <: AbstractGPUSparseMatrixCSR
        reduce(promote_type, map(i->eltype(bc.args[i].rowPtr), sparse_args))
    elseif sparse_typ <: AbstractGPUSparseMatrixCSC
        reduce(promote_type, map(i->eltype(bc.args[i].colPtr), sparse_args))
    elseif sparse_typ <: AbstractGPUSparseVector
        reduce(promote_type, map(i->eltype(bc.args[i].iPtr), sparse_args))
    end

    # determine the output type
    Tv = Broadcast.combine_eltypes(bc.f, eltype.(bc.args))
    if !Base.isconcretetype(Tv)
        error("""GPU sparse broadcast resulted in non-concrete element type $Tv.
                 This probably means that the function you are broadcasting contains an error or type instability.""")
    end

    # partially-evaluate the function, removing scalars.
    parevalf, passedsrcargstup = capturescalars(bc.f, bc.args)
    # check if the partially-evaluated function preserves zeros. if so, we'll only need to
    # apply it to the sparse input arguments, preserving the sparse structure.
    if all(arg->isa(arg, AbstractSparseArray), passedsrcargstup)
        fofzeros = parevalf(_zeros_eltypes(passedsrcargstup...)...)
        fpreszeros = _iszero(fofzeros)
    else
        fpreszeros = false
    end

    # the kernels below parallelize across rows or cols, not elements, so it's unlikely
    # we'll launch many threads. to maximize utilization, parallelize across blocks first.
    rows, cols = get(size(bc), 1, 1), get(size(bc), 2, 1) 
    # `size(bc, ::Int)` is missing
    # for AbstractGPUSparseVec, figure out the actual row range we need to address, e.g. if m = 2^20
    # but the only rows present in any sparse vector input are between 2 and 128, no need to
    # launch massive threads.
    # TODO: use the difference here to set the thread count
    overall_first_row = one(Ti)
    overall_last_row = Ti(rows)
    offsets = nothing
    # allocate the output container
    sparse_arg = bc.args[first(sparse_args)]
    if !fpreszeros && sparse_typ <: Union{AbstractGPUSparseMatrixCSR, AbstractGPUSparseMatrixCSC}
        # either we have dense inputs, or the function isn't preserving zeros,
        # so use a dense output to broadcast into.
        val_array = nonzeros(sparse_arg)
        output    = similar(val_array, Tv, size(bc))
        # since we'll be iterating the sparse inputs, we need to pre-fill the dense output
        # with appropriate values (while setting the sparse inputs to zero). we do this by
        # re-using the dense broadcast implementation.
        nonsparse_args = map(bc.args) do arg
            # NOTE: this assumes the broadcast is flattened, but not yet preprocessed
            if arg isa AbstractGPUSparseArray
                zero(eltype(arg))
            else
                arg
            end
        end
        broadcast!(bc.f, output, nonsparse_args...)
    elseif length(sparse_args) == 1 && sparse_typ <: Union{AbstractGPUSparseMatrixCSR, AbstractGPUSparseMatrixCSC}
        # we only have a single sparse input, so we can reuse its structure for the output.
        # this avoids a kernel launch and costly synchronization.
        if sparse_typ <: AbstractGPUSparseMatrixCSR
            offsets = rowPtr = sparse_arg.rowPtr
            colVal  = similar(sparse_arg.colVal)
            nzVal   = similar(sparse_arg.nzVal, Tv)
            output  = _sparse_array_type(sparse_typ)(rowPtr, colVal, nzVal, size(bc))
        elseif sparse_typ <: AbstractGPUSparseMatrixCSC
            offsets = colPtr = sparse_arg.colPtr
            rowVal  = similar(sparse_arg.rowVal)
            nzVal   = similar(sparse_arg.nzVal, Tv)
            output  = _sparse_array_type(sparse_typ)(colPtr, rowVal, nzVal, size(bc))
        end
    else
        # determine the number of non-zero elements per row so that we can create an
        # appropriately-structured output container
        offsets = if sparse_typ <: AbstractGPUSparseMatrixCSR
            ptr_array = sparse_arg.rowPtr
            similar(ptr_array, Ti, rows+1)
        elseif sparse_typ <: AbstractGPUSparseMatrixCSC
            ptr_array = sparse_arg.colPtr
            similar(ptr_array, Ti, cols+1)
        elseif sparse_typ <: AbstractGPUSparseVector
            ptr_array = sparse_arg.iPtr
            @allowscalar begin
                arg_first_rows = ntuple(Val(length(bc.args))) do i
                    bc.args[i] isa AbstractGPUSparseVector && return bc.args[i].iPtr[1]
                    return one(Ti)
                end
                arg_last_rows = ntuple(Val(length(bc.args))) do i
                    bc.args[i] isa AbstractGPUSparseVector && return bc.args[i].iPtr[end]
                    return Ti(rows)
                end
            end
            overall_first_row = min(arg_first_rows...)
            overall_last_row  = max(arg_last_rows...)
            similar(ptr_array, Pair{Ti, NTuple{length(bc.args), Ti}}, overall_last_row - overall_first_row + 1)
        end
        let
            args = if sparse_typ <: AbstractGPUSparseVector
                (sparse_typ, overall_first_row, overall_last_row, fpreszeros, offsets, bc.args...)
            else
                (sparse_typ, offsets, bc.args...)
            end
            kernel = compute_offsets_kernel(get_backend(bc.args[first(sparse_args)]))
            kernel(args...; ndrange=length(offsets))
        end
        # accumulate these values so that we can use them directly as row pointer offsets,
        # as well as to get the total nnz count to allocate the sparse output array.
        # cusparseXcsrgeam2Nnz computes this in one go, but it doesn't seem worth the effort
        if !(sparse_typ <: AbstractGPUSparseVector)
            @allowscalar accumulate!(Base.add_sum, offsets, offsets)
            total_nnz = @allowscalar last(offsets[end]) - 1
        else
            @allowscalar sort!(offsets; by=first)
            total_nnz = mapreduce(x->first(x) != typemax(first(x)), +, offsets)
        end
        output = if sparse_typ <: Union{AbstractGPUSparseMatrixCSR,AbstractGPUSparseMatrixCSC}
            ixVal = similar(offsets, Ti, total_nnz)
            nzVal = similar(offsets, Tv, total_nnz)
            sparse_typ(offsets, ixVal, nzVal, size(bc))
        elseif sparse_typ <: AbstractGPUSparseVector && !fpreszeros
            val_array = bc.args[first(sparse_args)].nzVal
            similar(val_array, Tv, size(bc))
        elseif sparse_typ <: AbstractGPUSparseVector && fpreszeros
            iPtr   = similar(offsets, Ti, total_nnz)
            nzVal  = similar(offsets, Tv, total_nnz)
            _sparse_array_type(sparse_arg){Tv, Ti}(iPtr, nzVal, rows)
        end
        if sparse_typ <: AbstractGPUSparseVector && !fpreszeros
            nonsparse_args = map(bc.args) do arg
                # NOTE: this assumes the broadcst is flattened, but not yet preprocessed
                if arg isa AbstractGPUSparseArray
                    zero(eltype(arg))
                else
                    arg
                end
            end
            broadcast!(bc.f, output, nonsparse_args...)
        end
    end
    # perform the actual broadcast
    if output isa AbstractGPUSparseArray
        args   = (bc.f, output, offsets, bc.args...)
        kernel = sparse_to_sparse_broadcast_kernel(get_backend(bc.args[first(sparse_args)]))
        ndrange = output.nnz
    else
        args   = sparse_typ <: AbstractGPUSparseVector ? (sparse_typ, bc.f, output, offsets, bc.args...) :
                                                         (sparse_typ, bc.f, output, bc.args...)
        kernel = sparse_to_dense_broadcast_kernel(get_backend(bc.args[first(sparse_args)]))
        ndrange = sparse_typ <: AbstractGPUSparseMatrixCSC ? size(output, 2) : size(output, 1)
    end
    kernel(args...; ndrange)
    return output
end
## COV_EXCL_START
@kernel function csr_reduce_kernel(f::F, op::OP, neutral, zeros_preserved::Bool, output::AbstractDeviceArray, args...) where {F, OP}
    # every thread processes an entire row
    row = @index(Global, Linear)
    if row ≤ size(output, 1)
        iter = @inbounds CSRIterator{Int}(row, args...)

        val = op(neutral, neutral)

        # reduce the values for this row
        for (col, ptrs) in iter
            I = CartesianIndex(row, col)
            vals = ntuple(Val(length(args))) do i
                arg = @inbounds args[i]
                ptr = @inbounds ptrs[i]
                _getindex(arg, I, ptr)
            end
            val = op(val, f(vals...))
        end
        if !zeros_preserved
            f_zero_val   = f(zero(neutral))
            next_row_ind = row+1
            nzs_this_row = ntuple(Val(length(args))) do i
                max_n_zeros = size(args[i], 2)
                arg_row_ptr = args[i].rowPtr
                nz_this_row = max_n_zeros - (@inbounds(arg_row_ptr[next_row_ind]) - @inbounds(arg_row_ptr[row]))
                nz_this_row * f_zero_val
            end
            val = op(val, nzs_this_row...)
        end

        @inbounds output[row] = val
    end
end

@kernel function csc_reduce_kernel(f::F, op::OP, neutral, zeros_preserved::Bool, output::AbstractDeviceArray, args...) where {F, OP}
    # every thread processes an entire column
    col = @index(Global, Linear) 
    if col ≤ size(output, 2)
        iter = @inbounds CSCIterator{Int}(col, args...)

        val = op(neutral, neutral)

        # reduce the values for this col
        for (row, ptrs) in iter
            I = CartesianIndex(row, col)
            vals = ntuple(Val(length(args))) do i
                arg = @inbounds args[i]
                ptr = @inbounds ptrs[i]
                _getindex(arg, I, ptr)
            end
            val = op(val, f(vals...))
        end
        if !zeros_preserved
            f_zero_val   = f(zero(neutral))
            next_col_ind = col+1
            nzs_this_col = ntuple(Val(length(args))) do i
                max_n_zeros = size(args[i], 1)
                arg_col_ptr = args[i].colPtr
                nz_this_col = max_n_zeros - (@inbounds(arg_col_ptr[next_col_ind]) - @inbounds(arg_col_ptr[col]))
                nz_this_col * f_zero_val
            end
            val = op(val, nzs_this_col...)
        end
        @inbounds output[col] = val
    end
end
## COV_EXCL_STOP

function _csc_type end
function _csr_type end

# TODO: implement mapreducedim!
function Base.mapreduce(f, op, A::AbstractGPUSparseMatrix; dims=:, init=nothing)
    # figure out the destination container type by looking at the initializer element,
    # or by relying on inference to reason through the map and reduce functions
    if init === nothing
        ET = Broadcast.combine_eltypes(f, (A,))
        ET = Base.promote_op(op, ET, ET)
        (ET === Union{} || ET === Any) &&
            error("mapreduce cannot figure the output element type, please pass an explicit init value")

        init = zero(ET)
    else
        ET = typeof(init)
    end

    f_preserves_zeros = ( f(zero(ET)) == zero(ET) )
    # we only handle reducing along one of the two dimensions,
    # or a complete reduction (requiring an additional pass)
    in(dims, [Colon(), 1, 2]) || error("only dims=:, dims=1 or dims=2 is supported")

    if A isa AbstractGPUSparseMatrixCSR && dims == 1
        A = _csc_type(A)(A)
    elseif A isa AbstractGPUSparseMatrixCSC && dims == 2
        A = _csr_type(A)(A)
    end
    m, n      = size(A)
    val_array = nonzeros(A)
    backend   = get_backend(A)
    output_dim = 0
    ndrange    = 0
    if A isa AbstractGPUSparseMatrixCSR
        output_dim = m
        ndrange = m
        kernel = csr_reduce_kernel(backend)
    elseif A isa AbstractGPUSparseMatrixCSC 
        output_dim = (1, n)
        ndrange = n
        kernel = csc_reduce_kernel(backend)
    end
    output = similar(val_array, ET, output_dim)
    kernel(f, op, init, f_preserves_zeros, output, A; ndrange = ndrange)
    if dims == Colon()
        return mapreduce(identity, op, output; init)
    else
        return output
    end
end
