# broadcasting

# broadcast container type promotion for combinations of sparse arrays and other types
struct GPUSparseVecStyle <: Broadcast.AbstractArrayStyle{1} end
struct GPUSparseMatStyle <: Broadcast.AbstractArrayStyle{2} end
Broadcast.BroadcastStyle(::Type{<:GPUSparseVector}) = GPUSparseVecStyle()
Broadcast.BroadcastStyle(::Type{<:GPUSparseMatrix}) = GPUSparseMatStyle()
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
Broadcast.extrude(x::GPUSparseArray) = x

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
@inline capturescalars(f, mixedargs::Tuple{GPUSparseArray, Ref{Type{T}}, Vararg{Any}}) where {T} =
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
            arg isa GPUSparseMatrixCSR && checkbounds(axes(arg, 1), row)
        end
    end

    col_ends = ntuple(Val(N)) do i
        arg = @inbounds args[i]
        if arg isa GPUSparseMatrixCSR
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
        if arg isa GPUSparseMatrixCSR
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
            if arg isa GPUSparseMatrixCSR
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
            arg isa GPUSparseMatrixCSC && checkbounds(axes(arg, 2), col)
        end
    end

    row_ends = ntuple(Val(N)) do i
        arg = @inbounds args[i]
        x = if arg isa GPUSparseMatrixCSC
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
        if arg isa GPUSparseMatrixCSC
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
            if arg isa GPUSparseMatrixCSC
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
function _getindex(arg::Union{GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseVector}, I, ptr)
    if ptr == 0
        zero(eltype(arg))
    else
        @inbounds arg.nzVal[ptr]
    end
end
@inline function _getindex(arg::DenseArray{Tv}, I, ptr)::Tv where {Tv}
    return @inbounds arg[I]::Tv
end
@inline _getindex(arg, I, ptr) = Broadcast._broadcast_getindex(arg, I)

## sparse broadcast implementation
iter_type(::Type{<:GPUSparseMatrixCSC}, ::Type{Ti}) where {Ti} = CSCIterator{Ti}
iter_type(::Type{<:GPUSparseMatrixCSR}, ::Type{Ti}) where {Ti} = CSRIterator{Ti}

_has_row(A, offsets, row, fpreszeros::Bool) = fpreszeros ? 0 : row
_has_row(A::AbstractDeviceArray, offsets, row, ::Bool) = row
# the position of `row` among the stored indices of a sparse vector, or 0 if it isn't stored
function _has_row(A::GPUSparseVector, offsets, row, ::Bool)
    ptr = searchsortedfirst(A.nzInd, row)
    return (ptr <= length(A.nzInd) && @inbounds(A.nzInd[ptr]) == row) ? ptr : 0
end

# for every row, determine which arguments have an entry there (see `_has_row`), and whether
# the output has one (`key` is the row, or `typemax(Ti)` if not)
@kernel function compute_offsets_kernel(::Type{<:GPUSparseVector}, fpreszeros::Bool,
                                        offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                        args...) where {Ti, N}
    row = @index(Global, Linear)
    if row ≤ length(offsets)
        arg_row_is_nnz = ntuple(Val(N)) do i
            arg = @inbounds args[i]
            _has_row(arg, offsets, row, fpreszeros) % Ti
        end
        row_is_nnz = false
        for i in 1:N
            row_is_nnz |= @inbounds(arg_row_is_nnz[i]) != 0
        end
        key = row_is_nnz ? row % Ti : typemax(Ti)
        @inbounds offsets[row] = key => arg_row_is_nnz
    end
end

# kernel to count the number of non-zeros in a row, to determine the row offsets
@kernel function compute_offsets_kernel(T::Type{<:Union{GPUSparseMatrixCSR, GPUSparseMatrixCSC}},
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

        # the counts are shifted by one (`offsets[1]` is set to 1 by the caller) so that
        # after accumulation they can be used as the rowPtr/colPtr array of a CSR/CSC matrix
        @inbounds offsets[leading_dim+1] = accum
    end
end

# `positions` is the inclusive prefix count of output entries, so an output entry for `row`
# goes to `positions[row]`.
@kernel function sparse_to_sparse_broadcast_kernel(f::F, output::GPUSparseVector{Tv,Ti},
                                                   offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                                   positions::AbstractVector{Ti},
                                                   args...) where {Tv, Ti, N, F}
    row = @index(Global, Linear)
    if row ≤ length(offsets)
        key, arg_ptrs = @inbounds offsets[row]
        if key != typemax(Ti)
            vals = ntuple(Val(N)) do i
                @inline
                arg = @inbounds args[i]
                # ptr is 0 if the sparse vector doesn't have an element at this row
                # ptr is 0 if the arg is a scalar AND f preserves zeros
                ptr = @inbounds arg_ptrs[i]
                _getindex(arg, row, ptr)
            end
            output_ix = @inbounds positions[row]
            @inbounds output.nzInd[output_ix]  = row
            @inbounds output.nzVal[output_ix] = f(vals...)
        end
    end
end

@kernel function sparse_to_sparse_broadcast_kernel(f, output::T, offsets::Union{<:AbstractArray,Nothing},
                                                   args...) where {Ti, T<:Union{GPUSparseMatrixCSR{<:Any,Ti},
                                                                                GPUSparseMatrixCSC{<:Any,Ti}}}
    # every thread processes an entire row
    leading_dim = @index(Global, Linear)
    leading_dim_size = output isa GPUSparseMatrixCSR ? size(output, 1) : size(output, 2)
    if leading_dim ≤ leading_dim_size
        iter = @inbounds iter_type(T, Ti)(leading_dim, args...)

        output_ptrs  = output isa GPUSparseMatrixCSR ? output.rowPtr : output.colPtr
        output_ivals = output isa GPUSparseMatrixCSR ? output.colVal : output.rowVal
        # fetch the row offset, and write it to the output
        @inbounds begin
            output_ptr = output_ptrs[leading_dim] = offsets[leading_dim]
            if leading_dim == leading_dim_size
                output_ptrs[leading_dim+one(eltype(leading_dim))] = offsets[leading_dim+one(eltype(leading_dim))]
            end
        end

        # set the values for this row
        for (sub_leading_dim, ptrs) in iter
            index_first  = output isa GPUSparseMatrixCSR ? leading_dim : sub_leading_dim
            index_second = output isa GPUSparseMatrixCSR ? sub_leading_dim : leading_dim
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
@kernel function sparse_to_dense_broadcast_kernel(T::Type{<:Union{GPUSparseMatrixCSR{Tv, Ti},
                                                                  GPUSparseMatrixCSC{Tv, Ti}}},
                                                  f, output::AbstractArray, args...) where {Tv, Ti}
    # every thread processes an entire row
    leading_dim = @index(Global, Linear)
    leading_dim_size = T <: GPUSparseMatrixCSR ? size(output, 1) : size(output, 2)
    if leading_dim ≤ leading_dim_size
        iter = @inbounds iter_type(T, Ti)(leading_dim, args...)

        # set the values for this row
        for (sub_leading_dim, ptrs) in iter
            index_first  = T <: GPUSparseMatrixCSR ? leading_dim : sub_leading_dim
            index_second = T <: GPUSparseMatrixCSR ? sub_leading_dim : leading_dim
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

@kernel function sparse_to_dense_broadcast_kernel(::Type{<:GPUSparseVector}, f::F,
                                                  output::AbstractArray{Tv},
                                                  offsets::AbstractVector{Pair{Ti, NTuple{N, Ti}}},
                                                  args...) where {Tv, F, N, Ti}
    row = @index(Global, Linear)
    if row ≤ length(output)
        arg_ptrs = @inbounds offsets[row][2]
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
        arg isa GPUSparseArray
    end
    sparse_types = unique(map(i->nameof(typeof(bc.args[i])), sparse_args))
    if length(sparse_types) > 1
        error("broadcast with multiple types of sparse arrays ($(join(sparse_types, ", "))) is not supported")
    end
    sparse_typ = typeof(bc.args[first(sparse_args)])
    sparse_typ <: Union{GPUSparseMatrixCSR,GPUSparseMatrixCSC,GPUSparseVector} ||
        error("broadcast with sparse arrays is currently only implemented for vectors and CSR and CSC matrices")
    Ti = if sparse_typ <: GPUSparseMatrixCSR
        reduce(promote_type, map(i->eltype(bc.args[i].rowPtr), sparse_args))
    elseif sparse_typ <: GPUSparseMatrixCSC
        reduce(promote_type, map(i->eltype(bc.args[i].colPtr), sparse_args))
    elseif sparse_typ <: GPUSparseVector
        reduce(promote_type, map(i->eltype(bc.args[i].nzInd), sparse_args))
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
    offsets = nothing
    positions = nothing
    # allocate the output container
    sparse_arg = bc.args[first(sparse_args)]
    if !fpreszeros && sparse_typ <: Union{GPUSparseMatrixCSR, GPUSparseMatrixCSC}
        # either we have dense inputs, or the function isn't preserving zeros,
        # so use a dense output to broadcast into.
        val_array = nonzeros(sparse_arg)
        output    = similar(val_array, Tv, size(bc))
        # since we'll be iterating the sparse inputs, we need to pre-fill the dense output
        # with appropriate values (while setting the sparse inputs to zero). we do this by
        # re-using the dense broadcast implementation.
        nonsparse_args = map(bc.args) do arg
            # NOTE: this assumes the broadcast is flattened, but not yet preprocessed
            if arg isa GPUSparseArray
                zero(eltype(arg))
            else
                arg
            end
        end
        broadcast!(bc.f, output, nonsparse_args...)
    elseif length(sparse_args) == 1 && sparse_typ <: Union{GPUSparseMatrixCSR, GPUSparseMatrixCSC}
        # we only have a single sparse input, so we can reuse its structure for the output.
        # this avoids a kernel launch and costly synchronization.
        if sparse_typ <: GPUSparseMatrixCSR
            offsets = rowPtr = copy(sparse_arg.rowPtr)
            colVal  = similar(sparse_arg.colVal)
            nzVal   = similar(sparse_arg.nzVal, Tv)
            output  = GPUSparseMatrixCSR(rowPtr, colVal, nzVal, size(bc))
        elseif sparse_typ <: GPUSparseMatrixCSC
            offsets = colPtr = copy(sparse_arg.colPtr)
            rowVal  = similar(sparse_arg.rowVal)
            nzVal   = similar(sparse_arg.nzVal, Tv)
            output  = GPUSparseMatrixCSC(colPtr, rowVal, nzVal, size(bc))
        end
    else
        # determine the number of non-zero elements per row so that we can create an
        # appropriately-structured output container
        offsets = if sparse_typ <: GPUSparseMatrixCSR
            ptr_array = sparse_arg.rowPtr
            fill!(similar(ptr_array, Ti, rows+1), one(Ti))
        elseif sparse_typ <: GPUSparseMatrixCSC
            ptr_array = sparse_arg.colPtr
            fill!(similar(ptr_array, Ti, cols+1), one(Ti))
        elseif sparse_typ <: GPUSparseVector
            ptr_array = sparse_arg.nzInd
            similar(ptr_array, Pair{Ti, NTuple{length(bc.args), Ti}}, rows)
        end
        let
            args = if sparse_typ <: GPUSparseVector
                (sparse_typ, fpreszeros, offsets, bc.args...)
            else
                (sparse_typ, offsets, bc.args...)
            end
            kernel = compute_offsets_kernel(get_backend(bc.args[first(sparse_args)]))
            # an empty launch fails on Metal (JuliaGPU/Metal.jl#980)
            isempty(offsets) || kernel(args...; ndrange=length(offsets))
        end
        # accumulate these values so that we can use them directly as row pointer offsets,
        # as well as to get the total nnz count to allocate the sparse output array.
        # cusparseXcsrgeam2Nnz computes this in one go, but it doesn't seem worth the effort
        if !(sparse_typ <: GPUSparseVector)
            accumulate!(Base.add_sum, offsets, offsets)
            total_nnz = @allowscalar offsets[end] - 1
        elseif fpreszeros
            # number the rows that have an output entry, which gives each its output position
            positions = similar(offsets, Ti)
            positions .= first.(offsets) .!= typemax(Ti)
            rows == 0 || accumulate!(Base.add_sum, positions, positions)
            total_nnz = rows == 0 ? 0 : Int(@allowscalar positions[end])
        end
        output = if sparse_typ <: Union{GPUSparseMatrixCSR,GPUSparseMatrixCSC}
            ixVal = similar(offsets, Ti, total_nnz)
            nzVal = similar(offsets, Tv, total_nnz)
            sparse_format_type(sparse_arg)(offsets, ixVal, nzVal, size(bc))
        elseif sparse_typ <: GPUSparseVector && !fpreszeros
            val_array = bc.args[first(sparse_args)].nzVal
            similar(val_array, Tv, size(bc))
        elseif sparse_typ <: GPUSparseVector && fpreszeros
            nzInd  = similar(offsets, Ti, total_nnz)
            nzVal  = similar(offsets, Tv, total_nnz)
            GPUSparseVector(nzInd, nzVal, rows)
        end
        if sparse_typ <: GPUSparseVector && !fpreszeros
            nonsparse_args = map(bc.args) do arg
                # NOTE: this assumes the broadcst is flattened, but not yet preprocessed
                if arg isa GPUSparseArray
                    zero(eltype(arg))
                else
                    arg
                end
            end
            broadcast!(bc.f, output, nonsparse_args...)
        end
    end
    # perform the actual broadcast
    if output isa GPUSparseArray
        args   = sparse_typ <: GPUSparseVector ? (bc.f, output, offsets, positions, bc.args...) :
                                                         (bc.f, output, offsets, bc.args...)
        kernel = sparse_to_sparse_broadcast_kernel(get_backend(bc.args[first(sparse_args)]))
        ndrange = if sparse_typ <: GPUSparseVector
                    rows
                  elseif sparse_typ <: GPUSparseMatrixCSC
                    size(output, 2)
                  else
                     size(output, 1)
                  end
    else
        args   = sparse_typ <: GPUSparseVector ? (sparse_typ, bc.f, output, offsets, bc.args...) :
                                                         (sparse_typ, bc.f, output, bc.args...)
        kernel = sparse_to_dense_broadcast_kernel(get_backend(bc.args[first(sparse_args)]))
        ndrange = sparse_typ <: GPUSparseMatrixCSC ? size(output, 2) : size(output, 1)
    end
    # an empty launch fails on Metal (JuliaGPU/Metal.jl#980)
    ndrange == 0 || kernel(args...; ndrange)
    return output
end
