"Automatic sparse algorithm selection, including supported vendor implementations."
struct Automatic end
"Portable sparse execution, including all subordinate sparse conversions."
struct Portable end
"Require vendor execution; reject unsupported inputs before modifying the destination."
struct Vendor end

struct UnsupportedSparseVendor
    reason::String
end

# Extensions return an already selected executor, with any vendor algorithm encoded in it.
# They must check types, both index widths, layout, placement and library capabilities.
vendor_spmv(y, tA, A, x, α, β) = UnsupportedSparseVendor("no supported sparse vendor extension for these operands")
vendor_spmm(C, tA, tB, A, B, α, β) = UnsupportedSparseVendor("no supported sparse vendor extension for these operands")
vendor_spgemm(A, B) = UnsupportedSparseVendor("no supported sparse vendor product for these operands")

select_sparse(::Portable, candidate) = Portable()
select_sparse(::Automatic, candidate::UnsupportedSparseVendor) = Portable()
select_sparse(::Automatic, candidate) = candidate
select_sparse(::Vendor, candidate::UnsupportedSparseVendor) = throw(ArgumentError(candidate.reason))
select_sparse(::Vendor, candidate) = candidate

function check_sparse_product(C, tA, tB, A, B)
    operand_terms(tA); operand_terms(tB)
    tA in ('S', 's', 'H', 'h') && LinearAlgebra.checksquare(A)
    tB in ('S', 's', 'H', 'h') && LinearAlgebra.checksquare(B)
    check_product_dims(size(C), op_size(A, tA), op_size(B, tB))
    check_storage_compatibility(map(storage_parent, (C, product_buffers(A)..., product_buffers(B)...))...)
    for input in (A, B)
        any(buffer -> Base.mightalias(C,buffer),product_buffers(input)) &&
            throw(ArgumentError("sparse mul! requires a destination independent of its inputs"))
    end
    return nothing
end

product_buffers(A::GPUSparseArray) = sparse_buffers(A)
product_buffers(A::AbstractGPUSparseArray) = (nonzeros(A),)
product_buffers(A::AbstractArray) = (A,)

const GPUSparseRootOperand = Union{AbstractGPUSparseMatrix,
    Transpose{<:Any,<:AbstractGPUSparseMatrix},Adjoint{<:Any,<:AbstractGPUSparseMatrix}}
# Preserve the wrapper's chosen triangle. LinearAlgebra's two-argument fallback can
# wrap an already triangular diagonal operand again and miss the vendor executor.
LinearAlgebra.ldiv!(A::LinearAlgebra.UpperOrLowerTriangular{<:Any,<:GPUSparseRootOperand},
                   B::Union{AnyGPUVector,AnyGPUMatrix}) = LinearAlgebra._ldiv!(B,A,B)

"""
    GPUArrays.spmv!(y, tA, A, x, α, β; algorithm=GPUArrays.Automatic())

Compute `y = α op(A) x + β y`. `tA` is a LinearAlgebra operand character.
Use `Portable()` to bypass vendor execution, or `Vendor()` to require it.
Validation and selection finish before output writes. An execution error is propagated;
it never triggers a retry against a potentially modified output.
"""
function spmv!(y, tA::AbstractChar, A::AbstractGPUSparseMatrix, x, α::Number, β::Number;
               algorithm=Automatic())
    check_sparse_product(y, tA, 'N', A, x)
    executor = algorithm isa Portable ? Portable() :
               select_sparse(algorithm, vendor_spmv(y, tA, A, x, α, β))
    return execute_spmv!(executor, y, tA, A, x, α, β)
end

"""
    GPUArrays.spmm!(C, tA, tB, A, B, α, β; algorithm=GPUArrays.Automatic())

Sparse × dense or dense × sparse matrix multiplication. Selection has the same contract
as [`spmv!`](@ref). Portable executors call portable conversions directly; they never
reenter selection for the same operation or a subordinate sparse conversion.
"""
function spmm!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number;
               algorithm=Automatic())
    check_sparse_product(C, tA, tB, A, B)
    executor = algorithm isa Portable ? Portable() :
               select_sparse(algorithm, vendor_spmm(C, tA, tB, A, B, α, β))
    return execute_spmm!(executor, C, tA, tB, A, B, α, β)
end

function execute_spmv!(::Portable, y,tA,A,x,α,β)
    A isa GPUSparseMatrix || throw(ArgumentError("portable SpMV is not implemented for this sparse format"))
    generic_spmv!(y,tA,A,x,α,β)
end
function execute_spmm!(::Portable,C,tA,tB,A,B,α,β)
    A isa GPUSparseMatrix || B isa GPUSparseMatrix ||
        throw(ArgumentError("portable SpMM is not implemented for this sparse format"))
    generic_spmm!(C,tA,tB,A,B,α,β)
end

"""
    GPUArrays.spgemm(A, B; algorithm=GPUArrays.Automatic(), max_temp_bytes=1<<30)

Sparse matrix product with central selection. The portable ESC executor rejects its
temporary-memory estimate above `max_temp_bytes`; vendor executors manage their own
workspace. Both integer widths of the output follow the input-promotion contract.
"""
function spgemm(A::GPUSparseMatrix,B::GPUSparseMatrix;algorithm=Automatic(),max_temp_bytes::Integer=1<<30)
    size(A,2) == size(B,1) || throw(DimensionMismatch("incompatible sparse product dimensions"))
    check_storage_compatibility(map(storage_parent,(sparse_buffers(A)...,sparse_buffers(B)...))...)
    executor = algorithm isa Portable ? Portable() : select_sparse(algorithm,vendor_spgemm(A,B))
    execute_spgemm(executor,A,B;max_temp_bytes)
end
execute_spgemm(::Portable,A,B;max_temp_bytes) = generic_spgemm(A,B;max_temp_bytes)

function spgemm!(C::GPUSparseMatrix,tA::AbstractChar,tB::AbstractChar,A::GPUSparseMatrix,B::GPUSparseMatrix,
                  α::Number,β::Number;algorithm=Automatic(),max_temp_bytes::Integer=1<<30)
    check_product_dims(size(C),op_size(A,tA),op_size(B,tB))
    check_sparse_mutation(C)
    P = spgemm(materialize(LinearAlgebra.wrap(A,tA)),materialize(LinearAlgebra.wrap(B,tB));
               algorithm,max_temp_bytes)
    R = iszero(β) ? α .* P : α .* P .+ β .* C
    copyto!(C,convert(matrix_format(C),R))
end

# Open sparse roots can participate through a selected vendor executor without inheriting
# traversals that only understand GPUArrays' canonical layouts.
LinearAlgebra.mul!(y::AnyGPUVector,tA::AbstractChar,A::AbstractGPUSparseMatrix,x::AnyGPUVector,α::Number,β::Number) =
    spmv!(y,tA,A,x,α,β)
LinearAlgebra.mul!(C::AnyGPUMatrix,tA::AbstractChar,tB::AbstractChar,A::AbstractGPUSparseMatrix,B::AnyGPUMatrix,α::Number,β::Number) =
    spmm!(C,tA,tB,A,B,α,β)

@static if VERSION < v"1.13.0-rc4"
    LinearAlgebra.generic_matvecmul!(C::AnyGPUVector,tA::AbstractChar,A::AbstractGPUSparseMatrix,B::AnyGPUVector,a::Number,b::Number) =
        LinearAlgebra.mul!(C,tA,A,B,a,b)
    LinearAlgebra.generic_matvecmul!(C::AnyGPUVector,tA::AbstractChar,A::AbstractGPUSparseMatrix,B::AnyGPUVector,_add::MulAddMul=MulAddMul()) =
        LinearAlgebra.mul!(C,tA,A,B,_add.alpha,_add.beta)
    LinearAlgebra.generic_matmatmul!(C::AnyGPUMatrix,tA::AbstractChar,tB::AbstractChar,A::AbstractGPUSparseMatrix,B::AnyGPUMatrix,a::Number,b::Number) =
        LinearAlgebra.mul!(C,tA,tB,A,B,a,b)
    LinearAlgebra.generic_matmatmul!(C::AnyGPUMatrix,tA::AbstractChar,tB::AbstractChar,A::AbstractGPUSparseMatrix,B::AnyGPUMatrix,_add::MulAddMul=MulAddMul()) =
        LinearAlgebra.mul!(C,tA,tB,A,B,_add.alpha,_add.beta)
end
