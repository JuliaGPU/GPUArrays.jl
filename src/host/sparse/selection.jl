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
    for input in (A, B)
        Base.mightalias(C, input) &&
            throw(ArgumentError("sparse mul! requires a destination independent of its inputs"))
    end
    return nothing
end

"""
    GPUArrays.spmv!(y, tA, A, x, α, β; algorithm=GPUArrays.Automatic())

Compute `y = α op(A) x + β y`. `tA` is a LinearAlgebra operand character.
Use `Portable()` to bypass vendor execution, or `Vendor()` to require it.
Validation and selection finish before output writes. An execution error is propagated;
it never triggers a retry against a potentially modified output.
"""
function spmv!(y, tA::AbstractChar, A::GPUSparseMatrix, x, α::Number, β::Number;
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

execute_spmv!(::Portable, args...) = generic_spmv!(args...)
execute_spmm!(::Portable, args...) = generic_spmm!(args...)
