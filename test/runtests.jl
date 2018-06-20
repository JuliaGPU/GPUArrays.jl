using GPUArrays, Base.Test
using GPUArrays: JLArray
using GPUArrays.TestSuite


@testset "Julia reference implementation:" begin
    TestSuite.run_tests(JLArray)
end

using CLArrays

using CuArrays
x = CuArray([5, 2, 3])
y = CuArray([1:5;])
A = y
I = (x,)
to_indices(A, I)
@which Base._getindex(IndexStyle(A), A, to_indices(A, I)...)
I = to_indices(A, I)
l = IndexStyle(A)
dest = similar(x)
using CLArrays
x = CLArray([5, 2, 3])
y = CLArray([1:5;])
GPUArrays.allowslow(false)
@which y[x]
A = y
I = (x,)
Base._getindex(IndexStyle(A), A, to_indices(A, I)...)
l = IndexStyle(A)
@which Base._unsafe_getindex(l, Base._maybe_reshape(l, A, I...), I...)
test(x, y) = @inbounds y[x]
test(x, y)
