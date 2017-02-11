typealias AccVecOrMat{T} Union{AbstractAccArray{T, 1}, AbstractAccArray{T, 2}}

# all backends need to define a blas_module function to map to the correct library
blas_module(A::AccVecOrMat) = blas_module(context(A))

# Interface that needs to be overwritten by backend
# Slightly difference behavior from buffer, since not all blas backends work directly with
# the gpu array buffer
function blasbuffer(ctx, A)
    error("$ctx doesn't support BLAS operations with $A")
end

for T in (Float32, Float64, Complex64, Complex128)
    @eval begin
        function Base.BLAS.gemm!(
                transA::Char, transB::Char, alpha::$T,
                A::AccVecOrMat{$T}, B::AccVecOrMat{$T},
                beta::$T, C::AccVecOrMat{$T}
            )
            ctx = context(A)
            blasmod = blas_module(ctx)
            result = blasmod.gemm!(
                transA, transB, alpha,
                blasbuffer(ctx, A), blasbuffer(ctx, B), beta, blasbuffer(ctx, C)
            )
            typeof(A)(result)
        end
    end
end
