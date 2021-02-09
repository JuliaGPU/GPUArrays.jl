@testsuite "uniformscaling" AT->begin
    eltypes = (ComplexF32, Float32)
    wrappers = (identity, UnitLowerTriangular, UnitUpperTriangular, LowerTriangular, UpperTriangular, Hermitian, Symmetric)

    @testcase "UniformScaling $f $T1 $T2" for T1 in eltypes, T2 in eltypes, f in wrappers
        x = ones(T1, 5, 5)
        y = AT(x)

        xw = f(x)
        yw = f(y)

        J = one(T2) * I

        # TODO: remove @allowscalar as soon as there is a proper implementation for Symmetric / Hermitian
        @test @allowscalar collect(xw + J) ≈ collect(yw + J)

        # Missing methods in Base it seems... -(x - I) can be removed when Base supports (I - x)
        @test @allowscalar collect(-(xw - J)) ≈ collect(J - yw)
    end
end
