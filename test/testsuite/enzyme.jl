using Enzyme

function scalarfirst(x)
	@allowscalar x[1]
end

@testsuite "Enzyme" (AT, eltypes)->begin
    for ET in eltypes
        T = AT{ET}
        @testset "Forward $ET" begin
	    x = T(ones(3))
	    dx = T(3*ones(3))
	    res = autodiff(Forward, scalarfirst, Duplicated(x, dx))
	    @test approx(res, 3) 
        end
    end
end
