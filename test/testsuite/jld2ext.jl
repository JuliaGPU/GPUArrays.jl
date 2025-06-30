@testsuite "ext/jld2" (AT, eltypes) -> begin
    mktempdir() do dir
        for ET in eltypes
            @testset "$ET" begin
                # Test with different array sizes and dimensions
                for dims in ((2,), (2, 2), (2, 2, 2))
                    # Create a random array
                    x = AT(rand(ET, dims...))

                    # Save to a temporary file
                    file = joinpath(dir, "test.jld2")

                    # Save and load
                    JLD2.save_object(file, x)
                    y = JLD2.load_object(file)

                    # Verify the loaded array matches the original
                    @test y isa AT{ET, length(dims)}
                    @test size(y) == size(x)
                    @test x == y
                end
            end
        end
    end
end
