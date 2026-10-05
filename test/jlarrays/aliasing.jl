using Test, JLArrays, GPUArrays

@testset "JLArray alias detection" begin
    function check_alias(A, B, expected)
        @test Base.mightalias(A, B) == expected
        @test Base.mightalias(B, A) == expected
    end

    buffer = JLArray{UInt8}(undef, 64)
    left = view(buffer, 1:16)
    right = view(buffer, 17:32)
    overlap = view(buffer, 9:24)
    inside = view(buffer, 5:8)

    # Contiguous views are dense JLArrays sharing one allocation.
    @test left isa JLArray
    @test right isa JLArray
    @test Base.dataids(left) == Base.dataids(right) == Base.dataids(buffer)
    check_alias(left, left, true)
    check_alias(left, right, false)
    check_alias(left, overlap, true)
    check_alias(right, overlap, true)
    check_alias(left, inside, true)
    check_alias(right, inside, false)
    check_alias(buffer, left, true)
    check_alias(left, copy(left), false)
    check_alias(left, reshape(left, 4, 4), true)
    check_alias(right, reshape(left, 4, 4), false)

    # Compare bytes rather than element offsets or counts after reinterpretation.
    words = reinterpret(UInt32, left)
    check_alias(words, left, true)
    check_alias(words, right, false)
    check_alias(words, overlap, true)
    check_alias(words, inside, true)
    check_alias(words, view(buffer, 16:16), true)

    # Empty ranges can start inside a nonempty array, or at an allocation boundary.
    for empty in (view(buffer, 9:8), view(buffer, 33:32), JLArray{UInt8}(undef, 0))
        check_alias(empty, buffer, false)
        check_alias(empty, left, false)
        check_alias(empty, empty, false)
    end
    check_alias(GPUArrays.derive(Nothing, buffer, (8,), 0), buffer, false)

    # Noncontiguous wrappers retain the conservative shared-storage detection.
    strided = view(reshape(buffer, 8, 8), 1:2:8, :)
    @test strided isa SubArray
    check_alias(buffer, strided, true)
    check_alias(left, strided, true)
    check_alias(copy(buffer), strided, false)

    # The broadcast regression that motivated shared-storage dataids (#716).
    A = JLArray(-ones(Float32, 3, 3))
    A .*= sign.(view(A, 1:4:9))
    @test Array(A) == ones(Float32, 3, 3)
end
