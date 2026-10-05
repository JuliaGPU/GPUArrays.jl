using Test, JLArrays, GPUArrays

@testset "JLArray alias detection" begin
    buffer = JLArray{UInt8}(undef, 32)
    left = view(buffer, 1:16)
    right = view(buffer, 17:32)
    overlap = view(buffer, 9:24)
    @test Base.dataids(left) == Base.dataids(right)
    @test Base.mightalias(left, left)
    @test !Base.mightalias(left, right)
    @test !Base.mightalias(right, left)
    @test Base.mightalias(left, overlap)
    @test Base.mightalias(overlap, left)
    @test !Base.mightalias(left, copy(left))
    # The last byte still overlaps after reinterpreting to a larger element type.
    @test Base.mightalias(reinterpret(UInt32, left), view(buffer, 16:16))
    empty = view(buffer, 9:8)
    @test !Base.mightalias(empty, left)
    @test !Base.mightalias(left, empty)
    @test !Base.mightalias(GPUArrays.derive(Nothing, buffer, (8,), 0), buffer)
end
