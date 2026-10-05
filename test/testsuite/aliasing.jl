@testsuite "aliasing" (AT, eltypes)->begin
    T = Float32 in eltypes ? Float32 : first(eltypes)
    buffer = AT{T}(undef, 32)
    left = view(buffer, 1:16)
    right = view(buffer, 17:32)
    overlap = view(buffer, 9:24)
    @test Base.mightalias(left, left)
    @test !Base.mightalias(left, right)
    @test !Base.mightalias(right, left)
    @test Base.mightalias(left, overlap)
    @test Base.mightalias(overlap, left)
    @test !Base.mightalias(left, copy(left))
    # The last element still overlaps after reinterpreting to a different element size.
    @test Base.mightalias(reinterpret(UInt8, left), view(buffer, 16:16))
    empty = view(buffer, 9:8)
    @test !Base.mightalias(empty, left)
    @test !Base.mightalias(left, empty)
end
