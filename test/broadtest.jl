# f.(args...) syntax (#15032)
let x = [1,3.2,4.7], y = [3.5, pi, 1e-4], α = 0.2342
    @test sin.(x) == broadcast(sin, x)
    @test sin.(α) == broadcast(sin, α)
    @test sin.(3.2) == broadcast(sin, 3.2) == sin(3.2)
    @test factorial.(3) == broadcast(factorial, 3)
    @test atan2.(x, y) == broadcast(atan2, x, y)
    @test atan2.(x, y') == broadcast(atan2, x, y')
    @test atan2.(x, α) == broadcast(atan2, x, α)
    @test atan2.(α, y') == broadcast(atan2, α, y')
end


# PR #17300: loop fusion
@test Array((x->x+1).((x->x+2).((x->x+3).(GPUArray(1:10))))) == collect(7:16)
let A = GPUArray([sqrt(i)+j for i = 1:3, j=1:4])
    @test cu.atan2.(log.(A), sum(A,1)) == broadcast(atan2, broadcast(log, A), sum(A, 1))
end
let x = sin.(1:10)
    @test atan2.((x->x+1).(x), (x->x+2).(x)) == broadcast(atan2, x+1, x+2) == broadcast(atan2, x.+1, x.+2)
    @test sin.(atan2.([x+1,x+2]...)) == sin.(atan2.(x+1,x+2))
    @test sin.(atan2.(x, 3.7)) == broadcast(x -> sin(atan2(x,3.7)), x)
    @test atan2.(x, 3.7) == broadcast(x -> atan2(x,3.7), x) == broadcast(atan2, x, 3.7)
end

# fusion with splatted args:
let x = sin.(1:10), a = [x]
    @test cos.(x) == cos.(a...)
    @test atan2.(x,x) == atan2.(a..., a...) == atan2.([x, x]...)
    @test atan2.(x, cos.(x)) == atan2.(a..., cos.(x)) == broadcast(atan2, x, cos.(a...)) == broadcast(atan2, a..., cos.(a...))
    @test ((args...)->cos(args[1])).(x) == cos.(x) == ((y,args...)->cos(y)).(x)
end
@test atan2.(3,4) == atan2(3,4) == (() -> atan2(3,4)).()
# fusion with keyword args:
let x = [1:4;]
    f17300kw(x; y=0) = x + y
    @test f17300kw.(x) == x
    @test f17300kw.(x, y=1) == f17300kw.(x; y=1) == f17300kw.(x; [(:y,1)]...) == x .+ 1
    @test f17300kw.(sin.(x), y=1) == f17300kw.(sin.(x); y=1) == sin.(x) .+ 1
    @test sin.(f17300kw.(x, y=1)) == sin.(f17300kw.(x; y=1)) == sin.(x .+ 1)
end
