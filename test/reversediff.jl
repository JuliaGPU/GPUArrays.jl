using Flux.Tracker
using GPUArrays
opencl(GPUArrays.is_gpu)

predict(x, W, b) = W*x .+ b
meansquarederror(ŷ, y) = sum((ŷ .- y).^2) ./ Float32(size(y, 2))
loss(x, y, W, b) = meansquarederror(predict(x, W, b), y)
function update!(ps, η = 0.1f0)
  for w in ps
    w.x .-= w.Δ .* η
    w.Δ .= 0f0
  end
end

w, h = 13, 503
init_b, init_w = [0f0], rand(Float32, 1, w)

b = track(copy(init_b))
W = track(copy(init_w))
t1 = rand(Float32, w, h)
t2 = rand(Float32, 1, h)


bg = track(GPUArray(init_b))
Wg = track(GPUArray(init_w))
t1g = GPUArray(t1)
t2g = GPUArray(t2)



for i =  1:3
  t3 = loss(t1, t2, W, b)
  @show t3.x
  @show t3.Δ
  @show b.Δ
  @show b.x
  back!(t3)
  update!((W, b))
  @show b.Δ
  @show b.x
  @show t3.x
  @show t3.Δ

  println("________________________")
  t3g = loss(t1g, t2g, Wg, bg)
  @show t3g.x
  @show t3g.Δ
  @show bg.Δ
  @show bg.x
  back!(t3g)
  update!((Wg, bg))
  @show bg.Δ
  @show bg.x
  @show t3g.x
  @show t3g.Δ
  println()
  @show Array(Wg.x) ≈ W.x
  println("##############################")
end


return $(Expr(:new, Type{CartesianIndex{2}},
:((Base.convert)(Type{Tuple{Int64,Int64}}, index))))

using Transpiler, Sugar

m = Transpiler.CLMethod((+, Tuple{CartesianIndex{2}, Int}))
getast!(m)
ast = Sugar.sugared(m.signature..., code_typed)
func = ast.args[6].args[1].args[1]
args = ast.args[6].args[1].args[2:end]
types = (map(x-> Sugar.expr_type(m, x), args)...)
FT = Sugar.expr_type(m, func)
return_type = Sugar.expr_type(m, ast.args[6].args[1])
Sugar.resolve_func(m, func)

f = if isclosure(FT)
    insert!(args, 2, func) # add self reference to call
    FT
else

end
