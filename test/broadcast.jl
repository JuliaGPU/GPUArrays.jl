using Iterators
using GPUArrays

a = rand(Float32, 4, 5, 3)
b = rand(Float32, 1, 5)
g1 = GPUArray(a);
g2 = GPUArray(b);

function test(a, b)
    a + b
end
lm = broadcast(test, g1, g2)


m = Transpiler.CLMethod((getindex, Tuple{Transpiler.cli.CLArray{Float32, 2}, Int32}))
Sugar.isintrinsic(m)
@which Transpiler.cli.clintrinsic(m.signature...)
@which Sugar.isintrinsic()
@code_warntype Base.argtail((1,))

lm = Transpiler.CLMethod((
    GPUArrays.broadcast_kernel!, (
    typeof(test),
    Transpiler.cli.CLArray{Float32,3},
    Tuple{Int32,Int32,Int32},
    Transpiler.cli.CLArray{Tuple{GPUArrays.BroadcastDescriptor{Array,3},GPUArrays.BroadcastDescriptor{Array,2}},1},
    Transpiler.cli.CLArray{Float32,3},
    Transpiler.cli.CLArray{Float32,2},
)))

lm = Transpiler.CLMethod((GPUArrays.gpu_sub2ind, (Tuple{Int, Int}, Tuple{Int, Int})))
println(Sugar.getsource!(lm))
println(code_typed(lm.signature..., optimize = false)[1])

Sugar.slotname(lm, SlotNumber(3))
ast = Sugar.getast!(lm).args
ast[11]
function print_deps(lm, level =  0)
    Sugar.isintrinsic(lm) && return
    println("    "^level, lm)
    for elem in Sugar.dependencies!(lm)
        print_deps(elem, level + 1)
    end
end
print_deps(lm)
slots = Sugar.getslots!(lm)

s2i = Transpiler.CLMethod((GPUArrays._sub2ind, (Tuple{Int32,Int32},  Int32, Int32, Int32, Int32)))

println(Sugar.has_varargs(s2i))

print_deps(s2i)
Sugar.slottype(lm, ast.args[11].args[1])
slots[9]
ast = Sugar.getsource!(lm)
println(ast)
ast.args[13].args[2].head
typeof(ast.args[1].args[1])
g3 = GPUArray(rand(Float32, 1, 5, 1)); a3 = Array(g3);

isapprox(Array(g1 .+ g3), a1 .+ a3)

out = rand(5, 5, 5)
B = rand(5, 5, 5)
A = rand(5, 5)

shape = Cint.(size(B))
keeps, Idefaults = Base.Broadcast.map_newindexer(shape, B, (A,))
args = (B, A)
Idefaults
descriptor_tuple = ntuple(length(args)) do i
    val, keep, idefault = args[i], keeps[i], Idefaults[i]
    N = length(keep)
    GPUArrays.BroadcastDescriptor{Base.Broadcast.containertype(val), N}(
        0f0,
        Cint.(size(val)),
        Cint.(keep),
        Cint.(idefault)
    )
end
descriptor_ref = [descriptor_tuple]
GPUArrays.linear_index(x) = Cint(1)
typeof(shape)
@code_warntype GPUArrays.broadcast_kernel!(test, out, shape, descriptor_ref, B, A)

code_warntype(
    GPUArrays.broadcast_kernel!, (
        typeof(test),
        Transpiler.cli.CLArray{Float32,3},
        Tuple{Int32,Int32,Int32},
        Transpiler.cli.CLArray{Tuple{GPUArrays.BroadcastDescriptor{Array,3},GPUArrays.BroadcastDescriptor{Array,2}},1},
        Transpiler.cli.CLArray{Float32,3},
        Transpiler.cli.CLArray{Float32,2},
    )#, optimize = false
)
@code_warntype(gpu_sub2ind)
