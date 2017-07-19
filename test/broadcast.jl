using GPUArrays
JLBackend.init()

ac = rand(Float32, 700)
bc = rand(Float32, 700)
outc = similar(ac)
a = GPUArray(ac)
b = GPUArray(bc)
out = GPUArray(outc)
function test(out, a, b)
    (out .= a .+ b)
end

test(out, a, b)

using BenchmarkTools
b1 = @benchmark test($out, $a, $b)
b2 = @benchmark test($outc, $ac, $bc)
judge(minimum(b1), minimum(b2))
Profile.clear()
@profile gpu_call(kernel, out, (out, a, b))
Profile.print()

@time gpu_call(kernel, out, ((out, a, b)))
@time test(outc, ac, bc)
@profile test(outc, ac, bc)
Profile.clear()
@profile test(outc, ac, bc)
Profile.print()
