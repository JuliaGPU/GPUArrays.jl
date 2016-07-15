using BenchmarkTools
using GPUArrays, ArrayFire
const g = GPUArrays
g.opencl_init()
function bench_op(op, A, B)
  op.(A, B)
end
function bench_op{T, N}(op, out, A::g.CLArray{T, N}, B::g.CLArray{T, N})
  broadcast!(op, out, A, B)
  out
end
function bench_op{T, N}(op, A::AFArray{T, N}, B::AFArray{T, N})
  r = op(A, B)
  ArrayFire.eval(r)
  ArrayFire.sync()
end
function bench_op{T, N}(op::typeof(/), A::AFArray{T, N}, B::AFArray{T, N})
  r = A ./ B
  ArrayFire.eval(r)
  ArrayFire.sync()
end
function bench_op{T, N}(op::typeof(*), A::AFArray{T, N}, B::AFArray{T, N})
  r = A .* B
  ArrayFire.eval(r)
  ArrayFire.sync()
end
function bench(N)
  a = rand(Float32, N);
  b = rand(Float32, N);
  cl_a = g.CLArray(a);
  cl_b = g.CLArray(b);
  cl_out = g.CLArray(b, :w);
  af_a = AFArray(a);
  af_b = AFArray(b);

  for op in (+, max, min, -, /, *)
    println("############################################")
    @time(bench_op(op, a, b))
    @time(bench_op(op, cl_out, cl_a, cl_b))
    @time(bench_op(op, af_a, af_b))
    println("############################################")
  end
end
