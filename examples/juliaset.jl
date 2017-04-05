# julia set
# (the familiar mandelbrot set is obtained by setting c==z initially)
# works only on 0.6 because of a stupid bug
@generated function julia{N}(z, maxiter::Val{N} = Val{16}())
    unrolled = Expr(:block)
    for i=1:N
        push!(unrolled.args, quote
            if abs2(z2) > 4.0
                return $(i-1)
            end
            z2 = z2 * z2 + c
        end)
    end
    quote
        c = Complex64(-0.5, 0.75)
        z2 = z
        $unrolled
        return N
    end
end
using GPUArrays

w = 2048 * 2;
h = 2048 * 2;
q = [Complex64(r, i) for i=1:-(2.0/w):-1, r=-1.5:(3.0/h):1.5];
m = similar(q, UInt8)
CLBackend.init()
mg = GPUArray(m)
qg = GPUArray(q)
mg .= julia.(qg)
using FileIO, Colors
save(Pkg.dir("GPUArrays", "examples", "juliaset.png"), Gray.(Array(mg) ./ 16.0))
