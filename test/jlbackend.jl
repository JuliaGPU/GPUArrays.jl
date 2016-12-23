using GPUArrays, BenchmarkTools
using GPUArrays.JLBackend
using Base.Test
import GPUArrays.JLBackend: JLArray, map_idx!
JLBackend.init()

t = zeros(100, 100, 100)
a = rand(100, 100, 100)
b = rand(100, 100, 100)

tc = JLArray(copy(t))
ac = JLArray(a)
bc = JLArray(b)

res1 = map!(+, t, a, b);
res2 = map!(+, tc, ac, bc);
@test buffer(res2) == res1

b1 = @benchmark map!($+, $t, $a, $b)
b2 = @benchmark map!($+, $tc, $ac, $bc)

judge(minimum(b1), minimum(b2))

tc .= (*).(ac, bc);
t .= (*).(a, b);
fft(tc);
fft!(tc);

using GeometryTypes

@inline function inner_velocity_one_form(i, velocity, idx_psi_hbar)
    idx2, psi, hbar = idx_psi_hbar
    i2 = (idx2[1][i[1]], idx2[2][i[2]], idx2[3][i[3]])
    @inbounds begin
        psi12  = psi[i[1],  i[2],  i[3]]
        psix12 = psi[i2[1], i[2],  i[3]]
        psiy12 = psi[i[1],  i2[2] ,i[3]]
        psiz12 = psi[i[1],  i[2],  i2[3]]
    end
    psi1n = Vec(psix12[1], psiy12[1], psiz12[1])
    psi2n = Vec(psix12[2], psiy12[2], psiz12[2])
    angle.(
        conj(psi12[1]) .* psi1n .+
        conj(psi12[2]) .* psi2n
    ) * hbar
end
function velocity_one_form!(velocity, psi)
    dims = size(psi)
    idx = ntuple(3) do i
        mod(1:dims[i], dims[i]) + 1
    end
    arg = (idx, psi, 1f0)
    map_idx!(inner_velocity_one_form, velocity, arg)
end

dims = (64, 32, 32)
psi1, psi2 = rand(Complex64, dims), rand(Complex64, dims);
psi = JLArray(map(identity, zip(psi1, psi2)));
velocity = JLArray(zeros(Vec3f0, dims));
v1 = velocity_one_form!(velocity, psi);

using BenchmarkTools
t1 = @benchmark velocity_one_form!($velocity, $psi);
t1
isa(velocity, JLArray{Vec3f0, 3})

methods(map_idx!)
