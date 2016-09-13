using GPUArrays
import GPUArrays: GPUArray, CUBackend, cu_map
using CUDAnative

cuctx = CUBackend.init()


@target ptx function mandel{T}(x0::T, y0::T, z0::T, n, iter)
    x,y,z = x0,y0,z0
    for i=1:iter
        r = CUDAnative.sqrt(x*x + y*y + z*z )
        theta = CUDAnative.atan2(CUDAnative.sqrt(x*x + y*y) , z)
        phi = CUDAnative.atan2(y,x)
        x1 = CUDAnative.pow(r, n) * CUDAnative.sin(theta*n) * CUDAnative.cos(phi*n) + x0
        y1 = CUDAnative.pow(r, n) * CUDAnative.sin(theta*n) * CUDAnative.sin(phi*n) + y0
        z1 = CUDAnative.pow(r, n) * CUDAnative.cos(theta*n) + z0
        (x1*x1 + y1*y1 + z1*z1) > n && return T(i)
        x,y,z = x1,y1,z1
    end
    T(iter)
end

dims = (100,100,100)
A = GPUArray(zeros(Float32, dims));
xrange, yrange, zrange = ntuple(i->linspace(-1f0, 1f0, dims[i]), 3)

@target ptx function eachmandel2(idx, a, x, y, z)
    @inbounds a[idx...] = mandel(x[idx[1]], y[idx[2]], z[idx[3]], 8f0, 50)
    nothing
end
CUBackend.map_eachindex(eachmandel2, A, xrange, yrange, zrange)
