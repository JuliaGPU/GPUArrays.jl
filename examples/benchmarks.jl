using GPUArrays

function mandelbulb{T}(x0::T, y0::T, z0::T, n, iter)
    x,y,z = x0,y0,z0
    for i=1:iter
        r = sqrt(x*x + y*y + z*z)
        theta = atan2(sqrt(x*x + y*y) , z)
        phi = atan2(y,x)
        rn = r^n
        x1 = rn * sin(theta*n) * cos(phi*n) + x0
        y1 = rn * sin(theta*n) * sin(phi*n) + y0
        z1 = rn * cos(theta*n) + z0
        (x1*x1 + y1*y1 + z1*z1) > n && return T(i)
        x,y,z = x1,y1,z1
    end
    T(iter)
end
dims = (50, 50, 50)

x, y, z = ntuple(3) do i
    # linearly spaced array (not dense) from -1 to 1
    GPUArray(reshape(linspace(-1f0, 1f0, dims[i]), ntuple(j-> j == i ? dims[i] : 1, 3)))
end
volume = GPUArray(zeros(Float32, dims))
volume .= mandelbulb.(x, y, z, Cint(8), Cint(25))
