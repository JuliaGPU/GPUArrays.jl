info("""
This process will figure out which acceleration Packages you have installed
and therefore which backends GPUArrays can offer.
Theoretically available:
:cudanative, :julia, :opencl

:julia is the default backend, which should always work.
Just start Julia with:
`JULIA_NUM_THREADS=8 julia -O3` to get it some threads.
8 is just an example and should be chosen depending on the processor you have.
`-O3` is completely optional, but when you're already fishing for multhithreaded
acceleration, you might as well want optimization level 3!
In the future, OpenCL, CUDA and OpenGL will be added as another backend.
""")

supported_backends = [:julia]

cudanative_dir = get(ENV, "CUDANATIVE_PATH", Pkg.dir("CUDAnative"))
install_cudanative = true

if !isdir(cudanative_dir)
    info("""
    Not installing CUDAnative backend. If you've installed CUDAnative.jl not in the
    default location, consider building GPUArrays like this:
    ```
    ENV[CUDANATIVE_PATH] = "path/to/CUDAnative/"
    Pkg.build("GPUArrays")
    ```
    If not installed, you can get CUDAnative like this:
    ```
    Install CUDA runtime
    Build Julia from the branch: tb/cuda.
    Then:
    Pkg.clone("https://github.com/JuliaGPU/CUDAnative.jl.git") #
    Pkg.test("CUDAnative")
    Pkg.checkout("CUDAdrv")
    Pkg.checkout("LLVM")
    ```
    """)
    install_cudanative = false
end

# Julia will always be available
info("julia added as a backend.")

install_cudanative = try
    using CUDAnative
    true
catch e
    info("CUDAnative doesn't seem to be usable and it won't be installed as a backend. Error: $e")
    info("If error fixed, try Pkg.build(\"GPUArrays\") again!")
    false
end
if install_cudanative
    info("cudanative added as backend!")
    push!(supported_backends, :cudanative)
end

try
    using OpenCL
    if is_apple() && isempty(cl.devices(:gpu))
        error("You're using OpenCL CPU implementation on OSX, which currently has errors that must be fixed")
    end
    device, ctx, queue = cl.create_compute_context()
    info("OpenCL added as backend!")
    push!(supported_backends, :opencl)
    true
catch e
    info("OpenCL not usable. Please install drivers and add OpenCL.jl: $e")
    false
end

# TODO add back OpenGL backend.. Currently not supported due too many driver bugs
# in OpenGL implementation
try
    using GLAbstraction, GLWindow
    # we need at least OpenGL 4.1
    ctx = GLWindow.create_glcontext("test", resolution = (10, 10), major = 3, minor = 3)
    if ctx.handle != C_NULL
        info("opengl added as backend!")
        push!(supported_backends, :opengl)
    else
        error("Not a high enough version of OpenGL available. Try upgrading the video driver!")
    end
catch e
    info("OpenGL not added as backend: $e")
end


supported_blas_libs = [:BLAS]
supported_fft_libs = [:FFT]
if :opencl in supported_backends
    try
        import CLBLAS
        push!(supported_blas_libs, :CLBLAS)
    catch e
        info("import of CLBLAS did not work, not added")
    end
    try
        import CLFFT
        push!(supported_fft_libs, :CLFFT)
    catch e
        info("import of CLFFT did not work, not added")
    end
end
if :cudanative in supported_backends
    try
        import CUBLAS
        push!(supported_blas_libs, :CUBLAS)
    catch e
        info("import of CUBLAS did work, not added")
    end
    try
        import CUFFT
        push!(supported_fft_libs, :CUFFT)
    catch e
        info("import of CUFFT did work, not added")
    end
end

file = joinpath(dirname(@__FILE__), "..", "src", "backends", "supported_backends.jl")

open(file, "w") do io
    backendstr = join(map(s-> string(":", s), supported_backends), ", ")
    println(io, "supported_backends() = ($backendstr,)")
    backendstr = join(map(s-> string(':', s), supported_blas_libs), ", ")
    println(io, "supported_blas_libs() = ($backendstr,)")
    backendstr = join(map(s-> string(':', s), supported_fft_libs), ", ")
    println(io, "supported_fft_libs() = ($backendstr,)")
    println(io, """
    is_backend_supported(sym::Symbol) = sym in supported_backends()
    is_blas_supported(sym) = sym in supported_blas_libs()
    is_fft_supported(sym) = sym in supported_fft_libs()
    """)
    for elem in supported_backends
        str = string(elem)
        path = escape_string(joinpath(str, str*".jl"))
        println(io, "include(\"$path\")")
    end
end
