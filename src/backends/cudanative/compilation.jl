# EXCLUDE FROM TESTING
import CUDArt

# Generate a temporary file with specific suffix
# NOTE: mkstemps is glibc 2.19+, so emulate its behavior
function mkstemps(suffix::AbstractString)
    base = tempname()
    filename = base * suffix
    # make sure the filename is unique
    i = 0
    while isfile(filename)
        i += 1
        filename = base * ".$i" * suffix
    end
    return (filename, Base.open(filename, "w"))
end


type CompileError <: Base.WrappedException
    message::String
    error
end

const builddir = joinpath(@__DIR__, ".cache")

function _compile(dev, kernel, code, containing_file)
    arch = CUDArt.architecture

    if !isdir(builddir)
        println("Writing build artifacts to $builddir")
        mkpath(builddir)
    end

    # Check if we need to compile
    codehash = hex(hash(code))
    output = "$builddir/$(kernel)_$(codehash)-$(arch).ptx"
    if isfile(output)
        need_compile = (stat(containing_file).mtime > stat(output).mtime)
    else
        need_compile = true
    end

    # Compile the source, if necessary
    if need_compile
        # Write the source into a compilable file
        (source, io) = mkstemps(".cu")
        write(io, """
extern "C"
{
$code
}
""")
        close(io)

        compile_flags = vcat(CUDArt.toolchain_flags, ["--gpu-architecture", arch])
        try
            # TODO: capture STDERR
            run(pipeline(`$(CUDArt.toolchain_nvcc) $(compile_flags) -ptx -o $output $source`, stderr=DevNull))
        catch ex
            isa(ex, ErrorException) || rethrow(ex)
            rethrow(CompileError("compilation of kernel $kernel failed (typo in C++ source?)", ex))
        finally
            rm(source)
        end

        if !isfile(output)
            error("compilation of kernel $kernel failed (no output generated)")
        end
    end

    # Pass the module to the CUDA driver
    mod = try
        CUDAdrv.CuModuleFile(output)
    catch ex
        rethrow(CompileError("loading of kernel $kernel failed (invalid CUDA code?)", ex))
    end

    # Load the function pointer
    func = try
        CUDAdrv.CuFunction(mod, kernel)
    catch ex
        rethrow(CompileError("could not find kernel $kernel in the compiled binary (wrong function name?)", ex))
    end

    return func
end
