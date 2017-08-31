#TODO tag CUDArt#103 and use that
import CUDArt
export CompileError

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

macro compile(dev, kernel, code)
    kernel_name = string(kernel)
    containing_file = @__FILE__

    return Expr(:toplevel,
        Expr(:export,esc(kernel)),
        :($(esc(kernel)) = _compile($(esc(dev)), $kernel_name, $code, $containing_file)))
end

immutable CompileError <: Exception
    message::String
end

const builddir = joinpath(@__DIR__, ".cache")

function _compile(dev, kernel, code, containing_file)
    arch = CUDArt.architecture

    if !isdir(builddir)
        println("Writing build artifacts to $builddir")
        mkpath(builddir)
    end

    # check if we need to compile
    codehash = hex(hash(code))
    output = "$builddir/$(kernel)_$(codehash)-$(arch).ptx"
    if isfile(output)
        need_compile = (stat(containing_file).mtime > stat(output).mtime)
    else
        need_compile = true
    end

    # compile the source, if necessary
    if need_compile
        # write the source to a compilable file
        (source, io) = mkstemps(".cu")
        write(io, """
extern "C"
{
$code
}
""")
        Base.close(io)

        compile_flags = vcat(CUDArt.toolchain_flags, ["--gpu-architecture", arch])
        err = Pipe()
        cmd = `$(CUDArt.toolchain_nvcc) $(compile_flags) -ptx -o $output $source`
        result = success(pipeline(cmd; stdout=DevNull, stderr=err))
        Base.close(err.in)
        rm(source)

        errors = readstring(err)
        if !result
            throw(CompileError("compilation of kernel $kernel failed\n$errors"))
        elseif !isempty(errors)
            warn("during compilation of kernel $kernel:\n$errors")
        end

        if !isfile(output)
            error("compilation of kernel $kernel failed (no output generated)")
        end
    end

    mod = CUDAdrv.CuModuleFile(output)
    return CUDAdrv.CuFunction(mod, kernel)
end

function clean_cache()
    if ispath(builddir)
        @assert isdir(builddir)
        rm(builddir; recursive=true)
    end
end
