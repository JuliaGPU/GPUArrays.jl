using GLAbstraction, ModernGL, OpenCL



function convert(::Type{CLSampler{T, N}}, A::GLSampler; access = CL_MEM_READ_WRITE)
    rawtex = buffer(A)
    mem = clCreateFromGLTexture(context, access, rawtex.texturetype, 0, rawtex.id, C_NULL)

end
function convert(::Type{CLArray{T, N}}, A::GLArray; access = CL_MEM_READ_WRITE)
    rawbuff = buffer(A)
    mem = clCreateFromGLBuffer(context, access, pbo, C_NULL)
end
