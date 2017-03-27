
using CUDArt, ModernGL
import CUDAdrv, CUDAnative, GLAbstraction
const cu = CUDArt.rt


cu_register(buff::GPUArray) = cu_register(buffer(buff))

let graphic_resource_map_buff = Dict{UInt32, Ref{cu.cudaGraphicsResource_t}}(), graphic_resource_map_tex = Dict{UInt32, Ref{cu.cudaGraphicsResource_t}}()
    function cu_register(
            buff::GLAbstraction.GLBuffer,
            flag=cu.cudaGraphicsMapFlagsNone
        )
        get!(graphic_resource_map_buff, buff.id) do
            graphics_ref = Ref{cu.cudaGraphicsResource_t}()
            cu.cudaGraphicsGLRegisterBuffer(
                graphics_ref, buff.id, flag
            )
            graphics_ref
        end
    end
    function cu_register(
            tex::GLAbstraction.Texture,
            flag=cu.cudaGraphicsMapFlagsNone
        )
        get!(graphic_resource_map_tex, tex.id) do
            graphics_ref = Ref{cu.cudaGraphicsResource_t}()
            cu.cudaGraphicsGLRegisterImage(
                graphics_ref, tex.id,
                tex.texturetype, flag
            )
            graphics_ref
        end
    end
end

function cu_map(f, buff::GPUArray)
    cu_array = cu_map(buff)
    f(cu_array)
    cu_unmap(buff)
    nothing
end

function cu_map{T, N}(A::GLBackend.GLArrayBuff{T, N}, graphics_ref=cu_register(A), ctx=CUBackend.current_context())
    cu.cudaGraphicsMapResources(1, graphics_ref, C_NULL)
    # Get a cuda device pointer to the opengl buffer memory location
    cubuff = Ref{Ptr{Void}}()
    _size = Ref{Csize_t}()
    cu.cudaGraphicsResourceGetMappedPointer(
        cubuff, _size,
        graphics_ref[],
    )
    cudevptr = Base.unsafe_convert(CUDAdrv.DevicePtr{T}, Ptr{T}(cubuff[]))
    sz = size(A)
    cuarr = CUDAdrv.CuArray{T}(sz, cudevptr)
    GPUArray(cuarr, sz, context=ctx)
end

function cu_map{T, N}(A::GLBackend.GLArrayTex{T, N}, graphics_ref=cu_register(A), ctx=CUBackend.current_context())
    cu.cudaGraphicsMapResources(1, graphics_ref, C_NULL)
    cubuff = Ref{Ptr{Void}}()
    cu.cudaGraphicsSubResourceGetMappedArray(
        cubuff,
        graphics_ref[],
        0, 0
    )
    cudevptr = Base.unsafe_convert(CUDAdrv.DevicePtr{T}, Ptr{T}(cubuff[]))
    sz = size(A)
    cuarr = CUDAdrv.CuArray{T}(sz, cudevptr)
    GPUArray(cuarr, sz, context=ctx)
end

function cu_unmap{T, N}(A::GLBackend.GLArray{T, N}, graphics_ref=cu_register(A))
    cu.cudaGraphicsUnmapResources(1, graphics_ref, C_NULL)
end






















# type CUDAGLBuffer{T} <: GPUArray{T, 1}
#     buffer::GLBuffer{T}
#     graphics_resource::Ref{cu.cudaGraphicsResource_t}
#     ismapped::Bool
# end
#
# function CUDAGLBuffer(buffer::GLBuffer, flag = 0)
#     cuda_resource = Ref{cu.cudaGraphicsResource_t}(C_NULL)
#     cu.cudaGraphicsGLRegisterBuffer(cuda_resource, buffer.id, flag)
#     CUDAGLBuffer(buffer, cuda_resource, false)
# end
# function map_resource(buffer::CUDAGLBuffer)
#     if !buffer.ismapped
#         cu.cudaGraphicsMapResources(1, buffer.graphics_resource, C_NULL)
#         buffer.ismapped = true;
#     end
#     nothing
# end
#
# function unmap_resource(buffer::CUDAGLBuffer)
#     if buffer.ismapped
#         cu.cudaGraphicsUnmapResources(1, buffer.graphics_resource, C_NULL)
#         buffer.ismapped = false
#     end
#     nothing
# end
#
# function copy_from_device_pointer{T}(
#         cuda_mem_ptr::Ptr{T},
#         cuda_gl_buffer::CUDAGLBuffer,
#     )
#     map_resource(cuda_gl_buffer)
#     buffersize = length(cuda_gl_buffer.buffer)*sizeof(eltype(cuda_gl_buffer.buffer))
#     if cuda_gl_buffer.buffer.buffertype == GL_RENDERBUFFER
#         array_ptr = Ref{cu.cudaArray_t}(C_NULL)
#         cu.cudaGraphicsSubResourceGetMappedArray(array_ptr, cuda_gl_buffer.graphics_resource[], 0, 0)
#         cu.cudaMemcpyToArray(array_ptr[], 0, 0, cuda_mem_ptr, buffersize, cu.cudaMemcpyDeviceToDevice)
#     else
#         opengl_ptr = Ref{Ptr{Void}}(C_NULL); size_ref = Ref{Csize_t}(buffersize)
#         cu.cudaGraphicsResourceGetMappedPointer(opengl_ptr, size_ref, cuda_gl_buffer.graphics_resource[])
#         cu.cudaMemcpy(opengl_ptr[], cuda_mem_ptr, buffersize, cu.cudaMemcpyDeviceToDevice)
#     end
#     unmap_resource(cuda_gl_buffer)
# end
#
# """
#  Gets the device pointer from the mapped resource
#  Sets is_mapped to true
# """
# function copy_to_device_pointer{T}(
#         cuda_mem_ptr::Ptr{T},
#         cuda_gl_buffer::CUDAGLBuffer,
#     )
#     map_resource(cuda_gl_buffer)
#     is_mapped = true
#     buffersize = length(cuda_gl_buffer.buffer)*sizeof(eltype(cuda_gl_buffer.buffer))
#     if cuda_gl_buffer.buffer.buffertype == GL_RENDERBUFFER
#         array_ptr = Ref{cu.cudaArray_t}(C_NULL);
#         cu.cudaGraphicsSubResourceGetMappedArray(array_ptr, cuda_gl_buffer.graphics_resource[], 0, 0)
#         cu.cudaMemcpyFromArray(cuda_mem_ptr, array_ptr[], 0, 0, buffersize, cu.cudaMemcpyDeviceToDevice)
#     else
#         opengl_ptr = Ref{Ptr{Void}}(C_NULL); size_ref = Ref{Csize_t}(buffersize)
#         cu.cudaGraphicsResourceGetMappedPointer(opengl_ptr, size_ref, cuda_gl_buffer.graphics_resource[])
#         cu.cudaMemcpy(cuda_mem_ptr, opengl_ptr, buffersize, cu.cudaMemcpyDeviceToDevice)
#     end
#     unmap_resource(cuda_gl_buffer)
# end
#
#
#
# function register_with_cuda{T, ND}(tex::GPUArray{Texture{T, ND}}, image::CUImage)
#     graphics_ref = Ref{cu.cudaGraphicsResource_t}()
#     cuGraphicsGLRegisterImage(
#         graphics_ref,
#         tex.id,
#         tex.texturetype,
#         CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST
#     )
#     cuGraphicsMapResources(1, graphics_ref, C_NULL);
#
#     # Bind the volume textures to their respective cuda arrays.
#     # Get a cuda device pointer to the opengl buffer memory location
#     cubuff = Ref{DevicePtr{T}}()
#     cuGraphicsSubResourceGetMappedArray(
#         cubuff,
#         graphics_ref,
#         0, 0
#     )
#     cuGraphicsUnmapResources(1, graphics_ref, C_NULL)
#     cubuff = CuArray(cubuff[], size(tex), length(tex))
#     GPUArray(cubuff, size(tex), cuctx)
# end
#
#
# function register_with_cuda{T}(
#         glbuffer::GPUArray{GLBuffer{T}};
#         cuctx = first(cuda_contexts()),
#         flag=cu.cudaGraphicsMapFlagsNone
#     )
#     graphics_ref = Ref{cu.cudaGraphicsResource_t}()
#     cu.cudaGraphicsGLRegisterBuffer(
#         graphics_ref, buffer(glbuffer).id, flag
#     )
#     cu.cudaGraphicsMapResources(1, graphics_ref, C_NULL);
#     # Get a cuda device pointer to the opengl buffer memory location
#     cubuff = Ref{Ptr{Void}}()
#     sizeta = Ref{Csize_t}()
#     cu.cudaGraphicsResourceGetMappedPointer(
#         cubuff,
#         sizeta,
#         graphics_ref[],
#     )
#
#     cudevptr = Base.unsafe_convert(DevicePtr{T}, Ptr{T}(cubuff[]))
#     sz = size(glbuffer)
#     cuarr = CuArray{T}(sz, cudevptr)
#     len = length(glbuffer)
#     threads = min(len, 1024)
#     blocks = floor(Int, len/threads)
#     @cuda (blocks, threads) kernel_vadd(cuarr)
#     cu.cudaGraphicsUnmapResources(1, graphics_ref, C_NULL)
#     cuarr
# end
#
