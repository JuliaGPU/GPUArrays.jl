# ArrayFire.AFArray
function Base.copy(source::ArrayFire.AFArray, target::CUDAGLBuffer)
    d_ptr = ArrayFire.af_device(source)
    copy_from_device_pointer(d_ptr, target)
end
function Base.copy(source::CUDAGLBuffer, target::ArrayFire.AFArray)
    d_ptr = ArrayFire.af_device(target)
    copy_to_device_pointer(d_ptr, target)
end
