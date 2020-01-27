# device management and properties

export AbstractGPUDevice

abstract type AbstractGPUDevice end

"""
    device(A::AbstractArray)

Gets the device associated to the Array `A`
"""
device(A::AbstractArray) = error("This array is not a GPU array") # COV_EXCL_LINE

"""
Hardware threads of device
"""
threads(::AbstractGPUDevice) = error("Not implemented") # COV_EXCL_LINE
