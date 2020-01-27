# device management and properties

export AbstractGPUDevice

abstract type AbstractGPUDevice end

"""
Hardware threads of device
"""
threads(::AbstractGPUDevice) = error("Not implemented") # COV_EXCL_LINE
