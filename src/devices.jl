# device management and properties

"""
    device(A::AbstractArray)

Gets the device associated to the Array `A`
"""
device(A::AbstractArray) = error("Not implemented") # COV_EXCL_LINE

"""
Hardware threads of device
"""
threads(device) = error("Not implemented") # COV_EXCL_LINE
